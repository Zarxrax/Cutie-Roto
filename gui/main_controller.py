from os import path
import logging
import re
from typing import Literal

import torch
try:
    from torch import mps
except:
    print('torch.MPS not available.')
from torch import autocast
from torchvision.transforms.functional import to_tensor
import numpy as np
from omegaconf import DictConfig, open_dict
from showinfm import show_in_file_manager
from PySide6.QtWidgets import QDialog, QMessageBox
from PySide6.QtCore import Qt
from gui.exporter_gui import Export_Dialog

from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore

from gui.interaction import ClickInteraction, Interaction
from gui.interactive_utils import torch_prob_to_numpy_mask, index_numpy_to_one_hot_torch, get_visualization, get_visualization_torch
from gui.resource_manager import ResourceManager
from gui.gui import GUI
from gui.click_controller import ClickController
from gui.reader import PropagationReader, get_data_loader
#from gui.exporter import convert_frames_to_video, convert_mask_to_binary
from scripts.download_models import download_models_if_needed

log = logging.getLogger()


class Dialog(QDialog):
    def __init__(self, parent=None, cfg=None):
        super().__init__(parent)
        self.cfg = cfg
        self.ui = Export_Dialog()
        self.ui.setupUi(self, self.cfg)

class MainController():
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.initialized = False

        # setting up the workspace
        if cfg["workspace"] is None:
            if cfg["images"] is not None:
                basename = path.basename(cfg["images"])
            elif cfg["video"] is not None:
                basename = path.basename(cfg["video"])[:-4]
            else:
                raise NotImplementedError('Either images, video, or workspace has to be specified')

            cfg["workspace"] = path.join(cfg['workspace_root'], basename)

        # reading arguments
        self.cfg = cfg
        self.num_objects = cfg['num_objects']
        self.device = cfg['device']
        self.amp = cfg['amp']

        # initializing the network(s)
        self.initialize_networks()

        # main components
        self.res_man = ResourceManager(cfg)
        self.processor = InferenceCore(self.cutie, self.cfg)
        self.gui = GUI(self, self.cfg)

        # initialize control info
        self.length: int = self.res_man.length
        self.interaction: Interaction = None
        self.interaction_type: str = 'Click'
        self.curr_ti: int = 0
        self.curr_object: int = 1
        self.propagating: bool = False
        self.propagate_direction: Literal['forward', 'backward', 'none'] = 'none'
        self.last_ex = self.last_ey = 0
        self.undo_stack = []

        # current frame info
        self.curr_frame_dirty: bool = False
        self.curr_image_np: np.ndarray = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.curr_image_torch: torch.Tensor = None
        self.curr_mask: np.ndarray = np.zeros((self.h, self.w), dtype=np.uint8)
        self.curr_prob: torch.Tensor = torch.zeros((self.num_objects + 1, self.h, self.w),
                                                   dtype=torch.float).to(self.device)
        self.curr_prob[0] = 1

        # visualization info
        self.vis_mode: str = 'overlay'
        self.vis_image: np.ndarray = None
        self.save_soft_mask: bool = True

        self.interacted_prob: torch.Tensor = None
        self.overlay_layer: np.ndarray = None
        self.overlay_layer_torch: torch.Tensor = None

        # Zoom parameters
        self.zoom_pixels = 150

        # the object id used for popup/layer overlay
        self.vis_target_objects = list(range(1, self.num_objects + 1))

        self.load_current_image_mask()
        self.show_current_frame()

        # initialize stuff
        self.update_memory_gauges()
        self.update_gpu_gauges()
        if cfg['use_long_term']:
            self.gui.work_mem_min.setValue(self.processor.memory.min_mem_frames)
            self.gui.work_mem_max.setValue(self.processor.memory.max_mem_frames)
            self.gui.long_mem_max.setValue(self.processor.memory.max_long_tokens)
        self.gui.mem_every_box.setValue(self.processor.mem_every)

        # set callbacks
        self.gui.on_mouse_motion_xy = self.on_mouse_motion_xy
        self.gui.click_fn = self.click_fn

        self.gui.showMaximized()
        self.gui.text('Initialized.')
        self.initialized = True

        # set the quality per the config
        self.on_quality_change()

        # try to load the default background layer
        if cfg.background_path:
            self._try_load_layer(cfg.background_path)
        #self.gui.set_object_color(self.curr_object)
        #self.update_config()

    def initialize_networks(self) -> None:
        download_models_if_needed()
        self.cutie = CUTIE(self.cfg).eval().to(self.device)
        model_weights = torch.load(self.cfg.weights, map_location=self.device)
        self.cutie.load_weights(model_weights)
        if self.cfg.ritm_use_anime is True:
            self.click_ctrl = ClickController(self.cfg.ritm_anime_weights, self.cfg.ritm_max_size, self.cfg.ritm_zoom_size, self.cfg.ritm_expansion_ratio, device=self.device)
        else:
            self.click_ctrl = ClickController(self.cfg.ritm_weights, self.cfg.ritm_max_size, self.cfg.ritm_zoom_size, self.cfg.ritm_expansion_ratio, device=self.device)

    def on_modelselect_change(self):
        if self.gui.comboBox_modelselect.currentText() == 'Standard':
            self.click_ctrl = ClickController(self.cfg.ritm_weights, self.cfg.ritm_max_size, self.cfg.ritm_zoom_size, self.cfg.ritm_expansion_ratio, device=self.device)
            self.gui.text('Standard segmentation model loaded.')
        else:
            self.click_ctrl = ClickController(self.cfg.ritm_anime_weights, self.cfg.ritm_max_size, self.cfg.ritm_zoom_size, self.cfg.ritm_expansion_ratio, device=self.device)
            self.gui.text('Anime segmentation model loaded.')

    def hit_number_key(self, number: int):
        if number == self.curr_object:
            return
        self.curr_object = number
        self.gui.object_dial.setValue(number)
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()
        self.gui.text(f'Current object changed to {number}.')
        self.gui.set_object_color(number)
        self.show_current_frame()

    def click_fn(self, action: Literal['left', 'right', 'middle'], x: int, y: int):
        if self.propagating:
            return

        last_interaction = self.interaction
        new_interaction = None

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            if action in ['left', 'right']:
                # left: positive click
                # right: negative click
                self.convert_current_image_mask_torch()
                image = self.curr_image_torch
                if (last_interaction is None or last_interaction.tar_obj != self.curr_object):
                    # create new interaction is needed
                    self.complete_interaction()
                    self.click_ctrl.unanchor()
                    new_interaction = ClickInteraction(image, self.curr_prob, (self.h, self.w),
                                                       self.click_ctrl, self.curr_object)
                    if new_interaction is not None:
                        self.interaction = new_interaction

                self.interaction.push_point(x, y, is_neg=(action == 'right'))
                self.interacted_prob = self.interaction.predict().to(self.device, non_blocking=True)
                self.update_interacted_mask()
                self.update_gpu_gauges()

            #elif action == 'middle':
                # middle: select a new visualization object
            #    target_object = self.curr_mask[int(y), int(x)]
            #    if target_object in self.vis_target_objects:
            #        self.vis_target_objects.remove(target_object)
            #    else:
            #        self.vis_target_objects.append(target_object)
            #    self.gui.text(f'Overlay target(s) changed to {self.vis_target_objects}')
            #    self.show_current_frame()
            #    return
            else:
                raise NotImplementedError

    def load_current_image_mask(self, no_mask: bool = False):
        self.curr_image_np = self.res_man.get_image(self.curr_ti)
        self.curr_image_torch = None

        if not no_mask:
            loaded_mask = self.res_man.get_mask(self.curr_ti)
            if loaded_mask is None:
                self.curr_mask.fill(0)
            else:
                self.curr_mask = loaded_mask.copy()
            self.curr_prob = None

    def convert_current_image_mask_torch(self, no_mask: bool = False):
        if self.curr_image_torch is None:
            self.curr_image_torch = to_tensor(self.curr_image_np).to(self.device, non_blocking=True)

        if self.curr_prob is None and not no_mask:
            self.curr_prob = index_numpy_to_one_hot_torch(self.curr_mask, self.num_objects + 1).to(
                self.device, non_blocking=True)

    def compose_current_im(self):
        self.vis_image = get_visualization(self.vis_mode, self.curr_image_np, self.curr_mask,
                                           self.overlay_layer, self.gui.bg_color)

    def update_canvas(self):
        self.gui.set_canvas(self.vis_image)

    def update_minimap(self):
        if not self.propagating:
            self.gui.set_minimap(self.vis_image)

    def update_current_image_fast(self, invalid_soft_mask: bool = False):
        # fast path, uses gpu. Changes the image in-place to avoid copying
        # thus current_image_torch must be voided afterwards
        self.vis_image = get_visualization_torch(self.vis_mode, self.curr_image_torch,
                                                 self.curr_prob, self.overlay_layer_torch, self.gui.bg_color)
        self.curr_image_torch = None
        self.vis_image = np.ascontiguousarray(self.vis_image)
        if self.save_soft_mask and not invalid_soft_mask:
            self.res_man.save_soft_mask(self.curr_ti, self.curr_prob.cpu().numpy())
        self.gui.set_canvas(self.vis_image)

    def show_current_frame(self, fast: bool = False, invalid_soft_mask: bool = False):
        # Re-compute overlay and show the image
        if fast:
            self.update_current_image_fast(invalid_soft_mask)
        else:
            self.compose_current_im()
            self.update_canvas()
            self.update_minimap()

        self.gui.update_slider(self.curr_ti)
        #self.gui.frame_name.setText(self.res_man.names[self.curr_ti] + '.jpg')

    def set_vis_mode(self):
        self.vis_mode = self.gui.combo.currentText()
        self.show_current_frame()

    def save_current_mask(self):
        # save mask to hard disk
        self.res_man.save_mask(self.curr_ti, self.curr_mask)
        #self.res_man.save_soft_mask(self.curr_ti, self.curr_prob.cpu().numpy())

    def on_slider_update(self):
        # if we are propagating, the on_run function will take care of everything
        # don't do duplicate work here
        self.curr_ti = self.gui.tl_slider.value()
        if not self.propagating:
            # with self.vis_cond:
            #     self.vis_cond.notify()
            if self.curr_frame_dirty:
                self.save_current_mask()
            self.curr_frame_dirty = False

            self.reset_this_interaction()
            self.curr_ti = self.gui.tl_slider.value()
            self.load_current_image_mask()
            self.show_current_frame()

    def on_full_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else: 
            self.on_clear_non_permanent_memory()
            self.curr_ti = 0
            self.on_forward_propagation()

    def on_forward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_fn = self.on_next_frame
            self.gui.forward_propagation_start()
            self.propagate_direction = 'forward'
            self.on_propagate(False)

    def on_backward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_fn = self.on_prev_frame
            self.gui.backward_propagation_start()
            self.propagate_direction = 'backward'
            self.on_propagate(False)
            
    def on_forward_one_propagation(self):
        self.propagate_fn = self.on_next_frame
        self.propagate_direction = 'forward'
        self.on_propagate(True)

    def on_backward_one_propagation(self):
        self.propagate_fn = self.on_prev_frame
        self.propagate_direction = 'backward'
        self.on_propagate(True)

    def on_pause(self):
        self.propagating = False
        self.gui.text(f'Propagation stopped at t={self.curr_ti}.')
        self.gui.pause_propagation()

    def on_propagate(self, singleFrame):
        # start to propagate
        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()

            self.gui.text(f'Propagation started at t={self.curr_ti}.')
            self.processor.clear_sensory_memory()
            self.curr_prob = self.processor.step(self.curr_image_torch,
                                                 self.curr_prob[1:],
                                                 idx_mask=False)
            self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
            # clear
            self.interacted_prob = None
            self.reset_this_interaction()
            
            # if start propagation on the first or last frame, update the softmask for that frame.
            if self.curr_ti == 0 or self.curr_ti == self.T - 1:
                self.show_current_frame(fast=True)
            else:
                self.show_current_frame(fast=True, invalid_soft_mask=True)

            self.propagating = True
            self.gui.clear_all_mem_button.setEnabled(False)
            self.gui.clear_non_perm_mem_button.setEnabled(False)
            self.gui.tl_slider.setEnabled(False)

            dataset = PropagationReader(self.res_man, self.curr_ti, self.propagate_direction)
            loader = get_data_loader(dataset, self.cfg.num_read_workers)

            # propagate till the end
            for data in loader:
                if not self.propagating:
                    break
                self.curr_image_np, self.curr_image_torch = data
                self.curr_image_torch = self.curr_image_torch.to(self.device, non_blocking=True)
                self.propagate_fn()

                self.curr_prob = self.processor.step(self.curr_image_torch)
                self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)

                self.save_current_mask()
                self.show_current_frame(fast=True)

                self.update_memory_gauges()
                self.gui.process_events()
                
                if singleFrame:
                    break
                if self.curr_ti == 0 or self.curr_ti == self.T - 1:
                    break

            self.propagating = False
            self.curr_frame_dirty = False
            self.on_pause()
            self.on_slider_update()
            self.gui.process_events()
            

    def pause_propagation(self):
        self.propagating = False

    def on_commit(self):
        if self.interacted_prob is None:
            # get mask from disk
            self.load_current_image_mask()
        else:
            # get mask from interaction
            self.complete_interaction()
            self.update_interacted_mask()

        # check if frame is already in reference list before adding to list
        items = self.gui.ref_listbox.findItems(str(self.curr_ti), Qt.MatchExactly)
        if len(items) == 0:
            self.gui.ref_listbox.addItem(str(self.curr_ti))
            self.gui.ref_listbox.sortItems()

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()
            self.gui.text(f'Permanent memory saved at {self.curr_ti}.')
            self.curr_prob = self.processor.step(self.curr_image_torch,
                                                 self.curr_prob[1:],
                                                 idx_mask=False,
                                                 force_permanent=True)
            self.update_memory_gauges()
            self.update_gpu_gauges()

    def on_play_video_timer(self):
        self.curr_ti += 1
        if self.curr_ti > self.T - 1:
            self.curr_ti = 0
        self.gui.tl_slider.setValue(self.curr_ti)

    def on_export_video(self):
        export_dialog = Dialog(None, cfg=self.cfg)
        export_dialog.exec()

    def on_object_dial_change(self):
        object_id = self.gui.object_dial.value()
        self.hit_number_key(object_id)

    def update_interacted_mask(self):
        self.undo_stack.append(self.curr_mask)
        self.curr_prob = self.interacted_prob
        self.curr_mask = torch_prob_to_numpy_mask(self.interacted_prob)
        self.save_current_mask()
        self.show_current_frame()
        self.curr_frame_dirty = False

    def reset_this_interaction(self):
        self.complete_interaction()
        self.interacted_prob = None
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()

    def on_reset_mask(self):
        self.curr_mask.fill(0)
        if self.curr_prob is not None:
            self.curr_prob.fill_(0)
        self.curr_frame_dirty = True
        self.undo_stack.clear()
        self.gui.undo_button.setEnabled(False)
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()

    def on_reset_object(self):
        self.curr_mask[self.curr_mask == self.curr_object] = 0
        if self.curr_prob is not None:
            self.curr_prob[self.curr_object] = 0
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()

    def complete_interaction(self):
        if self.interaction is not None:
            self.interaction = None

    def on_prev_frame(self):
        new_ti = max(0, self.curr_ti - 1)
        self.gui.tl_slider.setValue(new_ti)

    def on_next_frame(self):
        new_ti = min(self.curr_ti + 1, self.length - 1)
        self.gui.tl_slider.setValue(new_ti)

    def update_gpu_gauges(self):
        if 'cuda' in self.device:
            info = torch.cuda.mem_get_info()
            global_free, global_total = info
            global_free /= (2**30)
            global_total /= (2**30)
            global_used = global_total - global_free

            self.gui.gpu_mem_gauge.setFormat(f'{global_used:.1f} GB / {global_total:.1f} GB')
            self.gui.gpu_mem_gauge.setValue(round(global_used / global_total * 100))

            used_by_torch = torch.cuda.max_memory_allocated() / (2**30)
            self.gui.torch_mem_gauge.setFormat(f'{used_by_torch:.1f} GB / {global_total:.1f} GB')
            self.gui.torch_mem_gauge.setValue(round(used_by_torch / global_total * 100 / 1024))
            #Out of memory
            if (global_free / global_total) < 0.02 and self.cfg.clear_cache_when_full:
                self.gui.text(f't={self.curr_ti}: GPU memory is low. Clearing torch cache.')
                torch.cuda.empty_cache() #just clear cache instead of whole memory
                
                
        elif 'mps' in self.device:
            mem_used = mps.current_allocated_memory() / (2**30)
            self.gui.gpu_mem_gauge.setFormat(f'{mem_used:.1f} GB')
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat('N/A')
            self.gui.torch_mem_gauge.setValue(0)
        else:
            self.gui.gpu_mem_gauge.setFormat('N/A')
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat('N/A')
            self.gui.torch_mem_gauge.setValue(0)

    def on_gpu_timer(self):
        self.update_gpu_gauges()

    def update_memory_gauges(self):
        try:
            curr_perm_tokens = self.processor.memory.work_mem.perm_size(0)
            self.gui.perm_mem_gauge.setFormat(f'{curr_perm_tokens}')
            self.gui.perm_mem_gauge.setValue(100)

            max_work_tokens = self.processor.memory.max_work_tokens
            max_long_tokens = self.processor.memory.max_long_tokens

            curr_work_tokens = self.processor.memory.work_mem.non_perm_size(0)
            curr_long_tokens = self.processor.memory.long_mem.non_perm_size(0)

            self.gui.work_mem_gauge.setFormat(f'{curr_work_tokens} / {max_work_tokens}')
            self.gui.work_mem_gauge.setValue(round(curr_work_tokens / max_work_tokens * 100))

            self.gui.long_mem_gauge.setFormat(f'{curr_long_tokens} / {max_long_tokens}')
            self.gui.long_mem_gauge.setValue(round(curr_long_tokens / max_long_tokens * 100))

        except AttributeError:
            self.gui.work_mem_gauge.setFormat('Unknown')
            self.gui.long_mem_gauge.setFormat('Unknown')
            self.gui.work_mem_gauge.setValue(0)
            self.gui.long_mem_gauge.setValue(0)

    def on_work_min_change(self):
        if self.initialized:
            self.gui.work_mem_min.setValue(
                min(self.gui.work_mem_min.value(),
                    self.gui.work_mem_max.value() - 1))
            self.update_config()

    def on_work_max_change(self):
        if self.initialized:
            self.gui.work_mem_max.setValue(
                max(self.gui.work_mem_max.value(),
                    self.gui.work_mem_min.value() + 1))
            self.update_config()

    def on_quality_change(self):
        if self.initialized:
            if self.gui.comboBox_quality.currentText() == 'Low':
                self.gui.long_mem_max.setValue(self.cfg.LowQuality.max_num_tokens)
                self.gui.quality_box.setValue(self.cfg.LowQuality.max_internal_size)
            elif self.gui.comboBox_quality.currentText() == 'Normal':
                self.gui.long_mem_max.setValue(self.cfg.NormalQuality.max_num_tokens)
                self.gui.quality_box.setValue(self.cfg.NormalQuality.max_internal_size)
            elif self.gui.comboBox_quality.currentText() == 'High':
                self.gui.long_mem_max.setValue(self.cfg.HighQuality.max_num_tokens)
                self.gui.quality_box.setValue(self.cfg.HighQuality.max_internal_size)
            elif self.gui.comboBox_quality.currentText() == 'Ultra':
                self.gui.long_mem_max.setValue(self.cfg.UltraQuality.max_num_tokens)
                self.gui.quality_box.setValue(self.cfg.UltraQuality.max_internal_size)
            self.update_config()


    def update_config(self):
        if self.initialized:
            with open_dict(self.cfg):
                self.cfg.long_term['min_mem_frames'] = self.gui.work_mem_min.value()
                self.cfg.long_term['max_mem_frames'] = self.gui.work_mem_max.value()
                self.cfg.long_term['max_num_tokens'] = self.gui.long_mem_max.value()
                self.cfg['mem_every'] = self.gui.mem_every_box.value()
                self.cfg['max_internal_size'] = self.gui.quality_box.value()

            self.processor.update_config(self.cfg)

    def on_clear_memory(self):
        self.processor.clear_memory()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device:
            mps.empty_cache()
        self.gui.ref_listbox.clear()
        self.gui.text('Cleared all memory.')
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_clear_non_permanent_memory(self):
        self.processor.clear_non_permanent_memory()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device:
            mps.empty_cache()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_import_mask(self):
        file_names = self.gui.open_files('Select Mask(s)')
        pattern = re.compile(r'([0-9]+)')
        if len(file_names) == 0:
            return
        elif len(file_names) == 1: # load a single mask
            file_name = file_names[0]
            match = pattern.search(file_name)
            if match:
                frame_id = int(match.string[match.start():match.end()])
                if frame_id >= 0 and frame_id < self.T:
                    self.curr_ti = frame_id
                    self.gui.tl_slider.setValue(self.curr_ti)
                    mask = self.res_man.import_mask(file_name, size=(self.h, self.w))
                    shape_condition = ((len(mask.shape) == 2) and (mask.shape[-1] == self.w) and (mask.shape[-2] == self.h))
                    object_condition = (mask.max() <= self.num_objects)
                    if not shape_condition:
                        self.gui.text(f'Expected ({self.h}, {self.w}). Got {mask.shape} instead.')
                    elif not object_condition:
                        self.gui.text(f'Expected {self.num_objects} objects. Got {mask.max()} objects instead.')
                    else:
                        self.gui.text(f'Mask file {file_name} loaded at frame {self.curr_ti}.')
                        self.curr_image_torch = self.curr_prob = None
                        self.curr_mask = mask
                        self.show_current_frame()
                        self.save_current_mask()
                        self.res_man.save_queue.join() #wait for save to finish
                        self.on_commit() #commit mask to permanent memory
            else:
                mask = self.res_man.import_mask(file_name, size=(self.h, self.w))
                shape_condition = ((len(mask.shape) == 2) and (mask.shape[-1] == self.w) and (mask.shape[-2] == self.h))
                object_condition = (mask.max() <= self.num_objects)
                if not shape_condition:
                    self.gui.text(f'Expected ({self.h}, {self.w}). Got {mask.shape} instead.')
                elif not object_condition:
                    self.gui.text(f'Expected {self.num_objects} objects. Got {mask.max()} objects instead.')
                else:
                    self.gui.text(f'Mask file {file_name} loaded at frame {self.curr_ti}.')
                    self.curr_image_torch = self.curr_prob = None
                    self.curr_mask = mask
                    self.show_current_frame()
                    self.save_current_mask()
                    self.res_man.save_queue.join() #wait for save to finish
                    self.on_commit() #commit mask to permanent memory

        elif len(file_names) > 1: # when loading multiple files, make sure they all have good names
            all_correct = True
            frame_ids = []
            incorrect_files = []
            for file_name in file_names:
                match = pattern.search(file_name)
                if match:
                    frame_id = int(match.string[match.start():match.end()])
                    if frame_id >= 0 and frame_id < self.T:
                        frame_ids.append(frame_id)
                    else:
                        all_correct = False
                        incorrect_files.append(file_name)
                else:
                    all_correct = False
                    incorrect_files.append(file_name)
            if not all_correct:
                broken_file_names = '\n'.join(incorrect_files)
                QMessageBox.warning(None, "Incorrect File Names", f"When loading multiple masks, each filename must include the frame number.\nFiles with incorrect names:\n{broken_file_names}")
                return
            elif  frame_ids != sorted(frame_ids):
                QMessageBox.warning(None, "Incorrect File Names", "When loading multiple masks, each filename must include the frame number.\nSome of the files may have duplicate numbers or have an unexpected name format.")
                return

            reply = QMessageBox.question(None, "Save to Permanent Memory", "Would you like to commit all masks to permanent memory as reference frames?", QMessageBox.Yes | QMessageBox.No)
            for file_name in file_names:
                self.curr_ti = frame_ids[file_names.index(file_name)]
                self.gui.tl_slider.setValue(self.curr_ti)
                mask = self.res_man.import_mask(file_name, size=(self.h, self.w))
                shape_condition = ((len(mask.shape) == 2) and (mask.shape[-1] == self.w) and (mask.shape[-2] == self.h))
                object_condition = (mask.max() <= self.num_objects)
                if not shape_condition:
                    self.gui.text(f'Expected ({self.h}, {self.w}). Got {mask.shape} instead.')
                elif not object_condition:
                    self.gui.text(f'Expected {self.num_objects} objects. Got {mask.max()} objects instead.')
                else:    
                    self.gui.text(f'Mask file {file_name} loaded at frame {self.curr_ti}.')
                    self.curr_image_torch = self.curr_prob = None
                    self.curr_mask = mask
                    self.show_current_frame()
                    self.save_current_mask()
                    self.res_man.save_queue.join() #wait for save to finish
                    if reply == QMessageBox.Yes:
                        self.on_commit() #commit mask to permanent memory

    def on_open_workspace(self):
        show_in_file_manager(self.res_man.workspace)

    def on_import_layer(self):
        file_name = self.gui.open_file('Select Background Layer')
        if len(file_name) == 0:
            return

        self._try_load_layer(file_name)

    def _try_load_layer(self, file_name):
        try:
            layer = self.res_man.import_layer(file_name, size=(self.h, self.w))

            self.gui.text(f'Layer file {file_name} loaded.')
            self.overlay_layer = layer
            self.overlay_layer_torch = torch.from_numpy(layer).float().to(self.device) / 255
            self.show_current_frame()
        except FileNotFoundError:
            self.gui.text(f'{file_name} not found.')

    def on_bg_color(self):
        self.gui.choose_color()
        # disable background layer when choosing color
        self.overlay_layer: np.ndarray = None
        self.overlay_layer_torch: torch.Tensor = None
        self.show_current_frame()

    def on_save_soft_mask_toggle(self):
        self.save_soft_mask = self.gui.save_soft_mask_checkbox.isChecked()

    def on_mouse_motion_xy(self, x, y):
        self.last_ex = x
        self.last_ey = y
        self.update_minimap()

    def on_zoom_plus(self):
        self.zoom_pixels -= 50
        self.zoom_pixels = max(50, self.zoom_pixels)
        self.update_minimap()

    def on_zoom_minus(self):
        self.zoom_pixels += 50
        self.zoom_pixels = min(self.zoom_pixels, 300)
        self.update_minimap()

    def on_undo(self):
        self.curr_image_torch = self.curr_prob = None
        self.curr_mask = self.undo_stack.pop()
        self.show_current_frame()
        self.save_current_mask()
        self.complete_interaction()
        if len(self.undo_stack) == 0:
            self.gui.undo_button.setEnabled(False)

    @property
    def h(self) -> int:
        return self.res_man.h

    @property
    def w(self) -> int:
        return self.res_man.w

    @property
    def T(self) -> int:
        return self.res_man.T
    

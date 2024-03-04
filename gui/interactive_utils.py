# Modified from https://github.com/seoungwugoh/ivs-demo

from typing import Literal, List
import numpy as np

import torch
import torch.nn.functional as F
from cutie.utils.palette import davis_palette
from cutie.config.config import global_config


def image_to_torch(frame: np.ndarray, device: str = 'cuda'):
    # frame: H*W*3 numpy array
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float().to(device, non_blocking=True) / 255
    return frame


def torch_prob_to_numpy_mask(prob: torch.Tensor):
    mask = torch.max(prob, dim=0).indices
    mask = mask.cpu().numpy().astype(np.uint8)
    return mask


def index_numpy_to_one_hot_torch(mask: np.ndarray, num_classes: int):
    mask = torch.from_numpy(mask).long()
    return F.one_hot(mask, num_classes=num_classes).permute(2, 0, 1).float()


# set torch device for interactice segmentation
cfg = global_config
if cfg.force_cpu:
    device = torch.device("cpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
#print(f'Using click device: {device}')

# Some constants for visualization
color_map_np = np.frombuffer(davis_palette, dtype=np.uint8).reshape(-1, 3).copy()
# scales for better visualization
color_map_np = (color_map_np.astype(np.float32) * 1.5).clip(0, 255).astype(np.uint8)
color_map = color_map_np.tolist()
color_map_torch = torch.from_numpy(color_map_np).to(device) / 255

grayscale_weights = np.array([[0.3, 0.59, 0.11]]).astype(np.float32)
grayscale_weights_torch = torch.from_numpy(grayscale_weights).to(device).unsqueeze(0)


def get_visualization(mode: Literal['image', 'mask', 'overlay', 'background'], 
                      image: np.ndarray, mask: np.ndarray, layer: np.ndarray, color: List[int]) -> np.ndarray:
    if mode == 'image':
        return image
    elif mode == 'mask':
        return color_map_np[mask]
    elif mode == 'overlay':
        return overlay_davis(image, mask)
    elif mode == 'background':
        if layer is None:
            return overlay_bgcolor(image, mask, color)
        else:
            return overlay_layer(image, mask, layer)
    else:
        raise NotImplementedError


def get_visualization_torch(mode: Literal['image', 'mask', 'overlay',
                                          'background'], image: torch.Tensor, prob: torch.Tensor,
                            layer: torch.Tensor, color: List[int]) -> np.ndarray:
    if mode == 'image':
        image = image.permute(1, 2, 0)
        return (image * 255).byte().cpu().numpy()
    elif mode == 'mask':
        mask = torch.max(prob, dim=0).indices
        return (color_map_torch[mask] * 255).byte().cpu().numpy()
    elif mode == 'overlay':
        return overlay_davis_torch(image, prob)
    elif mode == 'background':
        if layer is None:
            return overlay_bgcolor_torch(image, prob, color)
        else:
            return overlay_layer_torch(image, prob, layer)
    else:
        raise NotImplementedError


def overlay_davis(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    """ Overlay segmentation on top of RGB image. from davis official"""
    im_overlay = image.copy()

    colored_mask = color_map_np[mask]
    foreground = image * alpha + (1 - alpha) * colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    return im_overlay.astype(image.dtype)


def overlay_layer(image: np.ndarray, mask: np.ndarray, layer: np.ndarray):
    # insert a background layer behind foreground
    # The CPU version is less accurate because we are using the hard mask
    # The GPU version has softer edges as it uses soft probabilities
    mask = mask.astype(np.float32)[:, :, np.newaxis]
    layer_rgb = layer[:, :, :3]
    im_overlay = (layer_rgb * (1 - mask) + image * mask).clip(0, 255)
    return im_overlay.astype(image.dtype)

def overlay_bgcolor(image: np.ndarray, mask: np.ndarray, color: List[int]):
    mask = mask.astype(np.float32)[:, :, np.newaxis]
    layer_rgb = np.full_like(image, color, dtype=np.uint8)
    im_overlay = (layer_rgb * (1 - mask) + image * mask).clip(0, 255)
    return im_overlay.astype(image.dtype)

def overlay_davis_torch(image: torch.Tensor,
                        prob: torch.Tensor,
                        alpha: float = 0.5):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # Changes the image in-place to avoid copying
    # NOTE: Make sure you no longer use image after calling this function
    image = image.permute(1, 2, 0)
    im_overlay = image
    mask = torch.max(prob, dim=0).indices

    colored_mask = color_map_torch[mask]
    foreground = image * alpha + (1 - alpha) * colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]

    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay

def overlay_bgcolor_torch(image: torch.Tensor, prob: torch.Tensor, color: List[int]):
    image = image.permute(1, 2, 0)
    mask = prob[0].unsqueeze(2)
    color = torch.tensor(color) / 255
    layer_rgb = color.expand_as(image).to(device)
    im_overlay = (layer_rgb * mask + image * (1 - mask)).clip(0, 1)
    
    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay

def overlay_layer_torch(image: torch.Tensor, prob: torch.Tensor, layer: torch.Tensor):
    # insert a background layer behind foreground
    # The CPU version is less accurate because we are using the hard mask
    # The GPU version has softer edges as it uses soft probabilities
    image = image.permute(1, 2, 0)
    mask = prob[0].unsqueeze(2)
    layer_rgb = layer[:, :, :3]
    im_overlay = (layer_rgb * mask + image * (1 - mask)).clip(0, 1)

    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay

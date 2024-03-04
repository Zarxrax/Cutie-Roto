import os
import sys
# fix for Windows
if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ:
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import signal
from argparse import ArgumentParser

try:
    import torch
except ModuleNotFoundError:
    sys.exit("Please execute \"install_pytorch.bat\" to install Pytorch, then try again.")
    
from omegaconf import open_dict
import logging
from PySide6.QtWidgets import QApplication, QDialog
import qdarktheme

from cutie.config.config import global_config
from gui.main_controller import MainController
from gui.launcher_gui import Launcher_Dialog

signal.signal(signal.SIGINT, signal.SIG_DFL)

class Dialog(QDialog, Launcher_Dialog):
    def __init__(self):
        super().__init__()
        self.cfg = cfg
        self.setupUi(self, self.cfg)


def get_arguments():
    parser = ArgumentParser()
    """
    Priority 1: If a "images" folder exists in the workspace, we will read from that directory
    Priority 2: If --images is specified, we will copy/resize those images to the workspace
    Priority 3: If --video is specified, we will extract the frames to the workspace (in an "images" folder) and read from there

    In any case, if a "masks" folder exists in the workspace, we will use that to initialize the mask
    That way, you can continue annotation from an interrupted run as long as the same workspace is used.
    """
    parser.add_argument('--images', help='Folders containing input images.', default=None)
    parser.add_argument('--video', help='Video file readable by OpenCV.', default=None)
    parser.add_argument('--workspace',
                        help='directory for storing buffered images (if needed) and output masks',
                        default=None)
    parser.add_argument('--num_objects', type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ in "__main__":
    log = logging.getLogger()

    # get the config
    cfg = global_config

    # input arguments
    args = get_arguments()

    # general setup
    torch.set_grad_enabled(False)
    if cfg.force_cpu:
        device = 'cpu'    
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    args.device = device
    log.info(f'Using device: {device}')

    # merge arguments into config
    args = vars(args)
    with open_dict(cfg):
        for k, v in args.items():
            assert k not in cfg, f'Argument {k} already exists in config'
            cfg[k] = v

    # prepare to start the gui
    app = QApplication(sys.argv)
    qdarktheme.setup_theme(theme="auto", additional_qss="QToolTip { border: 0px; }")    
    
    # if no input specified in args, make a launcher dialog to let the user choose a file to load.
    if not (cfg["video"] or cfg["images"] or cfg["workspace"] ):
        launcher = Dialog()
        result = launcher.exec()
        if result == QDialog.Accepted:
            print("Opening video: ", launcher.result)
            cfg["video"] = launcher.result
        else:
            print("User closed launcher")
            sys.exit()

    # launch the main window
    ex = MainController(cfg)
    sys.exit(app.exec())

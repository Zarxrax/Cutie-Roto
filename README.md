# Cutie Roto
Click image to view Video  
[![Cutie Roto Demo Video](http://img.youtube.com/vi/ZaC1lltmWxc/0.jpg)](http://www.youtube.com/watch?v=ZaC1lltmWxc "Cutie Roto Demo Video")

Cutie Roto is a fork of [Cutie](https://github.com/hkchengrex/Cutie), and is designed to be a user friendly tool for AI assisted rotoscoping of video clips. It serves as a free alternative to commercial solutions such as Adobe’s Roto Brush or DaVinci Resolve Magic Mask. It is generally less accurate than manual rotoscoping, but can usually give a pretty good result with little effort.

Also check out my newer project, [Sammie-Roto](https://github.com/Zarxrax/Sammie-Roto)

### Changes from Cutie:
- Downloadable package for windows users
- Launcher to select a video file to work on
- New export video dialog
- Undo support
- Minimap to show a zoomed in view of the image
- Ability to import black and white mattes as masks
- Additional click segmentation model trained on anime
- Simplified interface

### Installation (Windows):
- Download latest version from [releases](https://github.com/Zarxrax/Cutie-Roto/releases)
- Extract the zip archive.
- Run 'install_dependencies.bat' and follow the prompt.
- Run 'run_gui.bat' to launch the software.

### Manual Installation (Linux, Mac)
I can only test on Windows, so please let me know if there are any issues with this running on Linux or Mac.
#### Prerequisites:
* [Python](https://www.python.org/) (version 3.11 recommended)
* [Pytorch](https://pytorch.org) (version 2+ recommended)

#### Clone the repository and install dependencies:
```
git clone https://github.com/Zarxrax/Cutie-Roto.git
cd Cutie-Roto
pip install -r requirements.txt
```
#### Launch the application:
```
python cutie_roto.py
```

### How to use:
See the [wiki](https://github.com/Zarxrax/Cutie-Roto/wiki) for documentation

### Acknowledgements
* [Cutie](https://github.com/hkchengrex/Cutie) the foundation that Cutie-Roto is built on
* [Xmem2](https://github.com/max810/XMem2) a predecessor of cutie, inspired some features that I have added back into Cutie-Roto
* [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) the interactive segmentation architecture that is used

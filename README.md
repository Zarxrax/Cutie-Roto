# Cutie Roto
Click image to view Video  
[![Cutie Roto Demo Video](http://img.youtube.com/vi/ZaC1lltmWxc/0.jpg)](http://www.youtube.com/watch?v=ZaC1lltmWxc "Cutie Roto Demo Video")

Cutie Roto is a fork of [Cutie](https://github.com/hkchengrex/Cutie), and is designed to be a user friendly tool for AI assisted rotoscoping of video clips. It serves as a free alternative to commercial solutions such as Adobe’s Roto Brush or DaVinci Resolve Magic Mask. This tool is still in early development. It is generally less accurate than manual rotoscoping, but can usually give a pretty good result with little effort.

### Changes from Cutie:
- Downloadable package for windows users
- Launcher to select a video file to work on
- New export video dialog
- Zoomed minimap inspired from [XMem](https://github.com/hkchengrex/XMem)
- Ability to import black and white mattes as a mask
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
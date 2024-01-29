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

### Roadmap (planned features):
- Re-implement brush tool from [XMem](https://github.com/hkchengrex/XMem)
- Replace the minimap with the ability to zoom directly on the main canvas
- Continue to train better models for anime

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
You should trim your video down to the specific scene that you are trying to mask prior to loading it into the program. Some sample clips are included in the examples folder.

When you launch the program, you will be prompted to select a video file to load. A workspace folder will also be created for that video. Because the workspace is based on the filename, please avoid working on multiple files that have the same name. At any time, you can delete any workspace folders as long as you don't mind losing any work contained in them.

When the main application launches, you have a timeline where you can view all of the frames of the video. You can create a mask on a frame by left clicking to add areas to the mask (highlighted in red), or right click to remove areas from the mask. If you are satisfied with how a mask looks, press the 'commit to permanent memory' button to store this frame in the application's memory, so it can be used to help mask other frames.
Use the propagate forward and backward buttons to propagate the mask onto additional frames in the video.

You can import masks created in external software by clicking the 'import mask' button. The mask should be a black and white image file, with white indicating the object to be masked. The mask will be loaded onto the frame that is currently displayed, and it will automatically be committed to memory.

If you want to erase all masked frames and start over, just go to the first frame and press 'Reset frame', then press 'Reset all memory'. Finally, click 'full propagate' to remove the masks from all remaining frames. 

Once you have finished masking your clip, press the 'Export as video' button to create a video file that can be brought back into another application. Various image sequences will also be created in the workspace folder.

### Keyboard Shortcuts:
- Left/Right: Previous/Next frame
- Space: Play video
- Backspace: Reset mask on current frame
- Plus/Minus: Zoom in/out on the minimap
- Enter: Commit the current frame to memory

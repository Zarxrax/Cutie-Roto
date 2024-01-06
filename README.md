# Cutie Roto

Cutie Roto is a fork of [Cutie](https://github.com/hkchengrex/Cutie), and is designed to be a user friendly tool for AI assisted rotoscoping of video clips. It serves as a free alternative to commercial solutions such as Adobe’s Roto Brush or DaVinci Resolve Magic Mask. This tool is still in early development.

### Changes from Cutie:
- Downloadable package for windows users
- Launcher to select a video file to work on
- Ability to import black and white mattes as a mask
- Simplified interface

### Roadmap (planned features):
- Re-Implement brush tool and zoomed view from [XMem](https://github.com/hkchengrex/XMem).
- Add new export video dialog
- Automatic mask creation
- Implement a matting model for edge refinement
- Re-train all models for anime content (original models will also be kept)

### Installation (Windows):
- Download latest version from [releases](https://github.com/Zarxrax/Cutie-Roto/releases)
- Extract the 7z archive (you may need to install [7-zip](https://www.7-zip.org))
- Run 'install_pytorch.bat' and follow the prompt.
- Run 'run_gui.bat' to launch the software.

### How to use:
When you launch the program, you will be prompted to select a video file to load. A workspace folder will also be created for that video. Because the workspace is based on the filename, please avoid working on multiple files that have the same name. At any time, you can delete any workspace folders as long as you don't mind losing any work contained in them.

When the main application launches, you have a timeline where you can view all of the frames of the video. You can create a mask on a frame by left clicking to add areas to the mask (highlighted in red), or right click to remove areas from the mask. If you are satisfied with how a mask looks, press the 'commit to permanent memory' button to store this frame in the application's memory, so it can be used to help mask other frames.
Use the propagate forward and backward buttons to propagate the mask onto additional frames in the video.

You can import masks created in external software by clicking the 'import mask' button. The mask should be a black and white image file, with white indicating the object to be masked. The mask will be loaded onto the frame that is currently displayed, and it will automatically be committed to memory.

If you want to erase all masked frames and start over, just go to the first frame and press 'Reset frame', then press 'Reset all memory'. Finally, click 'propagate forward' to remove the masks from all remaining frames. 

Once you have finished masking your clip, press the 'Export binary masks' button. A binary masks folder will appear in the workspace, which contains an image sequence that you can import back into an editing application.

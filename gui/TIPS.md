### Tips

Core mechanism: annotate objects at one or more frames and use propagation to complete the video.
Use permanent memory to store accurate segmentation (commit good frames to it) for best results.
The first frame to enter the memory bank is always committed to the permanent memory.
Reset memory if needed.

- Use left-click for foreground annotation and right-click for background annotation.
- Use import mask to import a black and white mask that was created in other software.
- Use Propogate Forward and Backward buttons to propogate masks across the frames.
- Use Export as Video to save the output to a video file.
- Open Workspace to browse intermediate files that have been saved by the program.
- If you want to delete all work, delete the files in the workspace folder.
- Max internal resolution is the maximum vertical resolution that the frame propogation operates at. Increasing this value can increase accuracy at the cost of speed and VRAM usage. It is not recommended to go above 720.
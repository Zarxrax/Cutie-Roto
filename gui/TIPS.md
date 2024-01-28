### Tips


Basic workflow: annotate objects on one or more frames and use propagation to complete the video.
Commit good frames to permanent memory for best results. Reset memory if needed.

- Use left-click for foreground annotation and right-click for background annotation.
- If you do not start getting a good mask result after several clicks, it may be better to reset the frame and try again, clicking in different areas.
- The standard click segmentation model may work better on some anime scenes than the anime model.
- Use import mask to import a black and white mask that was created in other software.
- Use Propogate Forward and Backward buttons to propogate masks across the frames. Use full propagation to run it on the entire clip from the beginning.
- Use Export as Video to save the output to a video file.
- The refine edges option in the export video dialog should only be used after running a full propagation. Refine edges uses soft masks that are only generated/updated upon propgation.
- Open Workspace to browse intermediate files that have been saved by the program.
- If you want to delete all work, delete the files in the workspace folder and restart the software.
- Higher quality settings have a significant impact on the amount of VRAM used, and can cause the software to crash if your VRAM becomes full. You may be able to get away with using a higher setting for short clips.
- Committing frames to permanent memory uses some VRAM, so be careful not to save too many.
- You should commit frames that are very different from each other. It is not helpful to commit multiple frames that are similar.
# Stereo Vision Depth Camera
This project is all about required code to get simple stereo vision depth camera working!

# File Structure
- `main.py` - Main module to be executed to get depth camera running
- `cameraFeed.py` - A module that starts camera feed for individual cameras and handles clearing of resources once done.
- `depthCam.py` - A module that initiates different processes for individual cameras and handle output coming from triangulation module to generate final output.
- `depthCamEye.py` - A multiprocess module that works within its own process. It segments ball from the scene and calculates the offset and stores it in shared memory manager.
- `triangulation.py` - A module that consumes offset values and using that, calculates depth of the object and generates triangulation graphics to show on the output feed.

# Flow of the program
<img src="https://github.com/CPaladiya/StereoVisionDepthCam/blob/main/data/ProgramFlow.png" width="900">

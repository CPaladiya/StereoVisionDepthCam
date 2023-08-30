# Stereo Vision Depth Camera
Python modules required to tune depth camera to segment particular object using HSV channel thresholding coupled with calculation of depth of the object from camera. 

# Working demo
<img src="https://github.com/CPaladiya/StereoVisionDepthCam/blob/main/data/DepthCam.gif" width="400">

# File Structure
- `main.py`: Main execution module responsible for starting and managing the depth camera functionality.
- `cameraFeed.py`: Module dedicated to initializing camera feeds for individual cameras. Also responsible for clearing resources associated with camera feeds once they are no longer needed.
- `depthCam.py`: Module that coordinates various processes for individual cameras. It manages the output from the triangulation module to produce the final depth output.
- `depthCamEye.py`: A specialized multiprocess module operating within its own process. This module focuses on segmenting the ball from the scene, calculating its offset, and storing both the offset values and camera frames using the shared memory manager.
- `triangulation.py`: Module that takes in the calculated offset values and utilizes them to determine the depth of the object. This module generates the necessary triangulation graphics that are then incorporated into the output feed.

# Flow of the program
<img src="https://github.com/CPaladiya/StereoVisionDepthCam/blob/main/data/ProgramFlow.png" width="600">

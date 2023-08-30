# Stereo Vision Depth Camera
Python modules required to tune depth camera to segment particular object using HSV channel thresholding coupled with calculation of depth of the object from camera. 

# Working demo
<img src="https://github.com/CPaladiya/StereoVisionDepthCam/blob/main/data/DepthCam.gif" width="400">

# What you will need?
- Two cameras mounted on a frame with fixed known distance between them. Cameras should be aligned!
- device id of left and right camera. If you are not sure about device id, start with 0 for left and 1 for right, if fails, flip them, if fails, try 1/2 instead ans so on.
- Both left and right camera has to be identical.
- Field of view of camera

## Things to lookout for
- **Left** and **Right** of the camera is decided while looking from camera towards object.
- While working with machine that has its own webcam, its important to decide what device id belongs to which camera.
- Actual left camera id should be only used for `leftCam` argument in `DepthCam` object and actual right camera id for `rightCam` argument.
- Since we are working with parallel processes, `DepthCam` object should only be initiated and called under **`__main__`** block.
  - This can be achieved by implementing `if __name__ == "__main__":` in executing module. For an example please look at the `main.py`.
<img src="https://github.com/CPaladiya/StereoVisionDepthCam/blob/main/data/DepthCamDirection.png" width="400">

# File Structure
- `main.py`: Main execution module responsible for starting and managing the depth camera functionality.
- `cameraFeed.py`: Module dedicated to initializing camera feeds for individual cameras. Also responsible for clearing resources associated with camera feeds once they are no longer needed.
- `depthCam.py`: Module that coordinates various processes for individual cameras. It manages the output from the triangulation module to produce the final depth output.
- `depthCamEye.py`: A specialized multiprocess module operating within its own process. This module focuses on segmenting the ball from the scene, calculating its offset, and storing both the offset values and camera frames using the shared memory manager.
- `triangulation.py`: Module that takes in the calculated offset values and utilizes them to determine the depth of the object. This module generates the necessary triangulation graphics that are then incorporated into the output feed.

# Flow of the program
<img src="https://github.com/CPaladiya/StereoVisionDepthCam/blob/main/data/ProgramFlow.png" width="500">

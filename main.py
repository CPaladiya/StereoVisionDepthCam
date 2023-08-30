from depthCam import DepthCam as DC

distBetweenCameras_in = 6.0  # distance between left and right cameras in  inch
resOfCamera = (
    1920,
    1080,
)  # (width,height) - resolution of the camera - both cameras has to be the same
fieldOfView = 70.0  # Field of view in degrees - can be found in specs of the product

if (
    __name__ == "__main__"
):  # This block is important since we are working with processes
    depthCam = DC(
        fov=fieldOfView,
        distBtwnCameras=distBetweenCameras_in,
        leftCamID=2,
        rightCamID=1,
        widthRes=resOfCamera[0],
        heightRes=resOfCamera[1],
    )
    depthCam.calibrate(HSVon=True)
    depthCam.measureDepth()

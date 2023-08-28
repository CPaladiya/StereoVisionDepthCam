from depthCam import DepthCam as DC

distBetweenCameras_in = 6.0  # distance between cameras in  inch
resOfCamera = (
    1920,
    1080,
)  # (width,height) - resolution of the camera - both cameras has to be the same
fieldOfView = 70.0  # Field of view in degrees - Centon OTM Basics 360-Degree HD USB Webcam - can be found in specs of the product

if __name__ == "__main__":
    depthCam = DC(
        fov=fieldOfView,
        baseDist=distBetweenCameras_in,
        leftCam=2,
        rightCam=1,
        widthRes=resOfCamera[0],
        heightRes=resOfCamera[1],
    )
    #depthCam.calibrate(HSVon=True)
    depthCam.calibrateManually(53, 89, 66, 245, 56, 255)
    depthCam.measureDepth()

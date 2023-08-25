from depthCam import DepthCam as DC

distBetweenCameras_mm = 56.5                        # distance between cameras in mm
distBetweenCameras_in = distBetweenCameras_mm/25.4  # distance between cameras in  inch
resOfCamera = (1920,1080)                           # (width,height) - resolution of the camera - both cameras has to be the same
fieldOfView = 85                                    # Field of view in degrees - Centon OTM Basics 360-Degree HD USB Webcam - can be found in specs of the product

if __name__ == "__main__":
    depthCam = DC(fov=fieldOfView, 
                baseDist=distBetweenCameras_in,
                cam1=1,
                cam2=2,
                widthRes=resOfCamera[0], 
                heightRes=resOfCamera[1])
    #depthCam.calibrate()
    depthCam.calibrateManually(53,89,66,245,56,255)
    depthCam.measureDepth()
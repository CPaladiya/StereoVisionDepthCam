import cv2
import numpy as np
from CameraFeed import CameraFeed
from depthCamEye import DepthCamEye
import multiprocessing
import math
import time
from multiprocessing import Manager


class DepthCam:
    """Class used to tune depth camera and measure depth of specified objects"""

    # slots are used for memory optimization
    __slot__ = [
        "fov",
        "baseDist",
        "leftCam",
        "rightCam",
        "widthRes",
        "heightRes",
        "Hmin",
        "Smin",
        "Vmin",
        "Hmax",
        "Smax",
        "Vmax",
        "calibrationWindow",
        "focLenInPixels",
        "depthInInch",
        "lCamAngle",
        "rCamAngle",
        "cCamAngle",
    ]

    def __init__(
        self,
        fov: float,
        baseDist: float,
        leftCam: int,
        rightCam: int,
        widthRes: int = 640,
        heightRes: int = 480,
    ) -> None:
        """Initiate `DepthCam`

        Args:
            fov (float): HORIZONTAL Field of view of the camera in degrees
            baseDist (float): Center distance between two cameras
            leftCam (int) : Id of the left camera (looking from camera towards object)
            rightCam (int) : Id of the second camera (looking from camera towards object)
            widthRes (int): Width resolution of camera feed. Defaults to 640.
            heightRes (int): Height resolution of camera feed. Defaults to 480.
        """

        self.fov: float = fov  # field of view of camera
        self.baseDist: float = baseDist  # distance between two cameras
        self.leftCam: int = leftCam  # id of first camera
        self.rightCam: int = rightCam  # id of second camera
        self.widthRes: int = widthRes  # width resolution of the camera
        self.heightRes: int = heightRes  # height resolution of the camera
        self.Hmin: int = 0  # Hue-min color value threshold
        self.Smin: int = 0  # Sat-min color value threshold
        self.Vmin: int = 0  # Val-min color value threshold
        self.Hmax: int = 179  # Hue-max color value threshold
        self.Smax: int = 255  # Sat-max color value threshold
        self.Vmax: int = 255  # Val-max color value threshold
        self.calibrationWindow: str = "CalibrationWindow"
        self.focLenInPixels: int = 0
        self.depthInInch: float = 0.0

    def calcFocalLengthInPixels(self):
        """Calculates focal length in pixel which will be used in triagulation calculations"""
        halfOfImageWidth = self.widthRes / 2
        halfOfFOV = self.fov / 2
        self.focLenInPixels = int(halfOfImageWidth / math.tan(math.radians(halfOfFOV)))

    def startTrackers(self) -> None:
        """sets trackers on the main camera feed window"""

        # setting up sliders and tuning parameters
        def on_trackbar_hmin(val):
            nonlocal self
            self.Hmin = val

        def on_trackbar_hmax(val):
            nonlocal self
            self.Hmax = val

        def on_trackbar_smin(val):
            nonlocal self
            self.Smin = val

        def on_trackbar_smax(val):
            nonlocal self
            self.Smax = val

        def on_trackbar_vmin(val):
            nonlocal self
            self.Vmin = val

        def on_trackbar_vmax(val):
            nonlocal self
            self.Vmax = val

        cv2.namedWindow(self.calibrationWindow)
        # by default minimum value of slider is always 0
        cv2.createTrackbar("HueMin", self.calibrationWindow, 20, 179, on_trackbar_hmin)
        cv2.createTrackbar("HueMax", self.calibrationWindow, 150, 179, on_trackbar_hmax)
        cv2.createTrackbar("SatMin", self.calibrationWindow, 20, 255, on_trackbar_smin)
        cv2.createTrackbar("SatMax", self.calibrationWindow, 150, 255, on_trackbar_smax)
        cv2.createTrackbar("ValMin", self.calibrationWindow, 20, 255, on_trackbar_vmin)
        cv2.createTrackbar("ValMax", self.calibrationWindow, 150, 255, on_trackbar_vmax)

    def calibrateManually(self, Hmin, Hmax, Smin, Smax, Vmin, Vmax):
        self.Hmin = Hmin
        self.Hmax = Hmax
        self.Smin = Smin
        self.Smax = Smax
        self.Vmin = Vmin
        self.Vmax = Vmax

    def calibrate(self, HSVon:bool = True) -> None:
        """Sets value of threshold for Hue, Saturation and Value channels. It requires exactly two cameras.
        The video feed can be switch left to right by swapping leftCam and rightCam id.
        """
        # starting a video feed
        if HSVon:
            self.startTrackers()
        leftCamFeed, rightCamFeed = CameraFeed(self.leftCam), CameraFeed(self.rightCam)
        leftCamFeed.openCameraFeed()
        rightCamFeed.openCameraFeed()

        while True:
            leftFrame = leftCamFeed.retriveFrame()
            rightFrame = rightCamFeed.retriveFrame()
            # Draw vertical line
            if not HSVon:
                leftFrame = cv2.line(leftFrame, (leftFrame.shape[1]//2, 0), (leftFrame.shape[1]//2, leftFrame.shape[0]), (0, 0, 255), 2)
                rightFrame = cv2.line(rightFrame, (rightFrame.shape[1]//2, 0), (rightFrame.shape[1]//2, rightFrame.shape[0]), (0, 0, 255), 2)
                leftFrame = cv2.line(leftFrame, (0, leftFrame.shape[0]//2), (leftFrame.shape[1], leftFrame.shape[0]//2), (0, 0, 255), 2)
                rightFrame = cv2.line(rightFrame, (0, rightFrame.shape[0]//2), (rightFrame.shape[1], rightFrame.shape[0]//2), (0, 0, 255), 2)
            # connecting two frames into one
            frame = np.concatenate([leftFrame, rightFrame], axis=1)
            if HSVon:
                # converting rgb to hsv space and thresholding at the same time
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame = cv2.inRange(
                    frame,
                    (self.Hmin, self.Smin, self.Vmin),
                    (self.Hmax, self.Smax, self.Vmax),
                )
            # adding text to the image to quit once done tuning
                frame = cv2.putText(
                    img=frame,
                    text=f"press `q` once done calibrating!",
                    org=(10, 50),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1.5,
                    color=(88, 86, 93),
                    thickness=3,
                )
            cv2.imshow(self.calibrationWindow, frame)
            # quit when `q` is pressed
            if cv2.waitKey(1) == ord("q"):
                break
        leftCamFeed.release()
        rightCamFeed.release()
        cv2.destroyAllWindows()
        # get status on set values of HSV values
        print(
            f"Hmin set to {self.Hmin}\nHmax set to {self.Hmax}\nSmin set to {self.Smin}\nSmax set to {self.Smax}\nVmin set to {self.Vmin}\nVmax set to {self.Vmax}"
        )

    def measureDepth(self):
        self.calcFocalLengthInPixels()
        memManager = Manager()
        leftCamResDict, rightCamResDict = memManager.dict(), memManager.dict()
        leftCamResDict["frame"] = None
        leftCamResDict["xOffset"] = None
        rightCamResDict["frame"] = None
        rightCamResDict["xOffset"] = None
        leftEyeLock, rightEyeLock = memManager.Lock(), memManager.Lock()

        # starting both cameras
        leftEyeProcess = DepthCamEye(
            leftEyeLock,
            leftCamResDict,
            self.leftCam,
            [self.Hmin, self.Smin, self.Vmin, self.Hmax, self.Smax, self.Vmax],
            widthRes=self.widthRes,
            heightRes=self.heightRes,
        )
        rightEyeProcess = DepthCamEye(
            rightEyeLock,
            rightCamResDict,
            self.rightCam,
            [self.Hmin, self.Smin, self.Vmin, self.Hmax, self.Smax, self.Vmax],
            widthRes=self.widthRes,
            heightRes=self.heightRes,
        )
        leftEyeProcess.start(), rightEyeProcess.start()
        while True:
            leftFrame, rightFrame = None, None
            leftXoffset, rightXoffset = 0.0, 0.0
            with leftEyeLock, rightEyeLock:
                # have method to process the frames and X offsets
                leftFrame = leftCamResDict["frame"]
                leftXoffset = leftCamResDict["xOffset"]
                rightFrame = rightCamResDict["frame"]
                rightXoffset = rightCamResDict["xOffset"]
            if leftXoffset is not None and rightXoffset is not None:
                self.performTriangulation(leftXoffset, rightXoffset)
            if leftFrame is not None and rightFrame is not None:
                frame = np.concatenate([leftFrame, rightFrame], axis=1)
                # adding text to the image to quit once done tuning
                frame = cv2.putText(
                    img=frame,
                    text=f"Depth:{self.depthInInch:>4.2f}, aL:{self.lCamAngle:>4.1f}, aR:{self.rCamAngle:>4.1f}, aC:{self.cCamAngle:>4.1f}",
                    org=(10, 50),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1.5,
                    color=(255, 0, 0),
                    thickness=3,
                )
                cv2.imshow("ballInSpace", frame)
            else:
                pass
            # add a condition here to stop the processes!
            if cv2.waitKey(1) == ord("q"):
                break

        # appropreate tremination of the processes
        leftEyeProcess.terminate(), rightEyeProcess.terminate()
        leftEyeProcess.join(), rightEyeProcess.join()
        leftEyeProcess.close(), rightEyeProcess.close()
        cv2.destroyAllWindows()

    def performTriangulation(self, leftXoffset, rightXoffset):
        # inside angle of the leftCam in triangulation
        leftCamAngle = 0.0
        angle1 = math.degrees(math.atan(abs(leftXoffset) / self.focLenInPixels))
        if leftXoffset > 0.0:
            leftCamAngle = 90.0 - angle1
        elif leftXoffset < 0.0:
            leftCamAngle = 90.0 + angle1
        else:  # leftXoffset is zero
            leftCamAngle = 90.0
        self.lCamAngle = leftCamAngle
        # inside angle of the rightCam in triangulation
        rightCamAngle = 0.0
        angle2 = math.degrees(math.atan(abs(rightXoffset) / self.focLenInPixels))
        if rightXoffset > 0.0:
            rightCamAngle = 90.0 + angle2
        elif rightXoffset < 0.0:
            rightCamAngle = 90.0 - angle2
        else:  # rightXoffset is zero
            rightCamAngle = 90.0
        self.rCamAngle = rightCamAngle

        angleOpoToBaseLine = 180.0 - (leftCamAngle + rightCamAngle)
        self.cCamAngle = angleOpoToBaseLine

        # length of one of the side near to leftCam
        """
        Assuming triangle with three corners having A,B and C angles and sides opposite to it respectively, a,b and c,
        we can use LAW OF SINES
            -> a/sin(A) = b/sin(B) = c/sin(C)
        - In our case we have angle near leftCam, angle near rightCam and the distance between two cameras - baseDist
        - Now, we can find length of any one side using 
        """
        lineOpoToleftCamAngle = (
            self.baseDist * math.sin(math.radians(leftCamAngle))
        ) / math.sin(math.radians(angleOpoToBaseLine))
        self.depthInInch = lineOpoToleftCamAngle * math.sin(math.radians(leftCamAngle))
        print(f"Calculated depth in inch : {self.depthInInch}")

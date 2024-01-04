import cv2
import numpy as np
from cameraFeed import CameraFeed
from depthCamEye import DepthCamEye
import multiprocessing
import math
import time
from multiprocessing import Manager
from triangulation import Triangulation


class DepthCam:
    """Class used to tune depth camera and measure depth of thresholded objects"""

    # slots are used for memory optimization
    __slot__ = [
        "fov",
        "distBtwnCameras",
        "leftCamID",
        "rightCamID",
        "widthRes",
        "heightRes",
        "Hmin",
        "Smin",
        "Vmin",
        "Hmax",
        "Smax",
        "Vmax",
        "calibrationWindow",
    ]

    def __init__(
        self,
        fov: float,
        distBtwnCameras: float,
        leftCamID: int,
        rightCamID: int,
        widthRes: int = 640,
        heightRes: int = 480,
    ) -> None:
        """Initiate `DepthCam`

        Args:
            fov (float): HORIZONTAL Field of view of the camera in degrees
            distBtwnCameras (float): Center distance between two cameras
            leftCamID (int) : Id of the left camera (looking from camera towards object)
            rightCamID (int) : Id of the right camera (looking from camera towards object)
            widthRes (int): Width resolution of camera feed. Defaults to 640.
            heightRes (int): Height resolution of camera feed. Defaults to 480.
        """

        self.fov: float = fov  # field of view of camera
        self.distBtwnCameras: float = distBtwnCameras  # distance between two cameras
        self.leftCamID: int = leftCamID  # id of left camera
        self.rightCamID: int = rightCamID  # id of right camera
        self.widthRes: int = widthRes  # width resolution of the camera
        self.heightRes: int = heightRes  # height resolution of the camera
        self.Hmin: int = 0  # Hue-min color value threshold
        self.Smin: int = 0  # Saturation-min color value threshold
        self.Vmin: int = 0  # Value-min color value threshold
        self.Hmax: int = 179  # Hue-max color value threshold
        self.Smax: int = 255  # Saturation-max color value threshold
        self.Vmax: int = 255  # Value-max color value threshold
        self.calibrationWindow: str = (
            "CalibrationWindow"  # name of the calibration windwow
        )
        self.triangulation = Triangulation(fov, widthRes, distBtwnCameras)

    def startTrackers(self) -> None:
        """Sets trackers on the main camera feed window"""

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
        cv2.createTrackbar("HueMin", self.calibrationWindow, 20, 255, on_trackbar_hmin)
        cv2.createTrackbar("HueMax", self.calibrationWindow, 150, 255, on_trackbar_hmax)
        cv2.createTrackbar("SatMin", self.calibrationWindow, 20, 255, on_trackbar_smin)
        cv2.createTrackbar("SatMax", self.calibrationWindow, 150, 255, on_trackbar_smax)
        cv2.createTrackbar("ValMin", self.calibrationWindow, 20, 255, on_trackbar_vmin)
        cv2.createTrackbar("ValMax", self.calibrationWindow, 150, 255, on_trackbar_vmax)

    def calibrateManually(
        self, Hmin: int, Hmax: int, Smin: int, Smax: int, Vmin: int, Vmax: int
    ) -> None:
        """_summary_

        Args:
            Hmin (int): Hue minimum value
            Hmax (int): Hue maximum value
            Smin (int): Saturation minimum value
            Smax (int): Saturation maximum value
            Vmin (int): Value minimum value
            Vmax (int): Value maximum value
        """
        self.Hmin = Hmin
        self.Hmax = Hmax
        self.Smin = Smin
        self.Smax = Smax
        self.Vmin = Vmin
        self.Vmax = Vmax

    def addTextToCamImage(self, frame: np.ndarray, text: str) -> np.ndarray:
        """adds provided text to the image at predefied origin

        Args:
            frame (np.ndarray): frame fetched from the camera
            text (str): text to be added

        Returns:
            np.ndarray: frame with text added on it
        """
        # adding text to the image to quit once done tuning
        frame = cv2.putText(
            img=frame,
            text=text,
            org=(10, 50),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.25,
            color=(0, 0, 255),
            thickness=3,
        )
        return frame

    def calibrate(self, HSVon: bool = True) -> None:
        """Sets value of threshold for Hue, Saturation and Value channels for both left/right cameras.

        Args:
            HSVon (bool, optional): If False, HSV threshold values are set so that segements green ball from the camera frame. Defaults to True.
        """
        # starting a video feed
        if HSVon:
            self.startTrackers()
        leftCamFeed, rightCamFeed = CameraFeed(self.leftCamID), CameraFeed(
            self.rightCamID
        )
        leftCamFeed.openCameraFeed()
        rightCamFeed.openCameraFeed()

        while True:
            leftFrame = leftCamFeed.retriveFrame()
            rightFrame = rightCamFeed.retriveFrame()
            # Draw vertical line
            if not HSVon:
                leftFrame = cv2.line(
                    leftFrame,
                    (leftFrame.shape[1] // 2, 0),
                    (leftFrame.shape[1] // 2, leftFrame.shape[0]),
                    (0, 0, 255),
                    2,
                )
                rightFrame = cv2.line(
                    rightFrame,
                    (rightFrame.shape[1] // 2, 0),
                    (rightFrame.shape[1] // 2, rightFrame.shape[0]),
                    (0, 0, 255),
                    2,
                )
                leftFrame = cv2.line(
                    leftFrame,
                    (0, leftFrame.shape[0] // 2),
                    (leftFrame.shape[1], leftFrame.shape[0] // 2),
                    (0, 0, 255),
                    2,
                )
                rightFrame = cv2.line(
                    rightFrame,
                    (0, rightFrame.shape[0] // 2),
                    (rightFrame.shape[1], rightFrame.shape[0] // 2),
                    (0, 0, 255),
                    2,
                )
            # connecting two frames into one
            frame = np.concatenate([leftFrame, rightFrame], axis=1)
            if HSVon:
                # converting rgb to hsv space and thresholding at the same time
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
                frame = cv2.inRange(
                    frame,
                    (self.Hmin, self.Smin, self.Vmin),
                    (self.Hmax, self.Smax, self.Vmax),
                )
                # adding text to the image to quit once done tuning
                frame = self.addTextToCamImage(
                    frame, "press `q` once done calibrating!"
                )
            cv2.imshow(self.calibrationWindow, frame)
            # quit when `q` is pressed
            if cv2.waitKey(1) == ord("q"):
                break
        leftCamFeed.release()
        rightCamFeed.release()
        cv2.destroyAllWindows()
        if not HSVon:
            self.calibrateManually(53, 89, 66, 245, 56, 255)
        # get status on set values of HSV values
        print(
            f"Hmin set to {self.Hmin}\nHmax set to {self.Hmax}\nSmin set to {self.Smin}\nSmax set to {self.Smax}\nVmin set to {self.Vmin}\nVmax set to {self.Vmax}"
        )

    def measureDepth(self) -> None:
        """Starts individual camera with its own segmentation pipeline. Keeps checking memory manager for updated data and
        calculates depth usin triangulation followed by updates to the main output window.
        """
        # starting a memory manager to share data between processes
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
            self.leftCamID,
            [self.Hmin, self.Smin, self.Vmin, self.Hmax, self.Smax, self.Vmax],
            widthRes=self.widthRes,
            heightRes=self.heightRes,
        )
        rightEyeProcess = DepthCamEye(
            rightEyeLock,
            rightCamResDict,
            self.rightCamID,
            [self.Hmin, self.Smin, self.Vmin, self.Hmax, self.Smax, self.Vmax],
            widthRes=self.widthRes,
            heightRes=self.heightRes,
        )
        leftEyeProcess.start(), rightEyeProcess.start()

        # starting loop to fetch stored data in memory manager
        while True:
            leftFrame, rightFrame, triangleFrame = None, None, None
            leftXoffset, rightXoffset = 0.0, 0.0
            with leftEyeLock, rightEyeLock:
                # have method to process the frames and X offsets
                leftFrame = leftCamResDict["frame"]
                leftXoffset = leftCamResDict["xOffset"]
                rightFrame = rightCamResDict["frame"]
                rightXoffset = rightCamResDict["xOffset"]
            if leftXoffset is not None and rightXoffset is not None:
                self.triangulation.calcDepth(leftXoffset, rightXoffset)
                triangleFrame = self.triangulation.getImageWithTriangle()
            if leftFrame is not None and rightFrame is not None:
                frame = np.concatenate([leftFrame, rightFrame], axis=1)
                if triangleFrame is not None:
                    frame = np.concatenate([triangleFrame, frame], axis=0)
                frame = self.addTextToCamImage(
                    frame, f"{self.triangulation.depthInInch:>3.1f} in"
                )
                cv2.imshow("ballInSpace", frame)
            # add a condition here to stop the processes!
            if cv2.waitKey(1) == ord("q"):
                break
        # appropreate tremination of the processes and releasing resources
        leftEyeProcess.terminate(), rightEyeProcess.terminate()
        leftEyeProcess.join(), rightEyeProcess.join()
        leftEyeProcess.close(), rightEyeProcess.close()
        cv2.destroyAllWindows()

import multiprocessing
from typing import Any
from CameraFeed import CameraFeed
import numpy as np
import cv2
from typing import Optional


class depthCamEye(multiprocessing.Process):
    def __init__(
        self,
        lock: multiprocessing.Lock,
        resDict: dict,
        cam: int,
        threshParam: list[int],
        widthRes: int = 640,
        heightRes: int = 480,
    ) -> None:
        """A process class that continues to find object in the given image and saves result in `resDict` object.

        Args:
            lock (multiprocessing.Lock): lock object to save data into resDict while avoiding data race
            resDict (dict): dict object where xOffset and current Camera frame is saved
            cam (int): camera id
            threshParam (list[int]): HSV threshold values in order (Hmin, Smin, Vmin, Hmax, Smax, Vmax)
            widthRes (int, optional): width wise resolution of camera. Defaults to 640.
            heightRes (int, optional): height wise resolution of camera. Defaults to 480.
        """
        super().__init__()
        self.lock: multiprocessing.Lock = lock
        self.resDict: dict = resDict
        self.cam: int = cam
        self.threshParam: list[int] = threshParam
        self.widthRes: int = widthRes
        self.heightRes: int = heightRes

    def run(self):
        feed = CameraFeed(self.cam, self.widthRes, self.heightRes)
        feed.openCameraFeed()
        while True:
            # retrive, convert to HSV and threshold the given image
            frame = feed.retriveFrame()
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_threshold = cv2.inRange(
                frame_hsv,
                tuple(self.threshParam[:3]),  # values of (Hmin, Smin, Vmin)
                tuple(self.threshParam[3:]),  # values of (Hmax, Smax, Vmax)
            )

            # find contours in the image
            contours, _ = cv2.findContours(
                frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) != 0:
                largest_contour = max(contours, key=cv2.contourArea)
                # Find object and draw circle around it
                center, radius = cv2.minEnclosingCircle(largest_contour)
                center = tuple(map(int, center))
                radius = int(radius)
                cv2.circle(frame, center, radius, (255, 0, 0), 2)  # Green circle
                frame = cv2.resize(frame, (640, 480))
                xOffset = center[0] - (self.widthRes / 2)

                # saving the offset value of the ball and the frame with circle drawn
                with self.lock:
                    self.resDict["frame"] = frame
                    self.resDict["xOffset"] = xOffset
                    print(f"at {self.cam} prcess id of dict : {id(self.resDict)}, {id(self.resDict['frame'])}, {id(self.resDict['xOffset'])}")
                    print(f"at {self.cam}, id of locks : {id(self.lock)}")
            else:
                # saving the offset value of the ball and the frame with circle drawn
                with self.lock:
                    self.resDict["frame"] = None
                    self.resDict["xOffset"] = None
                    print("Nothing found!")
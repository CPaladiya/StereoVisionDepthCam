import multiprocessing
from typing import Any
from cameraFeed import CameraFeed
import numpy as np
import cv2
from typing import Optional


class DepthCamEye(multiprocessing.Process):
    "A class used to start camera feed, segment the object to obtain object center offset value and camera frame"
    def __init__(
        self,
        lock: multiprocessing.Lock,
        resultDict: dict,
        camID: int,
        threshParam: list[int],
        widthRes: int = 640,
        heightRes: int = 480,
    ) -> None:
        """A process class that loops to find object in the given image and saves result of xOffset value and camera frame in `resultDict` object.

        Args:
            lock (multiprocessing.Lock): lock object to save data into resultDict while avoiding data race
            resultDict (dict): dict object where xOffset and current Camera frame is saved
            camID (int): camera id
            threshParam (list[int]): HSV threshold values in order (Hmin, Smin, Vmin, Hmax, Smax, Vmax)
            widthRes (int, optional): width wise resolution of camera. Defaults to 640.
            heightRes (int, optional): height wise resolution of camera. Defaults to 480.
        """
        super().__init__()
        self.lock: multiprocessing.Lock = lock
        self.resultDict: dict = resultDict
        self.camID: int = camID
        self.threshParam: list[int] = threshParam
        self.widthRes: int = widthRes
        self.heightRes: int = heightRes

    def run(self):
        """Since `DepthCamEye` module is inheriting `multiprocessing.Process`. When starting this process with `DepthCamEye.start()`, this function will be called.
        This function applies HSV thresholding and segments the ball, draws boundry around it, finds xOffset and stores xOffset and camera frame into shared dictionary.
        """
        feed = CameraFeed(self.camID, self.widthRes, self.heightRes)
        feed.openCameraFeed()
        try:
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
                    cv2.circle(frame, center, radius, (0, 255, 255), 12)  # Green circle
                    frame = cv2.resize(frame, (self.widthRes // 6, self.heightRes // 6))
                    xOffset = center[0] - (self.widthRes / 2)

                    # saving the offset value of the ball and the frame with circle drawn
                    with self.lock:
                        self.resultDict["frame"] = frame
                        self.resultDict["xOffset"] = xOffset
                else:
                    # saving the offset value of the ball and the frame with circle drawn
                    with self.lock:
                        self.resultDict["frame"] = None
                        self.resultDict["xOffset"] = None
        # if anything goes wrong, release the camera resources
        except:
            feed.release()

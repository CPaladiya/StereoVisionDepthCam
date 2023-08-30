import cv2
import numpy as np


class CameraFeed:
    """Class that handles setup of camera resolution and closing/opening of camera feed along with releasing of resources once done"""

    # slots are used for memory optimization
    __slots__ = ["camID", "widthRes", "heightRes", "feed"]

    def __init__(self, camID: int, widthRes: int = 640, heightRes: int = 480) -> None:
        """Initiates CameraFeed object.

        Args:
            camID (int): Camera id
            widthRes (int, optional): width resolution of the camera. Defaults to 640.
            heightRes (int, optional): height resolution of the camera. Defaults to 480.
        """
        self.camID: int = camID
        self.widthRes: int = widthRes
        self.heightRes: int = heightRes
        self.feed: cv2.VideoCapture = None

    def openCameraFeed(self) -> None:
        """Open a camera feed, sets resolution and performs essential checks

        Raises:
            ValueError: Indicates if camera with `CameraFeed.camID` does not exist
            BrokenPipeError: Indicates if camera feed is not successfully opened
        """
        camCaputre = cv2.VideoCapture(self.camID)
        if camCaputre == None:
            raise ValueError(f"camera {self.camID} does not exists!")
        if not camCaputre.isOpened():
            raise BrokenPipeError(f"camera {self.camID} feed could not be opened!")
        self.feed = camCaputre

        if self.heightRes != None:
            self.feed.set(cv2.CAP_PROP_FRAME_HEIGHT, self.heightRes)
        if self.widthRes != None:
            self.feed.set(cv2.CAP_PROP_FRAME_WIDTH, self.widthRes)

    def retriveFrame(self) -> np.ndarray:
        """Retrive the frame from the camera feed
        Raises:
            BrokenPipeError: Raised if the frame is not being able to retrive

        Returns:
            np.ndarray: returning camera frame
        """
        retVal, frame = self.feed.read()
        if not retVal:
            raise BrokenPipeError(
                f"could not retrive frame from camera {self.feed.get(cv2.CAP_PROP_VIDEO_STREAM)}"
            )
        return frame

    def release(self) -> None:
        """Release resources occupied by this camera feed"""
        self.feed.release()

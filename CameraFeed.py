import cv2
import numpy as np

class CameraFeed():
    """Class that handles setup of camera resolution and closing/opening of camera feed"""
    
    # slots are used for memory optimization
    __slots__=[
        "cam",
        "widthRes",
        "heightRes",
        "feed"
    ]
    def __init__(self, cam: int, widthRes: int = 640, heightRes: int = 480) -> None:
        """Initiates CameraFeed object.

        Args:
            cam (int): Camera id
            widthRes (int, optional): width resolution of the camera. Defaults to 640.
            heightRes (int, optional): height resolution of the camera. Defaults to 480.
        """
        self.cam: int = cam
        self.widthRes: int = widthRes
        self.heightRes: int = heightRes
        self.feed: cv2.VideoCapture = None
        
    def openCameraFeed(self) -> None:
        """Open a camera feed, sets resolution and performs essential checks

        Raises:
            ValueError: Indicates if camera does not exist
            BrokenPipeError: Indicates if camera feed is not successfully opened
        """
        camCaputre = cv2.VideoCapture(self.cam)
        if camCaputre == None:
            raise ValueError(f"camera {self.cam} does not exists!")
        if not camCaputre.isOpened():
            raise BrokenPipeError(f"camera {self.cam} feed could not be opened!")
        self.feed = camCaputre
        
        if self.heightRes != None:
            self.feed.set(cv2.CAP_PROP_FRAME_HEIGHT, self.heightRes)
        if self.widthRes != None:
            self.feed.set(cv2.CAP_PROP_FRAME_WIDTH, self.widthRes)
            

    def retriveFrame(self) -> np.ndarray:
        """Retrive the frame from the camera feed

        Args:
            cameraFeed (cv2.VideoCapture): camera feed

        Raises:
            BrokenPipeError: Raised if the frame is not being able to retrive

        Returns:
            np.ndarray: returning frame
        """
        retVal, frame = self.feed.read()
        if not retVal:
            raise BrokenPipeError(
                f"could not retrive frame from camera {self.feed.get(cv2.CAP_PROP_VIDEO_STREAM)}"
            )
        return frame
    
    def release(self)->None:
        """Release resources occupied by this camera feed
        """
        self.feed.release()
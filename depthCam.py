import cv2
import numpy as np

class DepthCam():
    """Class used to start and tune depth camera
    """
    # slots are used for memory optimization
    __slot__ = ["FOV", "BaseDist", "widthRes", "Hmin", "Smin", "Vmin", "Hmax", "Smax", "Vmax", "namedWindow"]
    
    def __init__(self, FOV:float, BaseDist:float, widthRes:int, heightRes:int) -> None:
        
        self.FOV            :float  = FOV           # field of view of camera
        self.BaseDist       :float  = BaseDist      # distance between two cameras
        self.widthRes       :int    = widthRes      # width resolution of the camera
        self.heightRes      :int    = heightRes     # height resolution of the camera
        self.Hmin           :int    = 0             # Hue-min color value threshold
        self.Smin           :int    = 0             # Sat-min color value threshold
        self.Vmin           :int    = 0             # Val-min color value threshold
        self.Hmax           :int    = 179           # Hue-max color value threshold
        self.Smax           :int    = 255           # Sat-max color value threshold
        self.Vmax           :int    = 255           # Val-max color value threshold
        self.namedWindow    :str    = 'Feed'
    
    
    def tune(self, cam1:int=0, cam2:int=1) -> None:
        """Sets value of threshold for Hue, Saturation and Value channels. It requires exactly two cameras.
        The video feed can be switch left to right by swapping cam1 and cam2 id.

        Args:
            cam1 (int, optional): Device id of first camera. Defaults to 0.
            cam2 (int, optional): Device id of second camera. Defaults to 1.
        """ 
        # starting a video feed
        self.startTrackers()
        camCaputre1 = self.openCameraFeed(cam1)
        camCaputre2 = self.openCameraFeed(cam2)
        while True:
            frame1 = self.retriveFrame(camCaputre1)
            frame2 = self.retriveFrame(camCaputre2)
            # connecting two frames into one
            frame = np.concatenate([frame1, frame2], axis=1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame = cv2.inRange(frame, (self.Hmin, self.Smin, self.Vmin), (self.Hmax, self.Smax, self.Vmax))
            # adding text to the image to quit once done tuning
            frame = cv2.putText(
                                    img = frame,
                                    text = "press `q` once done tuning!",
                                    org = (10, 50),
                                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                                    fontScale = 1.5,
                                    color = (88, 86, 93),
                                    thickness = 3)
            cv2.imshow(self.namedWindow,frame)
            # quit when `q` is pressed
            if cv2.waitKey(1) == ord('q'):
                break
        camCaputre1.release()
        camCaputre2.release()
        cv2.destroyAllWindows()
        
    
    def startTrackers(self) -> None:
        """sets trackers on the main camera feed window
        """
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
        cv2.namedWindow(self.namedWindow)
        # by default minimum value of slider is always 0
        cv2.createTrackbar("HueMin", self.namedWindow, 20, 179, on_trackbar_hmin)
        cv2.createTrackbar("HueMax", self.namedWindow, 150, 179, on_trackbar_hmax)
        cv2.createTrackbar("SatMin", self.namedWindow, 20, 255, on_trackbar_smin)
        cv2.createTrackbar("SatMax", self.namedWindow, 150, 255, on_trackbar_smax)
        cv2.createTrackbar("ValMin", self.namedWindow, 20, 255, on_trackbar_vmin)
        cv2.createTrackbar("ValMax", self.namedWindow, 150, 255, on_trackbar_vmax)
        
    
    def openCameraFeed(self,cam:int) -> cv2.VideoCapture:
        """Open a camera feed and performs essential checks

        Args:
            cam (int): camera id

        Raises:
            ValueError: Indicates if camera does not exist
            BrokenPipeError: Indicates if camera feed is not successfully opened

        Returns:
            cv2.VideoCapture: cv2 video capture object
        """
        camCaputre = cv2.VideoCapture(cam)
        if camCaputre == None:
            raise ValueError(f"camera {cam} does not exists!")
        if not camCaputre.isOpened():
            raise BrokenPipeError(f"camera {cam} feed could not be opened!")
        return camCaputre
    
    
    def retriveFrame(self, cameraFeed:cv2.VideoCapture) -> np.ndarray:
        """Retrive the frame from the camera feed

        Args:
            cameraFeed (cv2.VideoCapture): camera feed

        Raises:
            BrokenPipeError: Raised if the frame is not being able to retrive

        Returns:
            np.ndarray: returning frame
        """
        retVal,frame = cameraFeed.read()
        if not retVal:
            raise BrokenPipeError(f"could not retrive frame from camera {cameraFeed.get(cv2.CAP_PROP_VIDEO_STREAM)}")
        return frame
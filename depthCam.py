import cv2
import numpy as np

class DepthCam():
    """Class used to start and tune depth camera
    """
    # slots are used for memory optimization
    __slot__ = ["FOV", "BaseDist", "widthRes", "heightRes", "RThres", "GThres", "BThres"]
    
    def __init__(self, FOV:float, BaseDist:float, widthRes:int, heightRes:int) -> None:
        self.FOV        :float  = FOV           # field of view of camera
        self.BaseDist   :float  = BaseDist      # distance between two cameras
        self.widthRes   :int    = widthRes      # width resolution of the camera
        self.heightRes  :int    = heightRes     # height resolution of the camera
        self.HThres     :int    = 255          # Hue color value threshold
        self.SThres     :int    = 255          # Sat color value threshold
        self.VThres     :int    = 255          # Val color value threshold
    
    
    def tune(self, cam1:int=0, cam2:int=1, H:bool=False, S:bool=False, V:bool=False) -> None:
        """Sets value of threshold for red, green and blue value.
        Function assumes that two cameras are connected and they are not built in pc webcams.

        Args:
            cam1 (int, optional): Device id of first camera. Defaults to 0.
            cam2 (int, optional): Device id of second camera. Defaults to 1.
            H (bool, optional): set `True` to tune tuning of Hue channel. Defaults to False.
            S (bool, optional): set `True` to tune tuning of Sat channel. Defaults to False.
            V (bool, optional): set `True` to tune tuning of Val channel. Defaults to False.
        """ 
            
        # starting a video feed
        camCaputre1 = self.OpenCameraFeed(cam1)
        camCaputre2 = self.OpenCameraFeed(cam2)
        hmin, smin, vmin = 0,0,0
        hmax, smax, vmax = 179,255,255
        
        def on_trackbar(val):
            print(val)
            
        cv2.namedWindow("HSVTrackBars")
        cv2.createTrackbar("HueMin", "HSVTrackBars", 0, 179, on_trackbar)
        
        while True:
            
            frame1 = self.retriveFrame(camCaputre1)
            frame2 = self.retriveFrame(camCaputre2)

            cv2.imshow(f"camId_{cam1}",frame1)
            cv2.imshow(f"camId_{cam2}",frame2)
            if cv2.waitKey(1) == ord('q'):
                break
        camCaputre1.release()
        camCaputre2.release()
        cv2.destroyAllWindows()
        
        
    def OpenCameraFeed(self,cam:int) -> cv2.VideoCapture:
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
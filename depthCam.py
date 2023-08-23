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
        self.RThres     :int    = 255          # Red color value threshold
        self.GThres     :int    = 255          # Gree color value threshold
        self.BThres     :int    = 255          # Blue color value threshold
    
    
    def tune(self, cam1:int=0, cam2:int=1, R:bool=False, G:bool=False, B:bool=False) -> None:
        """Sets value of threshold for red, green and blue value.
        Function assumes that two cameras are connected and they are not built in pc webcams.

        Args:
            cam1 (int, optional): Device id of first camera. Defaults to 0.
            cam2 (int, optional): Device id of second camera. Defaults to 1.
            R (bool, optional): set `True` to tune tuning of red channel. Defaults to False.
            G (bool, optional): set `True` to tune tuning of green channel. Defaults to False.
            B (bool, optional): set `True` to tune tuning of blue channel. Defaults to False.
        """ 
        # starting a video feed
        camCaputre1 = self.OpenCameraFeed(cam1)
        camCaputre2 = self.OpenCameraFeed(cam2)
        
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
                    # get the frame from the camera
            retVal,frame = cameraFeed.read()
            print(f"Getting frames from camera {cameraFeed.get(cv2.CAP_PROP_VIDEO_STREAM)}")
            if not retVal:
                raise BrokenPipeError(f"could not retrive frame from camera {cameraFeed.get(cv2.CAP_PROP_VIDEO_STREAM)}")
            return frame
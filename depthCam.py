
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
        for cam in [cam1, cam2]:
            pass
        
        
        
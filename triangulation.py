import cv2
import numpy as np
import math


class Triangulation:
    """Class used to calculate depth of the object from depth camera using offset values determined using left/right camera frame"""

    # using slots for memory optimization
    __slots__ = {
        "fov",
        "widthRes",
        "distBtwnCameras",
        "focLenInPixels",
        "maxAllowedTriangleSideLen",
        "lCamAngle",
        "rCamAngle",
        "cCamAngle",
        "lineOpoToRCamAngle_xComp",
        "lineOpoToRCamAngle_yComp",
        "triangleCoord",
        "depthInInch",
        "camImage",
        "ballImage",
    }

    def __init__(
        self, fov: float, widthRes: int = 640, distBtwnCameras: float = 480
    ) -> None:
        """Initiate triangulation object

        Args:
            fov (float): field of view of the camera.
            widthRes (int): width resolution of camera. Defaults to 640.
            distBtwnCameras (float): height resolution of camera. Defaults to 480.
        """
        self.fov = fov
        self.widthRes = widthRes
        self.distBtwnCameras = distBtwnCameras
        self.camImage = cv2.imread("data/camera.png")
        self.ballImage = cv2.imread("data/ball.jpeg")

    def calcFocalLengthInPixels(self) -> None:
        """Calculates focal length in pixel which will be used in triagulation calculations"""
        try:
            halfOfImageWidth = self.widthRes / 2
            halfOfFOV = self.fov / 2
            self.focLenInPixels = int(
                round(halfOfImageWidth / math.tan(math.radians(halfOfFOV)))
            )
        except:
            raise ValueError("Focal length in pixels could not be calculated")

    def calcAnglesFromOffsets(self, leftXoffset: int, rightXoffset: int) -> None:
        """Calculates all three angles of triangle using focal length in pixels and offset from both left and right cameras

        Args:
            leftXoffset (int): xOffset of object from left camera
            rightXoffset (int): xOffset of object from right camera
        """
        self.calcFocalLengthInPixels()
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
        self.cCamAngle = 180.0 - (self.lCamAngle + self.rCamAngle)

    def calcDepth(self, leftXoffset: int, rightXoffset: int) -> None:
        """Calculate depth of the object using offset of the object for left and right camera frame.

            Assuming triangle with three corners having A,B and C angles and sides opposite to it respectively, a,b and c,
            we can use LAW OF SINES -> a/sin(A) = b/sin(B) = c/sin(C)
                -> In our case, we have angle near leftCam, angle near rightCam and the distance between two cameras - distBtwnCameras.
                -> We can find length of any one side using LAW OF SINES

        Args:
            leftXoffset (int): _description_
            rightXoffset (int): _description_
        """
        try:
            # calculate angles from offsets
            self.calcAnglesFromOffsets(leftXoffset, rightXoffset)
            # calculate length of side using LAW OF SINES
            knownRatio = self.distBtwnCameras / math.sin(math.radians(self.cCamAngle))
            lineOpoToRCamAngle = math.sin(math.radians(self.rCamAngle)) * knownRatio
            # Finally, calculate depth
            self.depthInInch = lineOpoToRCamAngle * math.sin(
                math.radians(self.lCamAngle)
            )
            # Calculate X,Y component of triangle side opposite to right camera angle
            self.lineOpoToRCamAngle_yComp = self.depthInInch
            self.lineOpoToRCamAngle_xComp = lineOpoToRCamAngle * math.cos(
                math.radians(self.lCamAngle)
            )
        except Exception as e:
            self.depthInInch = 0.0
            print(f"error {e} encountered. Depth is set to zero!")

    def getImageWithTriangle(self) -> np.ndarray:
        """Draw a blank image with triangle drawn on it!

        Returns:
            np.ndarray: returns image with triangle drawn on it.
        """
        # get triangle coordinates
        self.calcTrianlgeCoordinates()
        if self.depthInInch > 0.0:
            # create blank image
            frame = np.zeros(
                (self.maxAllowedTriangleSideLen, self.maxAllowedTriangleSideLen, 3),
                dtype=np.uint8,
            )
            # Reshape vertices into shape required by cv2.polylines
            vertices = self.triangleCoord.reshape((-1, 1, 2))
            # Draw the triangle on the image
            frame = cv2.polylines(
                frame, [vertices], isClosed=True, color=(0, 0, 255), thickness=2
            )

            # add symbols for camera and ball along with relevant texts
            frame = self.addTextToTriangleImage(
                frame, self.triangleCoord[0] + [10, -10], f"{self.lCamAngle:>3.2f}"
            )
            frame = self.addTextToTriangleImage(
                frame, self.triangleCoord[0] + [-70, 30], f"(Left)"
            )
            frame = self.addImgSymbolToTriangleImage(
                frame, self.triangleCoord[0] + [-20, 10], self.camImage
            )
            frame = self.addTextToTriangleImage(
                frame, self.triangleCoord[1] + [-50, -10], f"{self.rCamAngle:>3.2f}"
            )
            frame = self.addTextToTriangleImage(
                frame, self.triangleCoord[1] + [20, 30], f"(Right)"
            )
            frame = self.addImgSymbolToTriangleImage(
                frame, self.triangleCoord[1] + [-20, 10], self.camImage
            )
            frame = self.addTextToTriangleImage(
                frame, self.triangleCoord[2] + [-20, 45], f"{self.cCamAngle:>3.2f}"
            )
            frame = self.addTextToTriangleImage(
                frame, self.triangleCoord[2] + [25, 5], f"(ball)"
            )
            frame = self.addImgSymbolToTriangleImage(
                frame, self.triangleCoord[2] + [-20, -20], self.ballImage
            )
            return frame
        else:
            return None

    def calcMaxAllowedTriSideLen(self) -> None:
        """calculate maximum allowed length of the biggest side of the triangle"""
        self.maxAllowedTriangleSideLen = 2 * (self.widthRes // 6)

    def calcScaleForTraingle(self) -> float:
        """calculates the scale amount that triangle needs to be scaled at to fit in the image perfectly

        Returns:
            float: returns the triangle scale value
        """
        self.calcMaxAllowedTriSideLen()
        dimThreshold = (
            0.8 * self.maxAllowedTriangleSideLen
        )  # traingle should be 80% of total image
        scale = 1.0
        # if triangle is acute, meaning all angles are less than 90 degrees OR rightangle triangle
        if self.cCamAngle <= 90 and self.lCamAngle <= 90 and self.rCamAngle <= 90:
            if self.distBtwnCameras > self.depthInInch:
                scale = dimThreshold / self.distBtwnCameras
            else:
                scale = dimThreshold / self.depthInInch
        # if triangle is obtuse, one angle >90
        else:
            reqHalfDist = abs(
                self.triangleCoord[2][0]
                - (self.triangleCoord[0][0] + self.distBtwnCameras / 2)
            )
            availHalfDist = dimThreshold / 2
            if (self.depthInInch / 2) > reqHalfDist:
                scale = dimThreshold / self.depthInInch
            else:
                scale = availHalfDist / reqHalfDist
        return scale

    def calcTrianlgeCoordinates(self) -> None:
        """calculate triangle coordinate so that triangle is scaled maximum possible to fit in the image
        while making sure that the center of the line connecting both cameras are always at the middle of the image
        """
        scale = self.calcScaleForTraingle()
        # calculating triangle coordinates
        self.triangleCoord = []
        self.triangleCoord.append([0.0, 0.0])  # left triangle point - camera1
        self.triangleCoord.append(
            [self.distBtwnCameras, 0.0]
        )  # right triangle point - camera2
        self.triangleCoord.append(
            [self.lineOpoToRCamAngle_xComp, self.lineOpoToRCamAngle_yComp]
        )
        # converting to numpy array
        self.triangleCoord = np.array(
            [[i[0], i[1]] for i in self.triangleCoord], np.float32
        )
        # scale the triangle
        self.triangleCoord = self.triangleCoord * [scale, scale]
        # shift trainlge to fit in the image
        triangleXShift = self.maxAllowedTriangleSideLen / 2 - (
            scale * self.distBtwnCameras / 2
        )
        triangleYShift = 0.1 * self.maxAllowedTriangleSideLen
        self.triangleCoord = self.triangleCoord + [triangleXShift, triangleYShift]
        # currently triangle has left bottom at (0,0) but images have top left at (0,0) so flip Y
        self.triangleCoord = [0, self.maxAllowedTriangleSideLen] - self.triangleCoord
        self.triangleCoord *= [-1.0, 1.0]
        # convert numpy array to int to be compatible with pixel positions.
        self.triangleCoord = self.triangleCoord.astype(np.int32)

    def addTextToTriangleImage(
        self, frame: np.ndarray, origin: np.ndarray, text: str
    ) -> np.ndarray:
        """Add text to the camera frame(image)

        Args:
            frame (np.ndarray): camara frame where text is to be added
            origin (np.ndarray): origin of the text in the image
            text (str): text to be added on the camera frame

        Returns:
            np.ndarray: image with text added on it
        """
        frame = cv2.putText(
            img=frame,
            text=text,
            org=origin,
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.50,
            color=(225, 225, 255),
            thickness=1,
        )
        return frame

    def addImgSymbolToTriangleImage(
        self, frame: np.ndarray, origin: np.ndarray, imageSymbol: np.ndarray
    ) -> np.ndarray:
        """Add image symbol to the camera frame image.

        Args:
            frame (np.ndarray): camara frame on which symbol needs to be added
            origin (np.ndarray): origin of the symbolic image on camera frame
            imageSymbol (np.ndarray): symbolic image

        Returns:
            np.ndarray: image with symbol added on it
        """
        # Get the dimensions of camera image
        height, width = imageSymbol.shape[:2]
        frame[
            origin[1] : origin[1] + height, origin[0] : origin[0] + width
        ] = imageSymbol
        return frame

import cv2
import numpy as np
import math

class Triangulation():
    # using slots for memory optimization
    __slots__ = {
        "fov",
        "widthRes",
        "distBtwnCameras",
        "focLenInPixels",
        "triangleImgMaxDim",
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
    
    def __init__(self, fov: float, widthRes: int, distBtwnCameras: float) -> None:
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
        
    def fetchAnglesFromOffset(self, leftXoffset: int, rightXoffset: int) -> None:
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

    def performTriangulation(self, leftXoffset: int, rightXoffset: int) -> None:
        """
        Assuming triangle with three corners having A,B and C angles and sides opposite to it respectively, a,b and c,
        we can use LAW OF SINES
            -> a/sin(A) = b/sin(B) = c/sin(C)
        - In our case we have angle near leftCam, angle near rightCam and the distance between two cameras - distBtwnCameras
        - Now, we can find length of any one side using
        """
        try:
            self.fetchAnglesFromOffset(leftXoffset, rightXoffset)
            knownRatio = self.distBtwnCameras / math.sin(math.radians(self.cCamAngle))
            lineOpoToRCamAngle = math.sin(math.radians(self.rCamAngle)) * knownRatio
            self.depthInInch = lineOpoToRCamAngle * math.sin(
                math.radians(self.lCamAngle)
            )
            self.lineOpoToRCamAngle_yComp = self.depthInInch
            self.lineOpoToRCamAngle_xComp = lineOpoToRCamAngle * math.cos(
                math.radians(self.lCamAngle)
            )
        except Exception as e:
            self.depthInInch = 0.0
            print(f"error {e} encountered!")
        if type(self.depthInInch) == float and self.depthInInch < 0.0:
            print("Most likely your right and left camera is in wrong order!")

    def drawImageWithTriangle(self) -> np.ndarray:
        self.getTrianlgeCoordinates()
        if self.depthInInch > 0.0:
            # draw this coordinates on a blank image - let's create blank image
            frame = np.zeros((self.triangleImgMaxDim, self.triangleImgMaxDim, 3), dtype=np.uint8)
            # Reshape vertices into shape required by cv2.polylines
            vertices = self.triangleCoord.reshape((-1, 1, 2))
            # Draw the triangle on the image
            frame = cv2.polylines(
                frame, [vertices], isClosed=True, color=(0, 0, 255), thickness=2
            )
            # adding text to the image to quit once done tuning
            frame = self.addTextToTriangleImage(
                frame, self.triangleCoord[0] + [10, -10], f"{self.lCamAngle:>3.2f}"
            )
            frame = self.addTextToTriangleImage(frame, self.triangleCoord[0] + [-70, 30], f"(Left)")
            frame = self.addImgToTriangleImage(
                self.camImage, frame, self.triangleCoord[0] + [-20, 10]
            )
            frame = self.addTextToTriangleImage(
                frame, self.triangleCoord[1] + [-50, -10], f"{self.rCamAngle:>3.2f}"
            )
            frame = self.addTextToTriangleImage(frame, self.triangleCoord[1] + [20, 30], f"(Right)")
            frame = self.addImgToTriangleImage(
                self.camImage, frame, self.triangleCoord[1] + [-20, 10]
            )
            frame = self.addTextToTriangleImage(
                frame, self.triangleCoord[2] + [-20, 45], f"{self.cCamAngle:>3.2f}"
            )
            frame = self.addTextToTriangleImage(frame, self.triangleCoord[2] + [25, 5], f"(ball)")
            frame = self.addImgToTriangleImage(
                self.ballImage, frame, self.triangleCoord[2] + [-20, -20]
            )

            return frame
        else:
            return None
    
    def getTriangleImgMaxDim(self):
        # decide image size
        self.triangleImgMaxDim = 2 * (self.widthRes // 6)
        
    def getTrianlgeCoordinates(self):
        self.getTriangleImgMaxDim()
        dimThreshold = 0.8 * self.triangleImgMaxDim  # traingle should be 80% of total image
        # calculating triangle coordinates
        self.triangleCoord = []
        self.triangleCoord.append([0.0, 0.0])  # left triangle point - camera1
        self.triangleCoord.append([self.distBtwnCameras, 0.0])  # right triangle point - camera2
        self.triangleCoord.append(
            [self.lineOpoToRCamAngle_xComp, self.lineOpoToRCamAngle_yComp]
        )  # calculated cetner point
        self.triangleCoord = np.array(
            [[i[0], i[1]] for i in self.triangleCoord], np.float32
        )  # converting to np array
        scale = 1.0
        # if triangle is acute, meaning all angles are less than 90 degrees OR rightangle triangle
        if self.cCamAngle <= 90 and self.lCamAngle <= 90 and self.rCamAngle <= 90:
            if self.distBtwnCameras > self.depthInInch:
                scale = dimThreshold / self.distBtwnCameras
            else:
                scale = dimThreshold / self.depthInInch
        # if triangle is obtuse, one angle >90
        else:
            reqHalfDist = abs(self.triangleCoord[2][0] - (self.triangleCoord[0][0] + self.distBtwnCameras / 2))
            availHalfDist = dimThreshold / 2
            if (self.depthInInch / 2) > reqHalfDist:
                scale = dimThreshold / self.depthInInch
            else:
                scale = availHalfDist / reqHalfDist
        self.triangleCoord = self.triangleCoord * [scale, scale]
        triangleXShift = self.triangleImgMaxDim / 2 - (scale * self.distBtwnCameras / 2)
        triangleYShift = 0.1 * self.triangleImgMaxDim
        self.triangleCoord = self.triangleCoord + [triangleXShift, triangleYShift]
        self.triangleCoord = [0, self.triangleImgMaxDim] - self.triangleCoord
        self.triangleCoord *= [-1.0, 1.0]
        self.triangleCoord = self.triangleCoord.astype(np.int32)

    def addTextToTriangleImage(
        self, frame: np.ndarray, origin: np.ndarray, text: str
    ) -> np.ndarray:
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

    def addImgToTriangleImage(
        self, image: np.ndarray, frame: np.ndarray, origin: np.ndarray
    ) -> np.ndarray:
        # Get the dimensions of camera image
        height, width = image.shape[:2]
        frame[origin[1] : origin[1] + height, origin[0] : origin[0] + width] = image
        return frame
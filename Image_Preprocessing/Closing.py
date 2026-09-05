import cv2
import numpy as np

class Closing:
    def CLS(self, spath, dpath):
        # Read grayscale class diagram image
        img = cv2.imread(spath)

        # Threshold to binary
        _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)

        # Define structuring element (small cross to preserve line structure)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        # Apply morphological closing
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Invert back (optional)
        result = cv2.bitwise_not(closed)

        # Save result
        cv2.imwrite(dpath, result)

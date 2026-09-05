import os
import cv2
import numpy as np
import pytesseract


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

os.environ["TESSDATA_PREFIX"] = (
    r"C:\Program Files\Tesseract-OCR\tessdata"
)


class Text_Extraction:
    """
    OCR and attribute extraction for preprocessed UI design images.

    Methodology notation:
        J_phi : extracted OCR text
        A_h   : extracted text attributes

    A_h contains:
        - recognized text
        - text bounding box
        - OCR confidence
        - estimated visual font size
        - line-based style characteristics
    """

    def __init__(self):
        self.extracted_text = ""
        self.attributes = []

    # ========================================================
    # OCR TEXT EXTRACTION
    # ========================================================

    def extract_text(self, image_path):
        """
        Extract text J_phi and attributes A_h from the input image.

        Parameters
        ----------
        image_path : str
            Path to the preprocessed UI image C_epsilon_n.

        Returns
        -------
        dict
            {
                "J_phi": complete extracted text,
                "A_h": list of extracted attributes
            }
        """

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                "Unable to read image: " + image_path
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # image_to_data provides recognized text together with
        # bounding boxes and OCR confidence scores.
        data = pytesseract.image_to_data(
            gray,
            lang="eng",
            output_type=pytesseract.Output.DICT,
            config="--psm 11"
        )

        extracted_words = []
        attributes = []

        for i in range(len(data["text"])):

            text = str(data["text"][i]).strip()

            try:
                confidence = float(data["conf"][i])
            except (ValueError, TypeError):
                confidence = -1.0

            # Ignore empty and invalid OCR detections.
            if not text or confidence < 0:
                continue

            # ------------------------------------------------
            # TEXT BOUNDING BOX
            # ------------------------------------------------

            x = int(data["left"][i])
            y = int(data["top"][i])
            width = int(data["width"][i])
            height = int(data["height"][i])

            x2 = x + width
            y2 = y + height

            extracted_words.append(text)

            # ------------------------------------------------
            # ESTIMATED VISUAL FONT SIZE
            # ------------------------------------------------

            font_size = self.estimate_font_size(height)

            # ------------------------------------------------
            # TEXT REGION FOR STYLE ANALYSIS
            # ------------------------------------------------

            text_region = gray[
                max(0, y):min(gray.shape[0], y2),
                max(0, x):min(gray.shape[1], x2)
            ]

            # ------------------------------------------------
            # HOUGH-LINE-BASED STYLE ANALYSIS
            # ------------------------------------------------

            style = self.detect_text_style(text_region)

            # ------------------------------------------------
            # ATTRIBUTE REPRESENTATION A_h
            # ------------------------------------------------

            attribute = {
                "text": text,

                "bounding_box": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "x2": x2,
                    "y2": y2
                },

                # Tesseract reports confidence on approximately 0-100 scale.
                "confidence": confidence / 100.0,

                # This is an image-space estimate in pixels.
                "estimated_font_size_px": font_size,

                "font_style": style
            }

            attributes.append(attribute)

        # J_phi: complete extracted text
        self.extracted_text = " ".join(extracted_words)

        # A_h: structured OCR attributes
        self.attributes = attributes

        return {
            "J_phi": self.extracted_text,
            "A_h": self.attributes
        }

    # ========================================================
    # FONT SIZE ESTIMATION
    # ========================================================

    @staticmethod
    def estimate_font_size(text_height):
        """
        Estimate visual font size using OCR bounding-box height.

        This value is expressed in image pixels and should not be
        interpreted as the original CSS px, Android sp, or point size.
        """

        if text_height <= 0:
            return 0.0

        return float(text_height)

    # ========================================================
    # HOUGH-LINE-BASED TEXT STYLE ANALYSIS
    # ========================================================

    @staticmethod
    def detect_text_style(text_region):
        """
        Detect visible horizontal line-based text characteristics.

        Hough analysis is used here for evidence of:
            - underline
            - strike-through

        It is not used to claim detection of font family, bold, or italic.
        """

        if text_region is None or text_region.size == 0:
            return {
                "underline": False,
                "strike_through": False,
                "horizontal_line_count": 0
            }

        edges = cv2.Canny(
            text_region,
            50,
            150
        )

        region_height, region_width = text_region.shape

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=10,
            minLineLength=max(
                5,
                int(region_width * 0.40)
            ),
            maxLineGap=3
        )

        underline = False
        strike_through = False
        horizontal_line_count = 0

        if lines is not None:

            for line in lines:

                x1, y1, x2, y2 = line[0]

                angle = abs(
                    np.degrees(
                        np.arctan2(
                            y2 - y1,
                            x2 - x1
                        )
                    )
                )

                # Horizontal or nearly horizontal line.
                if angle <= 10 or angle >= 170:

                    horizontal_line_count += 1

                    line_y = (y1 + y2) / 2.0

                    # Lower region: underline candidate.
                    if line_y >= region_height * 0.70:
                        underline = True

                    # Middle region: strike-through candidate.
                    elif (
                        region_height * 0.35
                        <= line_y
                        <= region_height * 0.65
                    ):
                        strike_through = True

        return {
            "underline": underline,
            "strike_through": strike_through,
            "horizontal_line_count": horizontal_line_count
        }


# ============================================================
# OPTIONAL STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    # Change this path to your preprocessed UI image.
    image_path = (
        r"..\Output\UI\Preprocessing"
        r"\sample_ForegroundAware_CLAHE.png"
    )

    extractor = Text_Extraction()

    result = extractor.extract_text(image_path)

    print("\nExtracted Text (J_phi)")
    print("======================")
    print(result["J_phi"])

    print("\nExtracted Attributes (A_h)")
    print("==========================")

    for attribute in result["A_h"]:
        print(attribute)


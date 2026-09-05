import os
import cv2
import numpy as np


class BlankAwareCLAHEPreprocessing:
    """
    Foreground-aware CLAHE preprocessing for UI design images.
    """

    @staticmethod
    def is_informative_tile(
            tile,
            white_threshold=245,
            min_foreground_ratio=0.03,
            min_variance=5.0):
        """
        Determine whether a local tile contains meaningful UI content.

        Returns
        -------
        informative : bool
        foreground_ratio : float
        variance : float
        """

        if tile is None or tile.size == 0:
            return False, 0.0, 0.0

        foreground_ratio = float(
            np.mean(tile < white_threshold)
        )

        variance = float(np.var(tile))

        informative = (
            foreground_ratio >= min_foreground_ratio
            and variance >= min_variance
        )

        return informative, foreground_ratio, variance

    @staticmethod
    def enhance(
            image_path,
            output_path,
            clip_limit=2.0,
            tiles_x=8,
            tiles_y=8,
            white_threshold=245,
            min_foreground_ratio=0.03,
            min_variance=5.0):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        height, width = gray.shape

        tile_w = int(np.ceil(width / tiles_x))
        tile_h = int(np.ceil(height / tiles_y))

        enhanced = gray.copy()

        informative_tiles = 0
        blank_tiles = 0

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(2, 2)
        )

        tile_statistics = []

        for row in range(tiles_y):
            for col in range(tiles_x):
                x1 = col * tile_w
                y1 = row * tile_h
                x2 = min(x1 + tile_w, width)
                y2 = min(y1 + tile_h, height)

                if x1 >= width or y1 >= height:
                    continue

                tile = gray[y1:y2, x1:x2]

                informative, foreground_ratio, variance = (
                    BlankAwareCLAHEPreprocessing.is_informative_tile(
                        tile,
                        white_threshold=white_threshold,
                        min_foreground_ratio=min_foreground_ratio,
                        min_variance=min_variance
                    )
                )

                if informative:
                    enhanced[y1:y2, x1:x2] = clahe.apply(tile)
                    informative_tiles += 1
                else:
                    enhanced[y1:y2, x1:x2] = tile
                    blank_tiles += 1

                tile_statistics.append({
                    "row": row,
                    "column": col,
                    "foreground_ratio": foreground_ratio,
                    "variance": variance,
                    "informative": informative
                })

        folder = os.path.dirname(output_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        cv2.imwrite(output_path, enhanced)

        return {
            "output_path": output_path,
            "informative_tiles": informative_tiles,
            "blank_tiles": blank_tiles,
            "tile_statistics": tile_statistics
        }

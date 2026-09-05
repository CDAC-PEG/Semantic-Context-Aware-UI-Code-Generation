import time
import numpy as np
import cv2


class Existing_AffineTransformation:
    model_name = "Affine Transformation"

    def align(self, source_points, target_points, all_points):
        """
        Estimate affine mapping from source_points -> target_points
        and transform all_points.

        source_points, target_points, all_points: Nx2 arrays
        """
        source_points = np.asarray(source_points, dtype=np.float32)
        target_points = np.asarray(target_points, dtype=np.float32)
        all_points = np.asarray(all_points, dtype=np.float32)

        start = time.perf_counter()

        if len(source_points) < 3:
            transformed = all_points.copy()
        else:
            matrix, _ = cv2.estimateAffine2D(
                source_points,
                target_points,
                method=cv2.LMEDS
            )

            if matrix is None:
                transformed = all_points.copy()
            else:
                homogeneous = np.hstack([
                    all_points,
                    np.ones((len(all_points), 1), dtype=np.float32)
                ])

                transformed = (
                    matrix @ homogeneous.T
                ).T

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return transformed, elapsed_ms


class Existing_ProjectiveTransformation:
    model_name = "Projective Transformation"

    def align(self, source_points, target_points, all_points):
        """
        Projective / homography-based coordinate transformation.
        """
        source_points = np.asarray(source_points, dtype=np.float32)
        target_points = np.asarray(target_points, dtype=np.float32)
        all_points = np.asarray(all_points, dtype=np.float32)

        start = time.perf_counter()

        if len(source_points) < 4:
            transformed = all_points.copy()
        else:
            H, _ = cv2.findHomography(
                source_points,
                target_points,
                method=cv2.RANSAC
            )

            if H is None:
                transformed = all_points.copy()
            else:
                pts = all_points.reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(
                    pts,
                    H
                ).reshape(-1, 2)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return transformed, elapsed_ms


class Existing_SimilarityTransformation:
    model_name = "Similarity Transformation"

    def align(self, source_points, target_points, all_points):
        """
        Similarity transform:
            rotation + uniform scale + translation
        """
        source_points = np.asarray(source_points, dtype=np.float32)
        target_points = np.asarray(target_points, dtype=np.float32)
        all_points = np.asarray(all_points, dtype=np.float32)

        start = time.perf_counter()

        if len(source_points) < 2:
            transformed = all_points.copy()
        else:
            matrix, _ = cv2.estimateAffinePartial2D(
                source_points,
                target_points,
                method=cv2.LMEDS
            )

            if matrix is None:
                transformed = all_points.copy()
            else:
                homogeneous = np.hstack([
                    all_points,
                    np.ones((len(all_points), 1), dtype=np.float32)
                ])

                transformed = (
                    matrix @ homogeneous.T
                ).T

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return transformed, elapsed_ms


class Existing_ThinPlateSplineTransformation:
    model_name = "Thin-Plate Splines Transformation"

    def __init__(self, regularization=1e-6):
        self.regularization = regularization

    @staticmethod
    def _kernel(r2):
        """
        TPS radial basis:
            U(r) = r^2 log(r^2)
        """
        r2 = np.asarray(r2, dtype=np.float64)

        return np.where(
            r2 > 0,
            r2 * np.log(r2 + 1e-12),
            0.0
        )

    def _fit(self, source_points, target_values):
        """
        Fit one TPS output coordinate.
        """
        X = np.asarray(source_points, dtype=np.float64)
        y = np.asarray(target_values, dtype=np.float64)

        n = X.shape[0]

        diff = X[:, None, :] - X[None, :, :]
        r2 = np.sum(diff ** 2, axis=2)

        K = self._kernel(r2)
        K += self.regularization * np.eye(n)

        P = np.column_stack([
            np.ones(n),
            X[:, 0],
            X[:, 1]
        ])

        O = np.zeros((3, 3))

        L = np.block([
            [K, P],
            [P.T, O]
        ])

        rhs = np.concatenate([
            y,
            np.zeros(3)
        ])

        params = np.linalg.solve(
            L,
            rhs
        )

        weights = params[:n]
        affine = params[n:]

        return weights, affine

    def _transform(self, source_control_points, points, weights, affine):
        X = np.asarray(source_control_points, dtype=np.float64)
        Pts = np.asarray(points, dtype=np.float64)

        diff = Pts[:, None, :] - X[None, :, :]
        r2 = np.sum(diff ** 2, axis=2)

        K = self._kernel(r2)

        values = (
            K @ weights
            + affine[0]
            + affine[1] * Pts[:, 0]
            + affine[2] * Pts[:, 1]
        )

        return values

    def align(self, source_points, target_points, all_points):
        """
        Thin-Plate Spline nonlinear transformation.
        """
        source_points = np.asarray(source_points, dtype=np.float64)
        target_points = np.asarray(target_points, dtype=np.float64)
        all_points = np.asarray(all_points, dtype=np.float64)

        start = time.perf_counter()

        if len(source_points) < 3:
            transformed = all_points.copy()
        else:
            wx, ax = self._fit(
                source_points,
                target_points[:, 0]
            )

            wy, ay = self._fit(
                source_points,
                target_points[:, 1]
            )

            x_new = self._transform(
                source_points,
                all_points,
                wx,
                ax
            )

            y_new = self._transform(
                source_points,
                all_points,
                wy,
                ay
            )

            transformed = np.column_stack([
                x_new,
                y_new
            ])

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return transformed, elapsed_ms

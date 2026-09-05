import os
import json
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


class Proposed_APKT:
    """
    Affine Transformation + Polynomial Kernel Transformation (APKT)
    for UI element coordinate alignment.

    Input:
        chi_v : detected UI elements

    Output:
        Phi_w : aligned UI elements
    """

    def __init__(self, polynomial_degree=2, kernel_constant=1.0,
                 alignment_tolerance=0.05, ridge_alpha=1e-3):
        self.polynomial_degree = polynomial_degree
        self.kernel_constant = kernel_constant
        self.alignment_tolerance = alignment_tolerance
        self.ridge_alpha = ridge_alpha

        self.poly = PolynomialFeatures(
            degree=self.polynomial_degree,
            include_bias=True
        )

        self.model_x = Ridge(alpha=ridge_alpha)
        self.model_y = Ridge(alpha=ridge_alpha)
        self.model_w = Ridge(alpha=ridge_alpha)
        self.model_h = Ridge(alpha=ridge_alpha)

        self.image_width = None
        self.image_height = None

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _bbox_from_component(component):
        center = component.get("bbox_center", {})

        g1 = Proposed_APKT._safe_float(center.get("g1", 0.0))
        g2 = Proposed_APKT._safe_float(center.get("g2", 0.0))
        w = Proposed_APKT._safe_float(center.get("width", 0.0))
        h = Proposed_APKT._safe_float(center.get("height", 0.0))

        if w <= 0 or h <= 0:
            box = component.get("bbox_xyxy", {})
            x1 = Proposed_APKT._safe_float(box.get("x1", 0.0))
            y1 = Proposed_APKT._safe_float(box.get("y1", 0.0))
            x2 = Proposed_APKT._safe_float(box.get("x2", 0.0))
            y2 = Proposed_APKT._safe_float(box.get("y2", 0.0))

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            g1 = x1 + w / 2.0
            g2 = y1 + h / 2.0

        return np.asarray([g1, g2, w, h], dtype=float)

    def extract_coordinates(self, components):
        if not components:
            return np.empty((0, 4), dtype=float)

        return np.vstack([
            self._bbox_from_component(c)
            for c in components
        ])

    @staticmethod
    def _robust_scale(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return 1.0

        spread = np.percentile(values, 90) - np.percentile(values, 10)
        return max(float(spread), 1.0)

    def _snap_values(self, values):
        values = np.asarray(values, dtype=float)

        if values.size <= 1:
            return values.copy()

        scale = self._robust_scale(values)
        tolerance = self.alignment_tolerance * scale

        order = np.argsort(values)
        sorted_values = values[order]

        groups = []
        current = [sorted_values[0]]

        for value in sorted_values[1:]:
            current_median = float(np.median(current))

            if abs(value - current_median) <= tolerance:
                current.append(value)
            else:
                groups.append(current)
                current = [value]

        groups.append(current)

        anchors = [float(np.median(g)) for g in groups]

        snapped = np.zeros_like(values)

        for i, value in enumerate(values):
            snapped[i] = min(
                anchors,
                key=lambda anchor: abs(value - anchor)
            )

        return snapped

    def build_alignment_targets(self, coordinates):
        if coordinates.size == 0:
            return coordinates.copy()

        target = coordinates.copy()
        target[:, 0] = self._snap_values(coordinates[:, 0])
        target[:, 1] = self._snap_values(coordinates[:, 1])
        target[:, 2] = self._snap_values(coordinates[:, 2])
        target[:, 3] = self._snap_values(coordinates[:, 3])

        return target

    def polynomial_kernel_value(self, chi_1, chi_2):
        chi_1 = np.asarray(chi_1, dtype=float)
        chi_2 = np.asarray(chi_2, dtype=float)

        return float(
            (np.dot(chi_1, chi_2) + self.kernel_constant)
            ** self.polynomial_degree
        )

    def estimate_canvas_size(self, coordinates):
        if coordinates.size == 0:
            self.image_width = 1.0
            self.image_height = 1.0
            return

        right = coordinates[:, 0] + coordinates[:, 2] / 2.0
        bottom = coordinates[:, 1] + coordinates[:, 3] / 2.0

        self.image_width = max(float(np.max(right)), 1.0)
        self.image_height = max(float(np.max(bottom)), 1.0)

    def normalize_coordinates(self, coordinates):
        normalized = coordinates.copy().astype(float)

        normalized[:, 0] /= self.image_width
        normalized[:, 1] /= self.image_height
        normalized[:, 2] /= self.image_width
        normalized[:, 3] /= self.image_height

        return normalized

    def denormalize_coordinates(self, coordinates):
        values = coordinates.copy().astype(float)

        values[:, 0] *= self.image_width
        values[:, 1] *= self.image_height
        values[:, 2] *= self.image_width
        values[:, 3] *= self.image_height

        return values

    def fit_transform(self, coordinates):
        if coordinates.shape[0] <= 1:
            return coordinates.copy()

        self.estimate_canvas_size(coordinates)

        source_norm = self.normalize_coordinates(coordinates)
        targets = self.build_alignment_targets(coordinates)
        target_norm = self.normalize_coordinates(targets)

        # D_c: higher-dimensional polynomial feature representation
        D_c = self.poly.fit_transform(source_norm)

        self.model_x.fit(D_c, target_norm[:, 0])
        self.model_y.fit(D_c, target_norm[:, 1])
        self.model_w.fit(D_c, target_norm[:, 2])
        self.model_h.fit(D_c, target_norm[:, 3])

        aligned_norm = np.column_stack([
            self.model_x.predict(D_c),
            self.model_y.predict(D_c),
            self.model_w.predict(D_c),
            self.model_h.predict(D_c)
        ])

        aligned = self.denormalize_coordinates(aligned_norm)
        aligned[:, 2] = np.maximum(aligned[:, 2], 1.0)
        aligned[:, 3] = np.maximum(aligned[:, 3], 1.0)

        return aligned

    @staticmethod
    def homogeneous_affine_matrix(source_xy, target_xy):
        source_xy = np.asarray(source_xy, dtype=float)
        target_xy = np.asarray(target_xy, dtype=float)

        if source_xy.shape[0] < 3:
            return np.eye(3, dtype=float)

        A = []
        b = []

        for (x, y), (xp, yp) in zip(source_xy, target_xy):
            A.append([x, y, 1, 0, 0, 0])
            A.append([0, 0, 0, x, y, 1])
            b.extend([xp, yp])

        params, _, _, _ = np.linalg.lstsq(
            np.asarray(A, dtype=float),
            np.asarray(b, dtype=float),
            rcond=None
        )

        a, b1, tx, c, d, ty = params

        return np.asarray([
            [a, b1, tx],
            [c, d, ty],
            [0.0, 0.0, 1.0]
        ])

    def align_ui_elements(self, input_json_path, output_json_path=None):
        if not os.path.exists(input_json_path):
            raise FileNotFoundError(
                f"Input JSON not found: {input_json_path}"
            )

        with open(input_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        components = []

        if "components" in payload:
            components = payload["components"]

        elif "beta_y" in payload:
            for group_name, members in payload["beta_y"].items():
                for member in members:
                    item = dict(member)
                    item["source_group"] = group_name
                    components.append(item)

        if not components:
            return {
                "method": "APKT",
                "Phi_w": []
            }

        source_coordinates = self.extract_coordinates(components)
        aligned_coordinates = self.fit_transform(source_coordinates)

        affine_matrix = self.homogeneous_affine_matrix(
            source_coordinates[:, :2],
            aligned_coordinates[:, :2]
        )

        aligned_elements = []

        for index, (component, source, aligned) in enumerate(
            zip(components, source_coordinates, aligned_coordinates)
        ):
            g1, g2, w, h = aligned

            aligned_elements.append({
                "component_id": component.get("component_id", index),
                "class_name": component.get("class_name", ""),
                "text": component.get("attributes", {}).get("text", ""),

                "original_coordinates": {
                    "g1": float(source[0]),
                    "g2": float(source[1]),
                    "width": float(source[2]),
                    "height": float(source[3])
                },

                "aligned_coordinates": {
                    "g1": float(g1),
                    "g2": float(g2),
                    "width": float(w),
                    "height": float(h),
                    "x1": float(g1 - w / 2.0),
                    "y1": float(g2 - h / 2.0),
                    "x2": float(g1 + w / 2.0),
                    "y2": float(g2 + h / 2.0)
                }
            })

        result = {
            "method": "APKT",
            "polynomial_degree_s": self.polynomial_degree,
            "kernel_constant_u": self.kernel_constant,
            "affine_homogeneous_matrix": affine_matrix.tolist(),
            "Phi_w": aligned_elements
        }

        if output_json_path:
            folder = os.path.dirname(output_json_path)
            if folder:
                os.makedirs(folder, exist_ok=True)

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

        return result


if __name__ == "__main__":

    input_json = (
        r"..\Output\UI\Stage2"
        r"\sample_stage2.json"
    )

    output_json = (
        r"..\Output\UI\Coordinate_Alignment"
        r"\sample_APKT.json"
    )

    apkt = Proposed_APKT(
        polynomial_degree=2,
        kernel_constant=1.0,
        alignment_tolerance=0.05,
        ridge_alpha=1e-3
    )

    result = apkt.align_ui_elements(
        input_json_path=input_json,
        output_json_path=output_json
    )

    print("\nAPKT Coordinate Alignment")
    print("=========================")

    print("\nAffine Homogeneous Matrix:")
    for row in result["affine_homogeneous_matrix"]:
        print(row)

    print("\nAligned UI elements (Phi_w):")
    for item in result["Phi_w"]:
        print(
            item["component_id"],
            item["class_name"],
            "->",
            item["aligned_coordinates"]
        )

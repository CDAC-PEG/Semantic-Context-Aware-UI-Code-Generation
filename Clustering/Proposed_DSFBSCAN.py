import os
import json
import math
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class Proposed_DSFBSCAN:
    """
    Density-Survival-Function-Based DBSCAN (DSFBSCAN)
    for grouping similar UI elements.

    Inputs
    ------
    A_h   : OCR/text attributes
            - text
            - OCR confidence
            - text bounding box
            - estimated font size
            - visible line/style attributes

    chi_v : detected UI element attributes
            - class
            - detection confidence
            - component bounding box

    Output
    ------
    beta_y : grouped similar UI elements

    Notes
    -----
    The survival function is used to adaptively derive:
        zeta (epsilon)
        P    (minPts)

    Instead of clustering document-level TF-IDF vectors, this implementation
    clusters UI-element-level multimodal features.
    """

    def __init__(
        self,
        survival_quantile=0.70,
        min_minpts=2,
        max_minpts=8
    ):
        self.survival_quantile = survival_quantile
        self.min_minpts = min_minpts
        self.max_minpts = max_minpts

        self.zeta = None
        self.P = None
        self.gamma_epsilon = None
        self.labels_ = None

    # ============================================================
    # UTILITY FUNCTIONS
    # ============================================================

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_text(text):
        if text is None:
            return ""
        return str(text).strip().lower()

    @staticmethod
    def _class_to_numeric(class_name):
        """
        Deterministic lightweight encoding of class labels.
        """
        if class_name is None:
            return 0.0

        text = str(class_name).strip().lower()

        if not text:
            return 0.0

        # stable simple string-to-number encoding
        value = sum((i + 1) * ord(ch) for i, ch in enumerate(text))
        return float(value % 1000) / 1000.0

    @staticmethod
    def _text_to_numeric(text):
        """
        Lightweight lexical encoding used only as one feature among
        visual/spatial/attribute features.

        This avoids document-level TF-IDF clustering and keeps grouping
        at UI-element level.
        """
        text = Proposed_DSFBSCAN._normalize_text(text)

        if not text:
            return [0.0, 0.0, 0.0]

        length = len(text)
        alpha_ratio = sum(c.isalpha() for c in text) / max(length, 1)
        digit_ratio = sum(c.isdigit() for c in text) / max(length, 1)

        # normalized deterministic lexical hash
        lexical_hash = (
            sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % 1000
        ) / 1000.0

        return [
            float(length),
            float(alpha_ratio),
            float(digit_ratio + lexical_hash)
        ]

    # ============================================================
    # FEATURE CONSTRUCTION FROM A_h AND chi_v
    # ============================================================

    def build_feature_vector(self, component):
        """
        Convert a UI component record into a multimodal numeric vector.

        Expected component format is compatible with the Stage-2 JSON:
        {
            "class_name": ...,
            "confidence": ...,
            "bbox_center": {...},
            "bbox_xyxy": {...},
            "attributes": {
                "text": ...,
                "ocr_confidence": ...,
                "estimated_font_size_px": ...,
                "font_style": {...}
            }
        }
        """

        attributes = component.get("attributes", {})
        bbox_center = component.get("bbox_center", {})
        bbox_xyxy = component.get("bbox_xyxy", {})

        text = attributes.get("text", "")
        text_features = self._text_to_numeric(text)

        class_numeric = self._class_to_numeric(
            component.get("class_name", "")
        )

        det_conf = self._safe_float(
            component.get("confidence", 0.0)
        )

        ocr_conf = self._safe_float(
            attributes.get("ocr_confidence", 0.0)
        )

        font_size = self._safe_float(
            attributes.get("estimated_font_size_px", 0.0)
        )

        font_style = attributes.get("font_style", {})

        underline = 1.0 if font_style.get("underline", False) else 0.0
        strike = 1.0 if font_style.get("strike_through", False) else 0.0

        g1 = self._safe_float(
            bbox_center.get("g1", 0.0)
        )
        g2 = self._safe_float(
            bbox_center.get("g2", 0.0)
        )
        width = self._safe_float(
            bbox_center.get("width", 0.0)
        )
        height = self._safe_float(
            bbox_center.get("height", 0.0)
        )

        # Fallback if bbox_center is absent
        if width == 0.0 and height == 0.0 and bbox_xyxy:
            x1 = self._safe_float(bbox_xyxy.get("x1", 0.0))
            y1 = self._safe_float(bbox_xyxy.get("y1", 0.0))
            x2 = self._safe_float(bbox_xyxy.get("x2", 0.0))
            y2 = self._safe_float(bbox_xyxy.get("y2", 0.0))

            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            g1 = x1 + width / 2.0
            g2 = y1 + height / 2.0

        aspect_ratio = width / height if height > 0 else 0.0
        area = width * height

        # Combined representation from A_h and chi_v
        feature_vector = [
            class_numeric,        # component semantic type
            det_conf,             # detection confidence
            g1,                   # center x
            g2,                   # center y
            width,
            height,
            aspect_ratio,
            area,
            text_features[0],     # text length
            text_features[1],     # alphabetic ratio
            text_features[2],     # digit + lexical representation
            ocr_conf,
            font_size,
            underline,
            strike
        ]

        return np.asarray(feature_vector, dtype=float)

    def build_feature_matrix(self, components):
        if not components:
            return np.empty((0, 15), dtype=float)

        return np.vstack([
            self.build_feature_vector(component)
            for component in components
        ])

    # ============================================================
    # SURVIVAL FUNCTION
    # ============================================================

    @staticmethod
    def survival_function(values):
        """
        Empirical survival function:

            S(r) = P(R > r) = 1 - F(r)

        Returns sorted values and their survival probabilities.
        """

        values = np.asarray(values, dtype=float)

        if values.size == 0:
            return np.array([]), np.array([])

        sorted_values = np.sort(values)

        n = len(sorted_values)

        survival_probabilities = np.array([
            (n - i) / n
            for i in range(n)
        ], dtype=float)

        return sorted_values, survival_probabilities

    def estimate_parameters(self, X_scaled):
        """
        Estimate zeta (epsilon) and P (minPts) adaptively.

        Similarity is computed using cosine similarity.
        Distance = 1 - similarity.

        zeta is selected from the empirical survival distribution of
        pairwise distances.

        P is adapted from neighborhood density under the selected zeta.
        """

        n_samples = X_scaled.shape[0]

        if n_samples <= 1:
            self.zeta = 0.5
            self.P = 1
            self.gamma_epsilon = self.zeta * self.P
            return self.zeta, self.P

        similarity = cosine_similarity(X_scaled)

        # convert similarity to dissimilarity/distance
        distance_matrix = 1.0 - similarity

        # upper triangular distances excluding diagonal
        distances = distance_matrix[
            np.triu_indices(n_samples, k=1)
        ]

        distances = distances[np.isfinite(distances)]

        if distances.size == 0:
            self.zeta = 0.5
        else:
            # Survival-informed threshold:
            # quantile of pairwise distance distribution.
            #
            # Higher quantile => more permissive clusters.
            self.zeta = float(
                np.quantile(
                    distances,
                    self.survival_quantile
                )
            )

        # Keep epsilon numerically valid.
        self.zeta = max(self.zeta, 1e-6)

        # Estimate minPts from local neighborhood density.
        neighbor_counts = []

        for i in range(n_samples):
            count = int(
                np.sum(distance_matrix[i] <= self.zeta)
            )

            # excludes self
            count = max(0, count - 1)

            neighbor_counts.append(count)

        if neighbor_counts:
            median_neighbors = int(
                round(np.median(neighbor_counts))
            )
        else:
            median_neighbors = 1

        self.P = max(
            self.min_minpts,
            median_neighbors
        )

        self.P = min(
            self.P,
            self.max_minpts,
            n_samples
        )

        # gamma_epsilon = zeta * P
        self.gamma_epsilon = (
            self.zeta * self.P
        )

        return self.zeta, self.P

    # ============================================================
    # DSFBSCAN CLUSTERING
    # ============================================================

    def cluster_components(self, components):
        """
        Group similar UI elements as beta_y.
        """

        if not components:
            return {
                "zeta": 0.0,
                "P": 0,
                "gamma_epsilon": 0.0,
                "labels": [],
                "groups": {}
            }

        X = self.build_feature_matrix(
            components
        )

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X
        )

        zeta, P = self.estimate_parameters(
            X_scaled
        )

        # DBSCAN uses Euclidean distance over normalized multimodal features.
        model = DBSCAN(
            eps=zeta,
            min_samples=P,
            metric="euclidean"
        )

        labels = model.fit_predict(
            X_scaled
        )

        self.labels_ = labels

        # beta_y groups
        groups = {}

        for index, label in enumerate(labels):
            key = (
                "noise"
                if int(label) == -1
                else f"beta_{int(label)}"
            )

            if key not in groups:
                groups[key] = []

            component = dict(
                components[index]
            )

            component["cluster_label"] = int(
                label
            )

            groups[key].append(
                component
            )

        return {
            "zeta": float(self.zeta),
            "P": int(self.P),
            "gamma_epsilon": float(
                self.gamma_epsilon
            ),
            "labels": [
                int(x) for x in labels
            ],
            "groups": groups
        }

    # ============================================================
    # LOAD STAGE-2 JSON AND RUN DSFBSCAN
    # ============================================================

    def clustering(
        self,
        input_json_path,
        output_json_path=None
    ):
        """
        Read Stage-2 JSON, run DSFBSCAN and optionally save beta_y.
        """

        if not os.path.exists(
            input_json_path
        ):
            raise FileNotFoundError(
                f"Input JSON not found: {input_json_path}"
            )

        with open(
            input_json_path,
            "r",
            encoding="utf-8"
        ) as file:
            payload = json.load(file)

        components = payload.get(
            "components",
            []
        )

        result = self.cluster_components(
            components
        )

        print(
            "\nSimilar UI Element Grouping "
            "using DSFBSCAN"
        )
        print(
            "================================="
        )

        print(
            f"Adaptive epsilon (zeta): "
            f"{result['zeta']:.6f}"
        )

        print(
            f"Adaptive minPts (P): "
            f"{result['P']}"
        )

        print(
            f"Core-point factor "
            f"(gamma_epsilon): "
            f"{result['gamma_epsilon']:.6f}"
        )

        for group_name, members in (
            result["groups"].items()
        ):
            print(
                f"\n{group_name}: "
                f"{len(members)} UI elements"
            )

            for member in members:
                print(
                    "   ",
                    member.get(
                        "class_name",
                        "unknown"
                    ),
                    "->",
                    member.get(
                        "attributes",
                        {}
                    ).get(
                        "text",
                        ""
                    )
                )

        # Preserve source information
        output_payload = {
            "method": "DSFBSCAN",
            "input_stage": input_json_path,

            # proposed parameters
            "zeta": result["zeta"],
            "P": result["P"],
            "gamma_epsilon":
                result["gamma_epsilon"],

            # beta_y
            "beta_y": result["groups"]
        }

        if output_json_path:
            folder = os.path.dirname(
                output_json_path
            )

            if folder:
                os.makedirs(
                    folder,
                    exist_ok=True
                )

            with open(
                output_json_path,
                "w",
                encoding="utf-8"
            ) as file:
                json.dump(
                    output_payload,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            print(
                "\nGrouped UI elements "
                "saved to:"
            )
            print(
                output_json_path
            )

        return output_payload


# ================================================================
# STANDALONE TEST
# ================================================================

if __name__ == "__main__":

    # Input generated by:
    # UIComponentTextAttributeStage.process(...)
    input_json = (
        r"..\Output\UI\Stage2"
        r"\sample_stage2.json"
    )

    output_json = (
        r"..\Output\UI\Component_Grouping"
        r"\sample_DSFBSCAN.json"
    )

    dsfbscan = Proposed_DSFBSCAN(
        survival_quantile=0.70,
        min_minpts=2,
        max_minpts=8
    )

    dsfbscan.clustering(
        input_json_path=input_json,
        output_json_path=output_json
    )

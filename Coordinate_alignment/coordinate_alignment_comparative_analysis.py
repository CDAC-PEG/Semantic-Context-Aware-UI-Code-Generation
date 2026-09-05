import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

from Proposed_APKT import Proposed_APKT

from existing_coordinate_alignment_methods import (
    Existing_AffineTransformation,
    Existing_ProjectiveTransformation,
    Existing_ThinPlateSplineTransformation,
    Existing_SimilarityTransformation
)


# ================================================================
# LOAD COMPONENT DATA
# ================================================================

def load_components(stage2_json_path):
    if not os.path.exists(stage2_json_path):
        raise FileNotFoundError(
            f"Input JSON not found: {stage2_json_path}"
        )

    with open(
        stage2_json_path,
        "r",
        encoding="utf-8"
    ) as file:
        payload = json.load(file)

    if "components" in payload:
        return payload["components"]

    if "beta_y" in payload:
        components = []

        for _, members in payload["beta_y"].items():
            components.extend(members)

        return components

    raise ValueError(
        "Input JSON must contain 'components' or 'beta_y'."
    )


def extract_center_coordinates(components):
    """
    Extract [g1, g2] centers from component boxes.
    """
    coords = []

    for component in components:
        center = component.get(
            "bbox_center",
            {}
        )

        if center:
            g1 = float(center.get("g1", 0.0))
            g2 = float(center.get("g2", 0.0))
        else:
            box = component.get(
                "bbox_xyxy",
                {}
            )

            x1 = float(box.get("x1", 0.0))
            y1 = float(box.get("y1", 0.0))
            x2 = float(box.get("x2", 0.0))
            y2 = float(box.get("y2", 0.0))

            g1 = (x1 + x2) / 2.0
            g2 = (y1 + y2) / 2.0

        coords.append([
            g1,
            g2
        ])

    return np.asarray(
        coords,
        dtype=float
    )


# ================================================================
# BUILD TARGET ALIGNMENT
# ================================================================

def build_target_alignment(components):
    """
    Use Proposed_APKT's own target construction logic
    so all baseline methods are evaluated against the
    same alignment target.
    """

    apkt = Proposed_APKT(
        polynomial_degree=2,
        kernel_constant=1.0,
        alignment_tolerance=0.05,
        ridge_alpha=1e-3
    )

    full_coordinates = (
        apkt.extract_coordinates(
            components
        )
    )

    target_coordinates = (
        apkt.build_alignment_targets(
            full_coordinates
        )
    )

    return (
        full_coordinates[:, :2],
        target_coordinates[:, :2]
    )


# ================================================================
# METRICS
# ================================================================

def calculate_regression_metrics(
    predicted,
    target
):
    predicted = np.asarray(
        predicted,
        dtype=float
    )

    target = np.asarray(
        target,
        dtype=float
    )

    error = predicted - target

    mse = float(
        np.mean(
            error ** 2
        )
    )

    rmse = float(
        np.sqrt(
            mse
        )
    )

    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    return mse, rmse, mae


def render_points_as_image(
    points,
    canvas_size=(512, 512),
    radius=4
):
    """
    Render coordinates into a binary image for SSIM.
    """
    points = np.asarray(
        points,
        dtype=float
    )

    image = np.zeros(
        canvas_size,
        dtype=np.uint8
    )

    if len(points) == 0:
        return image

    x = points[:, 0]
    y = points[:, 1]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x_scale = (
        (canvas_size[1] - 20)
        / max(x_max - x_min, 1.0)
    )

    y_scale = (
        (canvas_size[0] - 20)
        / max(y_max - y_min, 1.0)
    )

    for px, py in points:
        ix = int(
            10 + (px - x_min) * x_scale
        )

        iy = int(
            10 + (py - y_min) * y_scale
        )

        ix = min(
            max(ix, 0),
            canvas_size[1] - 1
        )

        iy = min(
            max(iy, 0),
            canvas_size[0] - 1
        )

        import cv2

        cv2.circle(
            image,
            (ix, iy),
            radius,
            255,
            -1
        )

    return image


def calculate_ssim(
    predicted,
    target
):
    predicted_image = (
        render_points_as_image(
            predicted
        )
    )

    target_image = (
        render_points_as_image(
            target
        )
    )

    return float(
        ssim(
            target_image,
            predicted_image,
            data_range=255
        )
    )


# ================================================================
# PROPOSED APKT
# ================================================================

def evaluate_apkt(
    components,
    target_xy
):
    model = Proposed_APKT(
        polynomial_degree=2,
        kernel_constant=1.0,
        alignment_tolerance=0.05,
        ridge_alpha=1e-3
    )

    source_full = (
        model.extract_coordinates(
            components
        )
    )

    start = time.perf_counter()

    aligned_full = (
        model.fit_transform(
            source_full
        )
    )

    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    predicted_xy = (
        aligned_full[:, :2]
    )

    mse, rmse, mae = (
        calculate_regression_metrics(
            predicted_xy,
            target_xy
        )
    )

    structural_similarity = (
        calculate_ssim(
            predicted_xy,
            target_xy
        )
    )

    return {
        "Method":
            "APKT",

        "MSE":
            mse,

        "RMSE":
            rmse,

        "MAE":
            mae,

        "SSIM":
            structural_similarity,

        "Alignment Time (ms)":
            elapsed_ms
    }


# ================================================================
# EXISTING METHODS
# ================================================================

def evaluate_existing_method(
    method,
    source_xy,
    target_xy
):
    predicted_xy, elapsed_ms = (
        method.align(
            source_xy,
            target_xy,
            source_xy
        )
    )

    mse, rmse, mae = (
        calculate_regression_metrics(
            predicted_xy,
            target_xy
        )
    )

    structural_similarity = (
        calculate_ssim(
            predicted_xy,
            target_xy
        )
    )

    return {
        "Method":
            method.model_name,

        "MSE":
            mse,

        "RMSE":
            rmse,

        "MAE":
            mae,

        "SSIM":
            structural_similarity,

        "Alignment Time (ms)":
            elapsed_ms
    }


# ================================================================
# GRAPH
# ================================================================

def generate_metric_graphs(
    df,
    output_directory
):
    os.makedirs(
        output_directory,
        exist_ok=True
    )

    metrics = [
        "MSE",
        "RMSE",
        "MAE",
        "SSIM"
    ]

    for metric in metrics:
        plt.figure()

        plt.bar(
            df["Method"],
            df[metric]
        )

        plt.xlabel(
            "Coordinate Alignment Method"
        )

        plt.ylabel(
            metric
        )

        plt.title(
            f"Comparison of {metric}"
        )

        plt.xticks(
            rotation=25,
            ha="right"
        )

        plt.tight_layout()

        output_path = os.path.join(
            output_directory,
            f"{metric}_comparison.png"
        )

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()


# ================================================================
# FULL COMPARATIVE ANALYSIS
# ================================================================

def comparative_analysis(
    stage2_json_path,
    output_json=None,
    output_csv=None,
    graph_directory=None
):
    components = load_components(
        stage2_json_path
    )

    source_xy, target_xy = (
        build_target_alignment(
            components
        )
    )

    methods = [
        Existing_AffineTransformation(),
        Existing_ProjectiveTransformation(),
        Existing_ThinPlateSplineTransformation(),
        Existing_SimilarityTransformation()
    ]

    results = []

    # Proposed APKT
    results.append(
        evaluate_apkt(
            components,
            target_xy
        )
    )

    # Existing approaches
    for method in methods:
        results.append(
            evaluate_existing_method(
                method,
                source_xy,
                target_xy
            )
        )

    df = pd.DataFrame(
        results
    )

    print(
        "\nCoordinate Alignment Comparative Analysis"
    )

    print(
        "========================================="
    )

    print(
        df.to_string(
            index=False
        )
    )

    if output_json:
        folder = os.path.dirname(
            output_json
        )

        if folder:
            os.makedirs(
                folder,
                exist_ok=True
            )

        with open(
            output_json,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                results,
                file,
                indent=4
            )

    if output_csv:
        folder = os.path.dirname(
            output_csv
        )

        if folder:
            os.makedirs(
                folder,
                exist_ok=True
            )

        df.to_csv(
            output_csv,
            index=False
        )

    if graph_directory:
        generate_metric_graphs(
            df,
            graph_directory
        )

    return df


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    comparative_analysis(

        stage2_json_path=
            r"..\Output\UI\Stage2"
            r"\sample_stage2.json",

        output_json=
            r"..\Output\UI\Coordinate_Alignment"
            r"\coordinate_alignment_comparison.json",

        output_csv=
            r"..\Output\UI\Coordinate_Alignment"
            r"\coordinate_alignment_comparison.csv",

        graph_directory=
            r"..\Graphs\Coordinate_Alignment"
    )

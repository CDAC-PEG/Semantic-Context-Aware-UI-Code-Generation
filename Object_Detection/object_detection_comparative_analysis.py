import os
import json
import numpy as np
import pandas as pd

from Proposed_SPPYOLO import Proposed_SPPYOLO

from existing_object_detectors import (
    Existing_YOLOv8,
    Existing_RetinaNet,
    Existing_EfficientDet,
    Existing_FCOS
)


def load_ground_truth(json_path):
    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    area_a = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )

    area_b = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    return (
        0.0
        if union <= 0
        else intersection / union
    )


def match_detections(
    predictions,
    ground_truth,
    iou_threshold=0.50
):
    predictions = sorted(
        predictions,
        key=lambda item:
            item["score"],
        reverse=True
    )

    matched_gt = set()

    tp = 0
    fp = 0

    matched_ious = []
    ranked_tp = []

    for prediction in predictions:

        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt in enumerate(
            ground_truth
        ):
            if gt_index in matched_gt:
                continue

            if (
                int(
                    prediction[
                        "class_id"
                    ]
                )
                !=
                int(
                    gt[
                        "class_id"
                    ]
                )
            ):
                continue

            iou = calculate_iou(
                prediction[
                    "box"
                ],
                gt[
                    "box"
                ]
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and
            best_iou >= iou_threshold
        ):
            tp += 1

            matched_gt.add(
                best_gt_index
            )

            matched_ious.append(
                best_iou
            )

            ranked_tp.append(
                1
            )

        else:
            fp += 1

            ranked_tp.append(
                0
            )

    fn = (
        len(ground_truth)
        - len(matched_gt)
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_ious":
            matched_ious,
        "ranked_tp":
            ranked_tp
    }


def average_precision(
    ranked_tp,
    total_gt
):
    if total_gt == 0:
        return 0.0

    tp_count = 0
    fp_count = 0

    precisions = []
    recalls = []

    for value in ranked_tp:

        if value == 1:
            tp_count += 1
        else:
            fp_count += 1

        precision = (
            tp_count
            / max(
                tp_count
                + fp_count,
                1
            )
        )

        recall = (
            tp_count
            / total_gt
        )

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

    if not recalls:
        return 0.0

    mrec = (
        [0.0]
        + recalls
        + [1.0]
    )

    mpre = (
        [0.0]
        + precisions
        + [0.0]
    )

    for i in range(
        len(mpre) - 2,
        -1,
        -1
    ):
        mpre[i] = max(
            mpre[i],
            mpre[i + 1]
        )

    ap = 0.0

    for i in range(
        1,
        len(mrec)
    ):
        if (
            mrec[i]
            !=
            mrec[i - 1]
        ):
            ap += (
                mrec[i]
                - mrec[i - 1]
            ) * mpre[i]

    return float(ap)


def evaluate_model(
    detector,
    dataset,
    iou_threshold=0.50
):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_ious = []
    image_aps = []

    exact_images = 0

    for sample in dataset:

        ground_truth = (
            sample.get(
                "objects",
                []
            )
        )

        predictions = (
            detector.predict(
                sample["image"]
            )
        )

        result = match_detections(
            predictions,
            ground_truth,
            iou_threshold
        )

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

        all_ious.extend(
            result[
                "matched_ious"
            ]
        )

        image_aps.append(
            average_precision(
                result[
                    "ranked_tp"
                ],
                len(
                    ground_truth
                )
            )
        )

        if (
            result["fn"] == 0
            and
            result["fp"] == 0
            and
            len(
                ground_truth
            ) > 0
        ):
            exact_images += 1

    precision = (
        total_tp
        / max(
            total_tp
            + total_fp,
            1
        )
    )

    recall = (
        total_tp
        / max(
            total_tp
            + total_fn,
            1
        )
    )

    f1 = (
        2
        * precision
        * recall
        / max(
            precision
            + recall,
            1e-12
        )
    )

    accuracy = (
        exact_images
        / max(
            len(dataset),
            1
        )
    )

    map50 = (
        float(
            np.mean(
                image_aps
            )
        )
        if image_aps
        else 0.0
    )

    mean_iou = (
        float(
            np.mean(
                all_ious
            )
        )
        if all_ious
        else 0.0
    )

    return {
        "Method":
            detector.model_name,

        "Accuracy":
            accuracy,

        "mAP@0.50":
            map50,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1-Score":
            f1,

        "IoU":
            mean_iou
    }


def comparative_analysis(
    detectors,
    ground_truth_json,
    output_json=None,
    output_csv=None
):
    dataset = load_ground_truth(
        ground_truth_json
    )

    results = []

    for detector in detectors:

        print(
            "\nEvaluating:",
            detector.model_name
        )

        metrics = evaluate_model(
            detector,
            dataset
        )

        results.append(
            metrics
        )

        print(
            "Accuracy:",
            f"{metrics['Accuracy']:.4f}"
        )

        print(
            "mAP@0.50:",
            f"{metrics['mAP@0.50']:.4f}"
        )

        print(
            "Recall:",
            f"{metrics['Recall']:.4f}"
        )

        print(
            "F1:",
            f"{metrics['F1-Score']:.4f}"
        )

        print(
            "IoU:",
            f"{metrics['IoU']:.4f}"
        )

    df = pd.DataFrame(
        results
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

    print(
        "\nComparative Analysis"
    )

    print(
        "===================="
    )

    print(
        df.to_string(
            index=False
        )
    )

    return df


if __name__ == "__main__":

    NUM_CLASSES = 7

    # Proposed SPPYOLO-v8
    proposed_sppyolo = (
        Proposed_SPPYOLO(
            model_path=
                "../Models/OB/PYOLO.pt",

            conf_threshold=
                0.25,

            iou_threshold=
                0.50,

            imgsz=
                640
        )
    )

    # Standard YOLO-v8 baseline
    yolo_v8 = Existing_YOLOv8(
        model_path=
            "../Models/OB/YOLOv8_best.pt"
    )

    # RetinaNet baseline
    retina_net = Existing_RetinaNet(
        checkpoint_path=
            "../Models/OB/RetinaNet_best.pth",

        num_classes=
            NUM_CLASSES
    )

    # FCOS baseline
    fcos = Existing_FCOS(
        checkpoint_path=
            "../Models/OB/FCOS_best.pth",

        num_classes=
            NUM_CLASSES
    )

    detectors = [
        proposed_sppyolo,
        yolo_v8,
        retina_net,
        fcos
    ]

    comparative_analysis(
        detectors=
            detectors,

        ground_truth_json=
            "../Dataset/UI/"
            "test_ground_truth.json",

        output_json=
            "../Output/UI/"
            "Object_Detection/"
            "comparative_results.json",

        output_csv=
            "../Output/UI/"
            "Object_Detection/"
            "comparative_results.csv"
    )

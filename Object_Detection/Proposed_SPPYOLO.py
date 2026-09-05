from ultralytics import YOLO
import cv2
import os
import json


class Proposed_SPPYOLO:
    """
    Proposed SPPYOLO-v8 UI component detector.

    This class wraps the existing trained model and exposes a reusable
    predict(image_path) method so it can be called from comparative
    evaluation modules.

    Returned prediction format:
        [
            {
                "box": [x1, y1, x2, y2],
                "class_id": int,
                "score": float
            },
            ...
        ]
    """

    model_name = "SPPYOLO-v8"

    def __init__(
        self,
        model_path="../Models/OB/PYOLO.pt",
        conf_threshold=0.25,
        iou_threshold=0.50,
        imgsz=640
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained SPPYOLO-v8 model not found: {self.model_path}"
            )

        self.model = YOLO(self.model_path)

    # ============================================================
    # COMPARATIVE-EVALUATION PREDICTION METHOD
    # ============================================================

    def predict(self, image_path):
        """
        Run SPPYOLO-v8 prediction for comparative evaluation.

        Parameters
        ----------
        image_path : str
            Path to the input/preprocessed UI image.

        Returns
        -------
        list
            Standardized predictions:
            {
                "box": [x1, y1, x2, y2],
                "class_id": int,
                "score": float
            }
        """

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Input image not found: {image_path}"
            )

        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            save=False,
            verbose=False
        )

        predictions = []

        for result in results:

            if result.boxes is None:
                continue

            boxes_xyxy = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            classes = (
                result.boxes.cls
                .detach()
                .cpu()
                .numpy()
            )

            confidences = (
                result.boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            for box, cls, conf in zip(
                boxes_xyxy,
                classes,
                confidences
            ):
                x1, y1, x2, y2 = box

                predictions.append({
                    "box": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ],
                    "class_id": int(cls),
                    "score": float(conf)
                })

        return predictions

    # ============================================================
    # FULL DETECTION + VISUALIZATION + JSON OUTPUT
    # ============================================================

    def detect_and_save(
        self,
        image_path,
        output_directory=r"..\Output\UI\Component_Detection",
        image_filename="Object_Detected.jpg",
        json_filename="Detected_Components.json"
    ):
        """
        Run prediction and additionally save:
            - detected/annotated UI image
            - detailed JSON component information
        """

        img = cv2.imread(image_path)

        if img is None:
            raise FileNotFoundError(
                f"Unable to read input image: {image_path}"
            )

        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            save=False,
            verbose=False
        )

        detected_components = []

        component_counter = 0

        for result in results:

            label_names = result.names

            if result.boxes is None:
                continue

            boxes_xyxy = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            boxes_xywh = (
                result.boxes.xywh
                .detach()
                .cpu()
                .numpy()
            )

            classes = (
                result.boxes.cls
                .detach()
                .cpu()
                .numpy()
            )

            confidences = (
                result.boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            for box, box_center, cls, conf in zip(
                boxes_xyxy,
                boxes_xywh,
                classes,
                confidences
            ):
                x1, y1, x2, y2 = map(int, box)

                g1, g2, width, height = box_center

                class_id = int(cls)

                label = label_names[class_id]

                confidence = float(conf)

                component = {
                    "component_id": component_counter,
                    "class_id": class_id,
                    "class_name": label,
                    "confidence": confidence,

                    "bbox_center": {
                        "g1": float(g1),
                        "g2": float(g2),
                        "width": float(width),
                        "height": float(height),
                        "probability": confidence
                    },

                    "bbox_xyxy": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                }

                detected_components.append(
                    component
                )

                component_counter += 1

                color = (0, 255, 0)

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                display_label = (
                    f"{label} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    img,
                    display_label,
                    (
                        x1,
                        max(
                            y1 - 10,
                            15
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        os.makedirs(
            output_directory,
            exist_ok=True
        )

        detected_image_path = (
            os.path.join(
                output_directory,
                image_filename
            )
        )

        json_output_path = (
            os.path.join(
                output_directory,
                json_filename
            )
        )

        cv2.imwrite(
            detected_image_path,
            img
        )

        with open(
            json_output_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                detected_components,
                file,
                indent=4
            )

        print(
            "\nSPP-YOLOv8 UI "
            "Component Detection"
        )

        print(
            "=================================="
        )

        print(
            "Total UI components detected:",
            len(detected_components)
        )

        for component in detected_components:
            print(
                component["component_id"],
                component["class_name"],
                "Confidence:",
                round(
                    component[
                        "confidence"
                    ],
                    3
                ),
                "Bounding Box:",
                component[
                    "bbox_xyxy"
                ]
            )

        print(
            "\nDetected image saved to:"
        )
        print(
            detected_image_path
        )

        print(
            "\nComponent information saved to:"
        )
        print(
            json_output_path
        )

        return {
            "components":
                detected_components,

            "detected_image_path":
                detected_image_path,

            "json_output_path":
                json_output_path
        }


# ================================================================
# STANDALONE EXECUTION
# ================================================================

if __name__ == "__main__":

    image_path = (
        r"..\Output\UI\Preprocessing"
        r"\sample_ForegroundAware_CLAHE.png"
    )

    detector = Proposed_SPPYOLO(
        model_path=
            "../Models/OB/PYOLO.pt",
        conf_threshold=
            0.25,
        iou_threshold=
            0.50,
        imgsz=
            640
    )

    detector.detect_and_save(
        image_path
    )

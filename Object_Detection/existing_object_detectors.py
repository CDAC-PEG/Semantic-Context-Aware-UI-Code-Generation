import cv2
import torch

from ultralytics import YOLO

from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    fcos_resnet50_fpn
)
from torchvision.transforms.functional import to_tensor


# ================================================================
# STANDARD YOLO-v8
# ================================================================

class Existing_YOLOv8:
    model_name = "YOLO-v8"

    def __init__(
        self,
        model_path,
        conf_threshold=0.25,
        iou_threshold=0.50,
        imgsz=640
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

    def predict(self, image_path):
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False
        )

        predictions = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()

            for box, cls, score in zip(boxes, classes, scores):
                predictions.append({
                    "box": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3])
                    ],
                    "class_id": int(cls),
                    "score": float(score)
                })

        return predictions


# ================================================================
# RETINANET
# ================================================================

class Existing_RetinaNet:
    model_name = "RetinaNet"

    def __init__(
        self,
        checkpoint_path,
        num_classes,
        score_threshold=0.25,
        device=None
    ):
        self.device = (
            device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = retinanet_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes
        )

        self.model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()

        self.score_threshold = score_threshold

    def predict(self, image_path):
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = to_tensor(image).to(self.device)

        with torch.no_grad():
            output = self.model([tensor])[0]

        predictions = []

        for box, label, score in zip(
            output["boxes"],
            output["labels"],
            output["scores"]
        ):
            score = float(score.item())

            if score < self.score_threshold:
                continue

            predictions.append({
                "box": box.detach().cpu().numpy().tolist(),
                "class_id": int(label.item()),
                "score": score
            })

        return predictions


# ================================================================
# FCOS
# ================================================================

class Existing_FCOS:
    model_name = "FCOS"

    def __init__(
        self,
        checkpoint_path,
        num_classes,
        score_threshold=0.25,
        device=None
    ):
        self.device = (
            device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = fcos_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes
        )

        self.model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()

        self.score_threshold = score_threshold

    def predict(self, image_path):
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = to_tensor(image).to(self.device)

        with torch.no_grad():
            output = self.model([tensor])[0]

        predictions = []

        for box, label, score in zip(
            output["boxes"],
            output["labels"],
            output["scores"]
        ):
            score = float(score.item())

            if score < self.score_threshold:
                continue

            predictions.append({
                "box": box.detach().cpu().numpy().tolist(),
                "class_id": int(label.item()),
                "score": score
            })

        return predictions


# ================================================================
# EFFICIENTDET
# ================================================================

class Existing_EfficientDet:
    """
    EfficientDet adapter.

    Because EfficientDet implementations vary, pass a prediction function
    having the signature:

        predict_fn(image_path) -> [
            {
                "box": [x1, y1, x2, y2],
                "class_id": int,
                "score": float
            }
        ]
    """

    model_name = "EfficientDet"

    def __init__(self, predict_fn):
        if predict_fn is None:
            raise ValueError(
                "Existing_EfficientDet requires a predict_fn."
            )

        self.predict_fn = predict_fn

    def predict(self, image_path):
        return self.predict_fn(image_path)

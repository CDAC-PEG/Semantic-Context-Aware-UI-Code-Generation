import torch
import numpy as np

from Proposed_VisualTABERT import Proposed_VisualTABERT


def cosine_similarity_vector(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def pad_visual_features(values, target_dim=12):
    values = list(values)

    if len(values) < target_dim:
        values = values + [0.0] * (target_dim - len(values))
    else:
        values = values[:target_dim]

    return values


class VisualTABERTRankingAdapter:
    """
    Evaluation adapter for Proposed_VisualTABERT.py

    This file does NOT reimplement VisualTABERT.
    It imports and calls the existing Proposed_VisualTABERT model.

    Expected use:
        model = Proposed_VisualTABERT(...)
        model.load_state_dict(torch.load(...))

        adapter = VisualTABERTRankingAdapter(model)
    """

    model_name = "VisualTABERT"

    def __init__(self, model=None, model_path=None, model_kwargs=None, device=None):

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if model is not None:
            self.model = model

        else:
            model_kwargs = model_kwargs or {}

            self.model = Proposed_VisualTABERT(
                **model_kwargs
            )

            if model_path is not None:
                state_dict = torch.load(
                    model_path,
                    map_location=self.device
                )

                self.model.load_state_dict(
                    state_dict
                )

        self.model.to(
            self.device
        )

        self.model.eval()

    def _create_component(
        self,
        text,
        visual_features,
        visual_position,
        class_name="ui"
    ):
        vf = pad_visual_features(
            visual_features,
            12
        )

        position = list(
            visual_position
        )

        while len(position) < 4:
            position.append(0.0)

        return {
            "class_name":
                class_name,

            "confidence":
                vf[1]
                if len(vf) > 1
                else 1.0,

            "bbox_center": {
                "g1":
                    float(position[0]),

                "g2":
                    float(position[1]),

                "width":
                    float(position[2]),

                "height":
                    float(position[3])
            },

            "attributes": {
                "text":
                    text,

                "estimated_font_size_px":
                    vf[8]
                    if len(vf) > 8
                    else 0.0,

                "ocr_confidence":
                    vf[9]
                    if len(vf) > 9
                    else 1.0,

                "font_style": {
                    "underline":
                        bool(vf[10] > 0.5)
                        if len(vf) > 10
                        else False,

                    "strike_through":
                        bool(vf[11] > 0.5)
                        if len(vf) > 11
                        else False
                }
            }
        }

    def encode_item(
        self,
        text,
        visual_features,
        visual_position,
        class_name="ui"
    ):
        component = self._create_component(
            text=text,
            visual_features=visual_features,
            visual_position=visual_position,
            class_name=class_name
        )

        with torch.no_grad():

            encoded = self.model.encode_group(
                [component]
            )

            if encoded is None:
                raise RuntimeError(
                    "VisualTABERT returned no semantic representation."
                )

            vector = (
                encoded["G_z"]
                .detach()
                .cpu()
                .numpy()
            )

        return vector

    def score_candidate(
        self,
        example,
        candidate
    ):
        query_vector = self.encode_item(
            text=
                example.query_text,

            visual_features=
                example.query_visual_features,

            visual_position=
                example.query_visual_position,

            class_name="query"
        )

        candidate_vector = self.encode_item(
            text=
                candidate.text,

            visual_features=
                candidate.visual_features,

            visual_position=
                candidate.visual_position,

            class_name=
                candidate.text.split()[0]
                if candidate.text
                else "ui"
        )

        return cosine_similarity_vector(
            query_vector,
            candidate_vector
        )

    def rank(self, example):
        scored = []

        for candidate in example.candidates:

            score = self.score_candidate(
                example,
                candidate
            )

            scored.append({
                "candidate_id":
                    candidate.candidate_id,

                "score":
                    float(score),

                "relevance":
                    float(
                        candidate.relevance
                    )
            })

        scored.sort(
            key=lambda item:
                item["score"],
            reverse=True
        )

        return scored

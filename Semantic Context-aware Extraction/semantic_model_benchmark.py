import os
import json
import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import List

from transformers import (
    BertTokenizer,
    VisualBertModel,
    LxmertModel,
    LxmertTokenizer
)


def reciprocal_rank(relevance_sorted):
    for rank, rel in enumerate(relevance_sorted, start=1):
        if rel > 0:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevance_sorted, k=None):
    if k is None:
        k = len(relevance_sorted)
    score = 0.0
    for rank, rel in enumerate(relevance_sorted[:k], start=1):
        score += (2 ** float(rel) - 1.0) / math.log2(rank + 1.0)
    return score


def ndcg_at_k(relevance_sorted, k=None):
    if k is None:
        k = len(relevance_sorted)
    ideal = sorted(relevance_sorted, reverse=True)
    actual_dcg = dcg_at_k(relevance_sorted, k)
    ideal_dcg = dcg_at_k(ideal, k)
    return 0.0 if ideal_dcg == 0 else actual_dcg / ideal_dcg


def evaluate_rankings(all_ranked_relevance, k=None):
    rr_values = [reciprocal_rank(r) for r in all_ranked_relevance]
    ndcg_values = [ndcg_at_k(r, k) for r in all_ranked_relevance]
    return {
        "MRR": float(np.mean(rr_values)) if rr_values else 0.0,
        "NDCG": float(np.mean(ndcg_values)) if ndcg_values else 0.0
    }


@dataclass
class CandidateUIElement:
    candidate_id: str
    text: str
    visual_features: List[float]
    visual_position: List[float]
    relevance: float


@dataclass
class RankingExample:
    query_text: str
    query_visual_features: List[float]
    query_visual_position: List[float]
    candidates: List[CandidateUIElement]


def cosine_similarity_vector(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def pad_visual_features(values, target_dim):
    values = list(values)
    return values + [0.0] * (target_dim - len(values)) if len(values) < target_dim else values[:target_dim]


class BaseRankingModel:
    model_name = "Base"

    def score_candidate(self, example, candidate):
        raise NotImplementedError

    def rank(self, example):
        scored = []
        for candidate in example.candidates:
            scored.append({
                "candidate_id": candidate.candidate_id,
                "score": float(self.score_candidate(example, candidate)),
                "relevance": float(candidate.relevance)
            })
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored


class VisualTABERTRankingAdapter(BaseRankingModel):
    model_name = "VisualTABERT"

    def __init__(self, proposed_model):
        self.model = proposed_model
        self.model.eval()

    def _to_component(self, text, visual_features, visual_position, class_name="ui"):
        vf = pad_visual_features(visual_features, 12)
        return {
            "class_name": class_name,
            "confidence": vf[1] if len(vf) > 1 else 1.0,
            "bbox_center": {
                "g1": visual_position[0],
                "g2": visual_position[1],
                "width": visual_position[2],
                "height": visual_position[3]
            },
            "attributes": {
                "text": text,
                "estimated_font_size_px": vf[8] if len(vf) > 8 else 0.0,
                "ocr_confidence": vf[9] if len(vf) > 9 else 1.0,
                "font_style": {
                    "underline": bool(vf[10] > 0.5) if len(vf) > 10 else False,
                    "strike_through": bool(vf[11] > 0.5) if len(vf) > 11 else False
                }
            }
        }

    def score_candidate(self, example, candidate):
        q = self._to_component(
            example.query_text,
            example.query_visual_features,
            example.query_visual_position,
            "query"
        )
        c = self._to_component(
            candidate.text,
            candidate.visual_features,
            candidate.visual_position,
            candidate.text.split()[0] if candidate.text else "ui"
        )

        with torch.no_grad():
            qv = self.model.encode_group([q])["G_z"].detach().cpu().numpy()
            cv = self.model.encode_group([c])["G_z"].detach().cpu().numpy()

        return cosine_similarity_vector(qv, cv)


class VisualBERTRankingAdapter(BaseRankingModel):
    model_name = "VisualBERT"

    def __init__(self, checkpoint="uclanlp/visualbert-vqa-coco-pre", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = VisualBertModel.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        self.visual_dim = self.model.config.visual_embedding_dim

    def _encode(self, text, visual_features):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        vf = pad_visual_features(visual_features, self.visual_dim)
        visual_embeds = torch.tensor([[vf]], dtype=torch.float32, device=self.device)
        visual_token_type_ids = torch.ones((1, 1), dtype=torch.long, device=self.device)
        visual_attention_mask = torch.ones((1, 1), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            output = self.model(
                **inputs,
                visual_embeds=visual_embeds,
                visual_token_type_ids=visual_token_type_ids,
                visual_attention_mask=visual_attention_mask,
                return_dict=True
            )

        vector = output.pooler_output[0] if output.pooler_output is not None else output.last_hidden_state[0, 0]
        return vector.detach().cpu().numpy()

    def score_candidate(self, example, candidate):
        return cosine_similarity_vector(
            self._encode(example.query_text, example.query_visual_features),
            self._encode(candidate.text, candidate.visual_features)
        )


class LXMERRankingAdapter(BaseRankingModel):
    model_name = "LXMERT"

    def __init__(self, checkpoint="unc-nlp/lxmert-base-uncased", visual_feat_dim=2048, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = LxmertTokenizer.from_pretrained(checkpoint)
        self.model = LxmertModel.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        self.visual_feat_dim = visual_feat_dim

    def _encode(self, text, visual_features, visual_position):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        vf = pad_visual_features(visual_features, self.visual_feat_dim)
        vp = pad_visual_features(visual_position, 4)

        visual_feats = torch.tensor([[vf]], dtype=torch.float32, device=self.device)
        visual_pos = torch.tensor([[vp]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                token_type_ids=inputs.get("token_type_ids"),
                visual_feats=visual_feats,
                visual_pos=visual_pos,
                return_dict=True
            )

        return output.language_output[0, 0].detach().cpu().numpy()

    def score_candidate(self, example, candidate):
        return cosine_similarity_vector(
            self._encode(
                example.query_text,
                example.query_visual_features,
                example.query_visual_position
            ),
            self._encode(
                candidate.text,
                candidate.visual_features,
                candidate.visual_position
            )
        )


class UNITERRankingAdapter(BaseRankingModel):
    model_name = "UNITER"

    def __init__(self, encoder_fn):
        if encoder_fn is None:
            raise ValueError("Connect the official/research UNITER encoder using encoder_fn.")
        self.encoder_fn = encoder_fn

    def score_candidate(self, example, candidate):
        q = self.encoder_fn(
            example.query_text,
            example.query_visual_features,
            example.query_visual_position
        )
        c = self.encoder_fn(
            candidate.text,
            candidate.visual_features,
            candidate.visual_position
        )
        return cosine_similarity_vector(q, c)


class ViLBERTRankingAdapter(BaseRankingModel):
    model_name = "ViLBERT"

    def __init__(self, encoder_fn):
        if encoder_fn is None:
            raise ValueError("Connect a compatible ViLBERT encoder using encoder_fn.")
        self.encoder_fn = encoder_fn

    def score_candidate(self, example, candidate):
        q = self.encoder_fn(
            example.query_text,
            example.query_visual_features,
            example.query_visual_position
        )
        c = self.encoder_fn(
            candidate.text,
            candidate.visual_features,
            candidate.visual_position
        )
        return cosine_similarity_vector(q, c)


class SemanticRankingBenchmark:
    def __init__(self, models):
        self.models = models

    def evaluate_model(self, model, examples, ndcg_k=None):
        ranked_relevance = []
        detailed_rankings = []

        for example_index, example in enumerate(examples):
            ranking = model.rank(example)
            ranked_relevance.append([item["relevance"] for item in ranking])
            detailed_rankings.append({
                "example_index": example_index,
                "query": example.query_text,
                "ranking": ranking
            })

        metrics = evaluate_rankings(ranked_relevance, k=ndcg_k)

        return {
            "model": model.model_name,
            "MRR": metrics["MRR"],
            "NDCG": metrics["NDCG"],
            "rankings": detailed_rankings
        }

    def evaluate_all(self, examples, ndcg_k=None):
        results = []

        for model in self.models:
            print(f"Evaluating {model.model_name}...")
            result = self.evaluate_model(model, examples, ndcg_k)
            results.append(result)
            print(f"  MRR  = {result['MRR']:.4f}")
            print(f"  NDCG = {result['NDCG']:.4f}")

        return results

    @staticmethod
    def save_results(results, output_path):
        folder = os.path.dirname(output_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)


def load_ranking_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    examples = []

    for item in raw:
        candidates = [
            CandidateUIElement(
                candidate_id=str(c["candidate_id"]),
                text=c.get("text", ""),
                visual_features=c.get("visual_features", []),
                visual_position=c.get("visual_position", [0, 0, 0, 0]),
                relevance=float(c.get("relevance", 0))
            )
            for c in item["candidates"]
        ]

        examples.append(
            RankingExample(
                query_text=item.get("query_text", ""),
                query_visual_features=item.get("query_visual_features", []),
                query_visual_position=item.get("query_visual_position", [0, 0, 0, 0]),
                candidates=candidates
            )
        )

    return examples


if __name__ == "__main__":
    print("Semantic ranking benchmark module loaded.")
    print("Connect trained model checkpoints and an annotated ranking dataset before evaluation.")

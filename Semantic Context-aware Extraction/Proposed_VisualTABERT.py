import os
import json
import math
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTokenizer:
    PAD_ID = 0
    CLS_ID = 1
    SEP_ID = 2
    UNK_ID = 3

    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size

    def _hash_token(self, token):
        token = str(token).lower().strip()
        if not token:
            return self.UNK_ID
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return 4 + (int(digest[:12], 16) % (self.vocab_size - 4))

    def encode(self, text, max_tokens=16):
        ids = [self.CLS_ID]
        for token in str(text).split()[:max_tokens]:
            ids.append(self._hash_token(token))
        ids.append(self.SEP_ID)
        return ids


class TopKAttention(nn.Module):
    def __init__(self, d_model=128, top_k=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)

        k = min(max(1, self.top_k), scores.size(-1))
        top_values, top_indices = torch.topk(scores, k=k, dim=-1)

        sparse_scores = torch.full_like(scores, float("-inf"))
        sparse_scores.scatter_(-1, top_indices, top_values)

        att = F.softmax(sparse_scores, dim=-1)
        att = torch.nan_to_num(att, nan=0.0)
        att = self.dropout(att)

        H = torch.matmul(att, V)
        H = self.out_proj(H)

        return H, att


class Proposed_VisualTABERT(nn.Module):
    def __init__(
        self,
        vocab_size=30000,
        d_model=128,
        max_position=512,
        num_heads=4,
        top_k=4,
        dropout=0.1,
        device=None
    ):
        super().__init__()

        self.d_model = d_model
        self.max_position = max_position
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = SimpleTokenizer(vocab_size)

        # J(beta_y) = iota + rho
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_position, d_model)

        # visual embedding = object projection + segment embedding + alignment embedding
        self.visual_feature_dim = 12
        self.object_projection = nn.Linear(self.visual_feature_dim, d_model)
        self.modality_embedding = nn.Embedding(2, d_model)
        self.alignment_projection = nn.Linear(4, d_model)

        self.topk_attention = TopKAttention(
            d_model=d_model,
            top_k=top_k,
            dropout=dropout
        )

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.to(self.device)

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _class_code(name):
        text = str(name).lower().strip()
        if not text:
            return 0.0
        value = sum((i + 1) * ord(ch) for i, ch in enumerate(text))
        return float(value % 1000) / 1000.0

    def component_visual_features(self, component):
        attrs = component.get("attributes", {})
        bbox = component.get("bbox_center", {})

        g1 = self._safe_float(bbox.get("g1", 0.0))
        g2 = self._safe_float(bbox.get("g2", 0.0))
        w = self._safe_float(bbox.get("width", 0.0))
        h = self._safe_float(bbox.get("height", 0.0))

        aspect = w / h if h > 0 else 0.0
        area = w * h

        style = attrs.get("font_style", {})

        return np.asarray([
            self._class_code(component.get("class_name", "")),
            self._safe_float(component.get("confidence", 0.0)),
            g1,
            g2,
            w,
            h,
            aspect,
            area,
            self._safe_float(attrs.get("estimated_font_size_px", 0.0)),
            self._safe_float(attrs.get("ocr_confidence", 0.0)),
            1.0 if style.get("underline", False) else 0.0,
            1.0 if style.get("strike_through", False) else 0.0
        ], dtype=np.float32)

    def alignment_features(self, component):
        bbox = component.get("bbox_center", {})
        g1 = self._safe_float(bbox.get("g1", 0.0))
        g2 = self._safe_float(bbox.get("g2", 0.0))
        w = self._safe_float(bbox.get("width", 0.0))
        h = self._safe_float(bbox.get("height", 0.0))

        scale = max(abs(g1), abs(g2), abs(w), abs(h), 1.0)

        return np.asarray(
            [g1 / scale, g2 / scale, w / scale, h / scale],
            dtype=np.float32
        )

    def embed_text(self, text, start_position=0):
        token_ids = self.tokenizer.encode(text)

        token_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=self.device
        )

        positions = torch.arange(
            start_position,
            start_position + len(token_ids),
            device=self.device
        )
        positions = torch.clamp(positions, max=self.max_position - 1)

        iota = self.token_embedding(token_ids)
        rho = self.position_embedding(positions)

        J = iota + rho

        # textual modality embedding
        text_modality = torch.zeros(
            len(token_ids),
            dtype=torch.long,
            device=self.device
        )
        J = J + self.modality_embedding(text_modality)

        return J, token_ids

    def embed_visual_component(self, component):
        visual = torch.tensor(
            self.component_visual_features(component),
            dtype=torch.float32,
            device=self.device
        )

        alignment = torch.tensor(
            self.alignment_features(component),
            dtype=torch.float32,
            device=self.device
        )

        oj = self.object_projection(visual)
        segment = self.modality_embedding(
            torch.tensor(1, dtype=torch.long, device=self.device)
        )
        aE = self.alignment_projection(alignment)

        return oj + segment + aE

    def encode_group(self, components):
        sequence = []
        metadata = []
        position_cursor = 0

        for index, component in enumerate(components):
            attrs = component.get("attributes", {})
            text = attrs.get("text", "")

            J, token_ids = self.embed_text(
                text,
                start_position=position_cursor
            )

            for j in range(J.size(0)):
                sequence.append(J[j])
                metadata.append({
                    "type": "text",
                    "component_index": index,
                    "token_id": int(token_ids[j].item())
                })

            position_cursor += J.size(0)

            visual_embedding = self.embed_visual_component(component)
            sequence.append(visual_embedding)

            metadata.append({
                "type": "visual",
                "component_index": index,
                "class_name": component.get("class_name", "")
            })

            position_cursor += 1

        if not sequence:
            return None

        X = torch.stack(sequence, dim=0).unsqueeze(0)

        H, topk_att = self.topk_attention(X)
        H = self.norm1(X + H)

        mha_out, mha_weights = self.multihead_attention(
            H, H, H,
            need_weights=True,
            average_attn_weights=False
        )

        Z = self.norm2(H + mha_out)
        Z = Z + self.feed_forward(Z)

        # G_z = group-level semantic context
        G_z = Z.mean(dim=1).squeeze(0)

        return {
            "G_z": G_z,
            "sequence_context": Z.squeeze(0),
            "topk_attention": topk_att.squeeze(0),
            "multihead_attention": mha_weights.squeeze(0),
            "metadata": metadata
        }

    @staticmethod
    def cosine(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return 0.0 if denom == 0 else float(np.dot(a, b) / denom)

    def extract_semantic_context(self, input_json_path, output_json_path=None):
        if not os.path.exists(input_json_path):
            raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

        with open(input_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        beta_y = payload.get("beta_y", {})

        semantic_context = {}
        group_vectors = {}

        self.eval()

        with torch.no_grad():
            for group_name, members in beta_y.items():
                if group_name == "noise" or not members:
                    continue

                encoded = self.encode_group(members)
                if encoded is None:
                    continue

                G_z = encoded["G_z"].cpu().numpy()
                group_vectors[group_name] = G_z.tolist()

                sparse_attention = encoded["topk_attention"].cpu().numpy()

                links = []
                for q in range(sparse_attention.shape[0]):
                    for k in range(sparse_attention.shape[1]):
                        value = float(sparse_attention[q, k])
                        if value > 0:
                            links.append({
                                "query_index": q,
                                "key_index": k,
                                "attention": value
                            })

                semantic_context[group_name] = {
                    "number_of_elements": len(members),
                    "G_z": G_z.tolist(),
                    "top_k_attention_links": links,
                    "sequence_metadata": encoded["metadata"]
                }

        relationships = []
        group_names = list(group_vectors.keys())

        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                g1 = group_names[i]
                g2 = group_names[j]

                relationships.append({
                    "source_group": g1,
                    "target_group": g2,
                    "semantic_similarity": self.cosine(
                        group_vectors[g1],
                        group_vectors[g2]
                    )
                })

        result = {
            "method": "VisualTABERT",
            "input_stage": input_json_path,
            "semantic_context": semantic_context,
            "semantic_relationships": relationships
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
        r"..\Output\UI\Component_Grouping"
        r"\sample_DSFBSCAN.json"
    )

    output_json = (
        r"..\Output\UI\Semantic_Relations"
        r"\sample_VisualTABERT.json"
    )

    model = Proposed_VisualTABERT(
        vocab_size=30000,
        d_model=128,
        max_position=512,
        num_heads=4,
        top_k=4,
        dropout=0.1
    )

    result = model.extract_semantic_context(
        input_json_path=input_json,
        output_json_path=output_json
    )

    print("\nVisualTABERT Semantic Context Extraction")
    print("========================================")

    for group_name, group_data in result["semantic_context"].items():
        print(
            group_name,
            "->",
            group_data["number_of_elements"],
            "elements"
        )

    print("\nSemantic relationships")

    for relation in result["semantic_relationships"]:
        print(
            relation["source_group"],
            "<->",
            relation["target_group"],
            ":",
            round(relation["semantic_similarity"], 4)
        )

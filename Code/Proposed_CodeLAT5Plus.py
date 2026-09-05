import os
import json
import math
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# SIMPLE TOKENIZER
# ================================================================

class SimpleCodeTokenizer:
    """
    Lightweight tokenizer used for a fully self-contained implementation.

    For publication-scale experiments, this tokenizer can later be replaced
    with a pretrained CodeT5/CodeT5+ tokenizer while keeping the same
    architecture interface.
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size

    def _hash_token(self, token):
        token = str(token).strip()

        if not token:
            return self.UNK_ID

        digest = hashlib.sha256(
            token.encode("utf-8")
        ).hexdigest()

        return 4 + (
            int(digest[:12], 16)
            % (self.vocab_size - 4)
        )

    def encode(self, text, max_length=1024):
        tokens = str(text).replace(
            "\n", " \n "
        ).split()

        ids = [self.BOS_ID]

        for token in tokens[:max_length - 2]:
            ids.append(
                self._hash_token(token)
            )

        ids.append(self.EOS_ID)

        return ids

    def decode(self, token_ids):
        """
        Hash-tokenization is not invertible.

        Therefore, this self-contained tokenizer is intended for model
        architecture testing. In real training/generation experiments,
        replace it with a reversible pretrained CodeT5 tokenizer.
        """
        return " ".join(
            f"<tok_{int(token_id)}>"
            for token_id in token_ids
            if int(token_id) not in {
                self.PAD_ID,
                self.BOS_ID,
                self.EOS_ID
            }
        )


# ================================================================
# STRUCTURED REPRESENTATION LINEARIZER
# ================================================================

class StructuredRepresentationLinearizer:
    """
    Converts F_lambda JSON into a deterministic sequence.
    """

    @staticmethod
    def linearize(F_lambda):
        parts = []

        parts.append("<UI_SCHEMA>")

        metadata = F_lambda.get(
            "metadata",
            {}
        )

        parts.append(
            f"<NUM_COMPONENTS> "
            f"{metadata.get('number_of_components', 0)}"
        )

        parts.append(
            f"<NUM_GROUPS> "
            f"{metadata.get('number_of_groups', 0)}"
        )

        for component in F_lambda.get(
            "components",
            []
        ):
            parts.append("<COMPONENT>")

            parts.append(
                f"<ID> "
                f"{component.get('component_id', '')}"
            )

            parts.append(
                f"<TYPE> "
                f"{component.get('class_name', '')}"
            )

            parts.append(
                f"<TEXT> "
                f"{component.get('text', '')}"
            )

            parts.append(
                f"<GROUP> "
                f"{component.get('group', '')}"
            )

            layout = component.get(
                "layout",
                {}
            )

            aligned = layout.get(
                "aligned_coordinates",
                {}
            )

            parts.append(
                "<LAYOUT> "
                f"x={aligned.get('x1', aligned.get('g1', 0))} "
                f"y={aligned.get('y1', aligned.get('g2', 0))} "
                f"w={aligned.get('width', 0)} "
                f"h={aligned.get('height', 0)}"
            )

            semantic = component.get(
                "semantic_context",
                {}
            )

            gz = semantic.get(
                "G_z",
                []
            )

            # Long semantic vectors are summarized to avoid huge sequences.
            if gz:
                gz_preview = gz[:8]

                parts.append(
                    "<SEMANTIC> "
                    + " ".join(
                        f"{float(v):.4f}"
                        for v in gz_preview
                    )
                )

            parts.append("</COMPONENT>")

        for relation in F_lambda.get(
            "semantic_relationships",
            []
        ):
            parts.append(
                "<RELATION> "
                f"{relation.get('source_group', '')} "
                f"{relation.get('target_group', '')} "
                f"{relation.get('semantic_similarity', 0.0)}"
            )

        parts.append("</UI_SCHEMA>")

        return "\n".join(parts)


# ================================================================
# LONGFORMER-STYLE SLIDING WINDOW ATTENTION
# ================================================================

class SlidingWindowSelfAttention(nn.Module):
    """
    Longformer-style local self-attention.

    Each token attends only to tokens within a fixed window.
    """

    def __init__(
        self,
        d_model=256,
        num_heads=8,
        window_size=64,
        dropout=0.1
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = (
            d_model // num_heads
        )
        self.window_size = window_size

        self.q_proj = nn.Linear(
            d_model,
            d_model
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def _split_heads(self, x):
        B, L, D = x.shape

        x = x.view(
            B,
            L,
            self.num_heads,
            self.head_dim
        )

        return x.transpose(
            1,
            2
        )

    def _merge_heads(self, x):
        B, H, L, Dh = x.shape

        x = x.transpose(
            1,
            2
        ).contiguous()

        return x.view(
            B,
            L,
            H * Dh
        )

    def forward(self, x):
        """
        x: [B, L, D]
        """

        B, L, D = x.shape

        Q = self._split_heads(
            self.q_proj(x)
        )

        K = self._split_heads(
            self.k_proj(x)
        )

        V = self._split_heads(
            self.v_proj(x)
        )

        outputs = []

        for i in range(L):

            start = max(
                0,
                i - self.window_size
            )

            end = min(
                L,
                i + self.window_size + 1
            )

            q_i = Q[:, :, i:i + 1, :]

            k_local = K[
                :,
                :,
                start:end,
                :
            ]

            v_local = V[
                :,
                :,
                start:end,
                :
            ]

            scores = torch.matmul(
                q_i,
                k_local.transpose(
                    -2,
                    -1
                )
            ) / math.sqrt(
                self.head_dim
            )

            att = F.softmax(
                scores,
                dim=-1
            )

            att = self.dropout(
                att
            )

            local_output = torch.matmul(
                att,
                v_local
            )

            outputs.append(
                local_output
            )

        Z = torch.cat(
            outputs,
            dim=2
        )

        Z = self._merge_heads(
            Z
        )

        return self.out_proj(
            Z
        )


# ================================================================
# ENCODER BLOCK
# ================================================================

class CodeLATEncoderBlock(nn.Module):
    """
    Encoder block:
        Longformer-style attention
        + FFN
        + LayerNorm
    """

    def __init__(
        self,
        d_model=256,
        num_heads=8,
        window_size=64,
        ff_dim=1024,
        dropout=0.1
    ):
        super().__init__()

        self.long_attention = (
            SlidingWindowSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                window_size=window_size,
                dropout=dropout
            )
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                d_model,
                ff_dim
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                ff_dim,
                d_model
            )
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):
        Z = self.long_attention(
            x
        )

        x = self.norm1(
            x + self.dropout(Z)
        )

        ffn_out = self.ffn(
            x
        )

        Z_tilde = self.norm2(
            x + self.dropout(
                ffn_out
            )
        )

        return Z_tilde


# ================================================================
# DECODER BLOCK
# ================================================================

class CodeLATDecoderBlock(nn.Module):
    """
    Decoder:
        causal self-attention
        + encoder-decoder cross-attention
        + FFN
    """

    def __init__(
        self,
        d_model=256,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1
    ):
        super().__init__()

        self.self_attention = (
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        )

        self.cross_attention = (
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.norm3 = nn.LayerNorm(
            d_model
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                d_model,
                ff_dim
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                ff_dim,
                d_model
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

    @staticmethod
    def causal_mask(
        length,
        device
    ):
        return torch.triu(
            torch.full(
                (length, length),
                float("-inf"),
                device=device
            ),
            diagonal=1
        )

    def forward(
        self,
        x,
        encoder_output
    ):
        L = x.size(1)

        mask = self.causal_mask(
            L,
            x.device
        )

        causal_out, _ = (
            self.self_attention(
                x,
                x,
                x,
                attn_mask=mask
            )
        )

        x = self.norm1(
            x + self.dropout(
                causal_out
            )
        )

        cross_out, _ = (
            self.cross_attention(
                x,
                encoder_output,
                encoder_output
            )
        )

        x = self.norm2(
            x + self.dropout(
                cross_out
            )
        )

        ffn_out = self.ffn(
            x
        )

        x = self.norm3(
            x + self.dropout(
                ffn_out
            )
        )

        return x


# ================================================================
# PROPOSED CODELAT5+
# ================================================================

class Proposed_CodeLAT5Plus(nn.Module):
    """
    Custom CodeLAT5+ architecture.

    Input:
        F_lambda

    Output:
        eta_ed

    Training objectives:
        X1 = autoregressive code generation loss
        X2 = text/code representation alignment loss

        Total loss = X1 + alpha * X2
    """

    def __init__(
        self,
        vocab_size=32000,
        d_model=256,
        max_position=2048,
        num_heads=8,
        window_size=64,
        ff_dim=1024,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        alignment_weight=0.1,
        device=None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_position = max_position
        self.alignment_weight = (
            alignment_weight
        )

        self.device = (
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.tokenizer = (
            SimpleCodeTokenizer(
                vocab_size=vocab_size
            )
        )

        self.token_embedding = (
            nn.Embedding(
                vocab_size,
                d_model
            )
        )

        self.position_embedding = (
            nn.Embedding(
                max_position,
                d_model
            )
        )

        self.encoder_blocks = (
            nn.ModuleList([
                CodeLATEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    window_size=window_size,
                    ff_dim=ff_dim,
                    dropout=dropout
                )
                for _ in range(
                    num_encoder_layers
                )
            ])
        )

        self.decoder_blocks = (
            nn.ModuleList([
                CodeLATDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout
                )
                for _ in range(
                    num_decoder_layers
                )
            ])
        )

        self.output_projection = (
            nn.Linear(
                d_model,
                vocab_size
            )
        )

        self.to(
            self.device
        )

    # ============================================================
    # EMBEDDING
    # ============================================================

    def embed_tokens(
        self,
        token_ids
    ):
        B, L = token_ids.shape

        positions = torch.arange(
            L,
            device=self.device
        ).unsqueeze(0).expand(
            B,
            L
        )

        positions = torch.clamp(
            positions,
            max=self.max_position - 1
        )

        return (
            self.token_embedding(
                token_ids
            )
            +
            self.position_embedding(
                positions
            )
        )

    # ============================================================
    # ENCODER
    # ============================================================

    def encode(
        self,
        source_ids
    ):
        x = self.embed_tokens(
            source_ids
        )

        for block in (
            self.encoder_blocks
        ):
            x = block(
                x
            )

        return x

    # ============================================================
    # DECODER
    # ============================================================

    def decode(
        self,
        target_ids,
        encoder_output
    ):
        x = self.embed_tokens(
            target_ids
        )

        for block in (
            self.decoder_blocks
        ):
            x = block(
                x,
                encoder_output
            )

        logits = (
            self.output_projection(
                x
            )
        )

        return logits, x

    # ============================================================
    # TRAINING OBJECTIVES X1 + X2
    # ============================================================

    def compute_loss(
        self,
        source_ids,
        target_ids
    ):
        encoder_output = self.encode(
            source_ids
        )

        decoder_input = (
            target_ids[:, :-1]
        )

        labels = (
            target_ids[:, 1:]
        )

        logits, decoder_hidden = (
            self.decode(
                decoder_input,
                encoder_output
            )
        )

        # --------------------------------------------------------
        # X1: autoregressive code generation loss
        # --------------------------------------------------------

        X1 = F.cross_entropy(
            logits.reshape(
                -1,
                self.vocab_size
            ),
            labels.reshape(-1),
            ignore_index=
                SimpleCodeTokenizer.PAD_ID
        )

        # --------------------------------------------------------
        # X2: representation alignment objective
        # --------------------------------------------------------

        encoder_pooled = (
            encoder_output.mean(
                dim=1
            )
        )

        decoder_pooled = (
            decoder_hidden.mean(
                dim=1
            )
        )

        cosine = F.cosine_similarity(
            encoder_pooled,
            decoder_pooled,
            dim=-1
        )

        X2 = (
            1.0
            - cosine
        ).mean()

        total_loss = (
            X1
            + self.alignment_weight * X2
        )

        return {
            "loss":
                total_loss,

            "X1_generation_loss":
                X1,

            "X2_alignment_loss":
                X2
        }

    # ============================================================
    # GREEDY GENERATION
    # ============================================================

    def generate_ids(
        self,
        source_ids,
        max_new_tokens=256
    ):
        self.eval()

        with torch.no_grad():

            encoder_output = (
                self.encode(
                    source_ids
                )
            )

            generated = torch.tensor(
                [[
                    SimpleCodeTokenizer.BOS_ID
                ]],
                dtype=torch.long,
                device=self.device
            )

            for _ in range(
                max_new_tokens
            ):

                logits, _ = self.decode(
                    generated,
                    encoder_output
                )

                next_token = (
                    torch.argmax(
                        logits[:, -1, :],
                        dim=-1,
                        keepdim=True
                    )
                )

                generated = torch.cat(
                    [
                        generated,
                        next_token
                    ],
                    dim=1
                )

                if int(
                    next_token.item()
                ) == (
                    SimpleCodeTokenizer.EOS_ID
                ):
                    break

        return generated

    # ============================================================
    # F_lambda -> eta_ed
    # ============================================================

    def generate_from_schema(
        self,
        input_json_path,
        output_code_path=None,
        max_source_length=1024,
        max_new_tokens=256
    ):
        """
        Load F_lambda JSON and generate eta_ed.
        """

        if not os.path.exists(
            input_json_path
        ):
            raise FileNotFoundError(
                f"Structured representation "
                f"not found: {input_json_path}"
            )

        with open(
            input_json_path,
            "r",
            encoding="utf-8"
        ) as file:

            F_lambda = json.load(
                file
            )

        linearized_input = (
            StructuredRepresentationLinearizer
            .linearize(
                F_lambda
            )
        )

        source_ids = (
            self.tokenizer.encode(
                linearized_input,
                max_length=
                    max_source_length
            )
        )

        source_tensor = torch.tensor(
            [source_ids],
            dtype=torch.long,
            device=self.device
        )

        generated_ids = (
            self.generate_ids(
                source_tensor,
                max_new_tokens=
                    max_new_tokens
            )
        )

        # eta_ed
        eta_ed = (
            self.tokenizer.decode(
                generated_ids[
                    0
                ].detach().cpu().tolist()
            )
        )

        result = {
            "method":
                "CodeLAT5+",

            "input_symbol":
                "F_lambda",

            "output_symbol":
                "eta_ed",

            "linearized_input":
                linearized_input,

            "generated_code":
                eta_ed
        }

        if output_code_path:

            folder = os.path.dirname(
                output_code_path
            )

            if folder:
                os.makedirs(
                    folder,
                    exist_ok=True
                )

            with open(
                output_code_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    eta_ed
                )

        return result


# ================================================================
# TRAINING HELPER
# ================================================================

def train_one_epoch(
    model,
    training_pairs,
    optimizer,
    max_source_length=1024,
    max_target_length=512
):
    """
    training_pairs:
        [
            {
                "schema_json": ".../sample_F_lambda.json",
                "reference_code": ".../sample.swift"
            }
        ]
    """

    model.train()

    total_loss = 0.0

    for pair in training_pairs:

        with open(
            pair["schema_json"],
            "r",
            encoding="utf-8"
        ) as file:

            F_lambda = json.load(
                file
            )

        source_text = (
            StructuredRepresentationLinearizer
            .linearize(
                F_lambda
            )
        )

        with open(
            pair["reference_code"],
            "r",
            encoding="utf-8"
        ) as file:

            target_code = file.read()

        source_ids = (
            model.tokenizer.encode(
                source_text,
                max_length=
                    max_source_length
            )
        )

        target_ids = (
            model.tokenizer.encode(
                target_code,
                max_length=
                    max_target_length
            )
        )

        source_tensor = torch.tensor(
            [source_ids],
            dtype=torch.long,
            device=model.device
        )

        target_tensor = torch.tensor(
            [target_ids],
            dtype=torch.long,
            device=model.device
        )

        optimizer.zero_grad()

        losses = model.compute_loss(
            source_tensor,
            target_tensor
        )

        losses["loss"].backward()

        optimizer.step()

        total_loss += float(
            losses[
                "loss"
            ].item()
        )

    return (
        total_loss
        / max(
            len(training_pairs),
            1
        )
    )


# ================================================================
# STANDALONE EXAMPLE
# ================================================================

if __name__ == "__main__":

    input_json = (
        r"..\Output\UI\Schema_Mapping"
        r"\sample_F_lambda.json"
    )

    output_code = (
        r"..\Output\Generated_Frontend_Code"
        r"\sample_generated_code.txt"
    )

    model = Proposed_CodeLAT5Plus(
        vocab_size=32000,
        d_model=256,
        max_position=2048,
        num_heads=8,
        window_size=64,
        ff_dim=1024,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        alignment_weight=0.1
    )

    result = model.generate_from_schema(
        input_json_path=input_json,
        output_code_path=output_code,
        max_source_length=1024,
        max_new_tokens=256
    )

    print(
        "\nCodeLAT5+ Front-End "
        "Code Generation"
    )

    print(
        "================================"
    )

    print(
        "\nGenerated code eta_ed:"
    )

    print(
        result[
            "generated_code"
        ]
    )

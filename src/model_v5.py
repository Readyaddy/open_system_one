"""Iteration 5: Multi-Question Poly-Encoder.

The gap this closes: every earlier iteration (v1-v4) answers exactly ONE
decision per forward pass -- one context, one candidate set, one output.
Jev's own documentation shows it answering several DIFFERENT typed
questions about the same input in a single pass ("you might define a
Choice over {billing, technical, sales, spam} and a Score for urgency
from 0 to 100, and Jev fills both in one pass"). That's the capability
this architecture adds.

Mechanism: v4's poly-encoder used `m` generic learned "code" vectors that
cross-attend over the context tokens to produce m parallel representations
(a technique with no particular meaning per code). Here that's made
explicit and task-aligned: instead of generic codes, there is one learned
query embedding PER QUESTION TYPE (intent / needs_human / urgency, with
room for more). All of them cross-attend over the SAME token hidden
states in a single batched attention call -- so the expensive transformer
forward pass over the context runs exactly once no matter how many
questions are asked, and each question gets its own dedicated read-out
vector from that one pass.

Each question type then routes to the head appropriate for its answer
type:
  - Choice ("intent"): dot-product compatibility against a bank of
    candidate outcome embeddings (same mechanism as v1-v4 -- still no
    decoding, candidates still independently encoded/cacheable).
  - Bool ("needs_human"): small MLP -> 1 logit -> sigmoid.
  - Score ("urgency", 5 ordinal bins): small MLP -> 5-way softmax,
    trained as ordinal classification so the output carries a calibrated
    probability per bin, not just a point estimate.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

BACKBONE = "sentence-transformers/all-roberta-large-v1"

QUESTION_TYPES = ["intent", "needs_human", "urgency"]  # extensible list; ids are the index into this list
QTYPE_TO_ID = {q: i for i, q in enumerate(QUESTION_TYPES)}
MAX_QUESTION_SLOTS = 8  # room to add more question types later without changing the embedding table shape


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def _freeze(backbone, freeze_layers: int):
    if freeze_layers <= 0:
        return
    for p in backbone.embeddings.parameters():
        p.requires_grad = False
    for layer in backbone.encoder.layer[:freeze_layers]:
        for p in layer.parameters():
            p.requires_grad = False


class OutcomeEncoder(nn.Module):
    """Unchanged from v4: one cacheable vector per Choice candidate,
    encoded independently of any context."""

    def __init__(self, out_dim: int, hidden_dim: int, backbone: str, freeze_layers: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size
        _freeze(self.backbone, freeze_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = _mean_pool(out.last_hidden_state, attention_mask)
        return self.head(pooled)


class MultiQuestionContextEncoder(nn.Module):
    """Runs the context transformer ONCE, then reads out one vector PER
    QUESTION TYPE via a single batched cross-attention call -- the actual
    "parallel outputs from one transformer pass" mechanism."""

    def __init__(self, out_dim: int, hidden_dim: int, backbone: str, freeze_layers: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size
        _freeze(self.backbone, freeze_layers)

        self.question_queries = nn.Embedding(MAX_QUESTION_SLOTS, hidden)
        self.question_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=8, batch_first=True)
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.out_dim = out_dim
        self.to_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, input_ids, attention_mask, question_type_ids):
        """question_type_ids: LongTensor (n_questions,) -- which question
        slots to read out, shared across the whole batch (every example in
        a batch gets asked the same set of questions here; see train_v5.py
        for why that's the right simplification for this dataset).

        Returns: (B, n_questions, hidden_dim) -- the shared hidden_dim
        representation BEFORE the final per-type head, since Bool/Score
        heads want to read from a shared space while the Choice head wants
        the smaller out_dim space matching the outcome encoder.
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = out.last_hidden_state  # (B, T, H)
        B = hidden_states.size(0)

        query = self.question_queries(question_type_ids).unsqueeze(0).expand(B, -1, -1)  # (B, Q, H)
        key_padding_mask = attention_mask == 0
        read_out, _ = self.question_attn(query, hidden_states, hidden_states,
                                          key_padding_mask=key_padding_mask, need_weights=False)
        return self.shared_proj(read_out)  # (B, Q, hidden_dim)

    def to_choice_space(self, shared_repr):
        return self.to_out(shared_repr)


class MultiQuestionPolyEncoderV5(nn.Module):
    def __init__(self, out_dim: int = 512, hidden_dim: int = 1024, temperature: float = 0.07,
                 backbone: str = BACKBONE, freeze_layers: int = 16):
        super().__init__()
        self.context_encoder = MultiQuestionContextEncoder(out_dim, hidden_dim, backbone, freeze_layers)
        self.outcome_encoder = OutcomeEncoder(out_dim, hidden_dim, backbone, freeze_layers)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))

        self.needs_human_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim // 2, 1)
        )
        self.urgency_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim // 2, 5)
        )

    def encode_context_all(self, input_ids, attention_mask):
        """One forward pass -> per-question shared representations for ALL
        registered question types, in the fixed QUESTION_TYPES order."""
        qids = torch.arange(len(QUESTION_TYPES), device=input_ids.device)
        return self.context_encoder(input_ids, attention_mask, qids)  # (B, 3, hidden_dim)

    def encode_outcome(self, input_ids, attention_mask):
        emb = self.outcome_encoder(input_ids, attention_mask)
        return nn.functional.normalize(emb, dim=-1)

    def intent_logits(self, shared_repr_intent, outcome_emb):
        ctx = nn.functional.normalize(self.context_encoder.to_choice_space(shared_repr_intent), dim=-1)
        temp = self.log_temperature.exp().clamp(min=1e-3)
        return (ctx @ outcome_emb.t()) / temp

    def needs_human_logits(self, shared_repr_nh):
        return self.needs_human_head(shared_repr_nh).squeeze(-1)  # (B,)

    def urgency_logits(self, shared_repr_urg):
        return self.urgency_head(shared_repr_urg)  # (B, 5)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_gradient_checkpointing(self):
        self.context_encoder.backbone.gradient_checkpointing_enable()
        self.outcome_encoder.backbone.gradient_checkpointing_enable()


def get_tokenizer(backbone: str = BACKBONE):
    return AutoTokenizer.from_pretrained(backbone)

"""Iteration 6: Joint-Sequence Marker-Readout model on ModernBERT-large.

This is the architecture deduced from TypeSafe's own documentation of Jev,
not an incremental tweak of the dual/poly-encoder family (v1-v5):

  - State (context) and ALL questions -- their instructions and their
    options/levels -- are packed into ONE sequence and processed by ONE
    transformer forward pass with FULL self-attention. State tokens can
    attend directly to option tokens and vice versa; there is no pooling
    bottleneck and no separate "outcome encoder" that never sees the raw
    context tokens. This follows directly from the docs stating "state and
    all questions together" share ONE combined context budget (64k
    tokens) -- a strong signal they're literally one sequence, not two
    independently-limited encoder inputs.
  - Each candidate option, ordinal level, or bool question gets a
    dedicated MARKER TOKEN inserted into the sequence. After the
    transformer pass, the final hidden state AT each marker position is
    that option/level/question's read-out representation -- this is what
    "evaluated in parallel and in isolation" plausibly means: every marker
    gets its own read-out from the one shared pass, and (because attention
    is computed once for the whole sequence) adding more markers is cheap,
    matching the docs' "adding questions barely changes response time."
  - Score's answer is a probability-weighted EXPECTED VALUE over its
    ordinal level markers -- not a separate regression head -- because
    that is the only way to produce the fractional outputs (e.g. `1.035`)
    TypeSafe's docs show, out of a small number of discrete described
    levels.
  - Noul (bool) is a single marker read out with a sigmoid, framed as an
    entailment-style question ("does state entail this claim") rather
    than a classification -- there IS no candidate set for Noul in Jev's
    own API, which is the evidence for this framing.

Backbone: ModernBERT-large (395M params, 8192-token native context,
RoPE + alternating local/global attention) -- chosen specifically because
this architecture needs a backbone that can hold state + several
questions' worth of options in one sequence cheaply, which a 512-token
2021-era encoder (what v1-v4 used) cannot do.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

BACKBONE = "answerdotai/ModernBERT-large"
MARKER_TOKENS = ["[OPT]", "[NOUL]", "[LVL]"]


def build_tokenizer(backbone: str = BACKBONE):
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    tokenizer.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
    return tokenizer


def build_sequence_text(request: dict) -> str:
    """Serializes one Jev-shaped request into the single packed sequence
    string. Marker tokens are inserted immediately before each option's
    text so the marker's own hidden state can attend to that specific
    option (and everything before it) while remaining a distinct,
    findable position after tokenization."""
    parts = [request["state"], "[SEP]"]

    parts.append(request["choice_instructions"])
    for opt_text in request["choice_options"]:
        parts.append("[OPT]")
        parts.append(opt_text)
    parts.append("[SEP]")

    parts.append(request["noul_instructions"])
    parts.append("[NOUL]")
    parts.append("[SEP]")

    parts.append(request["score_instructions"])
    for level_text in request["score_levels"]:
        parts.append("[LVL]")
        parts.append(level_text)

    return " ".join(parts)


def find_marker_positions(input_ids: torch.Tensor, marker_id: int, expected_count: int) -> torch.Tensor:
    """input_ids: (B, T). Returns (B, expected_count) LongTensor of the
    positions of each occurrence of marker_id in each row, in order.
    Assumes every row has exactly `expected_count` occurrences (true here
    since every request has a fixed option/level count) -- rows that don't
    (e.g. truncation) are padded with position 0 as a safe fallback."""
    B = input_ids.size(0)
    out = torch.zeros(B, expected_count, dtype=torch.long, device=input_ids.device)
    for b in range(B):
        positions = (input_ids[b] == marker_id).nonzero(as_tuple=True)[0]
        n = min(expected_count, positions.size(0))
        if n > 0:
            out[b, :n] = positions[:n]
    return out


class JointSequenceModelV6(nn.Module):
    def __init__(self, tokenizer, backbone: str = BACKBONE, freeze_layers: int = 0,
                 n_choice_options: int = 6, n_score_levels: int = 5):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.backbone.resize_token_embeddings(len(tokenizer))
        hidden = self.backbone.config.hidden_size

        if freeze_layers > 0:
            for p in self.backbone.embeddings.parameters():
                p.requires_grad = False
            for layer in self.backbone.layers[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

        self.choice_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2),
                                          nn.GELU(), nn.Linear(hidden // 2, 1))
        self.score_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2),
                                         nn.GELU(), nn.Linear(hidden // 2, 1))
        self.noul_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2),
                                        nn.GELU(), nn.Linear(hidden // 2, 1))

        self.n_choice_options = n_choice_options
        self.n_score_levels = n_score_levels
        self.opt_id = tokenizer.convert_tokens_to_ids("[OPT]")
        self.noul_id = tokenizer.convert_tokens_to_ids("[NOUL]")
        self.lvl_id = tokenizer.convert_tokens_to_ids("[LVL]")

    def forward(self, input_ids, attention_mask):
        """One joint forward pass. Returns raw hidden states -- callers
        gather marker positions and apply heads (kept separate so the
        expensive backbone pass is shared across all three question
        types, mirroring 'adding questions barely changes response
        time')."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, T, H)

    def read_choice(self, hidden_states, input_ids):
        pos = find_marker_positions(input_ids, self.opt_id, self.n_choice_options)  # (B, K)
        gathered = torch.gather(hidden_states, 1, pos.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
        return self.choice_head(gathered).squeeze(-1)  # (B, K) logits

    def read_score(self, hidden_states, input_ids):
        pos = find_marker_positions(input_ids, self.lvl_id, self.n_score_levels)
        gathered = torch.gather(hidden_states, 1, pos.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
        return self.score_head(gathered).squeeze(-1)  # (B, n_levels) logits

    def read_noul(self, hidden_states, input_ids):
        pos = find_marker_positions(input_ids, self.noul_id, 1)  # (B, 1)
        gathered = torch.gather(hidden_states, 1, pos.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
        return self.noul_head(gathered).squeeze(-1).squeeze(-1)  # (B,) logit

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_gradient_checkpointing(self):
        self.backbone.gradient_checkpointing_enable()


def score_expected_value(score_logits: torch.Tensor) -> torch.Tensor:
    """Probability-weighted expectation over ordinal level position (0-indexed
    here; add 1 at the call site if 1-indexed levels are wanted) -- this is
    how a fractional score like `1.035` falls out of a discrete softmax."""
    probs = torch.softmax(score_logits, dim=-1)
    levels = torch.arange(score_logits.size(-1), device=score_logits.device, dtype=probs.dtype)
    return (probs * levels).sum(dim=-1)

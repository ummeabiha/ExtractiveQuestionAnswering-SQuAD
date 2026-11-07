"""Answer a question with one of the trained SQuAD models.

    python inference.py --model-dir models/electra_56b9af30 \
        --question "In which year did Alan Turing introduce the Turing machine?" \
        --context "Alan Turing was a British mathematician and computer scientist. He introduced the concept of the Turing machine in 1936."
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


class ScratchQAModel(nn.Module):   # same as in the notebook
    def __init__(self, vocab_size, hidden=384, layers=6, heads=6, ff_dim=1536, max_len=256, dropout=0.1, pad_id=0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.seg_emb = nn.Embedding(2, hidden)
        self.match_emb = nn.Embedding(2, hidden)
        self.emb_norm = nn.LayerNorm(hidden)
        self.emb_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(hidden, heads, ff_dim, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(hidden), enable_nested_tensor=False)
        self.start_head = nn.Linear(hidden, 1)
        self.end_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        question = (token_type_ids == 0) & (attention_mask == 1)
        match = ((input_ids[:, :, None] == input_ids[:, None, :]) & question[:, None, :]).any(-1).long()
        x = self.token_emb(input_ids) + self.pos_emb(positions) + self.seg_emb(token_type_ids) + self.match_emb(match)
        h = self.encoder(self.emb_drop(self.emb_norm(x)), src_key_padding_mask=attention_mask == 0)
        padding = attention_mask == 0
        return self.start_head(h).squeeze(-1).masked_fill(padding, -1e4), self.end_head(h).squeeze(-1).masked_fill(padding, -1e4)


def make_windows(tokenizer, question, context, max_len, stride):
    windows, start = [], 0
    while True:
        cut = start - 1 if start > 0 and context[start - 1].isspace() else start
        enc = tokenizer(question, context[cut:], truncation="only_second", max_length=max_len,
                        padding="max_length", return_offsets_mapping=True)
        seq_ids = enc.sequence_ids()
        enc["offset_mapping"] = [(a + cut, b + cut) if s == 1 else None for (a, b), s in zip(enc["offset_mapping"], seq_ids)]
        windows.append(enc)
        if sum(enc["attention_mask"]) < max_len:
            return windows
        ctx = [i for i, s in enumerate(seq_ids) if s == 1]
        j = max(1, len(ctx) - stride)
        while j > 1 and not context[enc["offset_mapping"][ctx[j]][0] - 1].isspace():
            j -= 1
        start = enc["offset_mapping"][ctx[j]][0]


def load(model_dir, device):
    with open(os.path.join(model_dir, "qa_run_config.json")) as f:
        config = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if config["kind"] == "scratch":
        model = ScratchQAModel(**config["arch"])
        model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"), map_location="cpu"))
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    return model.to(device).eval(), tokenizer, config


@torch.no_grad()
def answer(question, context, model, tokenizer, config, n_best=20, max_answer_len=30):
    device = next(model.parameters()).device
    windows = make_windows(tokenizer, question.strip(), context, config["max_len"], config["stride"])
    inputs = {k: torch.tensor([w[k] for w in windows], device=device)
              for k in ["input_ids", "attention_mask", "token_type_ids"] if k in windows[0]}
    if isinstance(model, ScratchQAModel):
        start, end = model(**inputs)
    else:
        out = model(**inputs)
        start, end = out.start_logits, out.end_logits
    start, end = start.float().cpu().numpy(), end.float().cpu().numpy()
    best_text, best_score = "", -1e9
    for i, w in enumerate(windows):
        offsets = w["offset_mapping"]
        for s in np.argsort(start[i])[-n_best:]:
            for e in np.argsort(end[i])[-n_best:]:
                if offsets[s] is None or offsets[e] is None or e < s or e - s + 1 > max_answer_len:
                    continue
                if start[i][s] + end[i][e] > best_score:
                    best_text, best_score = context[offsets[s][0]:offsets[e][1]], float(start[i][s] + end[i][e])
    return best_text, best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load(args.model_dir, device)
    text, score = answer(args.question, args.context, model, tokenizer, config)
    print(f"Answer: {text} (score {score:.2f})")

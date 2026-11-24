"""Stand-alone interface: paste a critic's tasting note, get a WSET-style card from the trained student model.
Needs pack/wset_student.pt (written by the last cells of the notebook).  Run with run_app.bat or:  python app.py
Unofficial; trained on Wine Enthusiast notes (CC BY-NC-SA 4.0): non-commercial use only."""
import sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

CKPT = Path(__file__).with_name("pack") / "wset_student.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WsetStudent(nn.Module):                      # same layout as in the notebook, so the saved weights fit
    def __init__(self, model_name, n_binary, n_group_outputs):
        super().__init__()
        self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name))     # structure only; weights come from the checkpoint
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.binary_head = nn.Linear(h, n_binary)
        self.group_head = nn.Linear(h, n_group_outputs)

    def forward(self, input_ids, attention_mask):
        tokens = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(tokens.dtype)
        note = (tokens * m).sum(1) / m.sum(1).clamp(min=1)          # average over the real words
        return self.binary_head(note), self.group_head(note)


def load():
    if not CKPT.exists():
        sys.exit(f"Cannot find {CKPT}. Run the notebook to the end first (it saves the trained model there).")
    ck = torch.load(CKPT, map_location="cpu")
    schema = ck["schema"]; groups = schema["groups"]
    sizes = [len(v) for v in groups.values()]; ends = torch.tensor(sizes).cumsum(0).tolist(); slices = list(zip([0] + ends[:-1], ends))
    model = WsetStudent(ck["model_name"], len(schema["binary"]), sum(sizes))
    model.load_state_dict(ck["model"]); model.to(device).eval()
    return model, AutoTokenizer.from_pretrained(ck["model_name"]), ck, groups, slices


model, tok, ck, GROUPS, SLICES = load()
THRESH = ck["thresholds"]


@torch.no_grad()
def wset_card(note):
    e = tok([note], padding="max_length", truncation=True, max_length=ck["max_len"], return_tensors="pt")
    lb, lg = model(e["input_ids"].to(device), e["attention_mask"].to(device))
    pb = torch.sigmoid(lb.float())[0].cpu().numpy(); lg = lg.float()[0].cpu()
    out = ["### Structure and levels"]
    for q, (s, e_) in zip(GROUPS, SLICES):
        p = F.softmax(lg[s:e_], -1).numpy(); label = GROUPS[q][p.argmax()]
        if label != "not_stated":
            out.append(f"- **{q.replace('__', ': ').replace('_', ' ')}**: {label} ({p.max():.0%})")
    for scope in ("nose", "palate"):
        hits = [(pb[j], q) for j, q in enumerate(ck["schema"]["binary"]) if q.startswith(scope + "__") and pb[j] >= THRESH[j]]
        out.append(f"### {scope.title()} descriptors")
        out += [f"- {q.split('__')[-1].replace('_', ' ')} ({p:.0%})" for p, q in sorted(hits, reverse=True)[:12]] or ["- none found"]
    return "\n".join(out)


if __name__ == "__main__":
    import gradio as gr
    gr.Interface(fn=wset_card, inputs=gr.Textbox(lines=5, label="Critic's tasting note"), outputs=gr.Markdown(),
                 title="Critic note -> WSET card (small student model)",
                 description="Unofficial. Trained to copy a larger model's reading of Wine Enthusiast notes. Non-commercial.",
                 flagging_mode="never").launch()

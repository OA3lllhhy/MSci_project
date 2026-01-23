#!/usr/bin/env python3
import os
import math
import time
import pickle
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Fixed config from user
# -----------------------------
DEFAULT_PKL = "/ceph/submit/data/user/h/haoyun22/diffusion_data/epm_bkg_events.pkl"
DEFAULT_KMAX = 96


# -----------------------------
# Utils: sinusoidal time embedding
# -----------------------------
def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device, dtype=torch.float32) / half
    )
    t = t.float().unsqueeze(1)  # (B,1)
    angles = t * freqs.unsqueeze(0)  # (B,half)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # (B,dim)


# -----------------------------
# Diffusion schedule
# -----------------------------
@dataclass
class DiffusionSchedule:
    '''
    Define diffusion schedule with linear beta from beta_start to beta_end.
    Precompute alphas, abar, sqrt(abar), sqrt(1-abar) for efficiency.
    -----------------------------
    Attributes:
        T: int
            Total diffusion steps
        beta_start: float
            Starting beta value
        beta_end: float
            Ending beta value
    -----------------------------
    '''
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self):
        betas = torch.linspace(self.beta_start, self.beta_end, self.T)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.abar = abar
        self.sqrt_abar = torch.sqrt(abar)
        self.sqrt_one_minus_abar = torch.sqrt(1.0 - abar)

    def to(self, device):
        for k, v in list(self.__dict__.items()):
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self


# -----------------------------
# Dataset: e± only, pad + mask
# PDG already filtered in your pkl, but we assert anyway
# -----------------------------
class EPairsDataset(torch.utils.data.Dataset):
    def __init__(self, events: List[Dict], Kmax: int = 96):
        self.Kmax = Kmax
        self.events = []

        for ev in events:
            p = np.asarray(ev["p"], dtype=np.float32)   # (N,3) already scaled
            pdg = np.asarray(ev["pdg"], dtype=np.int32) # (N,)
            if p.ndim != 2 or p.shape[1] != 3:
                continue
            if pdg.ndim != 1 or len(pdg) != len(p):
                continue
            # keep only e±
            sel = (pdg == 11) | (pdg == -11)
            if not np.any(sel):
                continue
            p = p[sel]
            pdg = pdg[sel]
            self.events.append({"p": p, "pdg": pdg})

        if len(self.events) == 0:
            raise ValueError("No valid events after filtering.")

        # Option A PDG sampler pool: real PDG sequences
        self.pdg_seqs = [torch.as_tensor(ev["pdg"], dtype=torch.int64) for ev in self.events]

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ev = self.events[idx]
        p = torch.as_tensor(ev["p"], dtype=torch.float32)      # (N,3)
        pdg = torch.as_tensor(ev["pdg"], dtype=torch.int64)    # (N,)

        N = p.shape[0]
        K = min(N, self.Kmax)

        p_pad = torch.zeros(self.Kmax, 3, dtype=torch.float32)
        pdg_pad = torch.zeros(self.Kmax, dtype=torch.int64)
        mask = torch.zeros(self.Kmax, dtype=torch.bool)

        p_pad[:K] = p[:K]
        pdg_pad[:K] = pdg[:K]
        mask[:K] = True

        # Map PDG -> small ids for embedding: e-(11)->0, e+(-11)->1
        pdg_id = torch.zeros_like(pdg_pad)
        pdg_id[pdg_pad == 11] = 0
        pdg_id[pdg_pad == -11] = 1

        return {"p0": p_pad, "pdg_id": pdg_id, "mask": mask}

    def sample_real_pdg_list(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = random.choice(self.pdg_seqs)  # (N,)
        N = min(len(seq), self.Kmax)
        pdg_id = torch.zeros(self.Kmax, dtype=torch.int64)
        mask = torch.zeros(self.Kmax, dtype=torch.bool)
        pdg_id[:N] = torch.where(seq[:N] == 11, torch.tensor(0), torch.tensor(1))
        mask[:N] = True
        return pdg_id.to(device), mask.to(device)


# -----------------------------
# Model: PDG-conditioned Transformer denoiser
# -----------------------------
class MomentumDenoiser(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, pdg_vocab=2):
        super().__init__()
        self.d_model = d_model

        self.p_in = nn.Linear(3, d_model) # Embed input momentum
        self.pdg_emb = nn.Embedding(pdg_vocab, d_model)

        # time embedding dim == d_model
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 3)

    def forward(self, p_t, pdg_id, mask, t):
        x = self.p_in(p_t) + self.pdg_emb(pdg_id)

        t_emb = sinusoidal_time_embedding(t, dim=self.d_model).to(p_t.device)  # (B,d_model)
        x = x + self.t_mlp(t_emb).unsqueeze(1) #!

        key_padding_mask = ~mask
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.out(h)


# -----------------------------
# Training step
# -----------------------------
def train_steps(model, loader, optim, sched: DiffusionSchedule, device, num_steps: int, log_every: int = 20):
    model.train()
    it = iter(loader)
    losses = []

    for step in range(1, num_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        p0 = batch["p0"].to(device)         # (B,K,3)
        pdg_id = batch["pdg_id"].to(device) # (B,K)
        mask = batch["mask"].to(device)     # (B,K)

        B = p0.shape[0]
        t = torch.randint(0, sched.T, (B,), device=device)

        abar_t = sched.sqrt_abar[t].view(B, 1, 1)
        omabar_t = sched.sqrt_one_minus_abar[t].view(B, 1, 1)

        eps = torch.randn_like(p0)
        p_t = abar_t * p0 + omabar_t * eps

        eps_pred = model(p_t, pdg_id, mask, t)

        m = mask.unsqueeze(-1).float()
        loss = ((eps_pred - eps) ** 2 * m).sum() / (m.sum() * 3.0 + 1e-8)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        losses.append(loss.item())
        if step % log_every == 0 or step == 1:
            print(f"step {step:05d}/{num_steps} | loss={np.mean(losses[-log_every:]):.6f}")

    return float(np.mean(losses))


@torch.no_grad()
def sample_event(model, dataset: EPairsDataset, sched: DiffusionSchedule, device, steps: int = None):
    model.eval()
    if steps is None:
        steps = sched.T

    pdg_id, mask = dataset.sample_real_pdg_list(device=device)  # (K,), (K,)
    pdg_id = pdg_id.unsqueeze(0)  # (1,K)
    mask = mask.unsqueeze(0)      # (1,K)

    K = pdg_id.shape[1]
    p = torch.randn(1, K, 3, device=device) # put 10000 sth to accelerate, get rid of loop 不使用loop一个个events生成，而是一次生成一个batch的events

    for ti in reversed(range(steps)):
        """
        Reverse processing step
        """
        t = torch.full((1,), ti, device=device, dtype=torch.int64)
        eps_pred = model(p, pdg_id, mask, t)

        beta = sched.betas[ti]
        alpha = sched.alphas[ti]
        abar = sched.abar[ti]

        coef1 = 1.0 / torch.sqrt(alpha)
        coef2 = (1.0 - alpha) / torch.sqrt(1.0 - abar)
        mean = coef1 * (p - coef2 * eps_pred)

        if ti > 0:
            p = mean + torch.sqrt(beta) * torch.randn_like(p)
        else:
            p = mean

        p = p * mask.unsqueeze(-1)

    real = mask[0]
    p_out = p[0, real].detach().cpu().numpy()  # (N,3) in scaled units
    pdg_out = pdg_id[0, real].detach().cpu().numpy()  # (N,)
    return pdg_out, p_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true",
                    help="Run a very small sanity-check config (few steps, small model).")
    # ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--kmax", type=int, default=DEFAULT_KMAX)
    ap.add_argument("--pkl", type=str, default=DEFAULT_PKL)
    ap.add_argument("--outdir", type=str, default="/work/submit/haoyun22/FCC-Beam-Background/diffusion_model/model_out")
    args = ap.parse_args()

    sys.modules['onnx'] = None
    sys.modules['onnxruntime'] = None

    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["TORCH_DISABLE_CPP_PROTOS"] = "1"
    os.environ["TORCH_USE_RTLD_GLOBAL"] = "0"

    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading: {args.pkl}")
    with open(args.pkl, "rb") as f:
        events = pickle.load(f)

    # dataset expects scaled momenta already (you did /1e-3 in preprocessing)
    dataset = EPairsDataset(events, Kmax=args.kmax)
    print(f"Dataset events: {len(dataset)} | Kmax={args.kmax}")

    # toy vs normal
    if args.toy:
        T = 200
        d_model = 96
        nhead = 6
        num_layers = 2
        batch_size = 4
        num_steps = 300
        lr = 3e-4
        sample_steps = 200
        log_every = 20
        print("[MODE] --toy enabled")
    else:
        T = 1000
        d_model = 128
        nhead = 8
        num_layers = 4
        batch_size = 8
        num_steps = 3000
        lr = 2e-4
        sample_steps = 1000
        log_every = 50

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    sched = DiffusionSchedule(T=T).to(device)
    model = MomentumDenoiser(d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"device={device} | T={T} | d_model={d_model} | layers={num_layers} | bs={batch_size} | steps={num_steps}")

    t0 = time.time()
    avg_loss = train_steps(model, loader, optim, sched, device, num_steps=num_steps, log_every=log_every)
    dt = time.time() - t0
    print(f"Training done. avg_loss={avg_loss:.6f} | wall={dt/60:.2f} min")

    ckpt_path = os.path.join(args.outdir, "ckpt.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "sched_T": T,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "kmax": args.kmax,
            "seed": seed,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")

    # sample a few events and print |p| stats (still in scaled units)
    for i in range(3):
        pdg_out, p_out = sample_event(model, dataset, sched, device, steps=sample_steps)
        pnorm = np.linalg.norm(p_out, axis=1)
        print(f"[sample {i}] N={len(p_out)} |p| median={np.median(pnorm):.3f} mean={pnorm.mean():.3f} max={pnorm.max():.3f}")

    print("Note: momenta are in *scaled units* (same as training input). Multiply by 1e-3 to recover GeV if you used /1e-3 scaling.")


if __name__ == "__main__":
    main()
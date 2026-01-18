from podio import root_io
import ROOT
import glob
import pickle
import argparse
import os

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import functions

import glob, pickle
from collections import defaultdict

import numpy as np
from podio import root_io

ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument('--run', action='store_true', help='Run the data extraction from ROOT files')
args = parser.parse_args()

# -----------------------------
# Utils: sinusoidal time embedding
# -----------------------------
def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (B,) int64 or float32 in [0, T-1]
    return: (B, dim)
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / half
    )  # (half,)
    t = t.float().unsqueeze(1)  # (B,1)
    angles = t * freqs.unsqueeze(0)  # (B,half)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (B,2*half)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # (B,dim)


# -----------------------------
# Diffusion schedule
# -----------------------------
@dataclass
class DiffusionSchedule:
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
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self


# -----------------------------
# Dataset: keep only e-/e+ and pad with mask
# -----------------------------
class EPairsDataset(torch.utils.data.Dataset):
    """
    events: list of dicts: {"p": (Ni,3) float, "pdg": (Ni,) int}
    We filter to pdg in {11, -11}. If an event has no e±, we skip it.
    """
    def __init__(self, events: List[Dict], Kmax: int = 64):
        self.Kmax = Kmax
        self.events = []
        for ev in events:
            p = torch.as_tensor(ev["p"], dtype=torch.float32)
            pdg = torch.as_tensor(ev["pdg"], dtype=torch.int64)
            sel = (pdg == 11) | (pdg == -11)
            if sel.any():
                self.events.append({"p": p[sel], "pdg": pdg[sel]})
        if len(self.events) == 0:
            raise ValueError("No events contain e± after filtering.")

        # Option A PDG sampler pool: store real PDG sequences (signs) and multiplicities
        self.pdg_seqs = [ev["pdg"].clone() for ev in self.events]
        self.mults = [len(seq) for seq in self.pdg_seqs]

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ev = self.events[idx]
        p = ev["p"]          # (N,3)
        pdg = ev["pdg"]      # (N,)
        N = p.shape[0]
        K = min(N, self.Kmax)

        p_pad = torch.zeros(self.Kmax, 3, dtype=torch.float32)
        pdg_pad = torch.zeros(self.Kmax, dtype=torch.int64)
        mask = torch.zeros(self.Kmax, dtype=torch.bool)

        p_pad[:K] = p[:K]
        pdg_pad[:K] = pdg[:K]
        mask[:K] = True

        # Map PDG to small IDs for embedding: e- (11)->0, e+(-11)->1
        pdg_id = torch.zeros_like(pdg_pad)
        pdg_id[pdg_pad == 11] = 0
        pdg_id[pdg_pad == -11] = 1

        return {"p0": p_pad, "pdg_id": pdg_id, "mask": mask}

    # -------- Option A PDG sampler --------
    def sample_real_pdg_list(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          pdg_id: (Kmax,) in {0,1} for real tokens, 0 for padded (doesn't matter)
          mask:   (Kmax,) bool
        """
        seq = random.choice(self.pdg_seqs)  # (N,)
        N = min(len(seq), self.Kmax)
        pdg_id = torch.zeros(self.Kmax, dtype=torch.int64)
        mask = torch.zeros(self.Kmax, dtype=torch.bool)
        # map signs
        pdg_id[:N] = torch.where(seq[:N] == 11, torch.tensor(0), torch.tensor(1))
        mask[:N] = True
        return pdg_id.to(device), mask.to(device)


# -----------------------------
# Model: PDG-conditioned Transformer denoiser
# Input: p_t (B,K,3) + pdg_id (B,K) + time t (B,)
# Output: eps_pred (B,K,3)
# -----------------------------
class MomentumDenoiser(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, time_dim=128, pdg_vocab=2):
        super().__init__()
        self.d_model = d_model

        self.p_in = nn.Linear(3, d_model)
        self.pdg_emb = nn.Embedding(pdg_vocab, d_model)
        self.t_mlp = nn.Sequential(
            nn.Linear(time_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 3)

    def forward(self, p_t: torch.Tensor, pdg_id: torch.Tensor, mask: torch.Tensor, t: torch.Tensor):
        """
        p_t: (B,K,3)
        pdg_id: (B,K)
        mask: (B,K) bool, True=real token
        t: (B,)
        """
        B, K, _ = p_t.shape
        x = self.p_in(p_t) + self.pdg_emb(pdg_id)

        t_emb = sinusoidal_time_embedding(t, dim=self.d_model).to(p_t.device)  # (B,d)
        t_cond = self.t_mlp(t_emb).unsqueeze(1)  # (B,1,d)
        x = x + t_cond  # broadcast to all tokens

        # Transformer expects key_padding_mask where True means "ignore"
        key_padding_mask = ~mask  # (B,K)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B,K,d)
        eps = self.out(h)  # (B,K,3)
        return eps


# -----------------------------
# Training: DDPM eps-pred objective with masked loss
# -----------------------------
def train_one_epoch(model, loader, optim, sched: DiffusionSchedule, device):
    model.train()
    total = 0.0
    count = 0

    for batch in loader:
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

        # masked MSE over real tokens only
        m = mask.unsqueeze(-1).float()  # (B,K,1)
        loss = ((eps_pred - eps) ** 2 * m).sum() / (m.sum() * 3.0 + 1e-8)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        total += loss.item()
        count += 1

    return total / max(count, 1)


# -----------------------------
# Sampling: Option A PDG list from real events, generate momenta
# -----------------------------
@torch.no_grad()
def sample_event(model, dataset: EPairsDataset, sched: DiffusionSchedule, device, steps: int = None):
    model.eval()
    if steps is None:
        steps = sched.T

    pdg_id, mask = dataset.sample_real_pdg_list(device=device)  # (K,), (K,)
    pdg_id = pdg_id.unsqueeze(0)  # (1,K)
    mask = mask.unsqueeze(0)      # (1,K)

    K = pdg_id.shape[1]
    p = torch.randn(1, K, 3, device=device)  # start from noise x_T

    for ti in reversed(range(steps)):
        t = torch.full((1,), ti, device=device, dtype=torch.int64)
        eps_pred = model(p, pdg_id, mask, t)

        beta = sched.betas[ti]
        alpha = sched.alphas[ti]
        abar = sched.abar[ti]

        # DDPM mean (predict eps)
        coef1 = 1.0 / torch.sqrt(alpha)
        coef2 = (1 - alpha) / torch.sqrt(1 - abar)
        mean = coef1 * (p - coef2 * eps_pred)

        if ti > 0:
            noise = torch.randn_like(p)
            sigma = torch.sqrt(beta)
            p = mean + sigma * noise
        else:
            p = mean

        # keep padded tokens at zero (optional hygiene)
        p = p * mask.unsqueeze(-1)

    # return only real tokens
    real = mask[0]
    return {
        "pdg_id": pdg_id[0, real].detach().cpu(),   # (N,)
        "p": p[0, real].detach().cpu(),            # (N,3)
    }


# -----------------------------
# Minimal runner (you plug in events)
# -----------------------------
def run_training(events: List[Dict], Kmax=64, epochs=10, batch_size=128, lr=2e-4, T=1000, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    dataset = EPairsDataset(events, Kmax=Kmax)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    sched = DiffusionSchedule(T=T).to(device)
    model = MomentumDenoiser(d_model=128, nhead=8, num_layers=4).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, optim, sched, device)
        print(f"Epoch {ep:03d} | loss={loss:.6f}")

    # sample a few
    for i in range(3):
        out = sample_event(model, dataset, sched, device)
        print(f"[sample {i}] N={len(out['pdg_id'])}, pdg_id={out['pdg_id'].tolist()}")
        print(out["p"][:5])

    return model, dataset, sched


# === Geometry config ===
PITCH = functions.PITCH_MM
RADIUS = functions.RADIUS_MM
LAYER_RADII = [14, 36, 58]
TARGET_LAYER = 0

ELECTRON_PDGS = {11, -11}

def get_momentum_xyz(mc):
    """
    Robustly get (px,py,pz) from EDM4hep MCParticle.
    Adjust here if your interface differs.
    """
    mom = mc.getMomentum()  # often returns a vector-like with x,y,z
    # common patterns:
    if hasattr(mom, "x"):
        return float(mom.x), float(mom.y), float(mom.z)
    if hasattr(mom, "X"):
        return float(mom.X()), float(mom.Y()), float(mom.Z())
    # fallback: some bindings expose px/py/pz directly
    if hasattr(mc, "getPx"):
        return float(mc.getPx()), float(mc.getPy()), float(mc.getPz())
    raise RuntimeError("Cannot extract momentum. Please check MCParticle API in your environment.")

def extract_epm_events(files, limit_files=None):
    """
    Returns: events list
      event = {"p": np.ndarray (N,3), "pdg": np.ndarray (N,)}
    N = number of unique e± tracks that have at least one selected hit in target layer
    """
    events_out = []

    for i, filename in enumerate(files):
        if limit_files is not None and i >= limit_files:
            break
        print(f"Processing file {i+1}/{limit_files or len(files)}: {filename}")

        reader = root_io.Reader(filename)
        events = reader.get('events')

        for event in events:
            # trackID -> (pdg, (px,py,pz))
            track_dict = {}

            for hit in event.get('VertexBarrelCollection'):
                # layer selection
                if functions.radius_idx(hit, LAYER_RADII) != TARGET_LAYER:
                    continue
                # primary only
                if hit.isProducedBySecondary():
                    continue

                mc = hit.getMCParticle()
                if mc is None:
                    continue

                pid = int(mc.getPDG())
                if pid not in ELECTRON_PDGS:
                    continue

                trackID = mc.getObjectID().index
                # only need to record each track once
                if trackID in track_dict:
                    continue

                try:
                    px, py, pz = get_momentum_xyz(mc)
                except Exception as e:
                    # if some weird particle object lacks momentum
                    print(f"Skipping track {trackID} due to momentum error: {e}")
                    continue

                track_dict[trackID] = (pid, (px, py, pz))

            if len(track_dict) == 0:
                continue

            pdg = np.array([v[0] for v in track_dict.values()], dtype=np.int32)
            p = np.array([v[1] for v in track_dict.values()], dtype=np.float32)  # (N,3)
            p = p / 1e-3  # convert to MeV

            events_out.append({"p": p, "pdg": pdg})

    return events_out


if args.run:
    all_configs = {
        'background': {
            'files': glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root'),
            'outfile': '/ceph/submit/data/user/h/haoyun22/diffusion_data/epm_bkg_events.pkl',
            'limit': 1247, #1247,
        }
    }


    for label, cfg in all_configs.items():
        out_path = cfg["outfile"]
        out_dir = os.path.dirname(out_path)

        os.makedirs(out_dir, exist_ok=True)

        events_out = extract_epm_events(cfg["files"], limit_files=cfg["limit"])
        with open(cfg["outfile"], "wb") as f:
            pickle.dump(events_out, f)
        print(f"✅ Saved {label} e± events: N_events={len(events_out)} to {cfg['outfile']}")
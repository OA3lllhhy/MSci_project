#!/usr/bin/env python3
"""
Shared functions and classes for diffusion model pipeline
"""
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import old_work.functions as functions
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import ROOT and podio for data extraction
try:
    from podio import root_io
    import ROOT
    ROOT.gROOT.SetBatch(True)
    PODIO_AVAILABLE = True
except ImportError:
    PODIO_AVAILABLE = False
    print("Warning: podio/ROOT not available. Data extraction will be disabled.")
    print("Run source /work/submit/jaeyserm/software/FCCAnalyses/setup.sh to enable podio.")

FUNCTIONS_AVAILABLE = True


# -----------------------------
# Utils: sinusoidal time embedding
# -----------------------------
def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal time embedding for diffusion timesteps.
    
    Args:
        t: (B,) int64 or float32 in [0, T-1]
        dim: embedding dimension
    
    Returns:
        (B, dim) time embedding tensor
    """
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
    """
    Define diffusion schedule with linear beta from beta_start to beta_end.
    Precompute alphas, abar, sqrt(abar), sqrt(1-abar) for efficiency.
    
    Attributes:
        T: Total diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
    """
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
        """Move all tensors to specified device"""
        for k, v in list(self.__dict__.items()):
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self


# -----------------------------
# Dataset: e± only, pad + mask
# -----------------------------
class EPairsDataset(torch.utils.data.Dataset):
    """
    Dataset for electron/positron pairs with padding and masking.
    
    Args:
        events: List of dicts with keys "p" (N,3) and "pdg" (N,)
        Kmax: Maximum sequence length (padding size)
    """
    def __init__(self, events: List[Dict], Kmax: int = 96):
        self.Kmax = Kmax
        self.events = []

        for ev in events:
            p = np.asarray(ev["p"], dtype=np.float32)   # (N,3)
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

        # Store PDG sequences for sampling
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

        # Map PDG -> small ids: e-(11)->0, e+(-11)->1
        pdg_id = torch.zeros_like(pdg_pad)
        pdg_id[pdg_pad == 11] = 0
        pdg_id[pdg_pad == -11] = 1

        return {"p0": p_pad, "pdg_id": pdg_id, "mask": mask}

    def sample_real_pdg_list(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a real PDG sequence from the dataset.
        
        Returns:
            pdg_id: (Kmax,) tensor with PDG IDs
            mask: (Kmax,) bool tensor
        """
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
    """
    Transformer-based denoiser for momentum diffusion.
    Conditioned on PDG type and timestep.
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        pdg_vocab: Size of PDG vocabulary (2 for e+/e-)
    """
    def __init__(self, d_model=128, nhead=8, num_layers=4, pdg_vocab=2):
        super().__init__()
        self.d_model = d_model

        self.p_in = nn.Linear(3, d_model)
        self.pdg_emb = nn.Embedding(pdg_vocab, d_model)

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
        """
        Forward pass.
        
        Args:
            p_t: (B, K, 3) noisy momentum
            pdg_id: (B, K) PDG type IDs
            mask: (B, K) bool mask (True=real token)
            t: (B,) timestep
        
        Returns:
            (B, K, 3) predicted noise
        """
        x = self.p_in(p_t) + self.pdg_emb(pdg_id)

        t_emb = sinusoidal_time_embedding(t, dim=self.d_model).to(p_t.device)
        x = x + self.t_mlp(t_emb).unsqueeze(1)

        key_padding_mask = ~mask
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.out(h)


# -----------------------------
# Sampling function
# -----------------------------
@torch.no_grad()
def sample_event(model, dataset: EPairsDataset, sched: DiffusionSchedule, 
                 device, steps: int = None):
    """
    Sample one event using DDPM reverse process.
    
    Args:
        model: Trained MomentumDenoiser
        dataset: EPairsDataset for PDG sampling
        sched: DiffusionSchedule
        device: torch device
        steps: Number of sampling steps (defaults to sched.T)
    
    Returns:
        pdg_out: (N,) numpy array of PDG IDs
        p_out: (N, 3) numpy array of momenta
    """
    model.eval()
    if steps is None:
        steps = sched.T

    pdg_id, mask = dataset.sample_real_pdg_list(device=device)  # (K,), (K,)
    pdg_id = pdg_id.unsqueeze(0)  # (1,K)
    mask = mask.unsqueeze(0)      # (1,K)

    K = pdg_id.shape[1]
    p = torch.randn(1, K, 3, device=device)

    for ti in reversed(range(steps)):
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
    p_out = p[0, real].detach().cpu().numpy()
    pdg_out = pdg_id[0, real].detach().cpu().numpy()
    return pdg_out, p_out


#-----------------------------
# Data extraction from ROOT files
#-----------------------------

def get_momentum_xyz(mc):
    """Robustly get (px, py, pz) from EDM4hep MCParticle."""
    mom = mc.getMomentum()
    if hasattr(mom, "x"):
        return float(mom.x), float(mom.y), float(mom.z)
    if hasattr(mom, "X"):
        return float(mom.X()), float(mom.Y()), float(mom.Z())
    if hasattr(mc, "getPx"):
        return float(mc.getPx()), float(mc.getPy()), float(mc.getPz())
    raise RuntimeError("Cannot extract momentum. Check MCParticle API.")


def extract_epm_events(files, limit_files=None, target_layer=0):
    """
    Extract e+/e- events from ROOT files.
    
    Args:
        files: List of ROOT file paths
        limit_files: Maximum number of files to process
        target_layer: Target detector layer index
    
    Returns:
        List of event dicts with keys "p" (N,3) and "pdg" (N,)
    """
    if not PODIO_AVAILABLE:
        raise RuntimeError("podio/ROOT not available. Cannot extract data.")
    
    
    LAYER_RADII = [14, 36, 58]
    ELECTRON_PDGS = {11, -11}
    events_out = []

    for i, filename in enumerate(files):
        if limit_files is not None and i >= limit_files:
            break
        print(f"Processing file {i+1}/{limit_files or len(files)}: {filename}")

        reader = root_io.Reader(filename)
        events = reader.get('events')

        for event in events:
            track_dict = {}

            for hit in event.get('VertexBarrelCollection'):
                # Layer selection
                if FUNCTIONS_AVAILABLE:
                    if functions.radius_idx(hit, LAYER_RADII) != target_layer:
                        continue
                else:
                    # Simple radius check if functions not available
                    pass
                
                # Primary only
                if hit.isProducedBySecondary():
                    continue

                mc = hit.getMCParticle()
                if mc is None:
                    continue

                pid = int(mc.getPDG())
                if pid not in ELECTRON_PDGS:
                    continue

                trackID = mc.getObjectID().index
                if trackID in track_dict:
                    continue

                try:
                    px, py, pz = get_momentum_xyz(mc)
                except Exception as e:
                    print(f"Skipping track {trackID}: {e}")
                    continue

                track_dict[trackID] = (pid, (px, py, pz))

            if len(track_dict) == 0:
                continue

            pdg = np.array([v[0] for v in track_dict.values()], dtype=np.int32)
            p = np.array([v[1] for v in track_dict.values()], dtype=np.float32)
            p = p / 1e-3  # Convert to MeV

            events_out.append({"p": p, "pdg": pdg})

    return events_out



def train_steps(model, loader, optim, sched, device, num_steps, log_every=50):
    """Training loop for specified number of steps."""
    model.train()
    it = iter(loader)
    losses = []

    for step in range(1, num_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        p0 = batch["p0"].to(device)
        pdg_id = batch["pdg_id"].to(device)
        mask = batch["mask"].to(device)

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
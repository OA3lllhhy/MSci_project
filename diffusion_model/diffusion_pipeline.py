#!/usr/bin/env python3
"""
Unified pipeline for diffusion model: data extraction, training, and generation.

Usage:
    # Extract data from ROOT files
    python diffusion_pipeline.py extract --outfile /path/to/output.pkl

    # Train model
    python diffusion_pipeline.py train --pkl /path/to/data.pkl --outdir /path/to/model_out

    # Generate and compare
    python diffusion_pipeline.py generate --ckpt /path/to/ckpt.pt --pkl /path/to/data.pkl --outdir /path/to/plots
"""

import os
import sys
import time
import glob
import pickle
import random
import argparse
import old_work.functions as functions
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Import shared functions
from diffusion_functions import (
    DiffusionSchedule,
    EPairsDataset,
    MomentumDenoiser,
    sample_event,
)

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


# ============================================================================
# PART 1: DATA EXTRACTION
# ============================================================================

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


def run_extraction(args):
    """Run data extraction phase."""
    print("\n" + "="*60)
    print("PHASE 1: DATA EXTRACTION")
    print("="*60)
    
    if args.files is None:
        # Default file pattern
        pattern = '/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root'
        files = glob.glob(pattern)
        print(f"Using default pattern: {pattern}")
    else:
        files = glob.glob(args.files)
    
    if len(files) == 0:
        raise ValueError(f"No files found matching pattern")
    
    print(f"Found {len(files)} files")
    
    events = extract_epm_events(files, limit_files=args.limit_files, target_layer=args.target_layer)
    
    try:
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    except PermissionError:
        print(f"⚠️  Warning: Cannot create directory for {args.outfile}, trying current directory")
        args.outfile = "./epm_events_temp.pkl"
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    
    with open(args.outfile, "wb") as f:
        pickle.dump(events, f)
    
    print(f"\n✅ Saved {len(events)} e± events to: {args.outfile}")
    
    # Print statistics
    n_particles = sum(len(ev["pdg"]) for ev in events)
    mults = [len(ev["pdg"]) for ev in events]
    print(f"Total particles: {n_particles}")
    print(f"Multiplicity: min={min(mults)}, max={max(mults)}, mean={np.mean(mults):.2f}")


# ============================================================================
# PART 2: TRAINING
# ============================================================================

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


def run_training(args):
    """Run training phase."""
    print("\n" + "="*60)
    print("PHASE 2: TRAINING")
    print("="*60)
    
    # Disable problematic modules
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
        num_steps = 1500
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




# ============================================================================
# PART 3: GENERATION AND COMPARISON
# ============================================================================

def run_generation(args):
    """Run generation and comparison phase."""
    print("\n" + "="*60)
    print("PHASE 3: GENERATION AND COMPARISON")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")
    
    # Load real data
    print(f"Loading real data from: {args.pkl}")
    with open(args.pkl, "rb") as f:
        real_events = pickle.load(f)
    print(f"Real events: {len(real_events)}")
    
    # Real statistics
    real_N = np.array([len(np.asarray(ev["pdg"])) for ev in real_events], dtype=np.int32)
    real_p = np.concatenate([np.asarray(ev["p"], dtype=np.float32) for ev in real_events], axis=0)
    real_pnorm = np.linalg.norm(real_p, axis=1)
    
    print(f"Real multiplicity: min={real_N.min()} max={real_N.max()} mean={real_N.mean():.2f}")
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    T = int(ckpt["sched_T"])
    kmax = int(ckpt["kmax"])
    steps = T if args.use_steps < 0 else int(args.use_steps)
    
    model = MomentumDenoiser(
        d_model=int(ckpt["d_model"]),
        nhead=int(ckpt["nhead"]),
        num_layers=int(ckpt["num_layers"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    sched = DiffusionSchedule(T=T).to(device)
    dataset = EPairsDataset(real_events, Kmax=kmax)
    
    # Generate events
    print(f"\nGenerating {args.n_gen} events (steps={steps})...")
    gen_N = []
    gen_pnorm_list = []
    
    for i in range(args.n_gen):
        pdg_id, p_out = sample_event(model, dataset, sched, device, steps)
        gen_N.append(len(p_out))
        gen_pnorm_list.append(np.linalg.norm(p_out, axis=1))
        
        if (i + 1) % max(1, args.n_gen // 20) == 0:
            print(f"  Progress: {i+1}/{args.n_gen}")
    
    gen_N = np.array(gen_N, dtype=np.int32)
    gen_pnorm = np.concatenate(gen_pnorm_list, axis=0)
    
    print(f"\nGenerated multiplicity: min={gen_N.min()} max={gen_N.max()} mean={gen_N.mean():.2f}")
    print(f"\nTop 10 multiplicities (Real): {Counter(real_N.tolist()).most_common(10)}")
    print(f"Top 10 multiplicities (Gen):  {Counter(gen_N.tolist()).most_common(10)}")
    
    # Unit conversion
    if args.to_gev:
        real_pnorm_plot = real_pnorm * args.scale_back
        gen_pnorm_plot = gen_pnorm * args.scale_back
        p_unit = "GeV"
    else:
        real_pnorm_plot = real_pnorm
        gen_pnorm_plot = gen_pnorm
        p_unit = "scaled"
    
    if args.p_clip > 0:
        real_pnorm_plot = np.clip(real_pnorm_plot, 0, args.p_clip)
        gen_pnorm_plot = np.clip(gen_pnorm_plot, 0, args.p_clip)
    
    # Create plots
    try:
        os.makedirs(args.outdir, exist_ok=True)
    except PermissionError:
        print(f"⚠️  Warning: Cannot create {args.outdir}, trying current directory")
        args.outdir = "./checks_temp"
        os.makedirs(args.outdir, exist_ok=True)
    
    # 1. Multiplicity histogram
    nmin = int(min(real_N.min(), gen_N.min()))
    nmax = int(max(real_N.max(), gen_N.max()))
    binsN = np.linspace(nmin, nmax + 1, args.bins + 1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(real_N, bins=binsN, alpha=0.6, density=True, label="Real")
    plt.hist(gen_N, bins=binsN, alpha=0.6, density=True, label="Generated")
    plt.xlabel("Multiplicity N(e±)")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Multiplicity Distribution | real={len(real_N)}, gen={len(gen_N)}")
    plt.savefig(os.path.join(args.outdir, "multiplicity_hist.png"), dpi=200, bbox_inches="tight")
    print(f"\n✅ Saved: multiplicity_hist.png")
    
    # 2. |p| linear histogram
    pmin = float(min(real_pnorm_plot.min(), gen_pnorm_plot.min()))
    pmax = float(max(real_pnorm_plot.max(), gen_pnorm_plot.max()))
    binsP = np.linspace(pmin, pmax, args.momentum_bins + 1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(real_pnorm_plot, bins=binsP, alpha=0.6, density=True, label="Real")
    plt.hist(gen_pnorm_plot, bins=binsP, alpha=0.6, density=True, label="Generated")
    plt.xlabel(f"|p| ({p_unit})")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"|p| Distribution (Linear)")
    plt.savefig(os.path.join(args.outdir, "p_hist_linear.png"), dpi=200, bbox_inches="tight")
    print(f"✅ Saved: p_hist_linear.png")
    
    # 3. log10(|p|) histogram
    eps = 1e-12
    real_logp = np.log10(np.maximum(real_pnorm_plot, eps))
    gen_logp = np.log10(np.maximum(gen_pnorm_plot, eps))
    
    lpmin = float(min(real_logp.min(), gen_logp.min()))
    lpmax = float(max(real_logp.max(), gen_logp.max()))
    binsLP = np.linspace(lpmin, lpmax, args.logp_bins + 1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(real_logp, bins=binsLP, alpha=0.6, density=True, label="Real")
    plt.hist(gen_logp, bins=binsLP, alpha=0.6, density=True, label="Generated")
    plt.xlabel(f"log10(|p|) ({p_unit})")
    plt.ylabel("Density")
    plt.legend()
    plt.title("log10(|p|) Distribution")
    plt.savefig(os.path.join(args.outdir, "p_hist_log10.png"), dpi=200, bbox_inches="tight")
    print(f"✅ Saved: p_hist_log10.png")
    
    # Save arrays
    np.save(os.path.join(args.outdir, "real_N.npy"), real_N)
    np.save(os.path.join(args.outdir, "gen_N.npy"), gen_N)
    np.save(os.path.join(args.outdir, "real_pnorm.npy"), real_pnorm_plot)
    np.save(os.path.join(args.outdir, "gen_pnorm.npy"), gen_pnorm_plot)
    print(f"✅ Saved arrays to: {args.outdir}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified diffusion model pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract data
  python diffusion_pipeline.py extract --outfile /path/to/data.pkl --limit_files 100

  # Train model
  python diffusion_pipeline.py train --pkl /path/to/data.pkl --num_steps 3000

  # Generate and compare
  python diffusion_pipeline.py generate --ckpt model_out/ckpt.pt --pkl /path/to/data.pkl --n_gen 1000
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")
    
    # ==== EXTRACT subcommand ====
    extract_parser = subparsers.add_parser("extract", help="Extract data from ROOT files")
    extract_parser.add_argument("--files", type=str, default=None,
                                help="Glob pattern for ROOT files")
    extract_parser.add_argument("--outfile", type=str, 
                                default="/ceph/submit/data/user/h/haoyun22/diffusion_data/epm_bkg_events.pkl",
                                help="Output pickle file path")
    extract_parser.add_argument("--limit_files", type=int, default=None,
                                help="Limit number of files to process")
    extract_parser.add_argument("--target_layer", type=int, default=0,
                                help="Target detector layer")
    
    # ==== TRAIN subcommand ====
    train_parser = subparsers.add_parser("train", help="Train diffusion model")
    train_parser.add_argument("--pkl", type=str, 
                              default="/ceph/submit/data/user/h/haoyun22/diffusion_data/epm_bkg_events.pkl",
                              help="Input pickle file with training data")
    train_parser.add_argument("--outdir", type=str, 
                              default="/work/submit/haoyun22/FCC-Beam-Background/diffusion_model/model_out_kmax128",
                              help="Output directory for checkpoints")
    train_parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    train_parser.add_argument("--kmax", type=int, default=96, help="Maximum sequence length")
    train_parser.add_argument("--toy", action="store_true", help="Use toy config for quick test")
    train_parser.add_argument("--T", type=int, default=1000, help="Diffusion timesteps")
    train_parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    train_parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    train_parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    train_parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--num_steps", type=int, default=3000, help="Training steps")
    train_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    train_parser.add_argument("--log_every", type=int, default=50, help="Logging frequency")
    
    # ==== GENERATE subcommand ====
    gen_parser = subparsers.add_parser("generate", help="Generate samples and compare with real data")
    gen_parser.add_argument("--ckpt", type=str, 
                            default="/work/submit/haoyun22/FCC-Beam-Background/diffusion_model/model_out_kmax128/ckpt.pt",
                            help="Path to checkpoint")
    gen_parser.add_argument("--pkl", type=str, 
                            default="/ceph/submit/data/user/h/haoyun22/diffusion_data/epm_bkg_events.pkl",
                            help="Path to real data pickle")
    gen_parser.add_argument("--outdir", type=str, 
                            default="/work/submit/haoyun22/FCC-Beam-Background/diffusion_model/checks_model_kmax128",
                            help="Output directory for plots")
    gen_parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    gen_parser.add_argument("--n_gen", type=int, default=500, help="Number of events to generate")
    gen_parser.add_argument("--use_steps", type=int, default=-1, help="Sampling steps (-1 uses checkpoint T)")
    gen_parser.add_argument("--bins", type=int, default=40, help="Histogram bins")
    gen_parser.add_argument("--momentum_bins", type=int, default=50, help="Bins for |p| histogram")
    gen_parser.add_argument("--logp_bins", type=int, default=50, help="Bins for log10(|p|) histogram")
    gen_parser.add_argument("--p_clip", type=float, default=-1.0, help="Clip |p| for plotting")
    gen_parser.add_argument("--to_gev", action="store_true", help="Convert momentum to GeV for plots")
    gen_parser.add_argument("--scale_back", type=float, default=1e-3, help="Scale factor for GeV conversion")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate function
    if args.command == "extract":
        run_extraction(args)
    elif args.command == "train":
        run_training(args)
    elif args.command == "generate":
        run_generation(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

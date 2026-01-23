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
# import functions
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
    get_momentum_xyz,
    extract_epm_events,
    train_steps,
    validate,
    ExtractConfig,
    TrainConfig,
    GenerateConfig,
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
def run_extraction(config: ExtractConfig):
    """Run data extraction phase."""
    print("\n" + "="*60)
    print("PHASE 1: DATA EXTRACTION")
    print("="*60)
    
    if config.files is None:
        # Default file pattern
        pattern = '/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root'
        files = glob.glob(pattern)
        print(f"Using default pattern: {pattern}")
    else:
        files = glob.glob(config.files)
    
    if len(files) == 0:
        raise ValueError(f"No files found matching pattern")
    
    print(f"Found {len(files)} files")
    
    events = extract_epm_events(files, limit_files=config.limit_files, target_layer=config.target_layer)
    
    try:
        os.makedirs(os.path.dirname(config.outfile), exist_ok=True)
    except PermissionError:
        print(f"⚠️  Warning: Cannot create directory for {config.outfile}, trying current directory")
        config.outfile = "./epm_events_temp.pkl"
        os.makedirs(os.path.dirname(config.outfile), exist_ok=True)
    
    with open(config.outfile, "wb") as f:
        pickle.dump(events, f)
    
    print(f"\n✅ Saved {len(events)} e± events to: {config.outfile}")
    
    # Print statistics
    n_particles = sum(len(ev["pdg"]) for ev in events)
    mults = [len(ev["pdg"]) for ev in events]
    print(f"Total particles: {n_particles}")
    print(f"Multiplicity: min={min(mults)}, max={max(mults)}, mean={np.mean(mults):.2f}")


# ============================================================================
# PART 2: TRAINING
# ============================================================================
def run_training(config: TrainConfig):
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

    device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    os.makedirs(config.outdir, exist_ok=True)

    print(f"Loading: {config.pkl}")
    with open(config.pkl, "rb") as f:
        events = pickle.load(f)

    # dataset expects scaled momenta already (you did /1e-3 in preprocessing)
    dataset = EPairsDataset(events, Kmax=config.kmax)
    print(f"Dataset events: {len(dataset)} | Kmax={config.kmax}")
    from torch.utils.data import random_split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    print(f"Train events: {len(train_dataset)} | Val events: {len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    
    sched = DiffusionSchedule(T=config.T).to(device)
    model = MomentumDenoiser(d_model=config.d_model, nhead=config.nhead, num_layers=config.num_layers).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    print(f"device={device} | T={config.T} | d_model={config.d_model} | layers={config.num_layers} | bs={config.batch_size} | steps={config.num_steps}")

    t0 = time.time()
    history = train_steps(model, train_loader, optim, sched, device, num_steps=config.num_steps, log_every=config.log_every, val_loader=val_loader)
    dt = time.time() - t0
    print(f"Training done. avg_train_loss={history['avg_train_loss']:.6f} avg_val_loss={history['avg_val_loss']:.6f} | wall={dt/60:.2f} min")

    ckpt_path = os.path.join(config.outdir, "ckpt.pt")
    # torch.save(
    #     {
    #         "model": model.state_dict(),
    #         "sched_T": config.T,
    #         "d_model": config.d_model,
    #         "nhead": config.nhead,
    #         "num_layers": config.num_layers,
    #         "kmax": config.kmax,
    #         "seed": seed,
    #         "train_losses": history['train_losses'],
    #         "val_losses": history['val_losses'],
    #         "val_steps": history['val_steps'],
    #     },
    #     ckpt_path,
    # )
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,  # 保存整个配置
            "history": history,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")

    loss_path = os.path.join(config.outdir, "training_loss.npz")
    np.savez(loss_path, 
             train_losses=np.array(history['train_losses']),
             val_losses=np.array(history['val_losses']),
             val_steps=np.array(history['val_steps']))
    print(f"Saved loss history: {loss_path}")

    # sample a few events and print |p| stats (still in scaled units)
    for i in range(3):
        pdg_out, p_out = sample_event(model, dataset, sched, device, steps=config.sample_steps)
        pnorm = np.linalg.norm(p_out, axis=1)
        print(f"[sample {i}] N={len(p_out)} |p| median={np.median(pnorm):.3f} mean={pnorm.mean():.3f} max={pnorm.max():.3f}")

    print("Note: momenta are in *scaled units* (same as training input). Multiply by 1e-3 to recover GeV if you used /1e-3 scaling.")




# ============================================================================
# PART 3: GENERATION AND COMPARISON
# ============================================================================

def run_generation(config: GenerateConfig):
    """Run generation and comparison phase."""
    print("\n" + "="*60)
    print("PHASE 3: GENERATION AND COMPARISON")
    print("="*60)
    
    device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")
    
    # Load real data
    print(f"Loading real data from: {config.pkl}")
    with open(config.pkl, "rb") as f:
        real_events = pickle.load(f)
    print(f"Real events: {len(real_events)}")
    
    # Real statistics
    real_N = np.array([len(np.asarray(ev["pdg"])) for ev in real_events], dtype=np.int32)
    real_p = np.concatenate([np.asarray(ev["p"], dtype=np.float32) for ev in real_events], axis=0)
    real_pnorm = np.linalg.norm(real_p, axis=1)
    
    print(f"Real multiplicity: min={real_N.min()} max={real_N.max()} mean={real_N.mean():.2f}")
    
    # Load checkpoint
    print(f"Loading checkpoint from: {config.ckpt}")
    ckpt = torch.load(config.ckpt, map_location=device)
    T = int(ckpt["sched_T"])
    kmax = int(ckpt["kmax"])
    steps = T if config.use_steps < 0 else int(config.use_steps)
    
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
    print(f"\nGenerating {config.n_gen} events (steps={steps})...")
    gen_N = []
    gen_p_list = []
    gen_pnorm_list = []
    
    start_time = time.time()
    for i in range(config.n_gen):
        pdg_id, p_out = sample_event(model, dataset, sched, device, steps)
        gen_N.append(len(p_out)) # Number of particles in this event
        gen_p_list.append(p_out) # p_x, p_y, p_z for this event
        gen_pnorm_list.append(np.linalg.norm(p_out, axis=1)) # |p| for this event
        
        if (i + 1) % max(1, config.n_gen // 20) == 0:
            print(f"  Progress: {i+1}/{config.n_gen}")

    end_time = time.time()
    elapsed = end_time - start_time
    mean_time = elapsed / config.n_gen
    print(f"\nGeneration completed in {elapsed/60:.2f} min ({elapsed/config.n_gen:.2f} s/event)")
    print(f"Average time per event: {mean_time:.2f} s")
    
    gen_N = np.array(gen_N, dtype=np.int32)
    gen_pnorm = np.concatenate(gen_pnorm_list, axis=0)
    gen_p = np.concatenate(gen_p_list, axis=0)  # (total_particles, 3)
    
    print(f"\nGenerated multiplicity: min={gen_N.min()} max={gen_N.max()} mean={gen_N.mean():.2f}")
    print(f"\nTop 10 multiplicities (Real): {Counter(real_N.tolist()).most_common(10)}")
    print(f"Top 10 multiplicities (Gen):  {Counter(gen_N.tolist()).most_common(10)}")
    
    # Unit conversion
    if config.to_gev:
        real_pnorm_plot = real_pnorm * config.scale_back
        gen_pnorm_plot = gen_pnorm * config.scale_back
        p_unit = "GeV"
    else:
        real_pnorm_plot = real_pnorm
        gen_pnorm_plot = gen_pnorm
        p_unit = "scaled"
    
    if config.p_clip > 0:
        real_pnorm_plot = np.clip(real_pnorm_plot, 0, config.p_clip)
        gen_pnorm_plot = np.clip(gen_pnorm_plot, 0, config.p_clip)
    
    # Create plots
    try:
        os.makedirs(config.outdir, exist_ok=True)
    except PermissionError:
        print(f"⚠️  Warning: Cannot create {config.outdir}, trying current directory")
        config.outdir = "./checks_temp"
        os.makedirs(config.outdir, exist_ok=True)
    
    # 1. Multiplicity histogram
    # nmin = int(min(real_N.min(), gen_N.min()))
    # nmax = int(max(real_N.max(), gen_N.max()))
    # binsN = np.linspace(nmin, nmax + 1, config.bins + 1)
    
    # plt.figure(figsize=(10, 6))
    # plt.hist(real_N, bins=binsN, alpha=0.6, density=True, label="Real")
    # plt.hist(gen_N, bins=binsN, alpha=0.6, density=True, label="Generated")
    # plt.xlabel("Multiplicity N(e±)")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.title(f"Multiplicity Distribution | real={len(real_N)}, gen={len(gen_N)}")
    # plt.savefig(os.path.join(config.outdir, "multiplicity_hist.png"), dpi=200, bbox_inches="tight")
    # print(f"\n✅ Saved: multiplicity_hist.png")
    
    # Save arrays
    # np.save(os.path.join(config.outdir, "real_N.npy"), real_N)
    # np.save(os.path.join(config.outdir, "gen_N.npy"), gen_N)
    # np.save(os.path.join(config.outdir, "real_pnorm.npy"), real_pnorm_plot)
    # np.save(os.path.join(config.outdir, "gen_pnorm.npy"), gen_pnorm_plot)
    np.save(os.path.join(config.outdir, f"real_3d_p_{steps}s.npy"), real_p)
    np.save(os.path.join(config.outdir, f"gen_3d_p_{steps}s.npy"), gen_p)
    print(f"✅ Saved arrays to: {config.outdir}/")


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
  python diffusion_model/diffusion_pipeline.py extract --outfile /path/to/data.pkl --limit_files 100

  # Train model
  python diffusion_model/diffusion_pipeline.py train --pkl /path/to/data.pkl --num_steps 3000

  # Generate and compare
  python diffusion_model/diffusion_pipeline.py generate --ckpt model_out/ckpt.pt --pkl /path/to/data.pkl --n_gen 1000
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")
    
    # Extract
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--files", type=str, default=None)
    extract_parser.add_argument("--outfile", type=str, default=ExtractConfig.outfile)
    extract_parser.add_argument("--limit_files", type=int, default=None)
    extract_parser.add_argument("--target_layer", type=int, default=0)
    
    # Train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--pkl", type=str, default=TrainConfig.pkl)
    train_parser.add_argument("--outdir", type=str, default=TrainConfig.outdir)
    train_parser.add_argument("--kmax", type=int, default=TrainConfig.kmax)
    train_parser.add_argument("--num_steps", type=int, default=TrainConfig.num_steps)
    train_parser.add_argument("--sample_steps", type=int, default=TrainConfig.sample_steps)
    train_parser.add_argument("--toy", action="store_true")
    train_parser.add_argument("--device", type=str, default=TrainConfig.device)
    
    # Generate
    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--ckpt", type=str, default=GenerateConfig.ckpt)
    gen_parser.add_argument("--pkl", type=str, default=GenerateConfig.pkl)
    gen_parser.add_argument("--outdir", type=str, default=GenerateConfig.outdir)
    gen_parser.add_argument("--n_gen", type=int, default=GenerateConfig.n_gen)
    gen_parser.add_argument("--use_steps", type=int, default=-1)
    gen_parser.add_argument("--device", type=str, default=GenerateConfig.device)
    
    args = parser.parse_args()
    
    if args.command == "extract":
        config = ExtractConfig(**vars(args))
        run_extraction(config)
    
    elif args.command == "train":
        if args.toy:
            config = TrainConfig.toy_config()
            # 覆盖部分参数
            config.pkl = args.pkl
            config.outdir = args.outdir
        else:
            config = TrainConfig(**{k: v for k, v in vars(args).items() if k != 'command'})
        run_training(config)
    
    elif args.command == "generate":
        config = GenerateConfig(**{k: v for k, v in vars(args).items() if k != 'command'})
        run_generation(config)


if __name__ == "__main__":
    main()

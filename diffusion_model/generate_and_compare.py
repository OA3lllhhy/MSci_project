import os
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter

from dm_train import MomentumDenoiser, DiffusionSchedule, EPairsDataset, sample_event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="/work/submit/haoyun22/FCC-Beam-Background/diffusion_model/model_out/ckpt.pt", help="Path to ckpt.pt")
    ap.add_argument("--pkl", type=str, default="/ceph/submit/data/user/h/haoyun22/diffusion_data/epm_bkg_events.pkl",help="Path to real events pkl")
    ap.add_argument("--n_gen", type=int, default=1000, help="Number of generated events to sample")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--outdir", type=str, default="/work/submit/haoyun22/FCC-Beam-Background/diffusion_model/checks_model")
    ap.add_argument("--bins", type=int, default=40, help="Histogram bins (auto range)")
    ap.add_argument("--use_steps", type=int, default=-1, help="Override sampling steps; -1 uses ckpt T")
    ap.add_argument("--momentum_bins", type=int, default=50, help="Bins for |p| histogram")
    ap.add_argument("--logp_bins", type=int, default=50, help="Bins for log10(|p|) histogram")
    ap.add_argument("--p_clip", type=float, default=-1.0, help="If >0, clip |p| at this value for plotting")
    ap.add_argument("--to_gev", action="store_true",
                    help="Convert scaled momentum back to GeV for plots (assumes scaling factor=1e-3).")
    ap.add_argument("--scale_back", type=float, default=1e-3,
                    help="Scale factor used to recover GeV if --to_gev is set (default 1e-3).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"device={device}")

    # ----- load real data -----
    with open(args.pkl, "rb") as f:
        real_events = pickle.load(f)

    print(f"Real events loaded: {len(real_events)}")

    # real multiplicity
    real_N = np.array([len(np.asarray(ev["pdg"])) for ev in real_events], dtype=np.int32)
    print(f"Real N stats: min={real_N.min()} max={real_N.max()} mean={real_N.mean():.3f} median={np.median(real_N):.1f}")

    # real momentum |p| pool (particle-level)
    real_p = np.concatenate([np.asarray(ev["p"], dtype=np.float32) for ev in real_events], axis=0)
    real_pnorm = np.linalg.norm(real_p, axis=1)

    # ----- load checkpoint -----
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

    # Dataset only used for Option-A PDG/multiplicity sampling
    dataset = EPairsDataset(real_events, Kmax=kmax)

    # ----- generate -----
    gen_N = []
    gen_pnorm_list = []

    for i in range(args.n_gen):
        pdg_id, p_out = sample_event(model, dataset, sched, device=device, steps=steps)
        gen_N.append(len(p_out))
        gen_pnorm_list.append(np.linalg.norm(p_out, axis=1))

        if (i + 1) % max(1, args.n_gen // 20) == 0:
            print(f"Generated {i+1}/{args.n_gen} events...")

    gen_N = np.array(gen_N, dtype=np.int32)
    gen_pnorm = np.concatenate(gen_pnorm_list, axis=0).astype(np.float32)

    print(f"\nGenerated events: {len(gen_N)}")
    print(f"Gen N stats:  min={gen_N.min()} max={gen_N.max()} mean={gen_N.mean():.3f} median={np.median(gen_N):.1f}")

    print("\nTop multiplicities (REAL):", Counter(real_N.tolist()).most_common(10))
    print("Top multiplicities (GEN): ", Counter(gen_N.tolist()).most_common(10))

    # Optional unit conversion for plotting
    if args.to_gev:
        real_pnorm_plot = real_pnorm * args.scale_back
        gen_pnorm_plot = gen_pnorm * args.scale_back
        p_unit = "GeV"
    else:
        real_pnorm_plot = real_pnorm
        gen_pnorm_plot = gen_pnorm
        p_unit = "scaled"

    # Optional clip for nicer plots (keeps tail readable)
    if args.p_clip > 0:
        real_pnorm_plot = np.clip(real_pnorm_plot, 0, args.p_clip)
        gen_pnorm_plot = np.clip(gen_pnorm_plot, 0, args.p_clip)

    # ----- histogram: multiplicity -----
    nmin = int(min(real_N.min(), gen_N.min()))
    nmax = int(max(real_N.max(), gen_N.max()))
    binsN = np.linspace(nmin, nmax + 1, args.bins + 1)

    plt.figure()
    plt.hist(real_N, bins=binsN, alpha=0.6, density=True, label="Real")
    plt.hist(gen_N, bins=binsN, alpha=0.6, density=True, label="Generated")
    plt.xlabel("Multiplicity N(e±) per event")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Multiplicity | real={len(real_N)}, gen={len(gen_N)} | Kmax={kmax} | steps={steps}")
    out_pngN = os.path.join(args.outdir, "multiplicity_hist.png")
    plt.savefig(out_pngN, dpi=200, bbox_inches="tight")
    print(f"\nSaved multiplicity histogram: {out_pngN}")

    # ----- histogram: |p| (linear) -----
    pmin = float(min(real_pnorm_plot.min(), gen_pnorm_plot.min()))
    pmax = float(max(real_pnorm_plot.max(), gen_pnorm_plot.max()))
    # Avoid degenerate pmin=0 for log plots later
    binsP = np.linspace(pmin, pmax, args.momentum_bins + 1)

    plt.figure()
    plt.hist(real_pnorm_plot, bins=binsP, alpha=0.6, density=True, label="Real")
    plt.hist(gen_pnorm_plot, bins=binsP, alpha=0.6, density=True, label="Generated")
    plt.xlabel(f"|p| per particle ({p_unit})")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"|p| distribution (linear) | pooled particles | real={len(real_pnorm_plot)}, gen={len(gen_pnorm_plot)}")
    out_pngP = os.path.join(args.outdir, "p_hist_linear.png")
    plt.savefig(out_pngP, dpi=200, bbox_inches="tight")
    print(f"Saved |p| linear histogram: {out_pngP}")

    # ----- histogram: log10(|p|) -----
    # add epsilon to avoid log10(0)
    eps = 1e-12
    real_logp = np.log10(np.maximum(real_pnorm_plot, eps))
    gen_logp = np.log10(np.maximum(gen_pnorm_plot, eps))

    lpmin = float(min(real_logp.min(), gen_logp.min()))
    lpmax = float(max(real_logp.max(), gen_logp.max()))
    binsLP = np.linspace(lpmin, lpmax, args.logp_bins + 1)

    plt.figure()
    plt.hist(real_logp, bins=binsLP, alpha=0.6, density=True, label="Real")
    plt.hist(gen_logp, bins=binsLP, alpha=0.6, density=True, label="Generated")
    plt.xlabel(f"log10(|p|) ({p_unit})")
    plt.ylabel("Density")
    plt.legend()
    plt.title("log10(|p|) distribution | pooled particles")
    out_pngLP = os.path.join(args.outdir, "p_hist_log10.png")
    plt.savefig(out_pngLP, dpi=200, bbox_inches="tight")
    print(f"Saved log10(|p|) histogram: {out_pngLP}")

    # Save arrays for later
    np.save(os.path.join(args.outdir, "real_N.npy"), real_N)
    np.save(os.path.join(args.outdir, "gen_N.npy"), gen_N)
    np.save(os.path.join(args.outdir, "real_pnorm.npy"), real_pnorm_plot)
    np.save(os.path.join(args.outdir, "gen_pnorm.npy"), gen_pnorm_plot)
    print(f"\nSaved arrays to: {args.outdir}/(real_N, gen_N, real_pnorm, gen_pnorm).npy")


if __name__ == "__main__":
    main()
from podio import root_io
import ROOT
import glob
import argparse
import os
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

ROOT.gROOT.SetBatch(True)

# === Command-line args
parser = argparse.ArgumentParser()
parser.add_argument('--run', action='store_true', help='Run over ROOT files and save .pkl')
parser.add_argument('--plots', action='store_true', help='Generate scatterplots from .pkl files')
args = parser.parse_args()

# === File patterns
files_muons = sorted(glob.glob('/ceph/submit/data/user/j/jaeyserm/fccee/beam_backgrounds/CLD_o2_v05/mu_theta_0-180_p_50/*.root'))
files_signal = sorted(glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root'))
files_background = sorted(glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root'))

def process_files(files, label):
    energy_dict = defaultdict(list)
    for i, filename in enumerate(files):
        print(f"[{label}] Processing {i+1}/{len(files)}: {filename}")
        reader = root_io.Reader(filename)
        events = reader.get("events")
        for event in events:
            for hit in event.get("VertexBarrelCollection"):
                try:
                    mc = hit.getMCParticle()
                    if mc is None:
                        continue
                    pid = mc.getPDG()
                    energy = mc.getEnergy()
                    energy_dict[pid].append(energy)
                except Exception as e:
                    print(f"[{label}] Skipping hit due to error: {e}")
    return energy_dict

# === RUN mode
if args.run:
    print("🔁 Running extraction from ROOT files...")

    energy_muons = process_files(files_muons, "MuonGun")
    with open("muons.pkl", "wb") as f:
        pickle.dump(energy_muons, f)
    print("✅ Saved muons.pkl")

    energy_signal = process_files(files_signal, "WZ3P6 Signal")
    with open("signal.pkl", "wb") as f:
        pickle.dump(energy_signal, f)
    print("✅ Saved signal.pkl")

    energy_bkg = process_files(files_background, "Background")
    with open("background.pkl", "wb") as f:
        pickle.dump(energy_bkg, f)
    print("✅ Saved background.pkl")

# === PLOTS mode
if args.plots:
    print("📊 Generating PID-wise energy plots...")

    with open("muons.pkl", "rb") as f:
        energy_muons = pickle.load(f)
    with open("signal.pkl", "rb") as f:
        energy_signal = pickle.load(f)
    with open("background.pkl", "rb") as f:
        energy_bkg = pickle.load(f)

    all_pids = set(energy_muons) | set(energy_signal) | set(energy_bkg)
    outdir_scatter = "pid_energy_scatterplots"
    outdir_hist = "pid_energy_histograms"
    os.makedirs(outdir_scatter, exist_ok=True)
    os.makedirs(outdir_hist, exist_ok=True)

    print("\n📋 PID counts:")
    for label, dataset in [("muons", energy_muons), ("signal", energy_signal), ("background", energy_bkg)]:
        print(f"\n📁 {label.upper()}:")
        for pid in sorted(dataset):
            print(f"  PID {pid:>6} : {len(dataset[pid])} hits")

    for pid in sorted(all_pids):
        # === Scatterplot ===
        plt.figure(figsize=(8, 5))
        plotted = False
        if pid in energy_muons:
            plt.scatter(range(len(energy_muons[pid])), energy_muons[pid], label='muons', s=2, alpha=0.6, color='blue')
            plotted = True
        if pid in energy_signal:
            plt.scatter(range(len(energy_signal[pid])), energy_signal[pid], label='signal', s=2, alpha=0.6, color='green')
            plotted = True
        if pid in energy_bkg:
            plt.scatter(range(len(energy_bkg[pid])), energy_bkg[pid], label='background', s=2, alpha=0.6, color='red')
            plotted = True

        if plotted:
            plt.yscale("log")
            plt.xlabel("Hit Index")
            plt.ylabel("MC Particle Energy [GeV]")
            plt.title(f"MC Energy Scatter Plot for PDG ID {pid}")
            plt.legend(loc="upper right")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir_scatter, f"pid_{pid}_energy_scatter.png"))
            plt.close()

        # === Histogram ===
        plt.figure(figsize=(8, 5))
        plotted = False
        bins = np.logspace(-6, 2, 80)
        if pid in energy_muons:
            plt.hist(energy_muons[pid], bins=bins, alpha=0.6, label='muons', color='blue', histtype='stepfilled')
            plotted = True
        if pid in energy_signal:
            plt.hist(energy_signal[pid], bins=bins, alpha=0.6, label='signal', color='green', histtype='stepfilled')
            plotted = True
        if pid in energy_bkg:
            plt.hist(energy_bkg[pid], bins=bins, alpha=0.6, label='background', color='red', histtype='stepfilled')
            plotted = True

        if plotted:
            plt.xscale("log")
            plt.xlabel("MC Particle Energy [GeV]")
            plt.ylabel("Counts")
            plt.title(f"MC Energy Histogram for PDG ID {pid}")
            plt.legend(loc="upper right")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir_hist, f"pid_{pid}_energy_hist.png"))
            plt.close()

    # === Histogram of all top PIDs within each dataset ===


    def plot_full_energy_histogram(data, label, outdir="combined_energy_histograms", max_pids=15):
        os.makedirs(outdir, exist_ok=True)
        cmap = plt.colormaps["tab20"]
        bins = np.logspace(-6, 2, 100)

        sorted_pids = sorted(data, key=lambda k: len(data[k]), reverse=True)[:max_pids]
        hist_data = [data[pid] for pid in sorted_pids]
        labels = [str(pid) for pid in sorted_pids]
        colors = [cmap(i % 20) for i in range(len(hist_data))]

        # Stacked histogram
        plt.figure(figsize=(10, 6))
        plt.hist(hist_data, bins=bins, stacked=True, label=labels, color=colors, alpha=0.8)
        plt.xscale("log")
        plt.xlabel("MC Particle Energy [GeV]")
        plt.ylabel("Counts (stacked)")
        plt.title(f"{label.capitalize()} — Energy Histogram by PID")
        plt.legend(fontsize=7, loc="upper right", title="PDG ID")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{label}_energy_hist_stacked.png"))
        plt.close()

        # Overlaid histogram
        plt.figure(figsize=(10, 6))
        for i, pid in enumerate(sorted_pids):
            plt.hist(data[pid], bins=bins, histtype="step", color=colors[i], label=str(pid), linewidth=1.2)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("MC Particle Energy [GeV]")
        plt.ylabel("Counts")
        plt.title(f"{label.capitalize()} — Overlaid Energy Histogram by PID")
        plt.legend(fontsize=7, loc="upper right", title="PDG ID")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{label}_energy_hist_overlay.png"))
        plt.close()

    # Call for each dataset
    plot_full_energy_histogram(energy_muons, "muons")
    plot_full_energy_histogram(energy_signal, "signal")
    plot_full_energy_histogram(energy_bkg, "background")

    print(f"\n✅ Scatterplots saved to: {outdir_scatter}/")
    print(f"✅ Histograms saved to: {outdir_hist}/")
    print(f"✅ Combined PID histograms saved to: combined_energy_histograms/")
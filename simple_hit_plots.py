from podio import root_io
import ROOT
import glob
import pickle
import argparse
import os
import matplotlib.pyplot as plt
import functions
from collections import defaultdict

ROOT.gROOT.SetBatch(True)

# --- Argument parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--signal', action='store_true', help='Process signal files')
parser.add_argument('--background', action='store_true', help='Process background files')
parser.add_argument('--plots', action='store_true', help='Generate one plot per cell ID')
parser.add_argument('--maxFiles', type=int, default=100, help='Max files to process')
args = parser.parse_args()

# --- Constants ---
LAYER_RADII = [14, 36, 58]
TARGET_LAYER = 0  # Only plot from this layer

# --- Data collection ---
if args.signal or args.background:
    if args.signal:
        files = glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root')
        data_file = 'hit_xy_cellid_signal.pkl'
    elif args.background:
        files = glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root')
        data_file = 'hit_xy_cellid_background.pkl'

    xy_by_cellid = defaultdict(list)

    for i, filename in enumerate(files):
        if i >= args.maxFiles:
            break
        print(f"Processing file {i+1}/{args.maxFiles}: {filename}")
        reader = root_io.Reader(filename)
        events = reader.get('events')
        for event in events:
            for hit in event.get('VertexBarrelCollection'):
                try:
                    if functions.radius_idx(hit, LAYER_RADII) != TARGET_LAYER:
                        continue
                    if hit.isProducedBySecondary():
                        continue
                    pos = hit.getPosition()
                    cellID = hit.getCellID()
                    xy_by_cellid[cellID].append((pos.x, pos.y))
                except Exception as e:
                    print(f"Skipping hit due to error: {e}")

    with open(data_file, 'wb') as f:
        pickle.dump(xy_by_cellid, f)
    print(f"Saved hit (x, y) data for {len(xy_by_cellid)} cellIDs to {data_file}")

# --- Plotting ---
if args.plots:
    outdir = 'plots/cellid_xy_separate'
    os.makedirs(outdir, exist_ok=True)

    for label, data_file in [('Signal', 'hit_xy_cellid_signal.pkl'),
                             ('Background', 'hit_xy_cellid_background.pkl')]:
        if not os.path.exists(data_file):
            continue

        with open(data_file, 'rb') as f:
            xy_by_cellid = pickle.load(f)

        print(f"Loaded {len(xy_by_cellid)} cellIDs from {data_file}")

        for cellid, coords in xy_by_cellid.items():
            if not coords:
                continue
            xs, ys = zip(*coords)
            plt.figure(figsize=(6, 6))
            plt.scatter(xs, ys, s=3, alpha=0.6)
            plt.title(f"{label} — CellID {cellid}")
            plt.xlabel("x [mm]")
            plt.ylabel("y [mm]")
            plt.axis("equal")
            plt.tight_layout()
            outfile = os.path.join(outdir, f"{label.lower()}_cellid_{cellid}.png")
            plt.savefig(outfile)
            plt.close()
            print(f"Saved {outfile}")
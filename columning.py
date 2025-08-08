from podio import root_io
import glob
import pickle
import argparse
import functions
import math
import ROOT
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
ROOT.gROOT.SetBatch(True)
parser = argparse.ArgumentParser()
parser.add_argument('--calculate', help="Calculate", action='store_true')
parser.add_argument('--plots', help="Plot the energy deposits", action='store_true')
parser.add_argument("--maxFiles", type=int, default=1e99, help="Maximum files to run over")
args = parser.parse_args()

##########################################################################################
# this file is for plotting the number of hits in a 2D map of phi and z and purely as a
# function of phi and theta
##########################################################################################

folder = "/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z"
files = glob.glob(f"{folder}/*.root")


# layer_radii = [14, 23, 34.5, 141, 316] # IDEA approximate layer radii
# max_z = 96 # IDEA first layer

layer_radii = [14, 36, 58] # CLD approximate layer radii
max_z = 110 # CLD first layer

z_step = 2

if args.calculate:
    nFilled = 0
    nHits = 0

    phi_z_grid = np.zeros((functions.n_z_bins, functions.n_phi_bins))
    nEvents = 0

    for i, filename in enumerate(files):
        print(f"starting {filename} {i}/{len(files)}")
        podio_reader = root_io.Reader(filename)
        events = podio_reader.get("events")

        for event in events:
            nEvents += 1
            for hit in event.get("VertexBarrelCollection"):
                nHits += 1
                radius_idx = functions.radius_idx(hit, layer_radii)
                if radius_idx != 0:
                    continue  # use only first layer

                if hit.isProducedBySecondary():
                    continue

                x, y, z = hit.getPosition().x, hit.getPosition().y, hit.getPosition().z

                try:
                    z_idx, phi_idx = functions.get_grid_indices(x, y, z)
                    if 0 <= z_idx < functions.n_z_bins and 0 <= phi_idx < functions.n_phi_bins:
                        phi_z_grid[z_idx, phi_idx] += 1
                        nFilled += 1
                    else:
                        print(f"Out of bounds: z_idx={z_idx}, phi_idx={phi_idx}")
                except Exception as e:
                    print(f"Skipping hit due to error: {e}")

        if i > args.maxFiles:
            break

    # Normalize per event if desired
    #phi_z_grid /= nEvents
    print(f"Filled {nFilled} bins after {nEvents} events.")
    print(f"Parsed {nHits} hits")
    
    nonzero = phi_z_grid[phi_z_grid > 0]
    print("Max bin content:", phi_z_grid.max())
    if len(nonzero) > 0:
        print("Min (nonzero) bin content:", nonzero.min())
        print("Mean (nonzero):", nonzero.mean())
    else:
        print("No nonzero bins found!")
    
    with open("phi_z_grid.pkl", "wb") as f:
        pickle.dump(phi_z_grid, f)

if args.plots:
    outdir = "/home/submit/emmettf/FCCPhysics/beam_backgrounds/vtx"
    
    with open("phi_z_grid.pkl", "rb") as f:
        phi_z_grid = pickle.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))
    hep.style.use("ROOT")

    extent = [0, functions.n_phi_bins, 0, functions.n_z_bins]
    im = ax.imshow(phi_z_grid, aspect='auto', origin='lower', extent=extent, cmap='viridis')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average Hits per Event")

    ax.set_title("Hit Map in φ-z (25 µm Binning)")
    ax.set_xlabel("φ Columns (25 µm)")
    ax.set_ylabel("z Rows (25 µm)")

    fig.tight_layout()
    fig.savefig(f"{outdir}/phi_z_grid_heatmap.png")
    fig.savefig(f"{outdir}/phi_z_grid_heatmap.pdf")
    plt.close(fig)
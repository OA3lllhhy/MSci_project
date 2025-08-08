from podio import root_io
import ROOT
import glob
import pickle
import argparse
import os
import matplotlib.pyplot as plt
import functions

ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument('--calculate', action='store_true', help='Process files and compute cluster metrics')
parser.add_argument('--plots',     action='store_true', help='Plot histograms of cluster metrics')
parser.add_argument('--maxFiles',  type=int, default=10, help='Max files to process')
args = parser.parse_args()

# Geometry constants from functions.py
PITCH        = functions.PITCH_MM
RADIUS       = functions.RADIUS_MM
LAYER_RADII  = [14, 36, 58]      # CLD approximate layer radii
TARGET_LAYER = 0

# Output filenames
data_file = 'cluster_metrics.pkl'
outdir    = 'vtx_cluster_plots'

if args.calculate:
    cluster_metrics = []  # list of tuples: (elongation, phi_spread, z_extent, n_phi_rows)
    files = glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root')
    #Background =  /ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z
    #Signal = /ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/p


    for i, filename in enumerate(files):
        if i >= args.maxFiles:
            break
        print(f"Processing file {i+1}/{args.maxFiles}: {filename}")
        reader = root_io.Reader(filename)
        events = reader.get('events')

        for event in events:
            particles = {}
            for hit in event.get('VertexBarrelCollection'):
                try:
                    if functions.radius_idx(hit, LAYER_RADII) != TARGET_LAYER:
                        continue
                    if hit.isProducedBySecondary():
                        continue

                    pos = hit.getPosition()
                    mc  = hit.getMCParticle()
                    if mc is None:
                        continue
                    trackID = mc.getObjectID().index
                    energy  = mc.getEnergy()
                    #print(f"Track {trackID}: E = {energy} GeV, PDG = {mc.getPDG()}")

                    h = functions.Hit(x=pos.x, y=pos.y, z=pos.z,
                                      energy=energy, trackID=trackID)
                    if trackID not in particles:
                        particles[trackID] = functions.Particle(trackID)
                    particles[trackID].add_hit(h)
                except Exception as e:
                    print(f"Skipping hit due to error: {e}")

            for p in particles.values():
                if len(p.hits) < 3:
                    continue
                elong   = functions.compute_elongation_phi_z(p.hits, RADIUS)
                phi_sp  = p.phi_spread()
                z_ext   = p.z_extent()
                nrows   = p.n_phi_rows(PITCH, RADIUS)
                energy = p.hits[0].energy  # Approximate from first hit
                cluster_metrics.append((elong, phi_sp, z_ext, nrows, energy))

    with open(data_file, 'wb') as f:
        pickle.dump(cluster_metrics, f)
    print(f"Saved metrics for {len(cluster_metrics)} clusters to {data_file}")
 
if args.plots:
    # Load the computed metrics
    with open(data_file, 'rb') as f:
        metrics = pickle.load(f)
    elongations = [m[0] for m in metrics]
    phi_spreads = [m[1] for m in metrics]
    z_extents   = [m[2] for m in metrics]
    n_rows   = [m[3] for m in metrics]
    energies    = [m[4] for m in metrics]

    outdir = "vtx_energy_plots"
    os.makedirs(outdir, exist_ok=True)

    # Elongation with log-binned x-axis
    import numpy as np
    elong_min = max(1.0, min(e for e in elongations if e > 0))
    elong_max = max(elongations)
    elong_bins = np.logspace(np.log10(elong_min), np.log10(elong_max), 100)
    # Use math-text for subscripts and Greek letters
    functions.plot_hist_clusters(elongations,'elongation',r'Cluster Elongation ($\lambda_1/\lambda_2$)','Elongation',bins=elong_bins,logy=True,logx=True, outdir=outdir)
    # Z extent
    functions.plot_hist_clusters(z_extents,'z_extent',r'Z Extent',r'$\Delta z$ [mm]',bins=100,logy=True,outdir=outdir)
    # Integer-aligned bins for phi rows
    maxc = max(n_rows)
    functions.plot_hist_clusters(n_rows, 'n_phi_rows',r'Number of $\phi$ rows Hit','rows',bins=range(0, maxc + 2),logy=True,outdir=outdir)
    #Energy Plots
    functions.plot_energy_vs_metric(energies, elongations, 'energy_vs_elongation', r'Energy vs Elongation', 'Energy [GeV]', r'$\lambda_1/\lambda_2$', logx=True, logy=True, outdir=outdir)
    functions.plot_energy_vs_metric(energies, z_extents,    'energy_vs_z_extent', r'Energy vs Z Extent', 'Energy [GeV]', r'$\Delta z$ [mm]', logx=True, logy=True, outdir=outdir)
    functions.plot_energy_vs_metric(energies, n_rows,    'energy_vs_phi_rows', r'Energy vs φ rows Hit', 'Energy [GeV]', 'rows', logx=True, logy=True, outdir=outdir)

    #Combo Plots
    # Phi rows vs Z extent
    functions.plot_energy_vs_metric(z_extents,n_rows,'z_extent_vs_phi_rows',r'Z Extent vs $\phi$ rows',r'$\Delta z$ [mm]', 'rows',logx=False, logy=False,outdir=outdir)
    #elongation vs z
    functions.plot_energy_vs_metric(elongations, z_extents, 'elongation_vs_z_extent', r'Elongation vs Z Extent', r'$\lambda_1/\lambda_2$', r'$\Delta z$ [mm]', logx=True, logy=True, outdir=outdir)
    #elongation vs phi rows
    functions.plot_energy_vs_metric(elongations, n_rows, 'elongation_vs_phi_rows', r'Elongation vs $\phi$ rows Hit', r'$\lambda_1/\lambda_2$', 'rows', logx=True, logy=True, outdir=outdir)
    print(f"Plots saved in '{outdir}/'")
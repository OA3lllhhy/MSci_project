from podio import root_io
import ROOT
import glob
import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import functions
import math

ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument('--signal', action='store_true', help='Process signal files')
parser.add_argument('--background', action='store_true', help='Process background files')
parser.add_argument('--plots', action='store_true', help='Compare signal and background plots')
parser.add_argument('--maxFiles', type=int, default=100, help='Max files to process')
args = parser.parse_args()

PITCH = functions.PITCH_MM
RADIUS = functions.RADIUS_MM
LAYER_RADII = [14, 36, 58]
TARGET_LAYER = 0

if args.signal or args.background:
    if args.signal:
        files = glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root')
        data_file = 'cluster_metrics_signal.pkl'
    elif args.background:
        files = glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root')
        data_file = 'cluster_metrics_background.pkl'

    cluster_metrics = []  # list of tuples: (elongation, z_extent, n_phi_rows)
    two_hit_metrics = []      # (Δφ, Δz) for 2-hit clusters
    small_clusters=0
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
                    mc = hit.getMCParticle()
                    if mc is None:
                        continue
                    trackID = mc.getObjectID().index
                    energy = mc.getEnergy()
                    h = functions.Hit(x=pos.x, y=pos.y, z=pos.z, energy=energy, edep = 0, trackID=trackID)
                    if trackID not in particles:
                        particles[trackID] = functions.Particle(trackID)
                    particles[trackID].add_hit(h)
                except Exception as e:
                    print(f"Skipping hit due to error: {e}")
            
            for p in particles.values():
                multiplicity = len(p.hits)
                energies = [h.energy for h in p.hits]
                total_energy = sum(energies)

                if multiplicity == 2:
                    h1, h2 = p.hits
                    dphi = abs(h1.phi() - h2.phi())
                    if dphi > math.pi:
                        dphi = 2 * math.pi - dphi
                    dz = abs(h1.z - h2.z)
                    two_hit_metrics.append((dphi, dz))

                if multiplicity < 3:
                    small_clusters += 1
                    cluster_metrics.append((None, None, None, multiplicity, total_energy))
                    continue

                elong = functions.compute_elongation_phi_z(p.hits, RADIUS)
                z_ext = p.z_extent()
                nrows = p.n_phi_rows(PITCH, RADIUS)
                cluster_metrics.append((elong, z_ext, nrows, multiplicity, total_energy))
        

    with open(data_file, 'wb') as f:
        pickle.dump((cluster_metrics, two_hit_metrics), f)
    print(f"Saved metrics for {len(cluster_metrics)} clusters to {data_file}")
    print(f"Skipped {small_clusters} clusters with fewer than 3 hits")

if args.plots:
    import os
    import numpy as np
    from functions import plot_sig_bkg_hist, plot_overlay, plot_hist_clusters

    outdir = 'vtx_comparison_plots'
    os.makedirs(outdir, exist_ok=True)

    with open('cluster_metrics_signal.pkl', 'rb') as f:
        signal, signal_two_hit = pickle.load(f)
    with open('cluster_metrics_background.pkl', 'rb') as f:
        background, background_two_hit = pickle.load(f)

    # === Utility Filtering Functions ===
    def filter_metrics(clusters, indices):
        return [c for c in clusters if all(c[i] is not None for i in indices)]

    def filter_two_hit(data):
        return [(dphi, dz) for dphi, dz in data if dphi is not None and dz is not None]

    # === Filtered Data ===

    # 1. For plots that need elongation, z extent, phi rows, energy
    sig_full = filter_metrics(signal, [0, 1, 2, 4])
    bkg_full = filter_metrics(background, [0, 1, 2, 4])

    sig_elong, sig_z, sig_rows, _, sig_energy_full = zip(*sig_full)
    bkg_elong, bkg_z, bkg_rows, _, bkg_energy_full = zip(*bkg_full)

    # 2. For two-hit delta phi/z plots
    sig_dphi, sig_dz = zip(*filter_two_hit(signal_two_hit)) if signal_two_hit else ([], [])
    bkg_dphi, bkg_dz = zip(*filter_two_hit(background_two_hit)) if background_two_hit else ([], [])

    # 3. For multiplicity plots
    sig_mult = [c[3] for c in signal if c[3] is not None]
    bkg_mult = [c[3] for c in background if c[3] is not None]

    # 4. For energy-only plots
    sig_energy = [c[4] for c in signal if c[4] is not None]
    bkg_energy = [c[4] for c in background if c[4] is not None]
    
    print(f"Raw clusters: signal={len(signal)}, background={len(background)}")
    print(f"Filtered for plot: signal={len(sig_full)}, background={len(bkg_full)}")

    # === Plots ===

    # Histograms
    plot_sig_bkg_hist(sig_elong, bkg_elong, "elongation", "Elongation Comparison", r"$\lambda_1/\lambda_2$",
                      bins=np.logspace(np.log10(1.0), np.log10(max(sig_elong + bkg_elong)), 100), logx=True, logy=True, outdir=outdir)

    plot_sig_bkg_hist(sig_z, bkg_z, "z_extent", "Z Extent Comparison", r"$\Delta z$ [mm]",
                      bins=np.linspace(0, max(sig_z + bkg_z), 100), logy=True, outdir=outdir)

    plot_sig_bkg_hist(sig_rows, bkg_rows, "phi_rows", r"$\phi$ Rows Comparison", "Rows",
                      bins=range(0, max(max(sig_rows), max(bkg_rows)) + 2), logy=True, outdir=outdir)

    # Overlay scatter plots
    plot_overlay(sig_elong, sig_z, bkg_elong, bkg_z, "elongation_vs_z_extent", r"$\lambda_1/\lambda_2$", r"$\Delta z$ [mm]",
                 logx=True, logy=True, outdir=outdir)

    plot_overlay(sig_elong, sig_rows, bkg_elong, bkg_rows, "elongation_vs_phi_rows", r"$\lambda_1/\lambda_2$", "Rows",
                 logx=True, logy=True, outdir=outdir)

    plot_overlay(sig_z, sig_rows, bkg_z, bkg_rows, "z_extent_vs_phi_rows", r"$\Delta z$ [mm]", "Rows",
                 logx=False, logy=False, outdir=outdir)

    # Energy vs each metric
    plot_overlay(sig_energy_full, sig_elong, bkg_energy_full, bkg_elong, "energy_vs_elongation", "Energy [GeV]", r"$\lambda_1/\lambda_2$",
                 logx=True, logy=True, outdir=outdir)

    plot_overlay(sig_energy_full, sig_z, bkg_energy_full, bkg_z, "energy_vs_z_extent", "Energy [GeV]", r"$\Delta z$ [mm]",
                 logx=True, logy=True, outdir=outdir)

    plot_overlay(sig_energy_full, sig_rows, bkg_energy_full, bkg_rows, "energy_vs_phi_rows", "Energy [GeV]", "Rows",
                 logx=True, logy=False, outdir=outdir)

    # Multiplicity histograms
    plot_hist_clusters(sig_mult, "multiplicity_signal", "Signal Cluster Multiplicity", "Hits per Cluster",
                       bins=range(1, max(sig_mult)+2), logy=True, outdir=outdir)

    plot_hist_clusters(bkg_mult, "multiplicity_background", "Background Cluster Multiplicity", "Hits per Cluster",
                       bins=range(1, max(bkg_mult)+2), logy=True, outdir=outdir)

    plot_sig_bkg_hist(sig_mult, bkg_mult, "multiplicity", "Cluster Multiplicity Comparison", "Hits per Cluster",
                      bins=range(1, max(max(sig_mult), max(bkg_mult)) + 2), logy=False, outdir=outdir)

    # Two-hit φ–z difference
    plot_overlay(sig_dphi, sig_dz, bkg_dphi, bkg_dz, "two_hit_dphi_vs_dz", r"$\Delta\phi$ [rad]", r"$\Delta z$ [mm]",
                 logx=False, logy=False, outdir=outdir)

    plot_overlay(sig_dphi, sig_dz, [], [], "phi_z_diff_signal", r"$\Delta \phi$", r"$\Delta z$ [mm]",
                 logx=False, logy=False, outdir=outdir)

    plot_overlay([], [], bkg_dphi, bkg_dz, "phi_z_diff_background", r"$\Delta \phi$", r"$\Delta z$ [mm]",
                 logx=False, logy=False, outdir=outdir)

    # Energy histograms
    max_bkg_e = max(bkg_energy)
    max_sig_e = max(sig_energy)
    max_total_e = max(max_bkg_e, max_sig_e)

    plot_sig_bkg_hist(sig_energy, bkg_energy, "cluster_energy", "Total SimHit Energy per Cluster", "Energy [GeV]",
                      bins=np.logspace(-6, np.log10(max_total_e * 1.1), 100), logx=True, logy=False, outdir=outdir)

    plot_hist_clusters(sig_energy, "signal_cluster_energy", "Signal: Total SimHit Energy per Cluster", "Energy [GeV]",
                       bins=np.logspace(-6, np.log10(max_sig_e * 1.1), 100), logx=True, logy=False, outdir=outdir)

    plot_hist_clusters(bkg_energy, "background_cluster_energy", "Background: Total SimHit Energy per Cluster", "Energy [GeV]",
                       bins=np.logspace(-6, np.log10(max_bkg_e * 1.1), 100), logx=True, logy=False, outdir=outdir)

    print(f"Saved comparison plots in '{outdir}/'")
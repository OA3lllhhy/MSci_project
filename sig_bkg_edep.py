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
import random

ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument('--signal', action='store_true', help='Process signal files')
parser.add_argument('--background', action='store_true', help='Process background files')
parser.add_argument('--plots', action='store_true', help='Compare signal and background plots')
parser.add_argument('--maxFiles', type=int, default=100, help='Max files to process')
parser.add_argument('--classify', action='store_true', help='Train and evaluate an SVM classifier')
args = parser.parse_args()

PITCH = functions.PITCH_MM
RADIUS = functions.RADIUS_MM
LAYER_RADII = [14, 36, 58]
TARGET_LAYER = 0

if args.signal or args.background:
    zero_charge = 0
    pos_charge = 0
    neg_charge = 0
    big_clusters = 0
    if args.signal:
        files = glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root')
        data_file = 'sig_edep.pkl'
    elif args.background:
        files = glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root')
        data_file = 'bkg_edep.pkl'

    cluster_metrics = []  # list of tuples: (elongation, z_extent, n_phi_rows, multiplicity, total_edep, mc_energy, cos_theta)
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
                    try:
                        edep = hit.getEDep()
                    except AttributeError:
                        edep = 0
                        
                    h = functions.Hit(x=pos.x, y=pos.y, z=pos.z, energy=energy, edep=edep, trackID=trackID)
                    if trackID not in particles:
                        particles[trackID] = functions.Particle(trackID)
                    particles[trackID].add_hit(h)
                    
                except Exception as e:
                    print(f"Skipping hit due to error: {e}")
            
            for p in particles.values():
                multiplicity = len(p.hits)
                total_edep = p.total_energy()
                #total_charge = p.total_charge()
                x,y,z = functions.geometric_baricenter(p.hits)
                cos_theta = functions.cos_theta(x,y,z)
                mc_energy = h.energy
                z_ext = p.z_extent()
                nrows = p.n_phi_rows(PITCH, RADIUS)
                if multiplicity < 3:
                    small_clusters += 1
                    cluster_metrics.append((None, z_ext, nrows, multiplicity, total_edep, mc_energy, cos_theta))
                    continue
                big_clusters+=1
                elong = functions.compute_elongation_phi_z(p.hits, RADIUS)
                cluster_metrics.append((elong, z_ext, nrows, multiplicity, total_edep, mc_energy, cos_theta))
          
    with open(data_file, 'wb') as f:
        pickle.dump((cluster_metrics), f)
    print(f"Saved metrics for {len(cluster_metrics)} clusters to {data_file}")
    print(f"Analyzed {big_clusters} clusters with at least 3 hits")
    print(f"Skipped {small_clusters} clusters with fewer than 3 hits")
    #print(f"Positve charge: {pos_charge}, Zero charge: {zero_charge}, Negative charge: {neg_charge}")      

if args.plots:
    from functions import plot_sig_bkg_hist, plot_overlay, extract, plot_energy_vs_costheta_binned, plot_dz_vs_costheta_per_multiplicity
    import os, random
    outdir = 'vtx_comparison_edep_plots'
    random.seed(42)
    os.makedirs(outdir, exist_ok=True)
    with open('sig_edep.pkl', 'rb') as f:
        signal = pickle.load(f)
    with open('bkg_edep.pkl', 'rb') as f:
        background = pickle.load(f)

    # === Matched sampling ===
    bkg_all = background
    sig_all = random.sample(signal, len(bkg_all))
    sig_big = [c for c in signal if c[3] and c[3] >= 3]
    bkg_big_all = [c for c in background if c[3] and c[3] >= 3]
    bkg_big = random.sample(bkg_big_all, len(sig_big))
    print(f"All clusters: signal={len(sig_all)}, background={len(bkg_all)}")
    print(f"Big clusters (≥3 hits): signal={len(sig_big)}, background={len(bkg_big)}")

    # Big clusters
    sig_elong, sig_z, sig_rows, sig_mult, sig_edep, sig_mc_energy, sig_cos = extract(sig_big, 0, 1, 2, 3, 4, 5, 6)
    bkg_elong, bkg_z, bkg_rows, bkg_mult, bkg_edep, bkg_mc_energy, bkg_cos = extract(bkg_big, 0, 1, 2, 3, 4, 5, 6)
    #All Clusters
    sig_z_all, sig_rows_all, sig_mult_all, sig_edep_all, sig_mc_energy_all, sig_cos_all = extract(sig_all, 1, 2, 3, 4, 5, 6)
    bkg_z_all, bkg_rows_all, bkg_mult_all, bkg_edep_all, bkg_mc_energy_all, bkg_cos_all = extract(bkg_all, 1, 2, 3, 4, 5, 6)
    max_total_e = max(max(sig_edep_all), max(bkg_edep_all))
    #Big Clusters
    plot_sig_bkg_hist(sig_elong, bkg_elong, "elongation", "Elongation Comparison", r"$\lambda_1/\lambda_2$", bins=np.logspace(np.log10(1.0), np.log10(max(sig_elong + bkg_elong)), 100), logx=True, logy=True, outdir=outdir)
    plot_overlay(sig_elong, sig_z, bkg_elong, bkg_z, "elongation_vs_z_extent", r"$\lambda_1/\lambda_2$", r"$\Delta z$ [mm]", logx=True, logy=True, outdir=outdir)
    plot_overlay(sig_elong, sig_rows, bkg_elong, bkg_rows, "elongation_vs_phi_rows", r"$\lambda_1/\lambda_2$", "Rows", logx=True, logy=True, outdir=outdir)
    plot_overlay(sig_edep, sig_elong, bkg_edep, bkg_elong, "edep_vs_elongation", "Energy Deposited [GeV]", r"$\lambda_1/\lambda_2$", logx=True, logy=True, outdir=outdir)
    #Small Clusters
    plot_sig_bkg_hist(sig_z_all, bkg_z_all, "z_extent", "Z Extent Comparison", r"$\Delta z$ [mm]", bins=np.linspace(0, max(sig_z + bkg_z), 100), logy=True, outdir=outdir)
    plot_sig_bkg_hist(sig_rows_all, bkg_rows_all, "phi_rows", r"$\phi$ Rows Comparison", "Rows", bins=range(0, max(max(sig_rows), max(bkg_rows)) + 2), logy=True, outdir=outdir)
    plot_overlay(sig_z_all, sig_rows_all, bkg_z_all, bkg_rows_all, "z_extent_vs_phi_rows", r"$\Delta z$ [mm]", "Rows", logx=False, logy=False, outdir=outdir)
    plot_sig_bkg_hist(sig_cos_all, bkg_cos_all, "cos_theta", r"$\cos(\theta)$ Comparison", r"$\cos(\theta)$", bins=np.linspace(-1, 1, 100), logy=False, outdir=outdir)
    # Energy vs. geometry
    plot_overlay(sig_edep_all, sig_z_all, bkg_edep_all, bkg_z_all, "edep_vs_z_extent", "Energy Deposited [GeV]", r"$\Delta z$ [mm]", logx=True, logy=True, outdir=outdir)
    plot_overlay(sig_edep_all, sig_rows_all, bkg_edep_all, bkg_rows_all, "edep_vs_phi_rows", "Energy Deposited [GeV]", "Rows", logx=True, logy=False, outdir=outdir)
    #Cost theta vs ...
    plot_overlay(sig_cos_all, sig_edep_all, bkg_cos_all, bkg_edep_all, "cos_theta_vs_edep", "Cos(theta)", "Energy Deposited [GeV]", logx=False, logy=True, outdir=outdir)
    plot_overlay(np.array(sig_cos_all), sig_z_all, np.array(bkg_cos_all), bkg_z_all, "cos_theta_vs_z_extent", "cos(θ)", r"$\Delta z$ [mm]", logx=False,logy=True, outdir=outdir)
    plot_overlay(np.array(sig_cos), sig_elong, np.array(bkg_cos), bkg_elong, "cos_theta_vs_elongation", "cos(θ)", r"$\lambda_1 / \lambda_2$", logx=False,logy=True, outdir=outdir)
    plot_overlay(sig_cos_all, sig_rows_all, bkg_cos_all, bkg_rows_all, "cos_theta_vs_phi_rows", "cos(θ)", "Rows", logy=False, outdir=outdir)
    plot_overlay(sig_cos_all, sig_mult_all, bkg_cos_all, bkg_mult_all, "cos_theta_vs_multiplicity", "cos(θ)", "Cluster Multiplicity", logy=False, outdir=outdir)
    #Multiplicity
    plot_sig_bkg_hist(sig_mult_all, bkg_mult_all, "multiplicity", "Cluster Multiplicity Comparison", "Hits per Cluster", bins=range(1, max(max(sig_mult), max(bkg_mult)) + 2), logy=False, outdir=outdir)
    # Energy vs. MC energy
    plot_overlay(sig_edep_all, sig_mc_energy_all, bkg_edep_all, bkg_mc_energy_all, "edep_vs_energy", "Energy Deposited [GeV]", "MC Particle Energy [GeV]", logx=True, logy=True, outdir=outdir)
    # Edep histograms (all clusters)
    plot_sig_bkg_hist(sig_edep_all, bkg_edep_all, "cluster_edep", "Total SimHit Energy Deposited per Cluster", "Energy Deposited [GeV]", bins=np.logspace(-6, np.log10(max_total_e * 1.1), 100), logx=True, logy=False, outdir=outdir)
    cos_outdir = 'vtx_comparison_edep_plots/cos_edep_binned'
    os.makedirs(cos_outdir, exist_ok=True)
    plot_energy_vs_costheta_binned(sig_costheta=sig_cos_all, sig_edep=sig_edep_all, bkg_costheta=bkg_cos_all, bkg_edep=bkg_edep_all, nbins=10, outdir=cos_outdir)
    plot_dz_vs_costheta_per_multiplicity(sig_costheta=sig_cos_all, sig_dz=sig_z_all, sig_mult=sig_mult_all, bkg_costheta=bkg_cos_all, bkg_dz=bkg_z_all, bkg_mult=bkg_mult_all, multiplicities=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]) 
    #maybe add 1 to multiplicity array
    print(f"Saved comparison plots in '{outdir}/'")
    
if args.classify:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pickle

    print("Loading data for classification...")
    with open('sig_edep.pkl', 'rb') as f:
        sig = pickle.load(f)
    with open('bkg_edep.pkl', 'rb') as f:
        bkg = pickle.load(f)

    # Filter: only big clusters (multiplicity ≥ 3)
    #sig = [x for x in signal if x[3] and x[3] >= 3]
    #bkg = [x for x in background if x[3] and x[3] >= 3]
    print(f"Using {len(sig)} signal and {len(bkg)} background clusters")

    def extract_features(data):
        return np.array([
            [
                x[6],                                # cos(θ)
                x[4],                                # edep
                x[2],                                # phi extent
                x[1],                                # z extent
                x[0] if x[0] is not None else 0,     # elongation
                x[3],                                # multiplicity
                np.log10(x[1] + 1e-6),               # log(z extent)
                np.log10(x[4] + 1e-6),               # log(edep)
                np.log10((x[0] if x[0] is not None else 0) + 1e-6),  # log(elongation)
            ]
            for x in data
        ])

    X_sig = extract_features(sig)
    X_bkg = extract_features(bkg)
    y_sig = np.ones(len(X_sig))
    y_bkg = np.zeros(len(X_bkg))

    X = np.vstack([X_sig, X_bkg])
    y = np.hstack([y_sig, y_bkg])

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train SVM with strong bias toward signal
    clf = SVC(kernel='rbf', probability=False, class_weight={0: 1, 1: 20})
    clf.fit(X_train, y_train)

    # Predict on validation set
    y_pred = clf.predict(X_val)

    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n=== SVM Classification Report ===")
    print(f"Signal efficiency (TPR): {tpr:.4f}")
    print(f"Background rejection (1 - FPR): {1 - fpr:.4f}")
    print(f"False positives: {fp}, False negatives: {fn}")
    print(f"Confusion matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
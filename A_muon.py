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
from collections import defaultdict

ROOT.gROOT.SetBatch(True)   

parser = argparse.ArgumentParser()
parser.add_argument('--muons', action='store_true', help='Process muon files')
parser.add_argument('--background', action='store_true', help='Process background files')
parser.add_argument('--plots', action='store_true', help='Compare muons and background plots')
parser.add_argument('--maxFiles', type=int, default=100, help='Max files to process')
parser.add_argument('--classify', action='store_true', help='Train and evaluate an SVM classifier')
args = parser.parse_args()

PITCH = functions.PITCH_MM
RADIUS = functions.RADIUS_MM
LAYER_RADII = [14, 36, 58]
TARGET_LAYER = 0

if args.muons or args.background:
    if args.muons:
        files = glob.glob('/ceph/submit/data/user/j/jaeyserm/fccee/beam_backgrounds/CLD_o2_v05/mu_theta_0-180_p_50/*.root')
        data_file = 'muons_edep.pkl'
    elif args.background:
        files = glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root')
        data_file = 'bkg_edep.pkl'

    cluster_metrics = []
    for i, filename in enumerate(files):
        if i >= args.maxFiles:
            break
        print(f"Processing file {i+1}/{args.maxFiles}: {filename}")
        reader = root_io.Reader(filename)
        events = reader.get('events')

        for event in events:
            module_hits = defaultdict(list)

            for hit in event.get('VertexBarrelCollection'):
                try:
                    cellID = hit.getCellID()
                    if functions.radius_idx(hit, LAYER_RADII) != TARGET_LAYER:
                        continue
                    if hit.isProducedBySecondary():
                        continue
                    pos = hit.getPosition()
                    if functions.discard_AB(pos):
                        continue
                    mc = hit.getMCParticle()
                    if mc is None:
                        continue
                    #if args.muons and (mc.getEnergy() < 48):
                    #    continue

                    trackID = mc.getObjectID().index
                    energy = mc.getEnergy()
                    try:
                        edep = hit.getEDep()
                    except AttributeError:
                        edep = 0

                    h = functions.Hit(x=pos.x, y=pos.y, z=pos.z, energy=energy, edep=edep, trackID=trackID)
                    module_hits[cellID].append((trackID, h))

                except Exception as e:
                    print(f"Skipping hit due to error: {e}")

            # Cluster hits per module
            for cellID, hits in module_hits.items():
                particles = {}

                for trackID, h in hits:
                    key = (trackID, cellID)
                    if key not in particles:
                        particles[key] = functions.Particle(trackID=trackID, cellID=cellID)
                    particles[key].add_hit(h)

                for p in particles.values():
                    if len(p.hits) == 2:
                        p.hits = functions.merge_cluster_hits(p.hits)
                    multiplicity = len(p.hits)
                    total_edep = p.total_energy()
                    b_x, b_y, b_z = functions.geometric_baricenter(p.hits)
                    cos_theta = functions.cos_theta(b_x, b_y, b_z)
                    mc_energy = p.hits[0].energy
                    z_ext = p.z_extent()
                    nrows = p.n_phi_rows(PITCH, RADIUS)
                    cluster_metrics.append((z_ext, nrows, multiplicity, total_edep, mc_energy, cos_theta, b_x, b_y))

    with open(data_file, 'wb') as f:
        pickle.dump(cluster_metrics, f)
    print(f"Saved metrics for {len(cluster_metrics)} clusters to {data_file}")


if args.plots:
    from functions import (plot_sig_bkg_hist, plot_overlay, extract, plot_energy_vs_costheta_binned, plot_dz_vs_costheta_per_multiplicity)

    outdir = 'muon_vs_background_plots'
    random.seed(42)
    os.makedirs(outdir, exist_ok=True)

    with open('muons_edep.pkl', 'rb') as f:
        muons = pickle.load(f)
    with open('bkg_edep.pkl', 'rb') as f:
        background = pickle.load(f)

    bkg_all = background
    muon_all = random.sample(muons, len(bkg_all))
    print(f"All clusters: muons={len(muon_all)}, background={len(bkg_all)}")

    mu_z_all, mu_rows_all, mu_mult_all, mu_edep_all, mu_mc_energy_all, mu_cos_all, mu_b_x, mu_b_y = extract(muon_all, 0, 1, 2, 3, 4, 5, 6, 7)
    bkg_z_all, bkg_rows_all, bkg_mult_all, bkg_edep_all, bkg_mc_energy_all, bkg_cos_all, bkg_b_x, bkg_b_y = extract(background, 0, 1, 2, 3, 4, 5, 6, 7)

    max_total_e = max(max(mu_edep_all), max(bkg_edep_all))

    # Histogram and overlay plots
    plot_sig_bkg_hist(mu_z_all, bkg_z_all, "z_extent", "Z Extent Comparison", r"$\Delta z$ [mm]", bins=np.linspace(0, max(mu_z_all + bkg_z_all), 100), logy=True, outdir=outdir)
    plot_sig_bkg_hist(mu_rows_all, bkg_rows_all, "phi_rows", r"$\phi$ Rows Comparison", "Rows", bins=range(0, max(max(mu_rows_all), max(bkg_rows_all)) + 2), logy=True, outdir=outdir)
    plot_sig_bkg_hist(mu_mult_all, bkg_mult_all, "multiplicity", "Cluster Multiplicity Comparison", "Hits per Cluster", bins=range(1, max(max(mu_mult_all), max(bkg_mult_all)) + 2), outdir=outdir)
    plot_sig_bkg_hist(mu_edep_all, bkg_edep_all, "cluster_edep", "Total SimHit Energy Deposited", "Edep [GeV]", bins=np.logspace(-6, np.log10(max_total_e * 1.1), 100), logx=True, outdir=outdir)

    plot_overlay(mu_z_all, mu_rows_all, bkg_z_all, bkg_rows_all, "z_extent_vs_phi_rows", r"$\Delta z$ [mm]", "Rows", outdir=outdir, label_sig="muons", label_bkg="background")
    plot_overlay(mu_edep_all, mu_z_all, bkg_edep_all, bkg_z_all, "edep_vs_z_extent", "Edep [GeV]", r"$\Delta z$ [mm]", logx=True, logy=True, outdir=outdir, label_sig="muons", label_bkg="background")
    plot_overlay(mu_edep_all, mu_rows_all, bkg_edep_all, bkg_rows_all, "edep_vs_phi_rows", "Edep [GeV]", "Rows", logx=True, outdir=outdir, label_sig="muons", label_bkg="background")
    plot_overlay(mu_cos_all, mu_edep_all, bkg_cos_all, bkg_edep_all, "cos_theta_vs_edep", "cos(θ)", "Edep [GeV]", logy=True, outdir=outdir, label_sig="muons", label_bkg="background")
    plot_overlay(mu_cos_all, mu_z_all, bkg_cos_all, bkg_z_all, "cos_theta_vs_z_extent", "cos(θ)", r"$\Delta z$ [mm]", logy=True, outdir=outdir, label_sig="muons", label_bkg="background")
    plot_overlay(mu_cos_all, mu_rows_all, bkg_cos_all, bkg_rows_all, "cos_theta_vs_phi_rows", "cos(θ)", "Rows", outdir=outdir, label_sig="muons", label_bkg="background")
    plot_overlay(mu_cos_all, mu_mult_all, bkg_cos_all, bkg_mult_all, "cos_theta_vs_multiplicity", "cos(θ)", "Cluster Multiplicity", outdir=outdir, label_sig="muons", label_bkg="background")
    plot_overlay(mu_edep_all, mu_mc_energy_all, [], [], "edep_vs_energy", "Edep [GeV]", "MC Energy [GeV]", logx=True, logy=True, outdir=outdir, label_sig="muons", label_bkg="background")
    plot_overlay(mu_mc_energy_all, mu_mult_all, bkg_mc_energy_all, bkg_mult_all, name="mc_energy_vs_multiplicity", xlabel="MC Particle Energy [GeV]", ylabel="Cluster Multiplicity", logx=True, outdir=outdir, label_sig="muons", label_bkg="background")

    cos_outdir = os.path.join(outdir, 'cos_edep_binned')
    os.makedirs(cos_outdir, exist_ok=True)
    plot_energy_vs_costheta_binned(sig_costheta=mu_cos_all, sig_edep=mu_edep_all, bkg_costheta=bkg_cos_all, bkg_edep=bkg_edep_all, nbins=10, outdir=cos_outdir)

    mult_outdir = os.path.join(outdir, 'cos_z_mult')
    os.makedirs(mult_outdir, exist_ok=True)
    plot_dz_vs_costheta_per_multiplicity(sig_costheta=mu_cos_all, sig_dz=mu_z_all, sig_mult=mu_mult_all,
                                         bkg_costheta=bkg_cos_all, bkg_dz=bkg_z_all, bkg_mult=bkg_mult_all,
                                         multiplicities=list(range(2, 4)), outdir=mult_outdir)

    mu_bx_2 = [x for x, m in zip(mu_b_x, mu_mult_all) if m == 2]
    mu_by_2 = [y for y, m in zip(mu_b_y, mu_mult_all) if m == 2]
    bkg_bx_2 = [x for x, m in zip(bkg_b_x, bkg_mult_all) if m == 2]
    bkg_by_2 = [y for y, m in zip(bkg_b_y, bkg_mult_all) if m == 2]

    plot_overlay(mu_bx_2, mu_by_2, bkg_bx_2, bkg_by_2, name="barycenter_xy_multiplicity2", xlabel="b_x [mm]", ylabel="b_y [mm]", outdir=outdir, label_sig="muons", label_bkg="background")

    print(f"Saved muon vs background plots in '{outdir}/'")
    
if args.classify:
    import xgboost as xgb
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    import random
    from functions import extract

    outdir = 'muon_vs_background_plots'
    os.makedirs(outdir, exist_ok=True)
    random.seed(42)

    # === Load and sample data ===
    with open('muons_edep.pkl', 'rb') as f:
        muons = pickle.load(f)
    with open('bkg_edep.pkl', 'rb') as f:
        background = pickle.load(f)

    bkg_all = background
    muon_all = random.sample(muons, len(bkg_all))

    def get_features_and_labels(muons, background):
        signal = [(1, (z, rows, mult, edep, cos)) for (z, rows, mult, edep, _, cos, _, _) in muons]
        bkg = [(0, (z, rows, mult, edep, cos)) for (z, rows, mult, edep, _, cos, _, _) in background]
        data = signal + bkg
        random.shuffle(data)
        labels, features = zip(*data)
        return np.array(features), np.array(labels)

    print("\n=== Training Gradient Boosted Trees (XGBoost) ===")
    X, y = get_features_and_labels(muon_all, bkg_all)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=4)
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    threshold = 0.08 # try your optimal value here
    y_pred = (y_proba >= threshold).astype(int)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Background", "Signal"]))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    sig_eff = np.mean(y_pred[y_test == 1])
    bkg_rej = 1 - np.mean(y_pred[y_test == 0])
    print(f"Signal retention: {sig_eff:.4f}")
    print(f"Background rejection: {bkg_rej:.4f}")
    print(f"ROC AUC score: {roc_auc_score(y_test, y_proba):.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label="XGBoost (AUC = %.3f)" % roc_auc_score(y_test, y_proba))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Signal Efficiency)")
    plt.title("ROC Curve — XGBoost")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(outdir, "roc_curve_xgboost.png"))
    plt.close()
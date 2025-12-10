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

# === Command-line args ===
parser = argparse.ArgumentParser()
parser.add_argument('--run', action='store_true', help='Process and save muon, signal, and background files')
parser.add_argument('--plots', action='store_true', help='Plots')
parser.add_argument('--classify', action='store_true', help='Train and evaluate a classifier')
parser.add_argument('--grids', action ='store_true', help='Hyperparameter grid search for classifier')
parser.add_argument('--neural', action ='store_true', help='Neural Network Classifier')
args = parser.parse_args()

# === Geometry config ===
PITCH = functions.PITCH_MM
RADIUS = functions.RADIUS_MM
LAYER_RADII = [14, 36, 58]
TARGET_LAYER = 0


if args.run:
    all_configs = {
        'muons': {
            'files': glob.glob('/ceph/submit/data/user/j/jaeyserm/fccee/beam_backgrounds/CLD_o2_v05/mu_theta_0-180_p_50/*.root'),
            'outfile': '/ceph/submit/data/user/h/haoyun22/process_data_FCC_background/ABmuons_edep_xB_label_v3.pkl'
        },
        'signal': {
            'files': glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root'),
            'outfile': '/ceph/submit/data/user/h/haoyun22/process_data_FCC_background/ABsignal_edep_xB_label_v3.pkl'
        },
        'background': {
            'files': glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root'),
            'outfile': '/ceph/submit/data/user/h/haoyun22/process_data_FCC_background/ABbkg_edep_xB_label_v3.pkl'
        }
    }

    for label, config in all_configs.items():
        files = config['files']
        outfile = config['outfile']
        cluster_metrics = []
        limit = {
            'muons': 978,
            'signal': 100,
            'background': 1247
        }[label]

        for i, filename in enumerate(files):
            if i >= limit:
                break
            print(f"[{label.upper()}] Processing file {i+1}/{limit}: {filename}")
            # Open ROOT file with PODIO
            reader = root_io.Reader(filename)
            events = reader.get('events')
            
            # Iterate over events
            for event in events:
                particle_hits = defaultdict(list)

                # Extract hit information from detector collection
                for hit in event.get('VertexBarrelCollection'):
                    try:
                        # Separate by layer remove layer 2 and 3
                        if functions.radius_idx(hit, LAYER_RADII) != TARGET_LAYER:
                            continue
                        # Skip hits produced by secondary particles
                        if hit.isProducedBySecondary():
                            continue
                        pos = hit.getPosition() # 3D position
                        mc = hit.getMCParticle() # Monte Carlo truth particle
                        if mc is None:
                            continue
                        trackID = mc.getObjectID().index # Unique particle identifier
                        energy = mc.getEnergy() # Particle energy
                        pid = mc.getPDG() # Particle type
                        try:
                            edep = hit.getEDep() # Energy deposition
                        except AttributeError:
                            edep = 0
                        # Create Hit object
                        h = functions.Hit(x=pos.x, y=pos.y, z=pos.z, energy=energy, edep=edep, trackID=trackID)
                        # Group hits by particle trackID
                        particle_hits[trackID].append((trackID, h, pid))
                    except Exception as e:
                        print(f"Skipping hit due to error: {e}")

                # Build particle clusters from grouped hits
                for trackID, hit_group in particle_hits.items():
                    if not hit_group:
                        continue
                    _, _, pid = hit_group[0]
                    p = functions.Particle(trackID=trackID)
                    p.pid = pid
                    # p = functions.Particle(trackID=trackID, cellID=None, pid=pid)
                    for _, h, _ in hit_group:
                        p.add_hit(h)
                        
                    multiplicity = len(p.hits)
                    if multiplicity == 2:
                        p.hits = functions.merge_cluster_hits(p.hits)
                    total_edep = p.total_energy()
                    b_x, b_y, b_z = functions.geometric_baricenter(p.hits)
                    cos_theta = functions.cos_theta(b_x, b_y, b_z)
                    mc_energy = p.hits[0].energy
                    
                    
                    
                    z_ext = p.z_extent()
                    
                    # # Compute barycenter first
                    # b_x, b_y, b_z = functions.geometric_baricenter(p.hits)
                    # # Conservative Δz (actual hits)
                    # z_ext_raw = p.z_extent()
                    # # Optimistic geometric Δz
                    # z_ext_opt = functions.analytic_delta_z(p.hits, b_x, b_y, b_z)

                    # if z_ext_opt is not None:
                    #     z_ext = z_ext_opt
                    # else:
                    #     z_ext = z_ext_raw

                    if functions.discard_AB(pos):
                        cross_B = 1
                    else:
                        cross_B = 0
                    
                    nrows = p.n_phi_rows(PITCH, RADIUS)

                    cluster_metrics.append((z_ext, nrows, multiplicity, total_edep, mc_energy, cos_theta, b_x, b_y, pid, cross_B))

        with open(outfile, 'wb') as f:
            pickle.dump(cluster_metrics, f)
        print(f"✅ Saved {label} clusters to {outfile}")


if args.plots:
    from functions import extract, plot_overlay
    import os, pickle, random

    # outdir = 'cos_vs_z_extent'
    outdir = 'AB_Full_Classifier_outputs'
    os.makedirs(outdir, exist_ok=True)
    random.seed(42)

    with open('ABsignal_edep.pkl', 'rb') as f:
        signal = pickle.load(f)
    with open('ABbkg_edep.pkl', 'rb') as f:
        background = pickle.load(f)
    with open('ABmuons_edep.pkl', 'rb') as f:
        muons = pickle.load(f)

    bkg_all = background
    sig_all = random.sample(signal, len(bkg_all))

    sig_z_all, _, _, _, sig_edep_all, sig_cos_all, _, _ = extract(sig_all, 0, 1, 2, 3, 4, 5, 6, 7)
    bkg_z_all, _, _, _, bkg_edep_all, bkg_cos_all, _, _ = extract(bkg_all, 0, 1, 2, 3, 4, 5, 6, 7)

    fig_name = "z_extent_vs_cos_theta"

    plot_overlay(
        sig_cos_all, sig_z_all,
        bkg_cos_all, bkg_z_all,
        name=fig_name,
        xlabel="cos(θ)",
        ylabel=r"$\Delta z$ [mm]",
        logy=True,
        outdir=outdir,
        label_sig="signal",
        label_bkg="background"
    )

    print(f"✅ Plot saved to {os.path.join(outdir, fig_name + '.png')}")


if args.classify:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        precision_recall_fscore_support,
        roc_curve,
        ConfusionMatrixDisplay
    )
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    import random
    from functions import relabel_noise_clusters, get_features_and_labels

    outdir = 'Classification_AB'
    os.makedirs(outdir, exist_ok=True)
    random.seed(42)

    # === Load data ===
    with open('/ceph/submit/data/user/h/haoyun22/process_data_FCC_background/ABmuons_edep_xB_label_v3.pkl', 'rb') as f: # ABmuons_edep_xB_label.pkl
        muons = pickle.load(f)
    with open('/ceph/submit/data/user/h/haoyun22/process_data_FCC_background/ABsignal_edep_xB_label_v3.pkl', 'rb') as f: # ABsignal_edep_xB_label.pkl
        signal = pickle.load(f)
    with open('/ceph/submit/data/user/h/haoyun22/process_data_FCC_background/ABbkg_edep_xB_label_v3.pkl', 'rb') as f: # ABbkg_edep_xB_label.pkl
        background = pickle.load(f)

    # === Reassign noise-like clusters to background ===
    noise_pids = {11, -11, 13, -211, 22, 211, 2212, -2212}
    energy_cut = 0.01
    clean_muons, reassigned_muons = relabel_noise_clusters(muons, noise_pids, energy_cut)
    clean_signal, reassigned_signal = relabel_noise_clusters(signal, noise_pids, energy_cut)

    all_background = background + reassigned_muons + reassigned_signal
    all_signal = clean_muons + clean_signal
    sampled_signal = random.sample(all_signal, len(all_background))

    print(f"Clean muons: {len(clean_muons)}")
    print(f"Clean signal: {len(clean_signal)}")
    print(f"Reassigned to background: {len(reassigned_muons) + len(reassigned_signal)}")

    # === Feature extraction and split ===
    X, y = get_features_and_labels(sampled_signal, all_background)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        scale_pos_weight=0.5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # === Sweep thresholds to find one that preserves ≥99% signal
    thresholds = np.linspace(0.0, 1.0, 500)
    tpr_list, fpr_list, f1_list = [], [], []

    for thresh in thresholds:
        y_pred_temp = (y_proba >= thresh).astype(int)
        TP = np.sum((y_pred_temp == 1) & (y_test == 1))
        FP = np.sum((y_pred_temp == 1) & (y_test == 0))
        FN = np.sum((y_pred_temp == 0) & (y_test == 1))
        TN = np.sum((y_pred_temp == 0) & (y_test == 0))

        tpr = TP / (TP + FN) if TP + FN > 0 else 0
        fpr = FP / (FP + TN) if FP + TN > 0 else 0
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_temp, average='binary', zero_division=0)

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        f1_list.append(f1)

    target_tpr = 0.999
    best_thresh, best_fpr = None, 1.0
    for thresh, tpr, fpr in zip(thresholds, tpr_list, fpr_list):
        if tpr >= target_tpr and fpr < best_fpr:
            best_thresh, best_fpr = thresh, fpr

    if best_thresh is not None:
        print(f"Threshold for ≥{target_tpr*100:.1f}% signal retention: {best_thresh:.4f}")
        print(f"Background rejection at that threshold: {1 - best_fpr:.4f}")
    else:
        print(f"No threshold found that satisfies TPR ≥ {target_tpr*100:.1f}%")

    y_pred = (y_proba >= best_thresh).astype(int)

    print("\n=== Final Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Background", "Signal"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("ROC AUC score: %.4f" % roc_auc_score(y_test, y_proba))

    plt.plot(thresholds, tpr_list, label='TPR (Signal Retention)')
    plt.plot(thresholds, fpr_list, label='FPR (Background Acceptance)')
    if best_thresh is not None:
        plt.axvline(best_thresh, color='g', linestyle='--', label=f'TPR ≥ {target_tpr*100:.1f}% @ {best_thresh:.3f}')
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold Sweep — TPR, FPR (xB Dataset)")
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "threshold_sweep_metrics_xB_label_v3_0.999.png"))
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Final XGBoost Classifier (xB Dataset)")
    plt.legend()
    plt.grid(True)
    # plt.savefig(os.path.join(outdir, "presentation_ROC_curve_xB_label_v3.png"))
    plt.close()

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Background", "Signal"],
        cmap="Blues",
        values_format='d'
    )
    plt.title(f"Confusion Matrix @ Threshold = {best_thresh:.4f} xB Dataset")
    # plt.savefig(os.path.join(outdir, "presentation_confusion_matrix_xB_label_v3.png"))
    plt.close()
    
#     functions.plot_feature_importance(
#     clf.feature_importances_,
#     feature_names=[
#         r"$\log(\Delta z)$",
#         r"$\varphi$ extent",
#         r"multiplicity",
#         r"$\log(E_{\mathrm{dep}})$",
#         r"$\cos\theta$",
#         r"cross B boolean label"
#     ],
#     outdir="Classification_AB",
#     filename="feature_importance_xB_label_data_v3",
#     sort=True
# )
    
if args.grids:
    '''
    Hyperparameter grid search for XGBoost classifier on AB dataset
    1. Define parameter grid
    2. Iterate over combinations
    3. Train and evaluate model
    4. Record results
    5. Identify best parameters
    6. Save results to file
    7. Print best parameters and score
    8. (Optional) Visualize results
    '''
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import itertools
    import pandas as pd
    import matplotlib.pyplot as plt
    import pickle
    import os
    import random
    from functions import relabel_noise_clusters, get_features_and_labels

    outdir = 'Classification_AB/GridSearch'
    os.makedirs(outdir, exist_ok=True)
    random.seed(42)

    # === Load data ===
    with open('/ceph/submit/data/user/h/haoyun22/process_data_FCC_background/ABmuons_edep_xB_label_v2.pkl', 'rb') as f: # ABmuons_edep_xB_label.pkl
        muons = pickle.load(f)
    with open('/ceph/submit/data/user/h/haoyun22/process_data_FCC_background/ABsignal_edep_xB_label_v2.pkl', 'rb') as f: # ABsignal_edep_xB_label.pkl
        signal = pickle.load(f)
    with open('/ceph/submit/data/user/h/haoyun22/process_data_FCC_background/ABbkg_edep_xB_label_v2.pkl', 'rb') as f: # ABbkg_edep_xB_label.pkl
        background = pickle.load(f)

    # === Reassign noise-like clusters to background ===
    noise_pids = {11, -11, 13, -211, 22, 211, 2212, -2212}
    energy_cut = 0.01
    clean_muons, reassigned_muons = relabel_noise_clusters(muons, noise_pids, energy_cut)
    clean_signal, reassigned_signal = relabel_noise_clusters(signal, noise_pids, energy_cut)

    all_background = background + reassigned_muons + reassigned_signal
    all_signal = clean_muons + clean_signal
    sampled_signal = random.sample(all_signal, len(all_background))

    print(f"Clean muons: {len(clean_muons)}")
    print(f"Clean signal: {len(clean_signal)}")
    print(f"Reassigned to background: {len(reassigned_muons) + len(reassigned_signal)}")

    # === Feature extraction and split ===
    X, y = get_features_and_labels(sampled_signal, all_background)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Define hyperparameter grid ===
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01],
        'max_depth': [6, 8, 10, 12, 15],
        'scale_pos_weight': [0.5]
    }

    # === Grid search ===
    # save ROC AUC scores for each param combination
    results = []
    for n_estimators, learning_rate, max_depth, scale_pos_weight in itertools.product(
        param_grid['n_estimators'],
        param_grid['learning_rate'],
        param_grid['max_depth'],
        param_grid['scale_pos_weight']
    ):
        clf = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'scale_pos_weight': scale_pos_weight,
            'roc_auc': auc
        })
        print(f"Tested params: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}, scale_pos_weight={scale_pos_weight} => ROC AUC: {auc:.4f}")

    # === Identify best parameters ===
    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df['roc_auc'].idxmax()]
    best_params = {
        'n_estimators': best_row['n_estimators'],
        'learning_rate': best_row['learning_rate'],
        'max_depth': best_row['max_depth'],
        'scale_pos_weight': best_row['scale_pos_weight']
    }
    best_score = best_row['roc_auc']

    print("\n=== Grid Search Complete ===")
    print(f"Best Parameters: {best_params}")
    print(f"Best ROC AUC Score: {best_score:.4f}")

if args.neural:
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        precision_recall_fscore_support,
        roc_curve,
        ConfusionMatrixDisplay
    )
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    import random
    from functions import relabel_noise_clusters, get_features_and_labels, train_neural_network, evaluate_model, print_evaluation_report
    from functions import ParticleClassifier

    import sys
    sys.modules['onnx'] = None
    sys.modules['onnxruntime'] = None


    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["TORCH_DISABLE_CPP_PROTOS"] = "1"
    os.environ["TORCH_USE_RTLD_GLOBAL"] = "0"

    outdir = 'Classification_AB/GridSearch'
    os.makedirs(outdir, exist_ok=True)
    random.seed(42)

    # === Load data ===
    with open('/ceph/submit/data/user/h/haoyun22/process_data_FCC_background/ABmuons_edep_xB_label_v3.pkl', 'rb') as f: # ABmuons_edep_xB_label.pkl
        muons = pickle.load(f)
    with open('/ceph/submit/data/user/h/haoyun22/process_data_FCC_background/ABsignal_edep_xB_label_v3.pkl', 'rb') as f: # ABsignal_edep_xB_label.pkl
        signal = pickle.load(f)
    with open('/ceph/submit/data/user/h/haoyun22/process_data_FCC_background/ABbkg_edep_xB_label_v3.pkl', 'rb') as f: # ABbkg_edep_xB_label.pkl
        background = pickle.load(f)

    # === Reassign noise-like clusters to background ===
    noise_pids = {11, -11, 13, -211, 22, 211, 2212, -2212}
    energy_cut = 0.01
    clean_muons, reassigned_muons = relabel_noise_clusters(muons, noise_pids, energy_cut)
    clean_signal, reassigned_signal = relabel_noise_clusters(signal, noise_pids, energy_cut)

    all_background = background + reassigned_muons + reassigned_signal
    all_signal = clean_muons + clean_signal
    sampled_signal = random.sample(all_signal, len(all_background))

    print(f"Clean muons: {len(clean_muons)}")
    print(f"Clean signal: {len(clean_signal)}")
    print(f"Reassigned to background: {len(reassigned_muons) + len(reassigned_signal)}")
        
    # === Feature extraction and split ===
    features, labels = get_features_and_labels(sampled_signal, all_background)

    # Convert numpy to torch
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # (N,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    # === Model, loss, optimizer ===
    input_dim = X.shape[1]
    model = ParticleClassifier(input_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    experiment_name = 'Model_4'

    save_path = f'Classification_AB/NeuralNetwork/best_{experiment_name}.pth'

    # === Train model ===
    train_loss, test_loss = train_neural_network(
        model,
        train_loader,
        test_loader,
        save_path=save_path,
    )

    checkpoint_path = save_path
    save_dir = f"Classification_AB/NeuralNetwork/Evaluation_NN/ROC_Curves/{experiment_name}"
    threshold_plot_dir = f"Classification_AB/NeuralNetwork/Evaluation_NN/Threshold_Plots/{experiment_name}"
    cm_path = f"Classification_AB/NeuralNetwork/Evaluation_NN/Confusion_Matrix/{experiment_name}"

    # === Evaluate model ===
    results = evaluate_model(
        checkpoint_path=checkpoint_path,
        model_class=ParticleClassifier,
        input_dim=input_dim,
        test_loader=test_loader,
        save_dir=save_dir,
        threshold_plot_dir=threshold_plot_dir,
        cm_path=cm_path
    )

    # === Model Summary Report ===
    print_evaluation_report(results)
    
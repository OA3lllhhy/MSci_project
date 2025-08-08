from podio import root_io
import ROOT
import glob
import pickle
import argparse
import os
import functions

ROOT.gROOT.SetBatch(True)

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--signal', action='store_true', help='Process signal files')
parser.add_argument('--maxFiles', type=int, default=100, help='Max files to process')
parser.add_argument('--plots', action='store_true', help='Generate plots from saved data')
args = parser.parse_args()

# --- Constants ---
LAYER_RADII = [14, 36, 58]
TARGET_LAYER = 0  # layer index (0 = Layer 1)

# --- File selection ---
if args.signal:
    files = glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root')
    # --- Output containers ---
    cluster_data_all = []
    cluster_data_1A = []
    # --- Loop over files ---
    for i, filename in enumerate(files):
        if i >= args.maxFiles:
            break
        print(f"Processing file {i+1}/{args.maxFiles}: {filename}")
        reader = root_io.Reader(filename)
        events = reader.get('events')

        for event in events:
            particles = {}
            particles_1A = {}

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

                    # Add to all particles
                    if trackID not in particles:
                        particles[trackID] = functions.Particle(trackID)
                    particles[trackID].add_hit(h)

                    # Add to 1A subset
                    if not functions.discard_AB(pos):
                        if trackID not in particles_1A:
                            particles_1A[trackID] = functions.Particle(trackID)
                        particles_1A[trackID].add_hit(h)

                except Exception as e:
                    print(f"Skipping hit due to error: {e}")

            for p in particles.values():
                if len(p.hits) < 2:
                    continue
                barycenter = functions.geometric_baricenter(p.hits)
                z_ext = p.z_extent()
                r_ext = p.r_extent()
                cos_theta = functions.cos_theta(barycenter)
                cluster_data_all.append((z_ext, cos_theta, r_ext))

            for p in particles_1A.values():
                if len(p.hits) < 2:
                    continue
                barycenter = functions.geometric_baricenter(p.hits)
                z_ext = p.z_extent()
                r_ext = p.r_extent()
                cos_theta = functions.cos_theta(barycenter)
                cluster_data_1A.append((z_ext, cos_theta, r_ext))

    # --- Save output (once, after all files processed) ---
    print(f"Saved {len(cluster_data_all)} total clusters from Layer 1")
    print(f"Saved {len(cluster_data_1A)} clusters from Layer 1A only")

    with open('z_costheta_all_vs_1A.pkl', 'wb') as f:
        pickle.dump({'all': cluster_data_all, '1A': cluster_data_1A}, f)
        print("Wrote output to z_costheta_all_vs_1A.pkl")    

# --- Plotting ---
if args.plots:
    import matplotlib.pyplot as plt
    with open('z_costheta_all_vs_1A.pkl', 'rb') as f:
        data = pickle.load(f)

    layer_all = data['all']
    layer_1A = data['1A']

    if not layer_all or not layer_1A:
        raise ValueError("One or both cluster lists are empty.")

    # --- Unpack ---
    z_all, cos_all, r_all = zip(*layer_all)
    z_1A, cos_1A, r_1A= zip(*layer_1A)

    # --- Output directory ---
    outdir = "layer1_plots"
    os.makedirs(outdir, exist_ok=True)
    #Z_cos
    functions.plot_overlay(cos_1A, z_1A, [], [], "z_vs_costheta_L1A_only", r"cos(θ)", r"$\Delta z$ [mm]", logx=False, logy=True, outdir=outdir, label_sig="Layer 1A")
    functions.plot_overlay(cos_all, z_all, [], [], "z_vs_costheta_L1_only", r"cos(θ)", r"$\Delta z$ [mm]", logx=False, logy=True, outdir=outdir, label_sig="Layer 1")
    functions.plot_overlay(cos_1A, z_1A, cos_all, z_all, "z_vs_costheta_L1A_vs_L1", r"cos(θ)", r"$\Delta z$ [mm]", logx=False, logy=True, outdir=outdir, label_sig="Layer 1A", label_bkg="Layer 1")
    #R_cos
    functions.plot_overlay(cos_1A, r_1A, [], [], "r_vs_costheta_L1A_only", r"cos(θ)", r"$\Delta r$ [mm]", logx=False, logy=False, outdir=outdir, label_sig="Layer 1A")
    functions.plot_overlay(cos_all, r_all, [], [], "r_vs_costheta_L1_only", r"cos(θ)", r"$\Delta r$ [mm]", logx=False, logy=False, outdir=outdir, label_sig="Layer 1")
    functions.plot_overlay(cos_1A, r_1A, cos_all, r_all, "r_vs_costheta_L1A_vs_L1", r"cos(θ)", r"$\Delta r$ [mm]", logx=False, logy=False, outdir=outdir, label_sig="Layer 1A", label_bkg="Layer 1")
    #R_Z
    functions.plot_overlay(z_1A, r_1A, [], [], "r_vs_z_L1A_only", r"$\Delta z$ [mm]", r"$\Delta r$ [mm]", logx=True, logy=False, outdir=outdir, label_sig="Layer 1A")
    functions.plot_overlay(z_all, r_all, [], [], "r_vs_z_L1_only", r"$\Delta z$ [mm]", r"$\Delta r$ [mm]", logx=True, logy=False, outdir=outdir, label_sig="Layer 1")
    functions.plot_overlay(z_1A, r_1A, z_all, r_all, "r_vs_z_L1A_vs_L1", r"$\Delta z$ [mm]", r"$\Delta r$ [mm]", logx=True, logy=False, outdir=outdir, label_sig="Layer 1A", label_bkg="Layer 1")
    print(f"Saved: {outdir}/z_vs_costheta_L1A_vs_L1_overlay.png")
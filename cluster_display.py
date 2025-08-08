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
parser.add_argument('--maxFiles', type=int, default=100, help='Max files to process')
parser.add_argument('--plots', action='store_true', help='Generate φ-z event display plots')
args = parser.parse_args()

PITCH = functions.PITCH_MM
RADIUS = functions.RADIUS_MM
LAYER_RADII = [14, 36, 58]
TARGET_LAYER = 0

if args.signal or args.background:
    if args.signal:
        files = glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root')
        data_file_with_hits = 'sig_edep_with_hits.pkl'
    elif args.background:
        files = glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root')
        data_file_with_hits = 'bkg_edep_with_hits.pkl'

    cluster_metrics_with_hits = []  # list of tuples including hit positions

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
                cos_theta = functions.cos_theta(functions.geometric_baricenter(p.hits))
                mc_energy = p.hits[0].energy if p.hits else 0
                z_ext = p.z_extent()
                nrows = p.n_phi_rows(PITCH, RADIUS)
                phi_z_hits = [(h.phi(), h.z) for h in p.hits]

                elong = functions.compute_elongation_phi_z(p.hits, RADIUS) if multiplicity >= 3 else None
                cluster_metrics_with_hits.append((elong, z_ext, nrows, multiplicity, total_edep, mc_energy, cos_theta, phi_z_hits))

    with open(data_file_with_hits, 'wb') as f:
        pickle.dump(cluster_metrics_with_hits, f)
    print(f"Saved ϕ-z hit display data for {len(cluster_metrics_with_hits)} clusters to {data_file_with_hits}")
    
if args.plots:
    import os
    import numpy as np
    import pickle
    from functions import plot_cluster_scatter  # updated!

    outdir = "cluster_displays_phi_z"
    os.makedirs(outdir, exist_ok=True)

    with open("sig_edep_with_hits.pkl", "rb") as f:
        signal = pickle.load(f)
    with open("bkg_edep_with_hits.pkl", "rb") as f:
        background = pickle.load(f)

    cos_theta_target = 0.5
    tolerance = 0.001
    max_clusters_bkg = 25
    max_clusters_sig = 100

    def select_clusters(clusters, label, max_clusters):
        selected = []
        for c in clusters:
            if len(c) < 8 or c[7] is None:
                continue
            if abs(c[6] - cos_theta_target) < tolerance:
                selected.append({
                    "hits": c[7],  # list of (phi, z)
                    "cos_theta": c[6],
                    "cluster_id": len(selected),
                    "type": label
                })
            if len(selected) >= max_clusters:
                break
        return selected

    selected_signal = select_clusters(signal, "signal", max_clusters_sig)
    selected_background = select_clusters(background, "background", max_clusters_bkg)
    selected_clusters = selected_signal + selected_background

    for cluster in selected_clusters:
        hits = np.array(cluster["hits"])
        phi, z = hits[:, 0], hits[:, 1]

        outname = os.path.join(outdir, f"{cluster['type']}_cluster_{cluster['cluster_id']}_scatter.png")

        plot_cluster_scatter(phi, z, outname, cluster_type=cluster['type'], cluster_id=cluster['cluster_id'], cos_theta=cluster['cos_theta'], multiplicity=len(hits))
    print(f"Saved φ-z scatter plots for {len(selected_signal)} signal and {len(selected_background)} background clusters to '{outdir}/'")
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

# === Geometry config ===
PITCH = functions.PITCH_MM
RADIUS = functions.RADIUS_MM
LAYER_RADII = [14, 36, 58]
TARGET_LAYER = 0

configs = {
    'muons': {
        'files': glob.glob('/ceph/submit/data/user/j/jaeyserm/fccee/beam_backgrounds/CLD_o2_v05/mu_theta_0-180_p_50/output_10046_sim.root'),
        'outfile': ''
    },
    'signal': {
        'files': glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/proc_0_0.root'),
        'outfile': ''
    },
    'background': {
        'files': glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/'),
        'outfile': ''
    }
}

# Hit map size
Nphi = 21
Nz   = 50

Z_MIN = -110   # mm
Z_MAX = 110    # mm

def build_hitmap_from_event(event):
    hitmap = np.zeros((Nphi, Nz), dtype=np.float32)
    hits = event.get("VertexBarrelCollection")

    for hit in hits:
        pos = hit.getPosition()
        x, y, z = pos.x, pos.y, pos.z

        # φ coordinate
        phi = math.atan2(y, x)
        phi_idx = int((phi + math.pi) / (2 * math.pi) * Nphi)
        phi_idx = np.clip(phi_idx, 0, Nphi - 1)

        # z coordinate
        z_idx = int((z - Z_MIN) / (Z_MAX - Z_MIN) * Nz)
        z_idx = np.clip(z_idx, 0, Nz - 1)

        # use 1 or energy deposition
        try:
            edep = hit.getEDep()
        except:
            edep = 1.0

        hitmap[phi_idx, z_idx] += edep

    return hitmap


def visualize_hitmap(hitmap, title="Hit Map φ–z"):
    plt.figure(figsize=(8, 4))
    plt.imshow(
        hitmap,
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    plt.colorbar(label="Energy deposit / hit count")
    plt.xlabel("z bins")
    plt.ylabel("phi bins")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()


# ===============================
#       MAIN SCRIPT
# ===============================

rootfile = "/ceph/submit/data/user/j/jaeyserm/fccee/beam_backgrounds/CLD_o2_v05/mu_theta_0-180_p_50/output_10046_sim.root"
reader = root_io.Reader(rootfile)
events = reader.get("events")

event_index = 0  # pick the first event
event = events[event_index]

hitmap = build_hitmap_from_event(event)
visualize_hitmap(hitmap, title=f"Hit map event {event_index}")


import pickle
import numpy as np
import matplotlib.pyplot as plt
from functions import extract  # same one you're using in plots

# Load data
with open("muons_edep.pkl", "rb") as f:
    muons = pickle.load(f)

# Unpack using your standard extract format
mu_z, mu_rows, mu_mult, mu_edep, mu_mc_energy, mu_cos, mu_bx, mu_by = extract(muons, 0, 1, 2, 3, 4, 5, 6, 7)

# Get only multiplicity-2 barycenters
mu_bx_2 = [x for x, m in zip(mu_bx, mu_mult) if m == 2]
mu_by_2 = [y for y, m in zip(mu_by, mu_mult) if m == 2]

coords = np.column_stack((mu_bx_2, mu_by_2))

# Convert to polar
theta = np.arctan2(coords[:, 1], coords[:, 0])  # [-π, π]
theta_deg = np.degrees(theta)
theta_deg[theta_deg < 0] += 360  # [0, 360)

# Angular binning (16 bins = 22.5° each)
n_bins = 16
bin_edges = np.linspace(0, 360, n_bins + 1)
centers = []

for i in range(n_bins):
    in_bin = (theta_deg >= bin_edges[i]) & (theta_deg < bin_edges[i + 1])
    bin_coords = coords[in_bin]
    if len(bin_coords) > 0:
        avg = np.mean(bin_coords, axis=0)
        centers.append(tuple(avg))

# Print and optionally save
print(f"Found {len(centers)} overlap zones:")
for i, (x, y) in enumerate(centers):
    print(f"  Zone {i+1}: ({x:.3f}, {y:.3f})")

with open("overlap_centers.pkl", "wb") as f:
    pickle.dump(centers, f)

# Optional plot
plt.figure(figsize=(7.5, 7.5))  # Slightly bigger for better spacing
plt.scatter(mu_bx_2, mu_by_2, s=5, alpha=0.4, label='Multiplicity 2 Clusters')
plt.scatter(*zip(*centers), color='red', s=60, marker='x', label='Overlap Centers')
plt.xlabel('b_x [mm]', fontsize=12)
plt.ylabel('b_y [mm]', fontsize=12)
plt.title('Detected Overlap Zones (Angular Binning)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.axis('equal')
plt.tight_layout()  # Prevent text cutoff
plt.savefig("overlap_zones.png", dpi=300, bbox_inches='tight')
plt.show()
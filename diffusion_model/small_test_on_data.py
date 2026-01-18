import pickle
import numpy as np
from collections import Counter

pkl_path = "/ceph/submit/data/user/h/haoyun22/diffusion_data/epm_bkg_events.pkl"

with open(pkl_path, "rb") as f:
    events = pickle.load(f)

print(f"Loaded events: {len(events)}")

# -------- basic structure check --------
assert isinstance(events, list)
assert isinstance(events[0], dict)
assert "p" in events[0] and "pdg" in events[0]

# -------- multiplicity --------
mults = [len(ev["pdg"]) for ev in events]
print("Multiplicity summary:")
print("  min:", np.min(mults))
print("  max:", np.max(mults))
print("  mean:", np.mean(mults))

print("Multiplicity counts (top 10):")
print(Counter(mults).most_common(10))

# -------- PDG check --------
all_pdgs = np.concatenate([ev["pdg"] for ev in events])
unique_pdgs = set(all_pdgs.tolist())
print("Unique PDGs:", unique_pdgs)

assert unique_pdgs.issubset({11, -11})

# -------- momentum sanity --------
all_p = np.concatenate([ev["p"] for ev in events], axis=0)

p_norm = np.linalg.norm(all_p, axis=1)

print("\nMomentum |p| stats:")
print("  min |p|:", p_norm.min())
print("  max |p|:", p_norm.max())
print("  mean |p|:", p_norm.mean())
print("  median |p|:", np.median(p_norm))

# -------- numerical safety --------
print("\nNumerical checks:")
print("  any NaN in p:", np.isnan(all_p).any())
print("  any inf in p:", np.isinf(all_p).any())
print("  fraction |p| < 1e-6:", np.mean(p_norm < 1e-6))

all_E = []

for ev in events:
    for pid, p in zip(ev["pdg"], ev["p"]):
        pass  # we only have p here
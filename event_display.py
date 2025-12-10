from podio import root_io
import glob
import functions
import math
import ROOT
import os
ROOT.gROOT.SetBatch(True)

# --- Inputs ---
signal = '/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root'
background = '/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root'

# Choose which to run over
folder = signal  # <- change to `signal` if you want
files = glob.glob(f"{folder}")

# ===== DEBUG: Print file information =====
print(f"Looking for files in: {folder}")
print(f"Number of files found: {len(files)}")
if len(files) == 0:
    print("ERROR: No files found! Check the path.")
    exit(1)
else:
    print(f"First few files: {files[:3]}")


# --- Config ---
layer_radii = [14, 36, 58]  # CLD approximate layer radii
max_z = 110  # mm
TARGET_LAYER = 0            # Only show hits on this layer
outdir = "event_display/signal_events"  # Output directory
os.makedirs(outdir, exist_ok=True)
max_events = 50             # <-- run over 50 events total
event_counter = 0

# Binning (same as before)
nbins_z = int(max_z / 2)
nbins_phi = int(360 / 5)

# Make a cumulative histogram for summed multiplicity
cumulative = ROOT.TH2D("z_phi_sum", "", nbins_z, -max_z, max_z, nbins_phi, 0, 360)
cumulative.SetDirectory(0)

for i, filename in enumerate(files):
    print(f"Processing file {filename} ({i+1}/{len(files)})")
    podio_reader = root_io.Reader(filename)
    events = podio_reader.get("events")

    for event in events:
        if event_counter >= max_events:
            break

        particles = {}
        print(f"  Processing event {event_counter}")

        for hit in event.get("VertexBarrelCollection"):
            try:
                # keep only target layer and primaries
                if functions.radius_idx(hit, layer_radii) != TARGET_LAYER:
                    continue
                if hit.isProducedBySecondary():
                    continue

                pos = hit.getPosition()
                mc = hit.getMCParticle()
                if mc is None:
                    continue

                trackID = mc.getObjectID().index
                energy = mc.getEnergy()

                # edep not needed for multiplicity counting, but harmless to keep
                edep = 1000 * getattr(hit, "getEDep", lambda: 0)()

                h = functions.Hit(x=pos.x, y=pos.y, z=pos.z, energy=energy, edep=edep, trackID=trackID)
                if trackID not in particles:
                    particles[trackID] = functions.Particle(trackID)
                particles[trackID].add_hit(h)

            except Exception as e:
                print(f"Skipping hit due to error: {e}")

        # Skip empty events
        if not particles:
            continue

        # Per-event φ–z histogram of HIT COUNTS (weight = 1)
        hist = ROOT.TH2D(f"z_phi_evt{event_counter}", "", nbins_z, -max_z, max_z, nbins_phi, 0, 360)
        hist.SetDirectory(0)

        # Fill with +1 per hit
        for p in particles.values():
            for h in p.hits:
                phi_deg = h.phi() * 180.0 / math.pi
                # hist.Fill(h.z, phi_deg, 1.0)
                hist.Fill(h.z, phi_deg, h.edep)  # Use edep as weight

        # Add to cumulative
        cumulative.Add(hist)

        # Save per-event plot (title reflects multiplicity)
        outname = f"{outdir}/event_{event_counter:04d}_Energy.png"
        functions.plot_2dhist(
            hist,
            outname=outname,
            # title=f"Hit Multiplicity – Event {event_counter}",
            title=f"Hit Energy Deposition – Event {event_counter}",
            xMin=-max_z, xMax=max_z,
            xLabel="z (mm)", yLabel="Azimuthal angle φ (deg)"
        )

        event_counter += 1

    if event_counter >= max_events:
        print("Reached max number of events.")
        break

# Save the summed plot
sum_out = f"{outdir}/events_0_to_{event_counter-1}_SUM_Energy.png"
functions.plot_2dhist(
    cumulative,
    outname=sum_out,
    # title=f"Hit Multiplicity – Sum of {event_counter} signal events",
    title=f"Hit Energy Deposition – Sum of {event_counter} signal events",
    xMin=-max_z, xMax=max_z,
    xLabel="z (mm)", yLabel="Azimuthal angle φ (deg)",
    logBar=True
)

print(f"Done. Wrote {event_counter} per-event plots to {outdir} and a summed plot: {sum_out}")
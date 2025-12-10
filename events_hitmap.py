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

# Hit map configuration
PIXEL_SIZE_MM = 0.025  # 25 micrometers = 0.025 mm
MAP_SIZE_PHI = 5.0     # mm in phi direction (arc length)
MAP_SIZE_Z = 5.0       # mm in z direction

def calculate_average_barycenter(clusters):
    """
    Calculate the average barycenter from multiple clusters
    
    Args:
        clusters: list of functions.Particle objects
    
    Returns:
        tuple: (avg_x, avg_y, avg_z, avg_phi, avg_r)
    """
    if not clusters:
        return (0, 0, 0, 0, 0)
    
    sum_x, sum_y, sum_z = 0, 0, 0
    
    for cluster in clusters:
        b_x, b_y, b_z = functions.geometric_baricenter(cluster.hits)
        sum_x += b_x
        sum_y += b_y
        sum_z += b_z
    
    n = len(clusters)
    avg_x = sum_x / n
    avg_y = sum_y / n
    avg_z = sum_z / n
    avg_phi = math.atan2(avg_y, avg_x)
    avg_r = math.sqrt(avg_x**2 + avg_y**2)
    
    return (avg_x, avg_y, avg_z, avg_phi, avg_r)


def build_event_hitmap_centered(clusters, pixel_size_mm=0.025, 
                                 map_size_phi=0.625, map_size_z=0.625):
    """
    Build a hit-map for all clusters in an event, centered on average barycenter
    
    Args:
        clusters: list of functions.Particle objects
        pixel_size_mm: size of each pixel in mm (default 25 micrometers)
        map_size_phi: total size of map in phi direction (arc length in mm)
        map_size_z: total size of map in z direction (mm)
    
    Returns:
        hitmap: 2D numpy array (n_phi_bins, n_z_bins) in MeV
        avg_barycenter: tuple (avg_x, avg_y, avg_z, avg_phi, avg_r)
        extent: tuple for plotting
        cluster_barycenters: list of individual cluster barycenters
    """
    # Calculate average barycenter
    avg_x, avg_y, avg_z, avg_phi, avg_r = calculate_average_barycenter(clusters)
    
    # Calculate number of bins
    n_phi_bins = int(map_size_phi / pixel_size_mm)
    n_z_bins = int(map_size_z / pixel_size_mm)
    
    # Initialize hitmap
    hitmap = np.zeros((n_phi_bins, n_z_bins), dtype=np.float32)
    
    # Define coordinate ranges (centered on average barycenter)
    delta_phi = map_size_phi / avg_r  # arc_length = r * angle
    phi_min = avg_phi - delta_phi / 2
    phi_max = avg_phi + delta_phi / 2
    
    z_min = avg_z - map_size_z / 2
    z_max = avg_z + map_size_z / 2
    
    # Store individual cluster barycenters for visualization
    cluster_barycenters = []
    
    # Fill hitmap with hits from ALL clusters
    for cluster in clusters:
        # Get this cluster's barycenter
        c_x, c_y, c_z = functions.geometric_baricenter(cluster.hits)
        c_phi = math.atan2(c_y, c_x)
        c_r = math.sqrt(c_x**2 + c_y**2)
        cluster_barycenters.append((c_x, c_y, c_z, c_phi, c_r, cluster.pid))
        
        # Add all hits from this cluster
        for hit in cluster.hits:
            x, y, z = hit.x, hit.y, hit.z
            
            # Calculate phi coordinate
            phi = math.atan2(y, x)
            
            # Handle phi wraparound
            if phi - avg_phi > math.pi:
                phi -= 2 * math.pi
            elif phi - avg_phi < -math.pi:
                phi += 2 * math.pi
            
            # Check if hit is within map bounds
            if not (phi_min <= phi <= phi_max and z_min <= z <= z_max):
                continue
            
            # Convert to bin indices
            phi_idx = int((phi - phi_min) / (phi_max - phi_min) * n_phi_bins)
            z_idx = int((z - z_min) / (z_max - z_min) * n_z_bins)
            
            # Clip to valid range
            phi_idx = np.clip(phi_idx, 0, n_phi_bins - 1)
            z_idx = np.clip(z_idx, 0, n_z_bins - 1)
            
            # Add energy deposition in MeV
            hitmap[phi_idx, z_idx] += hit.edep * 1000  # GeV to MeV
    
    # Convert to plotting coordinates
    arc_min = -map_size_phi / 2
    arc_max = map_size_phi / 2
    z_plot_min = -map_size_z / 2
    z_plot_max = map_size_z / 2
    
    extent = (z_plot_min, z_plot_max, arc_min, arc_max)
    avg_barycenter = (avg_x, avg_y, avg_z, avg_phi, avg_r)
    
    return hitmap, avg_barycenter, extent, cluster_barycenters


def visualize_event_hitmap_centered(hitmap, avg_barycenter, extent, clusters,
                                     cluster_barycenters, event_idx,
                                     title="Event Hit Map", outdir='cluster_hitmaps',
                                     pixel_size_mm=0.025):
    """Visualize event hit-map centered on average barycenter"""
    os.makedirs(outdir, exist_ok=True)
    
    avg_x, avg_y, avg_z, avg_phi, avg_r = avg_barycenter
    z_min, z_max, arc_min, arc_max = extent
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot hitmap
    im = ax.imshow(
        hitmap,
        aspect='equal',
        origin='lower',
        cmap='viridis',
        extent=[z_min, z_max, arc_min, arc_max],
        interpolation='nearest'
    )
    
    # Mark average barycenter at origin (cyan cross)
    #ax.scatter(0, 0, color='cyan', s=300, marker='+', 
    #           linewidths=4, label='Avg Barycenter', zorder=6)
    
    # Mark individual cluster barycenters (different colors)
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(cluster_barycenters))))
    
    for idx, (c_x, c_y, c_z, c_phi, c_r, pid) in enumerate(cluster_barycenters):
        # Convert to relative coordinates
        # Handle phi wraparound
        c_phi_rel = c_phi - avg_phi
        if c_phi_rel > math.pi:
            c_phi_rel -= 2 * math.pi
        elif c_phi_rel < -math.pi:
            c_phi_rel += 2 * math.pi
        
        arc_length_rel = c_phi_rel * avg_r
        z_rel = c_z - avg_z
        
        # Only plot if within bounds
        if (z_min <= z_rel <= z_max and arc_min <= arc_length_rel <= arc_max):
            color = colors[idx % len(colors)]
            label = f'Cluster {idx} (PID={pid})' if idx < 5 else None  # Limit legend entries
            ax.scatter(z_rel, arc_length_rel, color=color, s=100, 
                      marker='o', edgecolors='white', linewidths=2, 
                      zorder=5, label=label, alpha=0.8)
    
    # Plot all hits
    total_hits = sum(len(c.hits) for c in clusters)
    for cluster in clusters:
        for hit in cluster.hits:
            hit_phi = math.atan2(hit.y, hit.x)
            
            # Handle phi wraparound
            if hit_phi - avg_phi > math.pi:
                hit_phi -= 2 * math.pi
            elif hit_phi - avg_phi < -math.pi:
                hit_phi += 2 * math.pi
            
            # Convert to arc length
            arc_length = (hit_phi - avg_phi) * avg_r
            z_rel = hit.z - avg_z
            
            # Only plot if within bounds
            if (z_min <= z_rel <= z_max and arc_min <= arc_length <= arc_max):
                ax.scatter(z_rel, arc_length, color='lime', s=10, 
                          alpha=0.5, edgecolors='none', zorder=4)
    
    # Add grid
    ax.grid(True, alpha=0.3, linewidth=0.5, color='white')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Energy deposit [MeV]")
    
    # Labels and title
    ax.set_xlabel("Δz [mm] (relative to average barycenter)", fontsize=12)
    ax.set_ylabel("Δs_φ [mm] (arc length, relative to average barycenter)", fontsize=12)
    ax.set_title(f"{title}\nPixel size: {pixel_size_mm*1000:.0f} μm, "
                f"Avg Barycenter: ({avg_x:.2f}, {avg_y:.2f}, {avg_z:.2f}) mm", 
                fontsize=11)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    # Add text with event info
    total_edep_mev = hitmap.sum()
    info_text = f"Event {event_idx}\n"
    info_text += f"Clusters: {len(clusters)}\n"
    info_text += f"Total Hits: {total_hits}\n"
    info_text += f"Total Edep: {total_edep_mev:.2f} MeV\n"
    info_text += f"Avg Radius: {avg_r:.2f} mm\n"
    info_text += f"Avg φ: {avg_phi:.3f} rad"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.9), fontsize=10)
    
    plt.tight_layout()
    
    save_path = os.path.join(outdir, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {save_path}")


def process_events_to_hitmaps(config_label='muons', max_events=10, 
                               min_hits_per_cluster=3,
                               pixel_size_mm=0.025, map_size_phi=5.0, 
                               map_size_z=5.0):
    """
    Process ROOT files and create one hit-map per event (all clusters combined)
    
    Args:
        config_label: 'muons', 'signal', or 'background'
        max_events: maximum number of events to process
        min_hits_per_cluster: minimum number of hits required for a cluster
        pixel_size_mm: pixel size in mm
        map_size_phi: map size in phi direction (arc length, mm)
        map_size_z: map size in z direction (mm)
    """
    configs = {
        'muons': {
            'files': glob.glob('/ceph/submit/data/user/j/jaeyserm/fccee/beam_backgrounds/CLD_o2_v05/mu_theta_0-180_p_50/*.root')[:3],
        },
        'signal': {
            'files': glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root')[:3],
        },
        'background': {
            'files': glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root')[:3],
        }
    }
    
    files = configs[config_label]['files']
    event_count = 0
    total_cluster_count = 0
    outdir = f'cluster_hitmaps_centered/{config_label}'
    
    print(f"\n=== Configuration ===")
    print(f"Dataset: {config_label}")
    print(f"Pixel size: {pixel_size_mm*1000:.1f} μm")
    print(f"Map size (phi): {map_size_phi} mm")
    print(f"Map size (z): {map_size_z} mm")
    print(f"Grid size: {int(map_size_phi/pixel_size_mm)} × {int(map_size_z/pixel_size_mm)} pixels")
    print(f"Min hits per cluster: {min_hits_per_cluster}\n")
    
    for file_idx, filename in enumerate(files):
        if event_count >= max_events:
            break
            
        print(f"\n📁 Processing file {file_idx + 1}/{len(files)}: {os.path.basename(filename)}")
        reader = root_io.Reader(filename)
        events = reader.get('events')
        
        for event_idx, event in enumerate(events):
            if event_count >= max_events:
                break
            
            print(f"\n  📊 Processing Event {event_count} (file event #{event_idx})")
            particle_hits = defaultdict(list)
            
            # Extract hits and group by trackID
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
                    pid = mc.getPDG()
                    
                    try:
                        edep = hit.getEDep()
                    except AttributeError:
                        edep = 0
                    
                    h = functions.Hit(x=pos.x, y=pos.y, z=pos.z, 
                                     energy=energy, edep=edep, trackID=trackID)
                    particle_hits[trackID].append((trackID, h, pid))
                    
                except Exception as e:
                    continue
            
            # Build all clusters for this event
            clusters = []
            for trackID, hit_group in particle_hits.items():
                if not hit_group:
                    continue
                    
                _, _, pid = hit_group[0]
                p = functions.Particle(trackID=trackID)
                p.pid = pid
                
                for _, h, _ in hit_group:
                    p.add_hit(h)
                
                # Only include clusters with enough hits
                if len(p.hits) >= min_hits_per_cluster:
                    clusters.append(p)
            
            if not clusters:
                print(f"     ⚠️  No valid clusters in this event, skipping...")
                continue
            
            print(f"     Found {len(clusters)} clusters with >={min_hits_per_cluster} hits")
            
            # Build single hit-map for entire event
            hitmap, avg_barycenter, extent, cluster_barycenters = build_event_hitmap_centered(
                clusters, pixel_size_mm, map_size_phi, map_size_z
            )
            
            # Skip empty hitmaps
            if hitmap.sum() == 0:
                print(f"     ⚠️  Empty hitmap, skipping...")
                continue
            
            # Create title
            title = f"{config_label}_event{event_count}_{len(clusters)}clusters"
            
            visualize_event_hitmap_centered(
                hitmap, avg_barycenter, extent, clusters, 
                cluster_barycenters, event_count,
                title=title, outdir=outdir, pixel_size_mm=pixel_size_mm
            )
            
            total_cluster_count += len(clusters)
            event_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Processing complete!")
    print(f"   Total events processed: {event_count}")
    print(f"   Total clusters: {total_cluster_count}")
    print(f"   Average clusters per event: {total_cluster_count/event_count:.1f}")
    print(f"   Output directory: {outdir}/")
    print(f"{'='*60}\n")


# ===============================
#       MAIN SCRIPT
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate event-level hit-maps centered on average cluster barycenter'
    )
    parser.add_argument('--label', type=str, default='muons', 
                       choices=['muons', 'signal', 'background'],
                       help='Dataset to process')
    parser.add_argument('--max_events', type=int, default=10,
                       help='Maximum number of events to process')
    parser.add_argument('--min_hits', type=int, default=3,
                       help='Minimum hits per cluster to include')
    parser.add_argument('--pixel_size', type=float, default=0.025,
                       help='Pixel size in mm (default: 0.025 = 25 μm)')
    parser.add_argument('--map_size_phi', type=float, default=5.0,
                       help='Map size in phi direction (arc length, mm)')
    parser.add_argument('--map_size_z', type=float, default=5.0,
                       help='Map size in z direction (mm)')
    args = parser.parse_args()
    
    process_events_to_hitmaps(
        config_label=args.label,
        max_events=args.max_events,
        min_hits_per_cluster=args.min_hits,
        pixel_size_mm=args.pixel_size,
        map_size_phi=args.map_size_phi,
        map_size_z=args.map_size_z
    )
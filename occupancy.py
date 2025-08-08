from podio import root_io
import glob
import pickle
import argparse
import functions
import math
import ROOT
ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument('--calculate', help="Calculate", action='store_true')
parser.add_argument("--maxFiles", type=int, default=1e99, help="Maximum files to run over")
args = parser.parse_args()


##########################################################################################
#  this file is for calculating the average and max occupancies in the first layer
##########################################################################################

folder = "/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/"
files = glob.glob(f"{folder}/*.root")


# layer_radii = [14, 23, 34.5, 141, 316] # IDEA approximate layer radii
# max_z = 96 # IDEA first layer

layer_radii = [14, 36, 58] # CLD approximate layer radii
max_z = 110 # CLD first layer


if args.calculate:

    cellCounts = {}
    nEvents = 0
    for i,filename in enumerate(files):

        print(f"starting {filename} {i}/{len(files)}")
        podio_reader = root_io.Reader(filename)

        events = podio_reader.get("events")
        for event in events:
            nEvents += 1
            for hit in event.get("VertexBarrelCollection"):
                radius_idx = functions.radius_idx(hit, layer_radii)
                if radius_idx != 0: # consider only hits on the first layer
                    continue

                if hit.isProducedBySecondary(): # remove mc particle not tracked
                    continue

                # the cellID corresponds to a cluster of pixels = module
                # the readout is done per module
                cellID = hit.getCellID()
                if not cellID in cellCounts:
                    cellCounts[cellID] = 0
                # increment the hit count for this module
                # strictly speaking all the primaries+secondaries within the dR should be treated as 1 hit
                cellCounts[cellID] += 1


        if i > args.maxFiles:
            break

    # normalize the hits over the number of events
    print(cellCounts)
    cellCounts_norm = [cellCounts[c]/nEvents for c in cellCounts]
    
    max_hits = max(cellCounts_norm)
    avg_hits = sum(cellCounts_norm)/len(cellCounts_norm)

    safety_factor = 3
    cluster_size = 5

    # this corresponds to the occupancy per module
    print(f"Maximum occupancy: {max_hits*safety_factor*cluster_size}")
    print(f"Average occupancy: {avg_hits*safety_factor*cluster_size}")


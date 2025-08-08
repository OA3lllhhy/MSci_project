# Impact of beam backgrounds on the vertex detector

Simulated campaigns

```
Detector: CLD_o2_v05
Samples: /ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/
Stack: /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2024-04-12
```


## Energy deposits

Each particle (whether signal or background) leaves a hit in the vertex detector layers when it crosses the layer. It's a small energy deposit in the order of keV. To compute the energy deposit in each layer, execute the following command:

```
python energy_deposit.py --calculate --maxFiles 100
```

In the ```energy_deposit.py```, you can specify your detector (IDEA or CLD) and the process (beam backgrounds or physics events, Z->hadrons in this case). The ```maxFiles``` argument is optional (remove it if you want to run over all the files and events). 


To plot the energy distributions for each layer, execute the following command:

```
python energy_deposit.py --plots
```

It generates the energy deposits for each layer. There are 3 types of energy deposit plots:

- Raw energy deposit for each hit on each layer (keV)
- Energy deposit per unit of crossed material (keV/mm)
- Same as above, but for a subset of hits that do not have secondary particles


## Hit maps

Hit maps are generated on the first layer of the Vertex detector

```
python hitmaps.py --calculate --maxFiles 100
python hitmaps.py --plots
```


## Occupancy


The occupancy is calculated with a characteristic area according to the CellID in the geometry definition:

```
python occupancy.py --calculate
```

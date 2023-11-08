"""
Copyright (c) 2023 - The University of Texas at Austin
This work was produced under contract #2317831 to National Technology and
Engineering Solutions of Sandia, LLC which is under contract
No. DE-NA0003525 with the U.S. Department of Energy.
"""
import yaml
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath((os.path.join(SCRIPT_DIR, os.pardir)))
sys.path.insert(0, PROJECT_DIR)
import sim
sys.path.insert(0, SCRIPT_DIR)
import latin_squares

# For a range of core sizes, the total number of neurons is always the same
core_counts = (8, 16, 32, 64, 128, 170, 256, 512)
compartment_counts = (16384, 8192, 4096, 2048, 1024, 768, 512, 256)

cores_per_tile = 4

run_experiment = True
plot_results = True

if run_experiment:
    with open("runs/dse/dvs_results.csv", "w") as dse_results:
        dse_results.write("cores,compartments,energy,latency\n")
    with open("runs/dse/latin_results.csv", "w") as dse_results:
        dse_results.write("cores,compartments,energy,latency\n")

    with open("arch/loihi_latin.yaml", "rb") as loihi_file:
        loihi_baseline_latin = yaml.safe_load(loihi_file)
    with open("arch/loihi_dvs.yaml", "rb") as loihi_file:
        loihi_baseline_dvs = yaml.safe_load(loihi_file)

    for cores, compartments in zip(core_counts, compartment_counts):
        mapping_file = f"{cores}c_{compartments}cx_mapping.net"
        mapping_path = os.path.join(PROJECT_DIR, "runs", "dse", "mappings",
                                    mapping_file)
        # Change the tile count, based on the number of cores. Otherwise, the loihi
        #  arch remains unchanged
        max_tile = cores // cores_per_tile
        tile_name = f"loihi_tile[0..{max_tile-1}]"
        loihi_baseline_latin["architecture"]["tile"][0]["name"] = tile_name
        loihi_baseline_dvs["architecture"]["tile"][0]["name"] = tile_name

        design_path_latin = os.path.join(PROJECT_DIR, "runs", "dse",
                                   f"loihi_{cores}c_latin.yaml")
        design_path_dvs = os.path.join(PROJECT_DIR, "runs", "dse",
                                   f"loihi_{cores}c_dvs.yaml")
        with open(design_path_latin, "w") as design_file:
            yaml.safe_dump(loihi_baseline_latin, design_file)
        with open(design_path_dvs, "w") as design_file:
            yaml.safe_dump(loihi_baseline_dvs, design_file)

        # **** Run the latin squares experiment first ****
        latin_squares.latin_square(15, tiles=cores//4, cores_per_tile=4,
                                   neurons_per_core=compartments)
        results = sim.run(design_path_latin, "runs/dse/latin_square_N15.net",
                          3000)
        with open("runs/dse/latin_results.csv", "a") as dse_results:
            dse_results.write(f"{cores},{compartments},{results['energy']},{results['time']}\n")

        # **** Run the DVS experiment next ****
        #"""
        # Concatenate the mapping with the DVS network into a temporary network file
        open("runs/dse/mappings/dvs_gesture_32x32_mapped.net", "w")
        with open("runs/dse/mappings/dvs_gesture_32x32_mapped.net", "a") as out:
            with open("runs/dse/mappings/dvs_gesture_32x32_unmapped.net", "r") as snn:
                for line in snn:
                    out.write(line)
            with open(mapping_path, "r") as snn:
                for line in snn:
                    out.write(line)

        results = sim.run(design_path_dvs,
                          "runs/dse/mappings/dvs_gesture_32x32_mapped.net", 128)

        with open("runs/dse/dvs_results.csv", "a") as dse_results:
            dse_results.write(f"{cores},{compartments},{results['energy']},{results['time']}\n")
        print(results)
        #"""

if plot_results:
    energies = []
    latencies = []
    with open("runs/dse/latin_results.csv", "r") as dse_results:
        reader = csv.reader(dse_results)
        next(reader)
        for line in reader:
            energies.append(float(line[2]))
            latencies.append(float(line[3]))
    plt.rcParams.update({'font.size': 7, 'lines.markersize': 1})

    labels = [f"<{c}c,{n}n>" for c, n in
            zip(core_counts, compartment_counts)]
    print(labels)
    df = pd.DataFrame({"Energies": np.array(energies) * 1.0e3, "Latencies": np.array(latencies)}, index=labels)
    #plt.bar(labels, np.array(energies) / energies[4], width=0.2)
    #plt.bar(labels, np.array(latencies) / latencies[4], width=0.2)

    df.plot.bar(rot=30, figsize=(3.6, 1.8), color=("#ff7f0e", "#1f77b4"), secondary_y="Latencies", legend=False)
    ax = plt.gca()
    plt.xlabel("Design Configuration")
    ax1, ax2 = plt.gcf().get_axes()
    ax1.minorticks_on()
    #ax1.set_ylim((0, 0.5))
    ax2.minorticks_on()
    #legend = ax2.legend()
    #print(legend)
    ax1.set_ylabel("Total Energy (mJ)")
    ax2.set_ylabel("Total Run-time (s)")
    ax1.tick_params(axis='x', which='minor', bottom=False)
    ax2.tick_params(axis='x', which='minor', bottom=False)
    #plt.xticks(rotation=90, ha="right")
    ax1.set_xticklabels(ax1.get_xticklabels(), ha="right")
    ax1.set_xlabel("Design Configuration")
    bars = ax1.containers
    bars += ax2.containers
    ax1.set_ylim((0, 25))
    plt.legend(bars, ("Energy", "Latency"), loc="upper center")
    # Make the Loihi configuration bold
    for lab in ax1.get_xticklabels():
        if lab.get_text() == "<128c,1024n>":
            lab.set_fontweight("bold")

    plt.tight_layout(pad=0.5)
    ax = plt.gca()
    ax_pos = ax.get_position()
    print(ax_pos)
    plt.savefig("runs/dse/latin_dse.pdf")

    energies = []
    latencies = []
    with open("runs/dse/dvs_results.csv", "r") as dse_results:
        reader = csv.reader(dse_results)
        next(reader)
        for line in reader:
            energies.append(float(line[2]))
            latencies.append(float(line[3]))

    labels = [f"<{c}c,{n}n>" for c, n in
            zip(core_counts, compartment_counts)]
    print(labels)
    plt.rcParams.update({'font.size': 7, 'lines.markersize': 1})
    plt.figure(figsize=(3.5, 2.0))
    plt.bar(labels, np.array(latencies) * 1.0e3, width=0.5)
    plt.xlabel("Design Configuration")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Total Latency (ms)")
    plt.tight_layout(pad=0.3)
    plt.savefig("runs/dse/dvs_latency.pdf")
    print(f"latencies={latencies}")

    plt.figure(figsize=(3.5, 2.0))
    plt.bar(labels, np.array(energies) * 1.0e3, width=0.5)
    plt.xlabel("Design Configuration")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Total Energy (mJ)")
    plt.tight_layout(pad=0.3)
    plt.savefig("runs/dse/dvs_energy.pdf")
    print(f"energies={energies}")

    #plt.figure(figsize=(3.5, 2.0))
    df = pd.DataFrame({"Energies": np.array(energies) * 1.0e3, "Latencies": np.array(latencies) * 1.0e3}, index=labels)
    #plt.bar(labels, np.array(energies) / energies[4], width=0.2)
    #plt.bar(labels, np.array(latencies) / latencies[4], width=0.2)

    df.plot.bar(rot=30, figsize=(3.6, 1.8), color=("#ff7f0e", "#1f77b4"), secondary_y="Latencies", legend=False)
    ax = plt.gca()
    plt.xlabel("Design Configuration")
    ax1, ax2 = plt.gcf().get_axes()
    ax1.minorticks_on()
    ax1.set_ylim((0, 0.5))
    ax2.minorticks_on()
    #legend = ax2.legend()
    #print(legend)
    ax1.set_ylabel("Total Energy (mJ)")
    ax2.set_ylabel("Total Run-time (ms)")
    ax1.tick_params(axis='x', which='minor', bottom=False)
    ax2.tick_params(axis='x', which='minor', bottom=False)
    #plt.xticks(rotation=90, ha="right")
    ax1.set_xticklabels(ax1.get_xticklabels(), ha="right")
    ax1.set_xlabel("Design Configuration")

    bars = ax1.containers
    bars += ax2.containers
    plt.legend(bars, ("Energy", "Latency"), loc="upper center")
    # Make the Loihi configuration bold
    for lab in ax1.get_xticklabels():
        if lab.get_text() == "<128c,1024n>":
            lab.set_fontweight("bold")
    plt.tight_layout(pad=0.3)
    ax1.set_position(ax_pos)
    ax2.set_position(ax_pos)
    plt.savefig("runs/dse/dvs_dse.pdf")

    plt.figure(figsize=(3.5, 2.0))
    plt.bar(labels, np.array(energies) * np.array(latencies))
    plt.xlabel("Configuration")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Energy Delay Product")
    plt.tight_layout(pad=0.3)
    plt.savefig("runs/dse/dvs_dse_product.png")

# TODO: range of tile configurations

"""
Copyright (c) 2023 - The University of Texas at Austin
This work was produced under contract #2317831 to National Technology and
Engineering Solutions of Sandia, LLC which is under contract
No. DE-NA0003525 with the U.S. Department of Energy.
"""
# External libraries, plotting
#import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Other external libraries
import numpy as np
import pandas as pd

# Python built-in libraries
import subprocess
import random
import csv
import yaml
import sys
import os

# SANA-FE libraries
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath((os.path.join(SCRIPT_DIR, os.pardir)))
sys.path.insert(0, PROJECT_DIR)
import sim

experiment = "nov19"
random_path = os.path.join(PROJECT_DIR, "runs", "random", experiment)

# Use a dumb seed to get consistent results
random.seed(1)

# Global experiment parameters
NETWORK_FILENAME = os.path.join(random_path, "random.net")
ARCH_FILENAME = os.path.join(PROJECT_DIR, "arch", "loihi.yaml")
LOIHI_CORES = 128
LOIHI_CORES_PER_TILE = 4
LOIHI_TILES = int(LOIHI_CORES / LOIHI_CORES_PER_TILE)
TIMESTEPS = 100

def create_random_network(cores, neurons_per_core, messages_per_neuron,
                          spikes_per_message):
    network = sim.Network(save_mappings=True)
    compartments = sim.init_compartments(LOIHI_TILES, LOIHI_CORES_PER_TILE,
                                           neurons_per_core)

    neurons = cores * neurons_per_core
    mappings = []
    for i in range(0, cores):
        m = (i/4, i%4)
        mappings.extend((m,) * neurons_per_core)

    print("Creating neuron population")
    population = sim.create_layer(network, neurons,
                                  compartments, log_spikes=0, log_potential=0,
                                  force_update=0, threshold=0.0, reset=0.0,
                                  leak=0.0, mappings=mappings)

    print("Generating randomized network connections")
    weight = 1.0
    print(f"Cores: {cores}, messages per neuron: {messages_per_neuron}")
    print(f"neurons per core: {neurons_per_core}, spikes per message: {spikes_per_message}")
    for n in range(0, neurons):
        src = population.neurons[n]
        # All neurons with outgoing connections should fire every timestep
        src.add_bias(1.0)
        if (n % 1024) == 0:
            print(f"Generating synaptic connections for neuron {n}")

        dest_core = random.sample(range(0, cores), messages_per_neuron)
        for c in dest_core:
            dest_neurons = random.sample(range(0, neurons_per_core),
                                         spikes_per_message)
            for d in dest_neurons:
                dest_id = (c * neurons_per_core) + d
                assert(dest_id < neurons)
                dest = population.neurons[dest_id]
                src.add_connection(dest, weight)

    network.save(NETWORK_FILENAME)


def run_sim(timesteps, cores, neurons_per_core, messages_per_core, spikes_per_message):
    create_random_network(cores, neurons_per_core, messages_per_core,
                          spikes_per_message)
    run_command = ("./sim", ARCH_FILENAME, NETWORK_FILENAME,
                   "{0}".format(timesteps))
    print("sana-fe command: {0}".format(" ".join(run_command)))
    subprocess.call(run_command)

    with open("run_summary.yaml", "r") as summary_file:
        summary = yaml.safe_load(summary_file)

    return summary


def onpick(event, df):
    N = len(event.ind)
    if not N:
        return
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.iloc[event.ind])


if __name__ == "__main__":
    run_experiments = True
    plot_experiments = True
    if run_experiments:
        with open(os.path.join(random_path, "loihi_random.csv"), "r") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames
            fieldnames.append("sim_energy")
            fieldnames.append("sim_latency")
            fieldnames.append("total_spikes")
            with open(os.path.join(random_path, "sim_random.csv"), "w") as out_file:
                writer = csv.DictWriter(out_file, fieldnames=fieldnames)
                writer.writeheader()

            for line in reader:
                results = sim.run(ARCH_FILENAME, line["network"], TIMESTEPS,
                                  perf_trace=True)
                print(results)
                df = pd.read_csv("perf.csv")
                line["total_spikes"] = df.loc[2, "fired"]
                #line["loihi_energy"] = float(line["loihi_energy"])
                #line["loihi_latency"] = float(line["loihi_latency"])
                line["sim_energy"] = results["energy"] / TIMESTEPS
                line["sim_latency"] = results["time"] / TIMESTEPS
                print(line)
                #with open("runs/random/tiny/sim_random.csv", "a") as out_file:
                with open(os.path.join(random_path, "sim_random.csv"), "a") as out_file:
                    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
                    writer.writerow(line)

    if plot_experiments:
        sim_df = pd.read_csv(os.path.join(random_path, "sim_random.csv"))
        loihi_df = pd.read_csv(os.path.join(random_path, "loihi_random.csv"))
        df = pd.merge(sim_df, loihi_df)
        # The smallest measurements hit the limits of Loihi's time measuring
        #  precision. Filter out these rows
        df = df[df["loihi_latency"] > 3.0e-6]
        #df = df[df["loihi_latency"] > 5.0e-6]
        # TODO: HACK, other filters
        #df = df[df["spiking_percentage"] == 100]
        #df = df[df["spikes_per_message"] <= 4]

        sim_energy = df["sim_energy"].values
        loihi_energy = df["loihi_energy"].values
        sim_latency = df["sim_latency"].values
        loihi_latency = df["loihi_latency"].values
        neurons_per_core = df["neurons_per_core"].values
        cores = df["cores"].values
        total_neurons = np.array(neurons_per_core * cores, dtype=float)
        total_spikes = df["total_spikes"].values
        spikes_per_message = df["spikes_per_message"]
        messages_per_neuron = df["messages_per_neuron"]

        #plt.rcParams.update({"font.size": 7, "lines.markersize": 3})
        plt.rcParams.update({"font.size": 7, "lines.markersize": 5})
        # Plot the simulated vs measured energy
        plt.figure(figsize=(3.0, 2.2))
        plt.minorticks_on()
        plt.gca().set_box_aspect(1)
        plt.xscale("log")
        plt.yscale("log")
        cm = plt.colormaps['coolwarm']
        plt.scatter(sim_energy, loihi_energy, marker="x", c=total_neurons,
                    cmap=cm, vmin=256, vmax=512, linewidths=1)
        plt.plot(np.linspace(min(sim_energy), max(sim_energy)),
                 np.linspace(min(sim_energy), max(sim_energy)),
                 "k--", alpha=0.5)
        plt.colorbar(label="Neurons", shrink=0.5)

        plt.minorticks_on()
        plt.xlabel("Simulated Energy (J)")
        plt.ylabel("Measured Energy (J)")
        plt.xticks((1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5))
        plt.yticks((1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5))
        plt.tight_layout(pad=0.3)
        plt.savefig(os.path.join(random_path, "random_energy.pdf"))
        plt.savefig(os.path.join(random_path, "random_energy.png"))

        # Plot the simulated vs measured latency
        fig = plt.figure(figsize=(3.0, 2.2))
        plt.minorticks_on()
        plt.gca().set_box_aspect(1)
        plt.xscale("log")
        plt.yscale("log")
        #plt.plot(sim_latency, loihi_latency, "x")
        plt.scatter(sim_latency, loihi_latency, marker="o", c=spikes_per_message,
                    #cmap=cm, vmin=256, vmax=512, linewidths=1,
                    cmap=cm, linewidths=1,
                    picker=True, pickradius=5)
                    #cmap=cm, vmin=256, vmax=8192, linewidths=1)

        plt.plot(np.linspace(min(sim_latency), max(sim_latency)),
                       np.linspace(min(sim_latency), max(sim_latency)), "k--")
        fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, df))
        pd.options.display.width = 300
        pd.options.display.max_colwidth = 300
        #plt.colorbar(label="Neurons", shrink=0.5)
        plt.colorbar(label="Spikes Per Message", shrink=0.5)
        plt.xlabel("Simulated Latency (s)")
        plt.ylabel("Measured Latency (s)")
        #plt.xticks((1.0e-6, 1.0e-5, 1.0e-4))
        #plt.yticks((1.0e-6, 1.0e-5, 1.0e-4))
        plt.tight_layout(pad=0.3)
        plt.savefig(os.path.join(random_path, "random_latency.pdf"))
        plt.savefig(os.path.join(random_path, "random_latency.png"))

        fig = plt.figure(figsize=(3.0, 2.2))
        plt.minorticks_on()
        plt.gca().set_box_aspect(1)
        plt.xscale("log")
        plt.yscale("log")
        #plt.plot(sim_latency, loihi_latency, "x")
        plt.scatter(sim_latency, loihi_latency, marker="o", c=cores,
                    #cmap=cm, vmin=256, vmax=512, linewidths=1,
                    cmap=cm, linewidths=1,
                    picker=True, pickradius=5)
                    #cmap=cm, vmin=256, vmax=8192, linewidths=1)

        plt.plot(np.linspace(min(sim_latency), max(sim_latency)),
                       np.linspace(min(sim_latency), max(sim_latency)), "k--")
        fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, df))
        pd.options.display.width = 300
        pd.options.display.max_colwidth = 300
        #plt.colorbar(label="Neurons", shrink=0.5)
        plt.colorbar(label="Cores", shrink=0.5)
        plt.xlabel("Simulated Latency (s)")
        plt.ylabel("Measured Latency (s)")
        #plt.xticks((1.0e-6, 1.0e-5, 1.0e-4))
        #plt.yticks((1.0e-6, 1.0e-5, 1.0e-4))
        plt.tight_layout(pad=0.3)
        plt.savefig(os.path.join(random_path, "random_latency_2.pdf"))
        plt.savefig(os.path.join(random_path, "random_latency_2.png"))

        fig = plt.figure(figsize=(3.0, 2.2))
        plt.minorticks_on()
        plt.gca().set_box_aspect(1)
        plt.xscale("log")
        plt.yscale("log")
        #plt.plot(sim_latency, loihi_latency, "x")
        plt.scatter(sim_latency, loihi_latency, marker="o", c=messages_per_neuron,
                    #cmap=cm, vmin=256, vmax=512, linewidths=1,
                    cmap=cm, linewidths=1,
                    picker=True, pickradius=5)
                    #cmap=cm, vmin=256, vmax=8192, linewidths=1)

        plt.plot(np.linspace(min(sim_latency), max(sim_latency)),
                       np.linspace(min(sim_latency), max(sim_latency)), "k--")
        fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, df))
        pd.options.display.width = 300
        pd.options.display.max_colwidth = 300
        #plt.colorbar(label="Neurons", shrink=0.5)
        plt.colorbar(label="Messages per neuron", shrink=0.5)
        plt.xlabel("Simulated Latency (s)")
        plt.ylabel("Measured Latency (s)")
        #plt.xticks((1.0e-6, 1.0e-5, 1.0e-4))
        #plt.yticks((1.0e-6, 1.0e-5, 1.0e-4))
        plt.tight_layout(pad=0.3)
        plt.savefig(os.path.join(random_path, "random_latency_3.pdf"))
        plt.savefig(os.path.join(random_path, "random_latency_3.png"))

        plt.show()

        absolute_latency_error = np.abs(loihi_latency - sim_latency) / loihi_latency
        absolute_energy_error = np.abs(loihi_energy - sim_energy) / loihi_energy

        print(df)
        print(np.sum(loihi_latency))
        print(np.sum(sim_latency))
        print(f"latency absolute mean error: {np.mean(absolute_latency_error) * 100.0}")
        print(f"energy absolute mean {np.mean(absolute_energy_error) * 100.0}")

        total_latency_error = (np.sum(loihi_latency) - np.sum(sim_latency)) / np.sum(loihi_latency)
        total_energy_error = (np.sum(loihi_energy) - np.sum(sim_energy)) / np.sum(loihi_energy)

        print(f"total latency error: {total_latency_error * 100.0} %")
        print(f"total energy error: {total_energy_error * 100.0} %")

# The old code for another experiment, where we measure simulator performance
#  across a range of SNNs. Currently just ignore all of this - I didn't figure
#  how to get meaningful plots
"""
# Run the simulation on SANA-FE, generating the network and immediately using it
#  Return the total runtime measured by Python, including setup and processing
#  time.

        cores = (1, 2, 4, 8, 16, 32, 64, 128)
        messages_per_neuron = (1, 2, 4, 8, 16, 32, 64, 128)
        spikes_per_message = (1, 4, 8)
        neurons_per_core = (8, 16, 32, 64, 128, 256, 512, 1024)

        # Part one is we try different numbers of cores and different numbers
        #  of messages per core
        data_points = []
        with open("runs/random/sanafe_perf.csv", "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(("cores", "neurons_per_core", "messages_per_neuron",
                             "spikes_per_message", "messages", "spikes",
                             "runtime"))
        for c in cores:
            for n in neurons_per_core:
                for m in messages_per_neuron:
                    if m <= c:
                        for s in spikes_per_message:
                            results = run_sim(TIMESTEPS, c, n, m, s)
                            row = (c, n, m, s, results["total_packets"],
                                results["total_spikes"],
                                results["wall_time"])
                            with open("runs/random/sanafe_perf.csv", "a") as csv_file:
                                writer = csv.writer(csv_file)
                                writer.writerow(row)
        print("Saved results to file")
"""


"""
def plot_results():
    df = pd.read_csv("runs/random/sanafe_perf.csv")
    plt.rcParams.update({'font.size': 14, 'lines.markersize': 5})

    plt.figure(figsize=(4.0, 4.0))
    for cores in df["cores"].unique():
        plt.plot(df.loc[(df["spikes_per_message"] == 1) & (df["neurons_per_core"] == 1024) & (df["cores"] == cores), "messages"],
                 df.loc[(df["spikes_per_message"] == 1) & (df["neurons_per_core"] == 1024) & (df["cores"] == cores), "runtime"],
                 "o-")

    legend_str = [f"{c} cores" for c in df["cores"].unique()]
    #plt.legend(legend_str, reverse=True)
    plt.legend(legend_str)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    plt.xlabel("Spike Messages")
    plt.ylabel("Run-time (s)")
    plt.tight_layout()
    plt.savefig("runs/random/sanafe_perf_1.png")
    plt.close()

    plt.figure(figsize=(4.0, 4.0))
    for spikes_per_message in df["spikes_per_message"].unique():
        plt.plot(df.loc[(df["spikes_per_message"] == spikes_per_message) & (df["neurons_per_core"] == 1024) & (df["cores"] == cores), "messages"],
                 df.loc[(df["spikes_per_message"] == spikes_per_message) & (df["neurons_per_core"] == 1024) & (df["cores"] == cores), "runtime"],
                 "o-")

    legend_str = [f"{s} spikes per message" for s in df["spikes_per_message"].unique()]
    plt.legend(legend_str)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    plt.xlabel("Spike Messages")
    plt.ylabel("Run-time (s)")
    plt.tight_layout()
    plt.savefig("runs/random/sanafe_perf_2.png")
    plt.close()

    plt.figure(figsize=(4.0, 4.0))
    for messages_per_neuron in (1, 4, 16):
        plt.plot(df.loc[(df["spikes_per_message"] == 4) & (df["neurons_per_core"] == 1024) & (df["messages_per_neuron"] == messages_per_neuron), "cores"],
                 df.loc[(df["spikes_per_message"] == 4) & (df["neurons_per_core"] == 1024) & (df["messages_per_neuron"] == messages_per_neuron), "runtime"],
                 "o-")

    legend_str = [f"{4*s} spikes / neuron" for s in (1, 4, 16)]
    plt.legend(legend_str)
    plt.xlabel("Loihi Core Count")
    plt.ylabel("Run-time (s)")
    plt.tight_layout()
    plt.savefig("runs/random/sanafe_perf_3.png")
    plt.close()

    plt.figure(figsize=(4.0, 4.0))
    #for messages_per_neuron in (1, 4, 16):
    #runtimes = np.array(df.loc[(df["spikes_per_message"] == 4) & (df["neurons_per_core"] == 1024) & (df["messages_per_neuron"] == messages_per_neuron), "runtime"])
    #neurons = np.array(df.loc[(df["spikes_per_message"] == 4) & (df["neurons_per_core"] == 1024) & (df["messages_per_neuron"] == messages_per_neuron), "cores"] * 1024)

    print("printing")
    runtimes = np.array(df.loc[((df["spikes_per_message"] == 1) &
    (df["neurons_per_core"] * df["messages_per_neuron"] * df["cores"] == 8192)), "runtime"])
    neurons = np.array(df.loc[((df["spikes_per_message"] == 1) &
    (df["neurons_per_core"] * df["messages_per_neuron"] * df["cores"] == 8192)), "cores"] * 1024)
    print(runtimes)
    print(neurons)
    print(runtimes / neurons)
    time_per_neuron = runtimes / neurons
    print("end print")

    plt.plot(neurons, time_per_neuron, "o")

    legend_str = [f"{4*s} spikes / neuron" for s in (1, 4, 16)]
    plt.legend(legend_str)
    plt.xlabel("Loihi Neuron Count")
    plt.ylabel("Run-time per Neuron (s)")
    plt.tight_layout()
    plt.savefig("runs/sanafe_perf_4.png")
    plt.close()

    plt.figure(figsize=(4.0, 4.0))
    for spikes_per_message in (1, 4, 8):
        runtimes = np.array(df.loc[(df["spikes_per_message"] == spikes_per_message) & (df["neurons_per_core"] == 1024) & (df["cores"] == 128), "runtime"])
        messages_per_neuron = np.array(df.loc[(df["spikes_per_message"] == spikes_per_message) & (df["neurons_per_core"] == 1024) & (df["cores"] == 128), "messages_per_neuron"])
        neurons = 128 * 1024
        print(messages_per_neuron)
        time_per_neuron = runtimes / neurons
        plt.plot(messages_per_neuron, time_per_neuron, "o-")

    legend_str = [f"{s} spikes per message" for s in (1, 4, 8)]
    plt.legend(legend_str)
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    plt.xlabel("Messages per Neuron")
    plt.ylabel("Run-time per Neuron (s)")
    plt.tight_layout()
    plt.savefig("runs/sanafe_perf_5.png")
    plt.close()

    return

"""

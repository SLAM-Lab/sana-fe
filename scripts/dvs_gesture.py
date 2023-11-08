"""
Copyright (c) 2023 - The University of Texas at Austin
This work was produced under contract #2317831 to National Technology and
Engineering Solutions of Sandia, LLC which is under contract
No. DE-NA0003525 with the U.S. Department of Energy.

Run DVS Gesture and extract some performance statistics
"""
#import matplotlib
#matplotlib.use('Agg')

import csv
import yaml
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath((os.path.join(SCRIPT_DIR, os.pardir)))

sys.path.insert(0, PROJECT_DIR)
import sim

ARCH_FILENAME = "loihi_dvs.yaml"
NETWORK_FILENAME = "dvs_gesture_32x32.net"
LOIHI_TIME_DATA_FILENAME = "loihi_gesture_32x32_time.csv"
LOIHI_ENERGY_DATA_FILENAME = "loihi_gesture_32x32_energy.csv"
SIM_TIME_DATA_FILENAME = "sim_gesture_32x32_time.csv"
SIM_ENERGY_DATA_FILENAME = "sim_gesture_32x32_energy.csv"

NETWORK_DIR = os.path.join(PROJECT_DIR, "runs", "dvs", "loihi_gesture_32x32")
DVS_RUN_DIR = os.path.join(PROJECT_DIR, "runs", "dvs")

ARCH_PATH = os.path.join(PROJECT_DIR, "arch", ARCH_FILENAME)
GENERATED_NETWORK_PATH = os.path.join(DVS_RUN_DIR, NETWORK_FILENAME)
LOIHI_TIME_DATA_PATH = os.path.join(DVS_RUN_DIR, LOIHI_TIME_DATA_FILENAME)
LOIHI_ENERGY_DATA_PATH = os.path.join(DVS_RUN_DIR, LOIHI_ENERGY_DATA_FILENAME)
SIM_TIME_DATA_PATH = os.path.join(DVS_RUN_DIR, SIM_TIME_DATA_FILENAME)
SIM_ENERGY_DATA_PATH = os.path.join(DVS_RUN_DIR, SIM_ENERGY_DATA_FILENAME)

def parse_stats(stats):
    print("Parsing statistics")
    total = stats.sum()
    pd.set_option('display.max_rows', None)
    analysis = {}
    analysis["times"] = stats.loc[:, "time"]
    analysis["hops"] = stats.loc[:, "hops"]
    analysis["fired"] = stats.loc[:, "fired"]
    analysis["packets"] = stats.loc[:, "packets"]
    analysis["total_energy"] = sum(stats.loc[:, "total_energy"])

    return analysis


def parse_loihi_spiketrains(total_timesteps):
    # Parse the CSV generated from the DVS gesture runs on Loihi
    #  The format is - first line is the neuron ID
    #  Second line is the timestep
    files = ("inputs.csv", "0Conv2D_15x15x16.csv", "1Conv2D_13x13x32.csv",
             "2Conv2D_11x11x64.csv", "3Conv2D_9x9x11.csv", "5Dense_11.csv")

    neurons = []
    timesteps = []

    for i in range(0, len(files)):
        f = files[i]
        path = os.path.join(DVS_RUN_DIR, "spiketrains", f)

        with open(path, "r") as spiketrain:
            reader = csv.reader(spiketrain)

            neurons += next(reader)
            timesteps += next(reader)

    spiketrain = {}
    for t in range(0, total_timesteps+1):
        spiketrain[t] = []

    for n, t in zip(neurons, timesteps):
        # Why are we -2 out of sync?
        #  Need to subtract 1, because the SNN toolbox starts from timestep 1,
        #   whereas my simulator goes from timestep 0
        t = int(t) - 2
        n = int(n)
        spiketrain[t].append(int(n))

    return spiketrain

if __name__ == "__main__":
    run_experiments = True
    plot_experiments = True
    experiment = "time"
    #experiment = "energy"

    neurons = []
    spiking_times = []
    spiking_update_energy = []
    spiking_spike_gen_energy = []
    spiking_synapse_energy = []
    spiking_network_energy = []
    times = np.array(())
    energies = np.array(())
    timesteps = 128
    frames = 100
    frame_increment = 1
    #frames = 1

    loihi_spiketrains = parse_loihi_spiketrains(timesteps)
    if run_experiments:
        neurons = ""
        groups = ""

        neuron_groups_filename = os.path.join(NETWORK_DIR, "neuron_groups.net")
        with open(neuron_groups_filename, "r") as group_file:
            group_data = group_file.read()

        snn_filename = os.path.join(NETWORK_DIR, "dvs_gesture.net")
        with open(snn_filename, "r") as snn_file:
            snn_data = snn_file.read()

        print("Reading mapping file")
        mappings_filename = os.path.join(NETWORK_DIR, "mappings.net")
        with open(mappings_filename, "r") as mappings_file:
            mapping_data = mappings_file.read()

        print("Reading input file")

        # Clear the data files
        if experiment == "energy":
            open(SIM_ENERGY_DATA_PATH, "w")
        elif experiment == "time":
            open(SIM_TIME_DATA_PATH, "w")

        for inputs in range(0, frames, frame_increment):
            print(f"Running for input: {inputs}")
            # First create the network file from the inputs and SNN
            input_filename = os.path.join(NETWORK_DIR, f"inputs{inputs}.net")
            with open(input_filename, "r") as input_file:
                input_data = input_file.read()

            data = (group_data + "\n" + input_data + "\n" + snn_data + "\n" +
                    mapping_data)
            with open(GENERATED_NETWORK_PATH, "w") as network_file:
                network_file.write(data)

            # Use a pre-generated network for a realistic use case i.e.
            #  dvs-gesture
            sim.run(ARCH_PATH, GENERATED_NETWORK_PATH, timesteps)
            # Parse the detailed perf statistics
            print("Reading performance data")
            stats = pd.read_csv(os.path.join(PROJECT_DIR, "perf.csv"))
            analysis = parse_stats(stats)
            times = np.append(times, analysis["times"])
            energies = np.append(energies, analysis["total_energy"] / timesteps)

            if experiment == "time":
                with open(SIM_TIME_DATA_PATH, "a") as time_file:
                    np.savetxt(SIM_TIME_DATA_PATH, times, delimiter=",")
            else:  # energy
                with open(SIM_ENERGY_DATA_PATH, "a") as energy_file:
                    np.savetxt(SIM_ENERGY_DATA_PATH, energies,
                               delimiter=",")

    if plot_experiments:
        """
        plt.figure(figsize=(5.5, 5.5))
        plt.bar(1, spiking_update_energy)
        plt.bar(2, spiking_spike_gen_energy, color="orange")
        plt.bar(3, spiking_synapse_energy, color="red")
        plt.bar(4, spiking_network_energy, color="green")
        plt.xticks([1,2,3,4], ["Update", "Spike Gen", "Synapse", "Network"])
        plt.ylabel("Energy (J)")
        plt.xlabel("Operation Type")
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
        plt.savefig("energy_breakdown.png")
        """
        # Plot the latency
        if experiment == "time":
            plt.rcParams.update({'font.size': 8, 'lines.markersize': 4})
            times = np.loadtxt(SIM_TIME_DATA_PATH, delimiter=",")
            loihi_data = pd.read_csv(LOIHI_TIME_DATA_PATH)
            #loihi_times = np.array(loihi_data.loc[:, "spiking"] / 1.0e6)
            loihi_times = np.array(loihi_data.loc[:, :] / 1.0e6)

            # There is a weird effect, that the first sample of all inputs > 1 is
            #  a 0 value. Just ignore the entries for both arrays (so we have
            #  timestep-1)
            times = np.delete(times,
                            list(range(timesteps, (timesteps*frames)//frame_increment, timesteps)))
            #loihi_times = np.delete(loihi_times,
            #                list(range(timesteps, timesteps*frames, timesteps)))

            total_times = np.zeros(frames // frame_increment)
            loihi_total_times = np.zeros(frames // frame_increment)
            for i in range(0, frames // frame_increment):
                total_times[i] = np.sum(times[i*(timesteps-1):(i+1)*(timesteps-1)])
                #loihi_total_times[i] = np.sum(loihi_times[i*(timesteps-1):(i+1)*(timesteps-1)])
                loihi_total_times[i] = np.sum(loihi_times[0:timesteps, i*frame_increment])

            """
            plt.figure(figsize=(7, 8))
            plt.subplot(311)
            plt.plot(np.arange(1, ((timesteps-1)*frames+1)), times[0:(timesteps-1)*frames], marker='x')
            plt.plot(np.arange(1, ((timesteps-1)*frames+1)), loihi_times[0:(timesteps-1)*frames], marker='x')
            plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
            plt.legend(("Simulated", "Measured on Loihi"))
            plt.ylabel("Latency (s)")
            plt.xlabel("Timestep")

            # Also plot underneath the different activity on the chip for both my
            #  simulator and Loihi
            plt.subplot(312)
            plt.plot(np.arange(1, timesteps+1), analysis["packets"], marker='x')
            plt.plot(np.arange(1, timesteps+1), analysis["hops"], marker='x')
            plt.legend(("Packets Sent", "Total Hops"))
            plt.xlabel("Timestep")

            plt.subplot(313)
            plt.plot(np.arange(1, timesteps+1), analysis["fired"], marker='x')
            # Figure out how many neurons fired in the Loihi data
            fired_count = [len(loihi_spiketrains[i]) for
                        i in range(0, len(loihi_spiketrains))]
            plt.plot(np.arange(1, timesteps+1), fired_count[0:timesteps], marker='x')
            plt.ylabel("Neurons Fired")
            plt.xlabel("Timestep")
            plt.legend(("Simulated", "Measured on Loihi"))

            print("diff = {}".format(
                analysis["fired"] - np.array(fired_count[0:timesteps])))
            plt.savefig("runs/dvs/dvs_gesture_sim_time2.png")
            """

            # Plot the latency
            times = np.loadtxt(SIM_TIME_DATA_PATH, delimiter=",")
            loihi_data = pd.read_csv(LOIHI_TIME_DATA_PATH)
            loihi_times = np.array(loihi_data.loc[:, :] / 1.0e6)
            times = np.delete(times,
                    list(range(timesteps-1, timesteps*frames//frame_increment, timesteps)))
            loihi_times = loihi_times[0:timesteps-1,:]
            plt.figure(figsize=(7.0, 1.7))

            ##plt.plot(np.arange(1, ((timesteps-1)*frames+1)), times[0:(timesteps-1)*frames], marker='x')
            ##plt.plot(np.arange(1, ((timesteps-1)*frames+1)), loihi_times[0:(timesteps-1), frames], marker='x')
            plt.rcParams.update({'font.size': 7})
            plt.plot(np.arange(1, timesteps-1), loihi_times[0:(timesteps-2), 0] * 1.0e6, "-")
            plt.plot(np.arange(1, timesteps-1), times[1:(timesteps-1)] * 1.0e6, "--x")
            plt.legend(("Measured on Loihi", "Simulated"), fontsize=7)
            plt.ylabel("Time-step Latency ($\mu$s)")
            plt.xlabel("Time-step")
            plt.yticks(np.arange(0, 61, 10))
            plt.tight_layout(pad=0.3)
            plt.savefig("runs/dvs/dvs_gesture_sim_time.pdf")
            plt.savefig("runs/dvs/dvs_gesture_sim_time.png")

            # Plot the correlation between simulated and measured time-step latency
            plt.figure(figsize=(1.7, 1.7))
            plt.minorticks_on()
            plt.gca().set_box_aspect(1)
            #plt.plot(times[0:frames*(timesteps-1)], loihi_times[0:frames*(timesteps-1)], "x")

            average_times = total_times / 128
            loihi_average_times = loihi_total_times / 128
            plt.rcParams.update({'font.size': 7, 'lines.markersize': 2})
            #plt.plot(average_times[0:frames] * 1.0e6, loihi_average_times[0:frames] * 1.0e6, "x")
            #plt.plot(np.linspace(min(average_times) * 1.0e6, max(average_times)) * 1.0e6,
            #         np.linspace(min(average_times) * 1.0e6, max(average_times)) * 1.0e6, "k--")

            plt.plot(average_times[0:frames]*1.0e6, loihi_average_times[0:frames]*1.0e6, "x")
            plt.plot(np.linspace(min(average_times)*1.0e6, max(average_times)*1.0e6),
                     np.linspace(min(average_times)*1.0e6, max(average_times)*1.0e6), "k--")
            #plt.xticks((1.0e-5, 1.5e-5, 2.0e-5, 2.5e-5, 3.0e-5))
            #plt.yticks((1.0e-5, 1.5e-5, 2.0e-5, 2.5e-5, 3.0e-5))
            plt.ylabel("Measured Latency ($\mu$s)")
            plt.xlabel("Simulated Latency ($\mu$s)")
            plt.xlim((10, 40))
            plt.ylim((10, 40))
            plt.xticks(np.arange(10, 41, 10))
            plt.yticks(np.arange(10, 41, 10))
            plt.tight_layout(pad=0.3)
            plt.savefig("runs/dvs/dvs_gesture_sim_correlation.pdf")
            plt.savefig("runs/dvs/dvs_gesture_sim_correlation.png")

            # Calculate total error
            relative_error = np.abs(loihi_total_times - total_times) / loihi_total_times
            mean_error = np.sum(relative_error) / len(relative_error)
            print("Time Absolute Mean error: {0} ({1} %)".format(mean_error, mean_error * 100))

            total_error =  (np.sum(loihi_total_times) - np.sum(total_times)) / np.sum(loihi_total_times)
            print("Time Total error: {0} ({1} %)".format(total_error, total_error * 100))

            """
            plt.plot(np.arange(1, timesteps+1), analysis["fired"], marker='x')
            # Figure out how many neurons fired in the Loihi data
            fired_count = [len(loihi_spiketrains[i]) for
                        i in range(0, len(loihi_spiketrains))]
            plt.plot(np.arange(1, timesteps+1), fired_count[0:timesteps], marker='x')
            plt.ylabel("Neurons Fired")
            plt.xlabel("Timestep")
            plt.legend(("Simulated", "Measured on Loihi"))

            print("diff = {}".format(
                analysis["fired"] - np.array(fired_count[0:timesteps])))
            plt.savefig("runs/dvs/dvs_gesture_sim_time2.png")
            """

        if experiment == "energy":
            plt.rcParams.update({'font.size': 7, 'lines.markersize': 2})
            loihi_data = pd.read_csv(LOIHI_ENERGY_DATA_PATH, delimiter=",")
            loihi_energies = np.array(loihi_data).flatten() * 1.0e6
            energies = np.loadtxt(SIM_ENERGY_DATA_PATH) * 1.0e6
            plt.figure(figsize=(1.7, 1.7))
            plt.minorticks_on()
            plt.gca().set_box_aspect(1)

            plt.plot(energies[0:frames], loihi_energies[0:frames], 'x')
            plt.plot(np.linspace(min(loihi_energies), max(loihi_energies)), np.linspace(min(loihi_energies), max(loihi_energies)), "k--")
            plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
            plt.xlim((1, 4))
            plt.ylim((1, 4))
            plt.ylabel("Measured Energy ($\mu$J)")
            plt.xlabel("Simulated Energy ($\mu$J)")
            plt.xticks(np.arange(1, 4.1, 1))
            plt.yticks(np.arange(1, 4.1, 1))
            plt.tight_layout(pad=0.5)
            plt.savefig("runs/dvs/dvs_gesture_sim_energy.png")
            plt.savefig("runs/dvs/dvs_gesture_sim_energy.pdf")

            relative_error = abs(loihi_energies[0:frames] - energies[0:frames]) / loihi_energies[0:frames]
            mean_error = sum(relative_error) / len(relative_error)
            print(relative_error)
            print(f"Energy Absolute Mean error: {mean_error} ({mean_error*100.0} %)")

            total_error = (sum(loihi_energies[0:frames]) - sum(energies[0:frames])) / sum(loihi_energies[0:frames])
            print(f"Energy Total error: {total_error} ({total_error*100.0} %)")
        """
        plt.figure()
        plt.plot(np.arange(1, timesteps+1), analysis["fired"], marker='x')
        # Figure out how many neurons fired in the Loihi data
        fired_count = [len(loihi_spiketrains[i]) for
                    i in range(0, len(loihi_spiketrains))]
        plt.plot(np.arange(1, timesteps+1), fired_count[0:timesteps], marker='x')
        plt.ylabel("Neurons Fired")
        plt.xlabel("Timestep")
        plt.legend(("Simulated", "Measured on Loihi"))
        print("diff = {}".format(
            analysis["fired"] - np.array(fired_count[0:timesteps])))
        plt.savefig("runs/dvs/dvs_gesture_spikes_i16.png")
        """
        exit()
        # These experiments not used for now
        # Plot the potential probes from simulation
        layers = ("inputs", "0Conv2D_15x15x16", "1Conv2D_13x13x32",
                "2Conv2D_11x11x64", "3Conv2D_9x9x11", "5Dense_11")
        layer_sizes = (1024, 3600, 5408, 7744, 891, 11)
        thresholds = (255, 293, 486, 510, 1729, 473)
        potential_data = pd.read_csv("probe_potential.csv")

        plt.rcParams.update({'font.size': 12, 'lines.markersize': 5})
        plot_neurons = {"inputs": [], "0Conv2D_15x15x16": [50, 67], "1Conv2D_13x13x32": [], "2Conv2D_11x11x64": [], "3Conv2D_9x9x11": [], "5Dense_11":[]}
        for layer_id, layer in enumerate(layers):
            layer_path = "runs/dvs/potentials/{0}".format(layer)
            if not os.path.exists(layer_path):
                os.makedirs(layer_path)

            for neuron_id in plot_neurons[layer]:
                potentials = potential_data.loc[:, "{0}.{1}".format(
                    layer_id, neuron_id)]
                plt.figure(figsize=(5.5, 5.5))
                plt.plot(np.arange(1, timesteps+1), potentials*64, "-x")
                plt.plot(np.arange(1, timesteps+1),
                        np.ones(timesteps) * thresholds[layer_id] * 64)
                #plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
                plt.ylabel("Membrane Potential")
                plt.xlabel("Time-step")
                #plt.ylim((-0.2, 1.0))
                plt.tight_layout()
                plt.savefig("{0}/probe_potential_{1}.png".format(layer_path,
                                                            neuron_id))
                plt.close()

        with open("run_summary.yaml", "r") as results_file:
            results = yaml.safe_load(results_file)
        #network_percentage = (results["network_time"] /
        #                                    results["time"]) * 100.0
        #print("Percentage of time used for only network: {0}".format(
        #      network_percentage))

        #plt.show()

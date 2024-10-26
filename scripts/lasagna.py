import csv
import torch
import os
import sys
import dill
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Try importing the installed sanafe library. If not installed, require a
#  fall-back to a local build of the sanafe Python library
try:
    import sanafe
except ImportError:
    # Not installed, fall-back to local build
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.abspath((os.path.join(SCRIPT_DIR, os.pardir)))
    sys.path.insert(0, PROJECT_DIR)
    import sanafe

timesteps = 25
num_inputs = 10
print("Loading models")
mnist_model = torch.load(
        os.path.join(PROJECT_DIR, "etc", "mnist.pt"),
        pickle_module=dill,
        map_location=torch.device("cpu"))
spiking_digits_model = torch.load(
        os.path.join(PROJECT_DIR, "etc", "spiking_digits.pt"),
        pickle_module=dill,
        map_location=torch.device("cpu"))
mnist_weights = {}
for param_name, param in mnist_model.named_parameters():
    mnist_weights[param_name] = param.detach().numpy()

mnist_inputs = np.loadtxt(os.path.join(PROJECT_DIR, "etc", "mnist.csv"),
                          delimiter=',')
#spiking_digits_inputs = np.loadtxt(os.path.join(PROJECT_DIR, "etc", "spiking_digits.npz"))

# PyTorch stores weights in an array with dims (num out x num in)
in_neurons = mnist_weights["fc1.weight"].shape[1]
hidden_neurons = mnist_weights["fc1.weight"].shape[0]
out_neurons = mnist_weights["fc2.weight"].shape[0]

#"""
# Load the LASAGNA architecture with analog neurons
arch = sanafe.load_arch("/home/james/code/lasagne/lasagne.yaml")

print("Creating network in SANA-FE")
network = sanafe.Network()
# TODO: mapping order still seems messed up..
print(f"in:{in_neurons} hidden:{hidden_neurons} out:{out_neurons}")
in_layer = network.create_neuron_group("in", in_neurons)
hidden_layer = network.create_neuron_group("hidden", hidden_neurons,
                                           soma_hw_name="soma")
out_layer = network.create_neuron_group("out", out_neurons, soma_hw_name="soma")

analog_neurons = True
for id, neuron in enumerate(in_layer.neurons):
    neuron.set_attributes(soma_hw_name=f"input[{id}]",
                          log_spikes=True)
    neuron.map_to_core(arch.tiles[0].cores[0])

for id, neuron in enumerate(hidden_layer.neurons):
    neuron.set_attributes(soma_hw_name=f"loihi", log_spikes=True,
                          model_parameters={"threshold": 1.0, "leak_decay": 0.85})
    if analog_neurons:
        neuron.set_attributes(soma_hw_name=f"analog_lif[{id}]", log_spikes=True)
    else:
        neuron.map_to_core(arch.tiles[0].cores[0])
    neuron.map_to_core(arch.tiles[0].cores[0])

for id, neuron in enumerate(out_layer.neurons):
    if analog_neurons:
        neuron.set_attributes(soma_hw_name=f"analog_lif[{id + hidden_neurons}]",
                              log_spikes=True,
                              log_potential=True)
    else:
        neuron.set_attributes(soma_hw_name=f"loihi",
                              log_spikes=True,
                              log_potential=True,
                              model_parameters={"threshold": 1.0, "leak_decay": 0.85})
    neuron.map_to_core(arch.tiles[0].cores[0])

# Connect neurons in both layers
for src in range(0, in_neurons):
    for dst in range(0, hidden_neurons):
        weight = mnist_weights["fc1.weight"][dst, src]
        if weight > 0.0:
            network.groups["in"].neurons[src].connect_to_neuron(
                network.groups["hidden"].neurons[dst], {"weight": weight})

for src in range(0, hidden_neurons):
    for dst in range(0, out_neurons):
        weight = mnist_weights["fc2.weight"][dst, src]
        if weight > 0.0:
            network.groups["hidden"].neurons[src].connect_to_neuron(
                network.groups["out"].neurons[dst], {"weight": weight})

# Run a simulation
print("Building h/w")
hw = sanafe.SpikingChip(arch, record_spikes=True, record_potentials=True)
hw.load(network)
print(f"Running simulation for {timesteps} timesteps")

mapped_inputs = hw.mapped_neuron_groups["in"]
for input in range(0, num_inputs):
    print(f"Simulating input: {input}")
    mnist_input = mnist_inputs[input, :]
    #plt.figure()
    #plt.imshow(mnist_input.reshape(28, 28), cmap="gray")
    #plt.colorbar()
    for id, mapped_neuron in enumerate(mapped_inputs):
        mapped_neuron.set_attributes(model_parameters={"poisson": mnist_input[id]})
    results = hw.sim(timesteps)
    print(results)
    hw.reset()
#"""

# Read in simulation results
with open("spikes.csv") as spike_csv:
    spike_data = csv.DictReader(spike_csv)

    in_spikes = [[] for _ in range(0, in_neurons)]
    hidden_spikes = [[] for _ in range(0, hidden_neurons)]
    out_spikes = [[] for _ in range(0, out_neurons)]
    for spike in spike_data:
        # Spike entry has the format <group_name.neuron_id,timestep>
        timestep, neuron = int(spike["timestep"]), spike["neuron"]
        group_name, neuron_id = neuron.split(".")
        neuron_id = int(neuron_id)
        # Track spikes for all three layers
        if group_name == "in":
            in_spikes[neuron_id].append(timestep)
        elif group_name == "hidden":
            hidden_spikes[neuron_id].append(timestep)
        elif group_name == "out":
            out_spikes[neuron_id].append(timestep)
        else:
            print(f"Warning: Group {group_name} not recognized!")

# 1) Count the total output spikes TODO: update now we have multiple input sequences
counts = [sum(spikes) for spikes in out_spikes]
print(f"Spike counts per class:{counts}")

# 2) Create a raster plot of all spikes
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(12.0, 8.0))
gs = GridSpec(2, 1, height_ratios=[1, 4], hspace=0.3)

# Top subplot for MNIST digits
ax_digits = fig.add_subplot(gs[0])
ax_digits.set_xticks([])
ax_digits.set_yticks([])

# Calculate positions for each digit
digit_width = 28  # MNIST digits are 28x28
time_per_digit = timesteps  # Each digit is presented for 50 timesteps
num_digits = len(mnist_inputs)

# Create a blank canvas for all digits
total_timesteps = timesteps * num_inputs
display_height = 8  # Adjust this factor to change digit height

# Place each digit at its corresponding time position
for i, digit in enumerate(mnist_inputs[0:num_inputs, :]):
    start_time = i * time_per_digit
    # Calculate extent for each digit: [left, right, bottom, top]
    # Width of each digit display is set to match the time window
    digit_extent = [start_time, start_time + time_per_digit, 0, display_height]
    ax_digits.imshow(digit.reshape(digit_width, digit_width),
                     cmap='gray', aspect='auto', extent=digit_extent)

# Display the digits
ax_digits.set_xlim(0, total_timesteps)
ax_digits.set_ylim(0, display_height)
ax_digits.set_title('Input MNIST Digits')

ax_spikes = fig.add_subplot(gs[1])
ax_spikes.set_xlim((0, total_timesteps))
ax_spikes.set_ylim((0, in_neurons + hidden_neurons + out_neurons))

total_neurons = 0
for neuron_id in range(0, in_neurons):
    ax_spikes.scatter(in_spikes[neuron_id],
                [total_neurons]*len(in_spikes[neuron_id]), c='r', s=2,
                marker='.', linewidths=0.5)
    total_neurons += 1

for neuron_id in range(0, hidden_neurons):
    ax_spikes.scatter(hidden_spikes[neuron_id],
                [total_neurons]*len(hidden_spikes[neuron_id]), c='b', s=2,
                marker='.', linewidths=0.5)
    total_neurons += 1

for neuron_id in range(0, out_neurons):
    ax_spikes.scatter(out_spikes[neuron_id],
                [total_neurons]*len(out_spikes[neuron_id]), c='k', s=2,
                marker='.', linewidths=0.5)
    total_neurons += 1

# Add vertical lines to show digit presentation boundaries
for i in range(num_inputs + 1):
    ax_spikes.axvline(x=i * time_per_digit, color='gray', linestyle='--', alpha=0.3)

ax_spikes.set_xlabel("Time-step")
ax_spikes.set_ylabel("Neuron")
plt.savefig("runs/lasagna/raster.png")

# 3) Create a potential plot of the output layer
fig = plt.figure(figsize=(12.0, 8.0))
gs = GridSpec(2, 1, height_ratios=[1, 4], hspace=0.3)

# Top subplot for MNIST digits
ax_digits = fig.add_subplot(gs[0])
ax_digits.set_xticks([])
ax_digits.set_yticks([])


# Create a blank canvas for all digits
display_height = 8  # Adjust this factor to change digit height

# Place each digit at its corresponding time position
for i, digit in enumerate(mnist_inputs[0:num_inputs, :]):
    start_time = i * time_per_digit
    # Calculate extent for each digit: [left, right, bottom, top]
    # Width of each digit display is set to match the time window
    digit_extent = [start_time, start_time + time_per_digit, 0, display_height]
    ax_digits.imshow(digit.reshape(digit_width, digit_width),
                     cmap='gray', aspect='auto', extent=digit_extent)

ax_digits.set_xlim(0, total_timesteps)
ax_digits.set_ylim(0, display_height)

# Plot the output neuron potentials
ax_potentials = fig.add_subplot(gs[1])
ax_potentials.set_xlim((0, total_timesteps))
potentials_df = pd.read_csv("potential.csv")
potentials_df.plot(x="timestep", y=["neuron out.0", "neuron out.1", "neuron out.2", "neuron out.3", "neuron out.4", "neuron out.5", "neuron out.6", "neuron out.7", "neuron out.8", "neuron out.9"], ax=ax_potentials)
# Add vertical lines to show digit presentation boundaries
for i in range(num_inputs + 1):
    ax_potentials.axvline(x=i*time_per_digit + 1, color='gray', linestyle='--', alpha=0.3)

# 4) Create a combined raster/potential plot of the hidden and output layers


print("Finished.")
plt.show()

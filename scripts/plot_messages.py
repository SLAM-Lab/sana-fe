import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

def tile_idx(x, y):
    return (x * 4) + y


def track_hops(src_x, src_y, dest_x, dest_y):
    # Loihi is 8x4 grid (8 on the x axis)
    spikes_processed_per_router = np.zeros(8*4, dtype=int)

    # Add spike to tile that generates the spike
    spikes_processed_per_router[tile_idx(src_x, src_y)] += 1

    # Account for all the hops in between the src and dest tile
    while src_x < dest_x:
        src_x += 1
        spikes_processed_per_router[tile_idx(src_x, src_y)] += 1

    while src_x > dest_x:
        src_x -= 1
        spikes_processed_per_router[tile_idx(src_x, src_y)] += 1

    while src_y < dest_y:
        src_y += 1
        spikes_processed_per_router[tile_idx(src_x, src_y)] += 1

    while src_y > dest_y:
        src_y -= 1
        spikes_processed_per_router[tile_idx(src_x, src_y)] += 1

    return spikes_processed_per_router


hops = np.zeros(32, dtype=int)
with open("messages.trace") as trace:
    reader = csv.DictReader(trace)
    for line in reader:
        hops += track_hops(int(line["src_x"]), int(line["src_y"]),
                           int(line["dest_x"]), int(line["dest_y"]))

plt.figure()

# Plot a heat map
x, y = np.meshgrid(np.linspace(0, 7, 8), np.linspace(0, 3, 4))
# Plot the circles

plt.scatter(x, y, c=hops, cmap="hot")
plt.colorbar()

df = pd.read_csv("messages.trace")
plt.figure()
plt.hist(df["hops"])

plt.figure()
plt.hist(df["blocking_latency"], bins=20)

plt.show()

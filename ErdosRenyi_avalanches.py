import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# --- Parameters ---
total_neurons = 1000      # total neurons in network
exc_ratio = 0.8           # fraction excitatory (80% E, 20% I)
Ne = int(total_neurons * exc_ratio)
Ni = total_neurons - Ne
simulation_time = 10000     # ms
dt = 1                   # ms per step
time_steps = int(simulation_time / dt)

# Izhikevich neuron parameters
params_exc = (0.02, 0.2, -65, 8)   # a, b, c, d for excitatory
params_inh = (0.1, 0.2, -65, 2)    # a, b, c, d for inhibitory

# List of connection probabilities to compare
p_connect_list = [0.01, 0.05, 0.1, 0.5, 0.7, 0.9]
# Define a set of distinct marker shapes for each p
marker_shapes = ['o', 's', '^', 'D', 'v', 'x']

# Function to run one simulation and detect avalanche sizes and durations
def run_simulation_and_detect(p_connect):
    # Build random network using erdos-renyi
    G = nx.erdos_renyi_graph(total_neurons, p_connect, seed=42)
    # Assign neuron types
    neuron_types = ['excitatory'] * Ne + ['inhibitory'] * Ni
    np.random.shuffle(neuron_types)
    # Initialize state variables
    v = np.full(total_neurons, -65.0)
    u = np.zeros(total_neurons)
    spike_times = []
    # Parameter arrays based on type
    a = np.array([params_exc[0] if t=='excitatory' else params_inh[0] for t in neuron_types])
    b = np.array([params_exc[1] if t=='excitatory' else params_inh[1] for t in neuron_types])
    c = np.array([params_exc[2] if t=='excitatory' else params_inh[2] for t in neuron_types])
    d = np.array([params_exc[3] if t=='excitatory' else params_inh[3] for t in neuron_types])
    # Synaptic weights
    weights = np.zeros((total_neurons, total_neurons))
    for i in range(total_neurons):
        for j in G.neighbors(i):
            weights[i, j] = np.random.uniform(0, 0.5) if neuron_types[j] == 'excitatory' else np.random.uniform(-0.5, 0)
    # Simulation loop
    for t in range(time_steps):
        I = np.random.normal(5, 2, total_neurons)
        fired = np.where(v >= 30)[0]
        spike_times.append(fired.size)
        v[fired] = c[fired]
        u[fired] += d[fired]
        I += np.sum(weights[:, fired], axis=1)
        v += dt * (0.04 * v**2 + 5 * v + 140 - u + I)
        u += dt * a * (b * v - u)

    # Detect avalanche sizes and durations
    sizes = []
    durations = []
    current_size = 0
    current_dur = 0
    for count in spike_times:
        if count > 0:
            current_size += count
            current_dur += 1
        else:
            if current_dur > 0:
                sizes.append(current_size)
                durations.append(current_dur)
                current_size = 0
                current_dur = 0
    if current_dur > 0:
        sizes.append(current_size)
        durations.append(current_dur)
    return sizes, durations

# Run for each p_connect and collect sizes and durations
all_sizes = {}
all_durations = {}
for p, marker in zip(p_connect_list, marker_shapes):
    sizes, durations = run_simulation_and_detect(p)
    all_sizes[p] = sizes
    all_durations[p] = durations

# Determine common bins in log-space for sizes
max_size = max(max(s) for s in all_sizes.values())
bins_size = np.logspace(np.log10(1), np.log10(max_size), 50)
# and for durations
max_dur = max(max(d) for d in all_durations.values())
bins_dur = np.logspace(np.log10(1), np.log10(max_dur), 50)

# Plot avalanche size distributions
plt.figure(figsize=(8, 6))
for marker, p in zip(marker_shapes, p_connect_list):
    hist, bin_edges = np.histogram(all_sizes[p], bins=bins_size, density=True)
    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    plt.plot(centers, hist, marker=marker, linestyle='None', markersize=4, label=f'p={p}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Avalanche Size')
plt.ylabel('Probability Density')
plt.title('Avalanche Size Distribution for Various Connection Probabilities')
plt.legend(title='p_connect')
plt.show()

# Plot avalanche duration distributions
plt.figure(figsize=(8, 6))
for marker, p in zip(marker_shapes, p_connect_list):
    hist, bin_edges = np.histogram(all_durations[p], bins=bins_dur, density=True)
    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    plt.plot(centers, hist, marker=marker, linestyle='None', markersize=4, label=f'p={p}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Avalanche Duration (time steps)')
plt.ylabel('Probability Density')
plt.title('Avalanche Duration Distribution for Various Connection Probabilities')
plt.legend(title='p_connect')
plt.show()

# --- Power-law exponent estimation ---
def estimate_power_law_exponent(data, x_min=1):
    """
    Estimate power-law exponent alpha via MLE for data >= x_min.
    alpha = 1 + n / sum(log(x_i/x_min)).
    Returns (alpha, n_samples).
    """
    data = np.array(data)
    filtered = data[data >= x_min]
    n = filtered.size
    if n < 2:
        return np.nan, n
    sum_log = np.sum(np.log(filtered / x_min))
    alpha = 1 + n / sum_log
    return alpha, n

# Calculate and print exponents for each p_connect
print("Power-law exponents for avalanche SIZE distributions:")
for p in p_connect_list:
    alpha_s, n_s = estimate_power_law_exponent(all_sizes[p], x_min=1)
    print(f"p={p}: alpha = {alpha_s:.2f}, n = {n_s}")

print("\nPower-law exponents for avalanche DURATION distributions:")
for p in p_connect_list:
    alpha_d, n_d = estimate_power_law_exponent(all_durations[p], x_min=1)
    print(f"p={p}: alpha = {alpha_d:.2f}, n = {n_d}")

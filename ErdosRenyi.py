import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists('figures'):
    os.makedirs('figures')

np.random.seed(0)

# Simulation parameters
T = 1000
Ne = 800
Ni = 200

# Initialize neuron parameters
re = np.random.rand(Ne, 1)
ri = np.random.rand(Ni, 1)

p_RS = [0.02, 0.2, -65, 8]
p_LTS = [0.02, 0.25, -65, 2]

a = np.vstack((p_RS[0]*np.ones((Ne, 1)), p_LTS[0]+0.08*ri))
b = np.vstack((p_RS[1]*np.ones((Ne, 1)), p_LTS[1]-0.05*ri))
c = np.vstack((p_RS[2]+15*re**2, p_LTS[2]*np.ones((Ni, 1))))
d = np.vstack((p_RS[3]-6*re**2, p_LTS[3]*np.ones((Ni, 1))))

# Function to create a random graph
def create_erdos_renyi_graph(N, p_connect):
    return np.random.rand(N, N) < p_connect

# Vary connection probability (p_connect) and synaptic weights
connection_probabilities = [0.1, 0.3, 0.5]
weights = [0.5, 1.0, 1.5]

# Initialize data for the plots
coherence_data = {}
firing_rate_data = {}

for p_connect in connection_probabilities:
    for weight in weights:
        S = create_erdos_renyi_graph(Ne + Ni, p_connect)
        S = S * weight

        # Initial neuron variables
        v = -65 * np.ones((Ne + Ni, 1))
        u = b * v
        firings = []

        # Simulation loop
        for t in range(T):
            I = np.vstack((5 * np.random.randn(Ne, 1), 2 * np.random.randn(Ni, 1)))
            if t > 0 and len(fired) > 0:
                I += np.sum(S[:, fired], axis=1).reshape(-1, 1)

            v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
            v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
            u += a * (b * v - u)

            fired = np.where(v >= 30)[0]
            if fired.size > 0:
                firings.extend([[t, idx] for idx in fired])
                v[fired] = c[fired]
                u[fired] += d[fired]

        firings = np.array(firings)


        time_bin = 50
        num_bins = T // time_bin
        binary_spikes = np.zeros((Ne + Ni, T))
        binary_spikes[firings[:, 1].astype(int), firings[:, 0].astype(int)] = 1
        global_coherence = np.zeros(num_bins)

        for idx in range(num_bins):
            start, end = idx * time_bin, (idx + 1) * time_bin
            spike_slice = binary_spikes[:, start:end]
            spike_counts = spike_slice.sum(axis=1)
            active_neurons = spike_counts > 0

            if np.sum(active_neurons) > 1:
                active_slice = spike_slice[active_neurons]
                numerator = np.dot(active_slice, active_slice.T)
                denominator = np.sqrt(np.outer(spike_counts[active_neurons], spike_counts[active_neurons]))
                np.fill_diagonal(denominator, 1)
                coherence_matrix = numerator / denominator
                np.fill_diagonal(coherence_matrix, 0)
                global_coherence[idx] = coherence_matrix.sum() / (np.sum(active_neurons) * (np.sum(active_neurons) - 1))

        coherence_time = np.arange(num_bins) * time_bin

        label = f'p={p_connect}, w={weight}'
        coherence_data[label] = (coherence_time, global_coherence)

        pop_firing_rate = binary_spikes.sum(axis=0) / (Ne + Ni)

        firing_rate_data[label] = (np.arange(T), pop_firing_rate)

plt.figure()
for label, (time, coherence) in coherence_data.items():
    plt.plot(time, coherence, label=label)

plt.xlabel('Time (ms)')
plt.ylabel('Global Coherence')
plt.title('Global Coherence vs. Time')
plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/global_coherence.png', dpi=120)
plt.show()

plt.figure()
for label, (time, rate) in firing_rate_data.items():
    plt.plot(time, rate, label=label)

plt.xlabel('Time (ms)')
plt.ylabel('Population Firing Rate')
plt.title('Population Firing Rate vs. Time')
plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/population_firing_rate.png', dpi=120)
plt.show()

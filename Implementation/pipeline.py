import pennylane as qp
from pennylane import numpy as np


dev = qp.device("default.qubit", wires=2)
num_qubits = 2
num_layers = 2


@qp.qnode(dev)
def ansatz(params, observable):
    qp.StronglyEntanglingLayers(weights=params, wires=range(num_qubits))
    return qp.expval(observable)


def run_vqa(initial_params, observable, optimiser=qp.GradientDescentOptimizer, step_size=0.01, steps=10000):
    optimiser = optimiser(stepsize=step_size)
    params = initial_params

    def cost_fn(params):
        return 1 - ansatz(params, observable)

    for _ in range(steps):
        params = optimiser.step(cost_fn, params)

    return params


@qp.qnode(dev)
def get_learned_state(params):
    qp.StronglyEntanglingLayers(weights=params, wires=range(num_qubits))
    return qp.state()


if __name__ == "__main__":
    # Define the target state as the Bell State |Φ+⟩ = (|00⟩ + |11⟩) / √2
    target_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    projector = np.outer(target_state, np.conj(target_state))   # |Φ+⟩⟨Φ+|
    observable_matrix = qp.Hermitian(projector, wires=range(num_qubits))

    # Run the VQA
    weights_shape = qp.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_qubits)
    initial_params = np.random.uniform(low=0, high=2 * np.pi, size=weights_shape, requires_grad=True)
    final_params = run_vqa(initial_params, observable_matrix)
    learned_wavefunction = get_learned_state(final_params)

    print("\nLEARNED WAVEFUNCTION")
    print("Target Wavefunction:  ", target_state)
    print("Learned Wavefunction: ", np.round(learned_wavefunction, decimals=4))

    # Calculate and print the physical probabilities for each basis state
    learned_probabilities = np.abs(learned_wavefunction) ** 2
    target_probabilities = np.abs(target_state) ** 2

    print("\nLEARNED PROBABILITIES")
    for i in range(2 ** num_qubits):
        basis_state = format(i, f'0{num_qubits}b')
        print(f"|{basis_state}⟩ | Target: {target_probabilities[i]:.1f} | Learned: {learned_probabilities[i]:.8f}")

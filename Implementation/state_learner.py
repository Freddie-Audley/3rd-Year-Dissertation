import pennylane as qp
from pennylane import numpy as np


class VariationalQuantumStateLearner:
    def __init__(self, num_qubits=2, num_layers=2):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.device = qp.device("default.qubit", wires=self.num_qubits)
        self._ansatz_qnode = qp.QNode(self._ansatz_circuit, self.device)
        self._state_qnode = qp.QNode(self._state_circuit, self.device)
        self.initialize_parameters()

    def _ansatz_circuit(self, params, observable):
        qp.StronglyEntanglingLayers(weights=params, wires=range(self.num_qubits))
        return qp.expval(observable)

    def _state_circuit(self, params):
        qp.StronglyEntanglingLayers(weights=params, wires=range(self.num_qubits))
        return qp.state()

    def initialize_parameters(self):
        weights_shape = qp.StronglyEntanglingLayers.shape(n_layers=self.num_layers, n_wires=self.num_qubits)
        self.params = np.random.uniform(low=0, high=2*np.pi, size=weights_shape, requires_grad=True)
        return self.params

    def train(self, observable, optimiser=qp.GradientDescentOptimizer, step_size=0.01, steps=100):
        optimiser = optimiser(stepsize=step_size)

        def cost_fn(params):
            return 1 - self._ansatz_qnode(params, observable)

        for _ in range(steps):
            self.params = optimiser.step(cost_fn, self.params)

        return self.params

    def get_state(self):
        return self._state_qnode(self.params)


if __name__ == "__main__":
    learner = VariationalQuantumStateLearner()

    # Define the target state as the Bell State |Φ+⟩ = (|00⟩ + |11⟩) / √2
    target_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    projector = np.outer(target_state, np.conj(target_state))  # |Φ+⟩⟨Φ+|
    observable_matrix = qp.Hermitian(projector, wires=range(learner.num_qubits))

    learner.train(observable_matrix, steps=1000)
    learned_wavefunction = learner.get_state()

    print("\nLEARNED WAVEFUNCTION")
    print("Target Wavefunction:  ", target_state)
    print("Learned Wavefunction: ", np.round(learned_wavefunction, decimals=4))

    learned_probabilities = np.abs(learned_wavefunction) ** 2
    target_probabilities = np.abs(target_state) ** 2

    print("\nLEARNED PROBABILITIES")
    for i in range(2 ** learner.num_qubits):
        basis_state = format(i, f"0{learner.num_qubits}b")
        print(f"|{basis_state}⟩ | Target: {target_probabilities[i]:.1f} | Learned: {learned_probabilities[i]:.8f}")

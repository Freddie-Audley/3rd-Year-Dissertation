import pennylane as qp
from pennylane import numpy as np

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt


def default_encoding(features, num_qubits):
    for i in range(num_qubits):
        qp.RY(features[i], wires=i)


class SupervisedQuantumLearner:
    def __init__(self, num_qubits, num_layers, encoding=default_encoding, ansatz=qp.StronglyEntanglingLayers, observable=qp.Z(wires=0), seed=None):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.observable = observable
        self.ansatz = ansatz
        self.encoding = encoding
        self.device = qp.device("default.qubit", wires=self.num_qubits)
        self._qnode = qp.QNode(self._circuit, self.device)
        self.loss_history = []
        self._rng = np.random.default_rng(seed)

        self.initialise_parameters()

    def _circuit(self, features, parameters):
        self.encoding(features, self.num_qubits)

        self.ansatz(weights=parameters, wires=range(self.num_qubits))

        return qp.expval(self.observable)

    def initialise_parameters(self):
        weights_shape = self.ansatz.shape(n_layers=self.num_layers, n_wires=self.num_qubits)
        self.parameters = self._rng.uniform(low=0, high=2*np.pi, size=weights_shape, requires_grad=True)
        return self.parameters

    def predict(self, features, parameters=None):
        if parameters is None:
            parameters = self.parameters
        return self._qnode(features, parameters)

    def train(self, feature_vectors, targets, optimiser=qp.GradientDescentOptimizer, cost_function=None, step_size=0.01, steps=100):
        if cost_function is None:
            cost_function = lambda predictions, targets: np.mean((predictions - targets) ** 2)  # Mean Squared Error

        optimiser = optimiser(stepsize=step_size)

        def objective_function(parameters):
            predictions = np.array([self.predict(features, parameters) for features in feature_vectors])
            return cost_function(predictions, targets)

        self.loss_history = []

        for _ in range(steps):
            self.parameters = optimiser.step(objective_function, self.parameters)
            self.loss_history.append(objective_function(self.parameters))

        return self.parameters

    def evaluate(self, feature_vectors, targets, cost_function=None):
        if cost_function is None:
            cost_function = lambda predictions, targets: np.mean((predictions - targets) ** 2)

        predictions = np.array([self.predict(features) for features in feature_vectors])
        return cost_function(predictions, targets)


if __name__ == "__main__":
    # Demo: f(x1, x2) = (cos(x1) + cos(x2)) / 2
    
    learner = SupervisedQuantumLearner(num_qubits=2, num_layers=2, observable=qp.Z(wires=0))

    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    x1, x2 = np.meshgrid(angles, angles, indexing="ij")
    feature_vectors = np.column_stack((x1.flatten(), x2.flatten()))
    targets = (np.cos(feature_vectors[:, 0]) + np.cos(feature_vectors[:, 1])) / 2

    rng = np.random.default_rng(seed=261104)
    indices = rng.permutation(len(feature_vectors))
    split = int(0.8 * len(feature_vectors))
    train_indices = indices[:split]
    test_indices = indices[split:]

    X_train, y_train = feature_vectors[train_indices], targets[train_indices]
    X_test, y_test = feature_vectors[test_indices], targets[test_indices]

    learner.train(X_train, y_train, steps=1000)

    train_loss = learner.evaluate(X_train, y_train)
    test_loss = learner.evaluate(X_test, y_test)

    print("Training Loss:", train_loss)
    print("Test Loss:", test_loss)
    print("Generalisation Gap:", test_loss - train_loss)

    plt.plot(learner.loss_history)
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Squared Error")
    plt.title("Training Loss")
    plt.grid()
    plt.show()

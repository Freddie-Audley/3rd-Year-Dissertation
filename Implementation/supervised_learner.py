import pennylane as qp
from pennylane import numpy as np

import matplotlib
matplotlib.use("QtAgg") # Arch Linux requires this backend
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

    def train(self, feature_vectors, targets, optimiser=qp.AdamOptimizer, cost_function=None, step_size=0.01, steps=100):
        if cost_function is None:
            cost_function = lambda predictions, targets: np.mean((predictions - targets) ** 2)  # Mean Squared Error

        optimiser = optimiser(stepsize=step_size)

        def objective_function(parameters):
            predictions = np.array([self.predict(features, parameters) for features in feature_vectors])
            return cost_function(predictions, targets)

        self.loss_history = []

        for step in range(steps):
            self.parameters, loss = optimiser.step_and_cost(
                objective_function, self.parameters
            )

            if step > 0:
                self.loss_history.append(loss)
                if (step - 1) % 100 == 0:
                    print(f"Step {step - 1}: Loss = {loss}")

        if steps > 0:
            final_loss = objective_function(self.parameters)
            self.loss_history.append(final_loss)
            if (steps - 1) % 100 == 0:
                print(f"Step {steps - 1}: Loss = {final_loss}")

        return self.parameters

    def evaluate(self, feature_vectors, targets, cost_function=None):
        if cost_function is None:
            cost_function = lambda predictions, targets: np.mean((predictions - targets) ** 2)

        predictions = np.array([self.predict(features) for features in feature_vectors])
        return cost_function(predictions, targets)


if __name__ == "__main__":
    # (cos(x1 + x2) + cos(x2 + x3) + cos(x3 + x1)) / 3
    def target_function(features):
        x1, x2, x3 = features.T
        return (np.cos(x1 + x2) + np.cos(x2 + x3) + np.cos(x3 + x1)) / 3

    learner = SupervisedQuantumLearner(num_qubits=3, num_layers=3, observable=(qp.Z(wires=0) + qp.Z(wires=1) + qp.Z(wires=2)) / 3)

    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    x1, x2, x3 = np.meshgrid(angles, angles, angles, indexing="ij")
    feature_vectors = np.column_stack((x1.flatten(), x2.flatten(), x3.flatten()))
    targets = target_function(feature_vectors)

    rng = np.random.default_rng(seed=261104)
    indices = rng.permutation(len(feature_vectors))
    split = int(0.8 * len(feature_vectors))
    train_indices = indices[:split]
    test_indices = indices[split:]

    X_train, y_train = feature_vectors[train_indices], targets[train_indices]
    X_test, y_test = feature_vectors[test_indices], targets[test_indices]

    learner.train(X_train, y_train, steps=100)

    train_loss = learner.evaluate(X_train, y_train)
    test_loss = learner.evaluate(X_test, y_test)

    print("Training Loss:", train_loss)
    print("Test Loss:", test_loss)
    print("Generalisation Gap:", test_loss - train_loss)

    plt.figure("Training Loss")
    plt.plot(learner.loss_history)
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Squared Error")
    plt.title("Training Loss")
    plt.grid()

    plot_angles = np.linspace(0, 2 * np.pi, 30)
    x1_plot, x2_plot = np.meshgrid(plot_angles, plot_angles, indexing="ij")
    slice_features = np.column_stack(
        (x1_plot.flatten(), x2_plot.flatten(), np.zeros(x1_plot.size))
    )
    learned_values = np.array([learner.predict(features) for features in slice_features])
    learned_values = learned_values.reshape(x1_plot.shape)
    target_values = target_function(slice_features).reshape(x1_plot.shape)

    function_figure = plt.figure("Target and learned functions")
    target_axis = function_figure.add_subplot(1, 2, 1, projection="3d")
    learned_axis = function_figure.add_subplot(1, 2, 2, projection="3d")

    target_axis.plot_surface(x1_plot, x2_plot, target_values, cmap="viridis")
    target_axis.set_title("Target function (x3 = 0)")
    target_axis.set_xlabel("x1")
    target_axis.set_ylabel("x2")
    target_axis.set_zlabel("f(x1, x2, 0)")

    learned_axis.plot_surface(x1_plot, x2_plot, learned_values, cmap="viridis")
    learned_axis.set_title("Learned function (x3 = 0)")
    learned_axis.set_xlabel("x1")
    learned_axis.set_ylabel("x2")
    learned_axis.set_zlabel("f(x1, x2, 0)")

    function_figure.tight_layout()
    plt.show()

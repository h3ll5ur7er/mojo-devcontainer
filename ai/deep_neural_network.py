""" Simple multi layer fully connected neural network without dependencies """
from typing import Any, Callable
from math import exp, log
import random

random.seed(42)

class Activation:
    def __init__(self, func: Callable[[float], float]):
        self.forward = func
        self.name = func.__name__
    def derivative(self, derivative: Callable[[float], float]) -> None:
        self.backward = derivative
    def backward(self):
        raise NotImplementedError
    def __str__(self) -> str:
        return f"Activation({self.name})"
    def __call__(self, x: float) -> float:
        return self.forward(x)
    def derive(self, x: float) -> float:
        return self.backward(x)



@Activation
def linear(x:float) -> float:
    return x
@linear.derivative
def linear_derivative(x: float) -> float:
    return 1

@Activation
def relu(x:float) -> float:
    return max(0, x)
@relu.derivative
def relu_derivative(x: float) -> float:
    return 0 if x <= 0 else 1

@Activation
def sigmoid(x:float) -> float:
    return 1 / (1 + exp(-x))
@sigmoid.derivative
def sigmoid_derivative(x: float) -> float:
    return x * (1 - x)

def cross_entropy_loss(values: list[float], labels: list[float]) -> float:
    return -sum([labels[i] * log(values[i]) + (1-labels[i]) * log(1-values[i]) for i in range(len(values))])

def shape(values: Any) -> tuple[int]:
    if isinstance(values, list):
        return (len(values),) + shape(values[0])
    else:
        return ()
def print_deep(values: Any, depth: int = 0) -> None:
    if isinstance(values, list):
        for value in values:
            print_deep(value, depth+1)
    else:
        print("  "*depth, values)

class Network:
    layer_sizes: list[int]
    weights: list[list[list[float]]]
    biases: list[list[float]]
    values: list[list[float]]
    activations: list[Activation]

    def __init__(self, layer_sizes: list[int], activations: list[Activation]):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weights = []
        self.values = []
        self.biases = []
        for i in range(len(layer_sizes)-1):
            self.weights.append([[random.random() for _ in range(layer_sizes[i+1])] for _ in range(layer_sizes[i])])
            self.biases.append([random.random() for _ in range(layer_sizes[i+1])])
        print("Weights: ", shape(self.weights))
        print_deep(self.weights)
        print("Biases: ", shape(self.biases))
        print_deep(self.biases)

    
    def forward_pass(self, inputs: list[float]) -> list[float]:
        self.values = [inputs.copy()]
        for i in range(len(self.layer_sizes)-1):
            values = [self.activations[i](sum([self.values[-1][j] * self.weights[i][j][k] for j in range(self.layer_sizes[i])]) + self.biases[i][k]) for k in range(self.layer_sizes[i+1])]
            self.values.append(values)
        return values

    def backpropagate(self, values: list[float], labels: list[float], learning_rate: float) -> float:
        """backpropagate with cross entropy loss"""
        # Forward pass
        output = self.forward_pass(values)
        # Backward pass
        deltas = []
        for i in range(len(self.layer_sizes)-1, 0, -1):
            if i == len(self.layer_sizes)-1:
                deltas.append([output[j] - labels[j] for j in range(self.layer_sizes[i])])
            else:
                delta = []
                for j in range(self.layer_sizes[i]):
                    gradient = self.activations[i].derive(self.values[i][j])
                    weighted_deltas = []
                    for k in range(self.layer_sizes[i+1]):
                        # print(f"{len(self.weights)}-> {i}, {len(self.weights[i])} -> {k}, {len(self.weights[i][k])} -> {j}",)
                        weighted_deltas.append(self.weights[i][j][k] * deltas[-1][k])
                    delta.append(sum(weighted_deltas) * gradient)
                deltas.append(delta)
        deltas.reverse()
        # Update weights and biases
        for i in range(len(self.layer_sizes)-1):
            for j in range(self.layer_sizes[i]):
                for k in range(self.layer_sizes[i+1]):
                    self.weights[i][j][k] -= learning_rate * deltas[i][k] * self.values[i][j]
            for k in range(self.layer_sizes[i+1]):
                self.biases[i][k] -= learning_rate * deltas[i][k]
        return cross_entropy_loss(output, labels)

    def train(self, data: list[tuple[list[float], list[float]]], epochs: int, learning_rate: float) -> None:
        for epoch in range(epochs):
            for values, labels in data:
                score = self.backpropagate(values, labels, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch} complete (score: {score})")
        
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(f"{len(self.layer_sizes)}\n")
            for size in self.layer_sizes:
                f.write(f"{size}\n")
            for i in range(len(self.layer_sizes)-1):
                for j in range(self.layer_sizes[i]):
                    for k in range(self.layer_sizes[i+1]):
                        f.write(f"{self.weights[i][j][k]}\n")
            for i in range(len(self.layer_sizes)-1):
                for j in range(self.layer_sizes[i+1]):
                    f.write(f"{self.biases[i][j]}\n")
    @staticmethod
    def load(path: str) -> "Network":
        with open(path, "r") as f:
            layer_count = int(f.readline())
            layer_sizes = [int(f.readline()) for _ in range(layer_count)]
            network = Network(layer_sizes, [sigmoid for _ in range(layer_count-1)])
            for i in range(layer_count-1):
                for j in range(layer_sizes[i]):
                    for k in range(layer_sizes[i+1]):
                        network.weights[i][j][k] = float(f.readline())
            for i in range(layer_count-1):
                for j in range(layer_sizes[i+1]):
                    network.biases[i][j] = float(f.readline())
            return network

def main():
    # xor dataset
    data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]
    network = Network(
        [2, 3, 1], 
        [sigmoid, sigmoid, sigmoid])
    network.train(data, 10001, 0.1)
    print(0, "xor", 0, "=", network.forward_pass([0, 0]))
    print(0, "xor", 1, "=", network.forward_pass([0, 1]))
    print(1, "xor", 0, "=", network.forward_pass([1, 0]))
    print(1, "xor", 1, "=", network.forward_pass([1, 1]))

if __name__ == "__main__":
    main()

import random


class Value:
    def __init__(self, data, prev=(), op=''):
        self.data = data
        self._prev = prev
        self._op = op
        self.grad = 0.0
        self._backward = lambda: None
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward

        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def backward(self):
        
        topological_tree = []
        visited_neurons = set()
        
        def build_topological_tree(neuron):
            if neuron not in visited_neurons:
                visited_neurons.add(neuron)
                for prev in neuron._prev:
                    build_topological_tree(prev)
                topological_tree.append(neuron)

        build_topological_tree(self)

        self.grad = 1
        for neuron in reversed(topological_tree):
            neuron._backward()
        
    def __repr__(self):
        return f'Value({self.data})'


class NNModule:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Neuron(NNModule):
    def __init__(self, dim):
        self.w = [Value(random.uniform(-1,1)) for _ in range(dim)]
        self.b = Value(0)
    
    def __call__(self, x):
        return sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

    def parameters(self):
        return self.w + [self.b]


class NNLayer(NNModule):
    def __init__(self, input_dim, output_dim):
        self.neurons = [Neuron(input_dim) for _ in range(output_dim)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]


class NeuralNet(NNModule):
    def __init__(self, layers):
        self.layers = [NNLayer(layer[0], layer[1]) for layer in layers]

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
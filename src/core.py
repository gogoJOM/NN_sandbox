class Value:
    def __init__(self, data, prev=(), op=''):
        self.data = data
        self._prev = prev
        self._op = op
        self.grad = 0.0
        self._backward = lambda: None
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        
        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward

        return out

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

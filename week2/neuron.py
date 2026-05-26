import random
import micrograd_engine as mg
value = mg.value

class neuron:
    def __init__(self, n_inputs):
        self.w = [value(random.uniform(-1,1)) for _ in range(n_inputs)]
        self.b = value(random.uniform(-1,1))

    def __call__(self, x):
        z = sum(wi*xi for wi,xi in zip(self.w,x)) + self.b
        out = z.sigmoid()
        return out
    
class layer:
    def __init__(self, n_input, n_output):
        self.neuron = [neuron(n_input) for _ in range(n_output)]

    def __call__(self, x):
        out = [n(x) for n in self.neuron]
        return out[0] if len(out)==1 else out

class MLP:
    def __init__(self, n_input, n_output):
        sz = [n_input]+n_output
        self.layers = [layer(sz[i], sz[i+1]) for i in range(len(n_output))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

import neuron
value = neuron.value
X = [
    [value(0), value(0)],
    [value(0), value(1)],
    [value(1), value(0)],
    [value(1), value(1)]
]
Y = [value(0), value(1), value(1), value(0)]
n = neuron.MLP(2, [4,4,1])
learning_rate = 1
for epoch in range (1000):
    prediction = [n(x) for x in X]
    losses = [(p + value(-1.0) * t) * (p + value(-1.0) * t) 
              for p, t in zip(prediction, Y)]
    total_loss = sum(losses[1:], losses[0])
    all_param = [p for layers in n.layers
                 for neuron in layers.neuron
                 for p in neuron.w + [neuron.b]]
    for p in all_param:
        p.grad = 0
    total_loss.backward()
    for p in all_param:
        p.data -= learning_rate * p.grad
    if epoch%100 == 0:
        print(f"epoch {epoch:4d} and loss={total_loss.data:.4f}")
print(f"final prediction is ")
for x, t in zip(X, Y):
    pred = n(x)
    print(f"input={[xi.data for xi in x]} | pred={pred.data:.4f} | target={t.data:.4f}")
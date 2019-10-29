import numpy as np

def nonlin(X, deriv = False):
    if (deriv == True):
        return X * (1 - X)
    else:
        return 1 / (1 + np.exp(-X))

def init_data():
    X = np.array([ [1, 1],
                   [0, 1],
                   [1, 0],
                   [0, 0] ])

    Y = np.array([ [0],
                   [1],
                   [1],
                   [0] ])
    
    w = 2 * (np.random.rand(np.shape(X)[0], np.shape(X)[1])) - 1

    return X, Y, w

def update(X, w, Y = init_data()[1]):
    net = X * w
    out = nonlin(net)

    dE = -(Y - out)
    dOut = nonlin(net, True)

    delta_w = dE * dOut * X

    return delta_w

def process(input, weights, delta=0):
    net = input * weights
    out = nonlin(net)
    delta += input * update(input, weights)
    updated_weights = weights + delta

    return out, updated_weights, delta

X, Y, w = init_data()
for epoch in range(100000):
    initial_input = X 
    initial_weights = w

    l1_output, l1_weights, l1_delta = process(initial_input, initial_weights)
    h1_output, h1_weights, h1_delta = process(l1_output, l1_weights, l1_delta)
    h2_output, h2_weights, h2_delta = process(h1_output, h1_weights, h1_delta)
    h3_output, h3_weights, h3_delta = process(h2_output, h2_weights, h2_delta)
    h4_output, h4_weights, h4_delta = process(h3_output, h3_weights, h3_delta)
    h5_output, h5_weights, h5_delta = process(h4_output, h4_weights, h4_delta)
    output = nonlin(h5_output * h5_weights)

print("Predicted:\n", output)
print("\nActual:\n", Y)
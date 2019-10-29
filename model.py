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

def process(input, weights):
    net = input * weights
    out = nonlin(net)
    delta = input * update(input, weights)
    updated_weights = weights + delta

    return out, updated_weights

X, Y, w = init_data()
for epoch in range(10000):
    initial_input = X 
    initial_weights = w

    l1_output, l1_weights = process(initial_input, initial_weights)
    h1_output, h1_weights = process(l1_output, l1_weights)
    h2_output, h2_weights = process(h1_output, h1_weights)
    output = nonlin(h2_output * h2_weights)

final = np.resize(output, (4, 1))
print("Predicted:\n", final)
print("\nActual:\n", Y)
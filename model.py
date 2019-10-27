import numpy as np

def nonlin(X, deriv = False):
    if (deriv == True):
        return X * (1 - X)
    else:
        return 1 / (1 + np.exp(-X))

def init_data():
    X = np.array([ [1], 
                   [0]])

    Y = np.array([ [1], 
                   [0]])

    return X, Y

def error(Y, out):
    return (1/2) * ((Y - out) ** 2)

def update(X, Y, w):
    net = np.dot(X, w)
    out = nonlin(net)
    dE = -(Y - out)
    dOut = net * (1 - net)
    delta_w = np.dot(np.dot(dE, dOut), X)
    return w - delta_w.T

X, Y = init_data()

w = []
for epoch in range(10000):
    if (epoch > 0):
        w.append(update(X, Y, w[len(w) - 1]))
        del(w[0])
    else:
        w_initial = 2 * (np.random.rand(np.shape(X)[1], np.shape(X)[0])) - 1
        w.append(w_initial)

result = nonlin(np.dot(X, w[len(w) - 1]))
print(result)

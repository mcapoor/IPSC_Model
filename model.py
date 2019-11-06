import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import data_processing

def nonlin(X, deriv = False):
    if (deriv == True):
        return X * (1 - X)
    else:
        return 1 / (1 + np.exp(-X))

def init_data():
    with open('data/data.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        headers = next(reader)
        data = list(reader)
        data = np.array(data)
        
        abbreviation = data[:0]
        cell = data[:1]
        signal = data[:2]
        p_value = data[:3]
        
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse=False)
    
        abbreviation_ie = label_encoder.fit_transform(abbreviation)
        abbreviation_ie = abbreviation_ie.reshape(len(abbreviation_ie), 1)
        abbreviation_encoded = onehot_encoder.fit_transform(abbreviation_ie)
        
        cell_ie = label_encoder.fit_transform(cell)
        cell_ie = abbreviation_ie.reshape(len(cell_ie), 1)
        cell_encoded = onehot_encoder.fit_transform(cell_ie)

        input = abbreviation_encoded + cell_encoded + p_value
        output = signal

    return input, output

def error(out, Y):
    sum = 0
    error = (1/2) * ((Y - out) ** 2)
    
    rows = np.shape(error)[0]
    columns = np.shape(error)[1]
    for row in range(rows):
        for column in range(columns):
            sum += out[row][column]

    mean = sum / (rows * columns)
    return round(mean * 100, 2)

def layer_out(layer_input, layer_weights):
    net = np.dot(layer_input, layer_weights)
    out = nonlin(net)
    return net, out

def update_weights(layer_input, layer_weights, Y):
    net, out = layer_out(layer_input, layer_weights)
    print(np.shape(Y), np.shape(net), np.shape(out), np.shape(layer_input))
    weight_delta = -(Y - out) * (net * (1 - net)) * layer_input
    return weight_delta

X, Y = init_data()
for epoch in range(100000):
    inputs = np.shape(X)[1]
    first_hidden_neurons = 2 * inputs
    second_hidden_neurons = 2 * first_hidden_neurons
    third_hidden_neurons = first_hidden_neurons
    output_neurons = np.shape(Y)[0]

    input_weights = 2 * (np.random.rand(inputs, first_hidden_neurons)) - 1
    first_hidden_weights = 2 * (np.random.rand(first_hidden_neurons, second_hidden_neurons)) - 1
    second_hidden_weights = 2 * (np.random.rand(second_hidden_neurons, third_hidden_neurons)) - 1
    third_hidden_weights = 2 * (np.random.rand(third_hidden_neurons, output_neurons)) - 1
    output_weights = 2 * (np.random.rand(output_neurons, np.shape(Y)[1])) - 1

    l1 = layer_out(X, input_weights)
    input_weights += update_weights(X, input_weights)
    
    h1 = layer_out(l1, first_hidden_weights)
    first_hidden_weights += update_weights(l1, first_hidden_weights)

    h2 = layer_out(h1, second_hidden_weights)
    second_hidden_weights += update_weights(h1, second_hidden_weights)

    h3 = layer_out(h2, third_hidden_weights)
    third_hidden_weights += update_weights(h2, third_hidden_weights)

    output = layer_out(h3, output_weights)
    output_weights += update_weights(h3, output_weights)
    
print("Predicted:\n", output)
print("\nActual:\n", Y)
print("\n Mean Error:\n", error(output, Y),"%")
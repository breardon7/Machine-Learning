# // Reardon, Bradley
# // 6202 HW5

# E.1
import numpy as np
import matplotlib.pyplot as plt
from math import exp


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = []
    hidden_layer = [{'weights': [np.random.rand() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [np.random.rand() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, inputs):
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Given approximation function
def approximation_function(p):
    result = np.e(-abs(p)) * np.pi * p
    return result


# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# test backpropagation of error
network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
expected = [0, 1]
backward_propagate_error(network, expected)
for layer in network:
	print(layer)

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    sse = []
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        sse.append(sum_error)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))



# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Stochastic Gradient Descent
def stochastic_gradient_descent(gradient, start, learning_rate, epochs, tolerance=1e-06):
    vector = start
    for _ in range(epochs):
        diff = -learning_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector

def batch_gradient_descent(X, y, theta, learning_rate, n_epochs):
    X_transpose = X.transpose()
    for i in range(0, n_epochs):
        prediction = np.dot(X, theta)
        loss = prediction - y
        gradient = np.dot(X_transpose, loss)
        theta = theta - learning_rate * gradient
    return theta

def plot_error_vs_outputs(trained_network, predict):
    outputs = predict["outputs"]
    sum_squared_errors = trained_network["sse"]
    plt.title("Sum Squared Errors vs Outputs")
    plt.xlabel("Sum Squared Errors")
    plt.ylabel("Outputs")
    plt.plot(sum_squared_errors, outputs)
    plt.show()

def plot_error_vs_epochs(trained_network):
    sum_squared_errors = trained_network["sse"]
    epochs = trained_network["n_epoch"]
    plt.title("Sum Squared Errors vs Epochs")
    plt.xlabel("Sum Squared Errors")
    plt.ylabel("Epochs")
    plt.plot(sum_squared_errors, np.arange(0, epochs))
    plt.show()

    def plot_stochastic_gradient(trained_network):
        sum_squared_errors = trained_network["sum_squared_errors"]
        learning_rate = trained_network["learning_rate"]
        number_of_epoch = trained_network["number_of_epoch"]
        stochastic_gradient = []
        for epoch in range(len(sum_squared_errors)):
            sum_squared_error = sum_squared_errors[epoch]
            stochastic_gradient_vector = stochastic_gradient_descent(gradient=lambda v: sum_squared_error * v,
                                                                     start=10.0,
                                                                     learning_rate=learning_rate,
                                                                     epochs=number_of_epoch)
            stochastic_gradient.append(stochastic_gradient_vector)
            print('>Epoch=%d, Learning Rate=%.3f, Stochastic Gradient Vector=%.3f' % (
                epoch, learning_rate, stochastic_gradient_vector))
        trained_network["sum_squared_errors"] = stochastic_gradient
        plot_error_vs_outputs(trained_network)
        plot_error_vs_epochs(trained_network)


def plot_batch_gradient(input_data, trained_network):
    sum_squared_errors = trained_network["sum_squared_errors"]
    learning_rate = trained_network["learning_rate"]
    number_of_epoch = trained_network["number_of_epoch"]
    theta = np.random.randn(len(sum_squared_errors), 1)
    X = np.array(input_data)
    Y = sum_squared_errors
    batch_gradient = batch_gradient_descent(X, Y, theta, learning_rate, number_of_epoch)
    trained_network["sum_squared_errors"] = batch_gradient[1]
    plot_error_vs_outputs(trained_network)
    plot_error_vs_epochs(trained_network)


epochs = 100
input_data = [[np.random.randint(-2, 2) for i in range(epochs)]]
number_of_inputs = 1
number_of_outputs = 1
learning_rate = 0.15

print("===========NETWORK 1============")
network_design1 = initialize_network(number_of_inputs, 2, number_of_outputs)
trained_network_output1 = train_network(network_design1, input_data, learning_rate, epochs, number_of_outputs)
plot_error_vs_outputs(trained_network_output1)
plot_error_vs_epochs(trained_network_output1)
plot_stochastic_gradient(trained_network_output1)
plot_batch_gradient(input_data, trained_network_output1)
print("===========NETWORK 2============")

network_design2 = initialize_network(number_of_inputs, 10, number_of_outputs)
trained_network_output2 = train_network(network_design1, input_data, learning_rate, epochs, number_of_outputs)
plot_error_vs_outputs(trained_network_output2)
plot_error_vs_epochs(trained_network_output2)
plot_stochastic_gradient(trained_network_output2)
plot_batch_gradient(input_data, trained_network_output2)
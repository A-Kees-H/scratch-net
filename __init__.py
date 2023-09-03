import math
import random
from decimal import Decimal
import collections

class Node:
    def __init__(self, ID, weights=[], bias=None):
        self.ID = ID
        self.weights = weights
        self.bias = bias
        self.last_output = None
        self.error = None
        self.current_tally = 0

    def change_weight(self, weight, index):
        self.weights[index] = weight

    def change_bias(self, bias):
        self.bias = bias

class OutputNode(Node):
    def __init__(self, ID, activation_function):
        super().__init__(ID)
        self.activation_function = activation_function

    def calculate(self, values):
        self.last_output = d(self.activation_function(sum(values)))

class InputNode(Node):
    def __init__(self, ID, weights):
        super().__init__(ID, weights=weights)

    def calculate(self, value):
        self.last_output = value
        self.weighted_outputs = [d(value) * d(weight) for weight in self.weights]
        return self.weighted_outputs

class HiddenNode(Node):
    def __init__(self, ID, weights, bias, activation_function):
        super().__init__(ID, weights=weights, bias=bias)
        self.activation_function = activation_function
        self.weighted_outputs = None

    def calculate(self):
        self.last_tally = self.current_tally
        biased = self.current_tally - self.bias
        self.current_tally = 0
        self.last_output = self.activation_function(biased)
        
        self.weighted_outputs = [d(weight) * d(self.last_output) for weight in self.weights]
        return self.weighted_outputs

class Layer(collections.abc.Sequence):
    def __init__(self, nodes):
        self.nodes = nodes

    def __getitem__(self, i):
        return self.nodes[i]

    def __len__(self, i):
        return len(self.nodes)

    def outputs(self):
        return [node.last_output for node in self.nodes]

class Network:
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, num_hidden_nodes, activation_function, n):
        self.node_count = 0
        self.activation_function = activation_function
        self.num_hidden_nodes = num_hidden_nodes #per layer
        self.num_outputs = num_outputs
        self.input_layer = self.generate_input_layer(num_inputs)
        self.hidden_layers = self.generate_hidden_layers(num_hidden_layers)
        self.output_layer = self.generate_output_layer(num_outputs)
        self.model = [self.input_layer, *self.hidden_layers, self.output_layer]
        self.nodes = self.get_nodes()

    def generate_input_layer(self, num_inputs):
        layer = []
        for num in range(num_inputs):
            weights = [get_random_decimal(-2, 2, 2) for _ in range(self.num_hidden_nodes)]
            node = InputNode(self.node_count, weights)
            layer.append(node)
            self.node_count += 1
        return Layer(layer)

    def generate_hidden_layers(self, num_hidden_layers):
        hidden_layers = []
        for num in range(num_hidden_layers):
            hidden_layers.append(self.generate_hidden_layer())
        return hidden_layers

    def generate_hidden_layer(self):
        layer = []
        for num in range(self.num_hidden_nodes):
            weights = [get_random_decimal(-2, 2, 2) for _ in range(self.num_hidden_nodes)]
            bias = get_random_decimal(0, 1, 3)
            node = HiddenNode(self.node_count, weights, bias, self.activation_function)
            layer.append(node)
            self.node_count += 1
        return Layer(layer)

    def generate_output_layer(self, num_outputs):
        layer = []
        for num in range(num_outputs):
            node = OutputNode(self.node_count, self.activation_function)
            layer.append(node)
            self.node_count += 1
        return Layer(layer)

    def get_nodes(self):
        nodes = []
        nodes += self.input_layer
        for layer in self.hidden_layers:
            nodes += layer
        nodes += self.output_layer
        return nodes

    def execute(self, inputs, desired_outputs):
        # input layer
        for i, node in enumerate(self.input_layer):
            node.calculate(inputs[i])

        # hidden layer
        layer_inputs = self.input_layer.outputs()
        for layer in self.hidden_layers:
            for i, node in enumerate(layer):
                node_outputs = node.calculate(layer_inputs)
            layer_inputs = layer_outputs

        # output layer
        for i, node in enumerate(self.output_layer):
            node.calculate(layer_inputs)

        self.get_errors(desired_outputs)
        self.adjust()



    def get_errors(self, desired_outputs):
        # start with output
        output_errors = []
        for i, output in enumerate(self.final_outputs):
            E = d(0.5 * (desired_outputs[i] - output)**2)
            self.output_layer[i].error = E
            output_errors.append(E)

        # now do hidden
        num_layers = len(hidden_layers_outputs)
        self.hidden_layers_errors = []
        errors_ahead = output_errors
        for i in range(num_layers):
            new_errors_ahead = []
            layer_outputs = self.hidden_layers_outputs[num_layers - (1 + i)]
            layer_nodes = self.hidden_layers[num_layers - (1 + i)]
            for i, o in enumerate(self.hidden_layers_outputs):
                node = layer_nodes[i]
                En = o * (1 - o) * sum([weight * errors_ahead[i] for i, weight in enumerate(node.weights)])
                node.error = En
                new_errors_ahead.append(En)
            self.hidden_layers_errors.insert(0, new_errors_ahead)
            errors_ahead = new_errors_ahead

        self.input_layer_errors = []
        # now do the input nodes
        for i, node in enumerate(self.input_layer):
            o = self.input_values[i]
            En = o * (1 - o) * sum([weight * errors_ahead[i] for i, weight in enumerate(node.weights)])
            node.error = En
            self.input_layer_errors.append(En)
        return E

    def adjust(self):
        # formula
        "new weight = old weight + adjustment variable * error of the node it came out of * output that it weighted"
        adjustable_layers = [self.input_layer] + self.hidden_layers
        for layer in adjustable_layers:
            for node in layer:
                error = node.error
                last_output = node.last_output
                for i, old_weight in enumerate(node.weights):
                    new_weight = old_weight + n * error * last_output
                    node.change_weight(new_weight, i)
                new_bias = node.bias + n * error * last_output
                node.change_bias(new_bias)





def step_activation(value):
    return value >= 0

def sigmoid_activation(value):
    return d(1/(1+math.e**(-int(value))))

def linear_activation(coefficient, value):
    return coefficient * value

def ReLU_activation(value):
    return max((0, value))

def get_random_decimal(bottom, top, ndp):
    return Decimal(str(round(random.uniform(bottom, top), ndp)))

def csv_print(values):
    for value in values:
        print(value, end=",")
    print()

def print_results(inputs, desired_outputs, total_errors):
    print("\ninputs:")
    csv_print(inputs)
    print("desired outputs:")
    csv_print(desired_outputs)
    print("actual outputs:")
    csv_print(self.final_outputs)
    print("total errors:")
    csv_print(total_errors)

def d(num):
    return Decimal(str(num))

def generate_ABC_training_data(num_examples):
    inputs = []
    outputs = []
    each = round(num_examples**(1/3))
    for A in range(each):
        for B in range(each):
            for C in range(each):
                inputs.append((A, B, C))
                total = A+B+C
                outputs.append([total])

    for i, inp in enumerate(inputs):
        if sum(inp) != outputs[i][0]:
            print("fail")
    return inputs, outputs


if __name__ == "__main__":
    # let's try to simulate the function A + B + C
    # so one output node
    num_outputs = 1
    # three input nodes
    num_inputs = 3
    # let's say 3 hidden layers
    num_hidden_layers = 3
    # let's say 5 nodes per layer
    num_hidden_nodes = 5

    inputs, outputs = generate_ABC_training_data(100)
    test_net = Network(num_inputs, num_outputs, num_hidden_layers, num_hidden_nodes, sigmoid_activation, 0.01)
    for i, input_triple in enumerate(inputs):
        answer = outputs[i]
        test_net.execute(input_triple, answer)

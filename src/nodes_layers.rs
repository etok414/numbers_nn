extern crate rand;

use rand::Rng;

#[derive(Clone)]
pub struct Node {
    // A node/neuron's bias and the weights of its connections to the previous layer.
    pub bias: f32, //A value to adjust the value by independent of inputs.
    pub weights: Vec<f32>, //For each node in the previous layer, one of these weights correspond to it and adjusts the amount that this nodes value is affected by that node.
    pub personal_pos: usize, //The position within its own layer. Only used in find_delta, but find_delta is used a lot.
}

impl Node {
    pub fn new(number_of_weights: usize, personal_pos: usize) -> Node {
        //Generates a new node with a random bias and random weights.
        let mut rng = rand::thread_rng();

        let mut init_weights = vec![0.0; number_of_weights];

        for n in 0..number_of_weights {
            let x: f32 = rng.gen();  // Random number in the interval [0; 1[
            init_weights[n] = 2.0 * x  - 1.0;  // The initial weights will be in [-1; 1[
        }
        let x: f32 = rng.gen();  // Random number in the interval [0; 1[
        let init_bias = 2.0 * x  - 1.0;  // The initial bias will be in [-1; 1[

        Node {
            bias: init_bias,
            weights: init_weights, // vec![0.0; number_of_weights],
            personal_pos: personal_pos,
            // bias_adjust: None,
            // weight_adjusts: Vec::new()
        }
    }

    pub fn calculate(&self, previous_layer_values:&Vec<f32>) -> f32 {
        //Calculates the value of the node based on the values of the previous layer and the node's bias and weights.
        let mut value = self.bias;
        let previous_layer_len = previous_layer_values.len();
        if self.weights.len() != previous_layer_len {
            panic!("The number of weights ({}) doesn't match the number of values ({})", self.weights.len(), previous_layer_len);
        }
        for pos_num in 0..previous_layer_len {
            value += previous_layer_values[pos_num] * self.weights[pos_num];
        }
        let norm_value = 1.0 / (1.0 + (-value).exp());
        // if norm_value < 0.0 || norm_value > 1.0 {
        //     panic!{"Math is broken, the sigmoid functions returns value outside [0; 1]"}
        // }
        norm_value
    }

    pub fn find_delta(&self, personal_value:f32, next_layer: &Layer, d_values: &Vec<f32>) -> f32 {
        //Finds delta and returns it.
        //d_values stands either for desired values, a vector of the desired output values,
        //or for delta values, a vector of the deltas of the next_layer.
        let mut delta = 0.0;
        if next_layer.node_count == 0 {
            delta = (personal_value - d_values[self.personal_pos]) * personal_value * (1.0 - personal_value)
        } else {
            for num in 0..next_layer.node_count {
                delta += d_values[num] * next_layer.nodes[num].weights[self.personal_pos];
            }
            delta *= personal_value * (1.0 - personal_value);
        }
        delta
    }

    // pub fn single_adjust(&mut self, delta: f32, learning_rate: f32, previous_layer_value: f32, relevant_weight: usize) {
    //     if relevant_weight == self.weights.len() {
    //         self.bias -= delta * learning_rate;
    //     } else {
    //         self.weights[relevant_weight] -= delta * previous_layer_value * learning_rate;
    //     }
    // }
}

impl IntoIterator for Node {
    //This trait is implemented so that the nodes can easily be written to file.
    //The bias and weights are turned into strings so that the file writer will accept them.
    //The bias and weights are converted from f32 to u32 before they're converted into strings, so that they're easier to turn back later.
    type Item = String;
    type IntoIter = ::std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let mut output = Vec::new();
        output.push(self.bias.to_bits().to_string());
        for weight in self.weights {
            output.push(weight.to_bits().to_string())
        }
        output.into_iter()
    }
}

#[derive(Clone)]
pub struct Layer {
    //A layer of nodes, their biases, and the weights of their connections to the previous layer.
    pub nodes: Vec<Node>, //A Vec of all the nodes in the layer.
    pub node_count: usize, //Should be equal to nodes.len() and shouldn't change.
    pub learning_rate: f32, //Reduces the amount it learns from each training example so that it doesn't fluctuate as wildly, and goes down a bit more steadily.
}

impl Layer {
    pub fn new(previous_layer_nodes: usize, number_of_nodes: usize, learning_rate: f32) -> Layer {
        //Generates a layer of nodes, each with a random bias and a number of random weights equal to the number of nodes in the previous layer.
        let mut nodes = Vec::new();
        for num in 0..number_of_nodes {
            nodes.push(Node::new(previous_layer_nodes, num));
        }
        let node_count = nodes.len();
        Layer {
            nodes: nodes,
            node_count: node_count,
            learning_rate: learning_rate,
        }
    }

    pub fn calculate(&self, previous_layer_values:&Vec<f32>) -> Vec<f32> {
        //Calculates the values of the nodes based on the values of the previous layer and the nodes' weights and biases.
        let mut values = Vec::new();
        for node_num in 0..self.node_count {
            values.push(self.nodes[node_num].calculate(previous_layer_values));
        }
        values
    }

    pub fn find_deltas(&self, values:&Vec<f32>, desired_values:&Vec<f32>, next_layer:&Layer, next_layer_deltas:&Vec<f32>) -> Vec<f32> {
        //Finds out how the nodes' weights and biases should be adjusted, based either on a list of desired values, or how the next layer is set to be adjusted.
        //The function does this by calling find_adjusts for each node.
        let d_values = if next_layer.node_count > 0 {
                next_layer_deltas
            } else {
                desired_values
            };
        let mut deltas = Vec::new();
        for node_num in 0..self.node_count {
            deltas.push(self.nodes[node_num].find_delta(
                values[node_num],
                next_layer,
                d_values,
                // else {&vec![desired_values[node_num]]}
                )
            );
        }
        deltas
    }

    pub fn adjust(&mut self, deltas: &Vec<f32>, previous_layer_values: &Vec<f32>) {
        // Adjusts the weights and biases of the nodes based on the deltas and the values of the previous layer.
        let previous_layer_len = previous_layer_values.len();
        for prev_layer_num in 0..previous_layer_len {
            for num in 0..self.node_count {
                self.nodes[num].weights[prev_layer_num] -= deltas[num] * previous_layer_values[prev_layer_num] * self.learning_rate;
            }
        }
        for num in 0..self.node_count {
            self.nodes[num].bias -= deltas[num] * self.learning_rate;
        }
    }
}

#[derive(Clone)]
pub struct Network {
    // A struct to organize the layers, and to make them work together.
    // The represented layers are the hidden layers and the output layer, since the weights of the connections are stored in the latter of the connected layers, and the input layer doesn't need biases anyway.
    pub layers: Vec<Layer>, //The layers organized in a Vec
    pub layer_count: usize, //Should be equal to layers.len() and shouldn't change.
    // pub learning_rate: f32, //Reduces the amount it learns from each training example so that it doesn't fluctuate as wildly, and goes down a bit more steadily. Moved to be in each layer instead.
}

impl Network {
    pub fn new(node_nums:&[usize], learning_rate: f32) -> Network {
        //Makes a completely new network with random weights and biases.
        //node_nums tells how many nodes there should be in each layer. The nodes will have a number of weights equal to the number of nodes in the previous layer.
        //The network will have a layer for each member of node_nums except for the 0th one, since that's the input layer.
        //If you want to load in a network from a series of files, use inout::read_network instead.
        let mut layers = Vec::new();
        for layer_num in 1..node_nums.len() {
            layers.push(Layer::new(node_nums[layer_num-1], node_nums[layer_num], learning_rate));
        }
        let layer_count = layers.len();
        Network {
            layers: layers,
            layer_count: layer_count,
            // learning_rate: learning_rate,
        }
    }

    pub fn calculate(&self, inputs: &Vec<f32>) -> Vec<Vec<f32>> {
        //Calculates the values of all nodes based on the active training data and the weights and biases.
        //The outer vector of the output is the layer, the inner vector is the position in the layer.
        //To get the output layer from values, say "values[values.len() - 1]" or "values[self.layer_count - 1]" if working inside the network struct.
        let mut values = vec![self.layers[0].calculate(inputs)];
        for num in 1..self.layer_count {
            values.push(self.layers[num].calculate(&values[num-1]));
        }
        values
    }

    pub fn find_make_adjust(&mut self, inputs: &Vec<f32>, desired_outputs:&Vec<f32>) {
    //Finds out the deltas of all the nodes, puts them in delta_matrix, then adjusts the weights and biases of the nodes based on that.
        let values = self.calculate(inputs);
        let mut delta_matrix = vec![Vec::new(); self.layer_count];
        delta_matrix[self.layer_count-1] = self.layers[self.layer_count-1].find_deltas(&values[self.layer_count-1], desired_outputs, &Layer::new(0, 0, 0.0), &Vec::new());
        for num in (0..self.layer_count-1).rev() {
        //This has to be done in reverse order because the deltas of a hidden layer is based on the deltas of the following layer.
        //The deltas of the output layer is based on the desired output values and is determined immediately above this for loop.
            delta_matrix[num] = self.layers[num].find_deltas(&values[num], &Vec::new(), &self.layers[num+1], &delta_matrix[num+1]);
        }

        self.layers[0].adjust(&delta_matrix[0], inputs);
        for num in 1..self.layer_count {
            self.layers[num].adjust(&delta_matrix[num], &values[num-1])
        }
    }

    pub fn compare_success(&self, inputs: &Vec<f32>, desired_outputs:&Vec<f32>, margin_of_error:f32) -> bool {
    //Compares whether the all the outputs are correct within a margin of error. Has no effect on the actual training.
        let output_values = &self.calculate(inputs)[self.layer_count - 1];
        let mut within_margin = true;
        for num in 0..desired_outputs.len() {
            if (desired_outputs[num] - output_values[num]).powi(2) > margin_of_error.powi(2) {
                within_margin = false;
            }
        }
        within_margin
    }
}

pub fn find_biggest(numbers:&[f32]) -> (f32, Vec<usize>) {
    //Finds the positions of each instance of the biggest number in a vector and returns them in a vector
    let mut biggest_number = std::f32::NEG_INFINITY; //The biggest number starts as negative infinity
    let mut biggest_number_positions = Vec::new();
    let num_len = numbers.len();
    for num in 0..num_len {
        if numbers[num] > biggest_number {
            biggest_number = numbers[num];
            biggest_number_positions = vec![num];
        } else if numbers[num] == biggest_number {
            biggest_number_positions.push(num)
        }
    }
    (biggest_number, biggest_number_positions)
}

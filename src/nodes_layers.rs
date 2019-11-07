extern crate rand;

use rand::Rng;

#[derive(Clone)]
pub struct Node {
// A node/neuron's bias and the weights of its connections to the previous layer.
    bias: f32,
    weights: Vec<f32>,
    personal_pos: usize,
    // bias_adjust: Option<f32>,
    // weight_adjusts: Vec<f32>,
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
        let init_bias = 2.0 * x  - 1.0;  // The initial weights will be in [-1; 1[

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

#[derive(Clone)]
pub struct Layer {
//A layer of nodes, their biases, and the weights of their connections to the previous layer.
    nodes: Vec<Node>,
    node_count: usize, //Should be equal to nodes.len() and shouldn't change.
}

impl Layer {
    pub fn new(previous_layer_nodes: usize, number_of_nodes: usize) -> Layer {
    //Generates a layer of nodes, each with a random bias and a number of random weights equal to the number of nodes in the previous layer.
        let mut nodes = Vec::new();
        for num in 0..number_of_nodes {
            nodes.push(Node::new(previous_layer_nodes, num));
        }
        let node_count = nodes.len();
        Layer {
            nodes: nodes,
            node_count: node_count,
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
        let mut deltas = Vec::new();
        for node_num in 0..self.node_count {
            deltas.push(self.nodes[node_num].find_delta(
                                                        values[node_num],
                                                        next_layer,
                                                        if next_layer.node_count > 0 {next_layer_deltas}
                                                        else {desired_values}
                                                        // else {&vec![desired_values[node_num]]}
                                                        )
                        );
        }
        deltas
    }

    pub fn adjust(&mut self, deltas: &Vec<f32>, previous_layer_values: &Vec<f32>, learning_rate: f32) {
        let previous_layer_len = previous_layer_values.len();
        for prev_layer_num in 0..previous_layer_len {
            for num in 0..self.node_count {
                self.nodes[num].weights[prev_layer_num] -= deltas[num] * previous_layer_values[prev_layer_num] * learning_rate;
            }
        }
        for num in 0..self.node_count {
            self.nodes[num].bias -= deltas[num] * learning_rate;
        }
    }
}

pub struct Network {
    layers: Vec<Layer>,
    layer_count: usize,
    learning_rate: f32,
}

impl Network {
    pub fn new(node_nums:Vec<usize>, learning_rate: f32) -> Network {
        let mut layers = Vec::new();
        for layer_num in 1..node_nums.len() {
            layers.push(Layer::new(node_nums[layer_num-1], node_nums[layer_num]));
        }
        let layer_count = layers.len();
        Network {
            layers: layers,
            layer_count: layer_count,
            learning_rate: learning_rate,
        }
    }

    pub fn calculate(&self, inputs: &Vec<f32>) -> Vec<Vec<f32>> {
    //Calculates the values of all nodes based on the active training data and the weights and biases.
    //The outer vector of the output is the layer, the inner vector is the position in the layer. To get the output layer from values, say values[values.len() - 1]
        let mut values = vec![self.layers[0].calculate(inputs)];
        for num in 1..self.layer_count {
            values.push(self.layers[num].calculate(&values[num-1]));
        }
        values
    }

    pub fn find_make_adjust(&mut self, inputs: &Vec<f32>, desired_outputs:&Vec<f32>) {
        let values = self.calculate(inputs);
        let mut delta_matrix = vec![Vec::new(); self.layer_count];
        delta_matrix[self.layer_count-1] = self.layers[self.layer_count-1].find_deltas(&values[self.layer_count-1], desired_outputs, &Layer::new(0, 0), &Vec::new());
        for num in (0..self.layer_count-1).rev() {
            delta_matrix[num] = self.layers[num].find_deltas(&values[num], &Vec::new(), &self.layers[num+1], &delta_matrix[num+1]);
        }

        self.layers[0].adjust(&delta_matrix[0], inputs, self.learning_rate);
        for num in 1..self.layer_count {
            self.layers[num].adjust(&delta_matrix[num], &values[num-1], self.learning_rate)
        }
    }
}

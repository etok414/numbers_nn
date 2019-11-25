extern crate csv;

use std::error::Error;
use std::fs;

use crate::nodes_layers;

pub fn unpack_labels(filename: &str) -> Vec<u8> {
    let content = fs::read(filename)
        .expect("Something went wrong while reading labels")
        [8..].to_vec();
    content
}

pub fn turn_to_result(input: Vec<u8>) -> Vec<Vec<f32>> {
    let mut output = Vec::new();
    for number in input {
        let mut element = vec![0.0;10];
        element[number as usize] = 1.0;
        output.push(element);
    }
    output
}

pub fn unpack_images(filename: &str) -> Vec<Vec<f32>> {
        let raw_content = fs::read(filename)
            .expect("Something went wrong while reading images")
            [16..].to_vec();
        let raw_content = turn_to_float(raw_content);
        let content_as_iter = raw_content.chunks_exact(28*28);
        if content_as_iter.remainder().len() > 0 {
            panic!("An image was only partially formed. It had only {:?} elements, when it should have {:?}", content_as_iter.remainder().len(), 28*28)
        }
        let mut content = Vec::new();
        for chonk in content_as_iter {
            content.push(chonk.to_vec());
        }
        content
}

pub fn turn_to_float(input: Vec<u8>) -> Vec<f32> {
    let mut output = Vec::new();
    for number in input {
        output.push(number as f32 / 255.0)
    }
    output
}


pub fn write_network(network: nodes_layers::Network, file_paths: Vec<&str>) -> Result<(), Box<dyn Error>> {
    if network.layer_count != file_paths.len() {
        panic!("Missing proper error implementation, and the number of file paths doesn't match the number of layers")
    }
    for num in 0..network.layer_count {
        let mut wtr = csv::Writer::from_path(file_paths[num])?;
        for node in network.layers[num].nodes.clone() {
            wtr.write_record(node)?;
        }
        wtr.flush()?;
    }
    Ok(())
}

pub fn read_network(file_paths: Vec<&str>, learning_rate:f32) -> Result<nodes_layers::Network, Box<dyn Error>> {
    let mut layers = Vec::new();
    for file_path in file_paths{
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(file_path)
        ?;
        let mut nodes = Vec::new();
        for result in rdr.records() {
            let record = result?;
            let bias: u32 = record[0].parse()?;
            let bias = f32::from_bits(bias);
            let mut weights = Vec::new();
            for num in 1..record.len() {
                let weight: u32 = record[num].parse()?;
                weights.push(f32::from_bits(weight));
            }
            nodes.push(
                nodes_layers::Node{
                    bias: bias,
                    weights: weights,
                    personal_pos: nodes.len(),
                }
            )
        }
        let node_count = nodes.len();
        layers.push(
            nodes_layers::Layer{
                nodes: nodes,
                node_count: node_count,
            }
        )
    }
    let layer_count = layers.len();
    Ok(nodes_layers::Network{
        layers: layers,
        layer_count: layer_count,
        learning_rate: learning_rate,
    })
}

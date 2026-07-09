use microgradrs::layer::{self, Layer};
use microgradrs::nn::Neuron;
use microgradrs::value::{Value, back_propagate, to_dot, topo_sort};
use std::fs;
use std::process::Output;
fn main() {
    let data = vec![Value::new(3.0), Value::new(2.0), Value::new(-3.0)];

    let layer = Layer::new(3, 1);
    let mut output = layer.forward(&data, true);
    let output_len = output.len() as i32;
    output = Layer::new(output_len, 1).forward(&output, false);

    let final_output = &back_propagate(output[0].clone())[0];
    fs::write("graph.dot", to_dot(final_output));
}

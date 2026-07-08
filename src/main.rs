use microgradrs::layer::{self, Layer};
use microgradrs::nn::Neuron;
use microgradrs::value::{Value, back_propagate, to_dot, topo_sort};
use std::fs;
fn main() {
    let data = vec![Value::new(-5.0), Value::new(-5.0), Value::new(5.0)];

    let layer = Layer::new(3, 1);
    let output = layer.forward(&data);
    let mut final_output = Value::new(0.0);
    for val in output.iter() {
        final_output += val;
    }
    let final_output = &back_propagate(final_output)[0];
    fs::write("graph.dot", to_dot(final_output));
}

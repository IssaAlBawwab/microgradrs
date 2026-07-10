use crate::{nn::Neuron, value::Value};
#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(num_inputs: i32, num_neurons: i32) -> Layer {
        let mut neurons = Vec::new();
        for _ in 0..num_neurons {
            neurons.push(Neuron::new(num_inputs));
        }
        Layer { neurons }
    }

    pub fn forward(&self, data: &[Value], activation: bool) -> Vec<Value> {
        let mut output = Vec::new();
        for neuron in &self.neurons {
            output.push(neuron.forward(data, activation));
        }
        output
    }
}

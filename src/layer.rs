use crate::value::Value;
use ndarray::Array2;
use rand::{Rng, RngExt};
#[derive(Debug)]
pub struct Layer {
    weights: Value,
    bias: Value,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Layer {
        let mut rng = rand::rng();
        let weights = Array2::from_shape_fn((input_size, output_size), |idx| {
            let number: f32 = rng.random();
            number
        });
        let bias = Array2::from_shape_fn((1, output_size), |idx| {
            let number: f32 = rng.random();
            number
        });
        Layer {
            weights: Value::new(weights),
            bias: Value::new(bias),
        }
    }

    pub fn forward(&self, data: &Value, activation: bool) -> Value {
        let mut output = data.matmul(&self.weights);
        output += &self.bias;
        if activation {
            output = output.relu();
        }
        output
    }
    pub fn parameters(&self) -> Vec<Value> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}

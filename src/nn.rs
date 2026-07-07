use crate::value::Value;
use rand::{Rng, RngExt};
use std::ops::{Add, Mul};
#[derive(Debug)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(input_size: i32) -> Neuron {
        let data = || {
            let mut rng = rand::rng();
            let num: f32 = rng.random();
            num
        };
        let mut weights = Vec::new();
        for _ in 0..input_size {
            weights.push(Value::new(data()));
        }
        Self {
            weights,
            bias: Value::new(data()),
        }
    }

    pub fn forward(&mut self, data: &[Value]) -> Value {
        let mut total = self.bias.clone();
        for (w, x) in self.weights.iter().zip(data.iter()) {
            let product = w * x;
            total += &product;
        }
        total
    }
}

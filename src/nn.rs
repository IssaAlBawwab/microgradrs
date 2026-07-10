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

    pub fn forward(&self, data: &[Value], activation: bool) -> Value {
        let mut total = self.bias.clone();
        for (w, x) in self.weights.iter().zip(data.iter()) {
            let product = w * x;
            total += &product;
        }
        if activation { total.relu() } else { total }
    }
    pub fn parameters(&self) -> Vec<Value> {
        let mut params = vec![self.bias.clone()];
        for weight in &self.weights {
            params.push(weight.clone());
        }
        params
    }
}

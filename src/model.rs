use ndarray::{Array, Array2, Zip, iter};

use crate::{
    layer::Layer,
    value::{Value, back_propagate},
};

#[derive(Debug)]
pub struct Model {
    layers: Vec<Layer>,
}

impl Model {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }
    pub fn forward(&self, data: &Value) -> Value {
        let mut pred = data.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            if i == self.layers.len() - 1 {
                pred = layer.forward(&pred, false);
            } else {
                pred = layer.forward(&pred, true);
            }
        }
        pred
    }
    pub fn parameters(&self) -> Vec<Value> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    pub fn fit(&mut self, epoch: i32, xs: Vec<Array2<f32>>, ys: Vec<Array2<f32>>, lr: f32) {
        let xs: Vec<Value> = xs.into_iter().map(|val| Value::new(val)).collect();
        let ys: Vec<Value> = ys.into_iter().map(|val| Value::new(val)).collect();

        for _ in 0..epoch {
            let mut loss_accumulation = 0.0;
            let mut total_losses = 0;
            for (x, y) in xs.iter().zip(ys.iter()) {
                let pred = self.forward(x);
                let mut loss = Value::new(Array::zeros(y.data().raw_dim()));
                let pred_len: usize = pred.data().len();
                let diff = &pred - &y;
                loss += &(&diff * &diff);
                let mut div_arr = Array::zeros(loss.data().raw_dim());
                div_arr.fill(1.0 / pred_len as f32);
                loss = &loss * &Value::new(div_arr);
                total_losses += 1;
                loss_accumulation += loss.data().sum();
                let loss_dim = loss.gradient().raw_dim();
                loss.update_gradient(Array::ones(loss_dim));
                back_propagate(loss);
                let params = self.parameters();
                for param in &params {
                    let update = lr * &*param.gradient();
                    param.subtract_value(update);
                }
                for param in &params {
                    param.zero_gradient();
                }
                println!("{}", loss_accumulation / total_losses as f32)
            }
        }
    }
}

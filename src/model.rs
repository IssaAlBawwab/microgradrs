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
    pub fn forward(&self, data: &[Value]) -> Vec<Value> {
        let mut pred: Vec<Value> = data.to_vec();
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

    pub fn fit(&mut self, epoch: i32, xs: Vec<Vec<f32>>, ys: Vec<Vec<f32>>, lr: f32) {
        let xs: Vec<Vec<Value>> = xs
            .into_iter()
            .map(|row| row.into_iter().map(|val| Value::new(val)).collect())
            .collect();
        let ys: Vec<Vec<Value>> = ys
            .into_iter()
            .map(|row| row.into_iter().map(|val| Value::new(val)).collect())
            .collect();
        for _ in 0..epoch {
            let mut loss_accumulation = 0.0;
            let mut total_losses = 0;
            for (x, y) in xs.iter().zip(ys.iter()) {
                let pred = self.forward(x);
                let mut loss = Value::new(0.0);
                let pred_len = pred.len();
                for (predicition, truth) in pred.iter().zip(y) {
                    let diff = predicition - truth;
                    loss += &(&diff * &diff);
                }
                loss = &loss * &Value::new(1.0 / pred_len as f32);
                total_losses += 1;
                loss_accumulation += loss.data();
                loss.update_gradient(1.0);
                back_propagate(loss);
                let params = self.parameters();
                for param in &params {
                    param.subtract_value(lr * param.gradient());
                }
                for param in &params {
                    param.zero_gradient();
                }
                println!("{}", loss_accumulation / total_losses as f32)
            }
        }
    }
}

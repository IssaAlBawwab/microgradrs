use crate::{
    layer::{self, Layer},
    value::Value,
};

#[derive(Debug)]
pub struct Model {
    layers: Vec<Layer>,
}

impl Model {
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
    pub fn fit(&mut self, epoch: i32, xs: Vec<Vec<f32>>, ys: Vec<Value>) {
        for i in 0..epoch {
            for (x, y) in xs.iter().zip(ys.iter()) {
                let pred = self.forward(x);
            }
        }
    }
}

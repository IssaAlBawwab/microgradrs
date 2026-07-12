use microgradrs::layer::Layer;
use microgradrs::model::Model;
use rand::{Rng, RngExt};
use ndarray::Array2;
fn poly(a: f32, b: f32, c: f32) -> f32 {
    2.0 * a + 3.0 * b - c
}

fn main() {
    let mut rng = rand::rng();
    let sample_count = 64;

    let mut data: Vec<Vec<f32>> = Vec::new();
    let mut truth: Vec<Vec<f32>> = Vec::new();

    for _ in 0..sample_count {
        let a: f32 = rng.random();
        let b: f32 = rng.random();
        let c: f32 = rng.random();
        data.push(vec![a, b, c]);
        truth.push(vec![poly(a, b, c)]);
    }

    let layer = Layer::new(3, 8);
    let layer_2 = Layer::new(8, 1);
    let mut model = Model::new(vec![layer, layer_2]);

    model.fit(200, data, truth, 0.05);
}

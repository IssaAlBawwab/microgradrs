use microgradrs::layer::Layer;
use microgradrs::model::Model;
use ndarray::{Array2, array};
use rand::{Rng, RngExt};
fn poly(a: f32, b: f32, c: f32) -> f32 {
    2.0 * a + 3.0 * b - c
}

fn main() {
    let mut rng = rand::rng();
    let sample_count = 64;

    let mut data: Vec<Array2<f32>> = Vec::new();
    let mut truth: Vec<Array2<f32>> = Vec::new();

    for _ in 0..sample_count {
        let a: f32 = rng.random();
        let b: f32 = rng.random();
        let c: f32 = rng.random();
        data.push(array![[a, b, c]]);
        truth.push(array![[poly(a, b, c)]]);
    }

    let layer = Layer::new(3, 8);
    let layer_2 = Layer::new(8, 1);
    let mut model = Model::new(vec![layer, layer_2]);

    model.fit(200, data, truth, 0.05);
}

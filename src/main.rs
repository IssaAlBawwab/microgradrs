use microgradrs::nn::Neuron;
use microgradrs::value::{Value, back_propagate};
fn main() {
    let mut x1 = Neuron::new(3);
    let data = vec![Value::new(5.0), Value::new(5.0), Value::new(5.0)];
    let output = back_propagate(x1.forward(&data));

    println!("{:#?}", output);
}

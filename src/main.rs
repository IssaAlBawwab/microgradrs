use microgradrs::nn::Neuron;
use microgradrs::value::{Value, back_propagate};
fn main() {
    let neuron = Neuron::new(3);
    println!("{:#?}", neuron);
}

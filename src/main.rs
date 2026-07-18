use microgradrs::data::{argmax, load_mnist};
use microgradrs::layer::Layer;
use microgradrs::model::Model;
use microgradrs::value::Value;

fn main() {
    let train_count = 1000;
    let (images, labels) = load_mnist(train_count);

    let eval_images: Vec<_> = images.iter().take(100).cloned().collect();
    let eval_labels: Vec<_> = labels.iter().take(100).cloned().collect();

    let mut model = Model::new(vec![Layer::new(784, 64), Layer::new(64, 10)]);

    model.fit(5, images, labels, 0.01);

    let mut correct = 0;
    for (img, label) in eval_images.iter().zip(eval_labels.iter()) {
        let pred = model.forward(&Value::new(img.clone()));
        println!("{}", pred.data());
        let out = pred.data();
        if argmax(&out) == argmax(label) {
            correct += 1;
        }
    }
    println!("accuracy: {}/{}", correct, eval_images.len());
}

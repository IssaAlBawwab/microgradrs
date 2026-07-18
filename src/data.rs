use mnist::MnistBuilder;
use ndarray::Array2;

pub fn load_mnist(count: usize) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {
    let mnist = MnistBuilder::new()
        .base_path("data")
        .label_format_digit()
        .training_set_length(count as u32)
        .validation_set_length(0)
        .test_set_length(0)
        .finalize();

    let images = (0..count)
        .map(|i| {
            let start = i * 28 * 28;
            let pixels: Vec<f32> = mnist.trn_img[start..start + 28 * 28]
                .iter()
                .map(|&p| p as f32 / 255.0)
                .collect();
            Array2::from_shape_vec((1, 28 * 28), pixels).expect("image reshape")
        })
        .collect();

    let labels = (0..count)
        .map(|i| {
            let digit = mnist.trn_lbl[i] as usize;
            let mut one_hot = vec![0.0f32; 10];
            one_hot[digit] = 1.0;
            Array2::from_shape_vec((1, 10), one_hot).expect("label reshape")
        })
        .collect();

    (images, labels)
}

pub fn argmax(row: &Array2<f32>) -> usize {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

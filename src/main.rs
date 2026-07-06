use microgradrs::{Value, back_propagate};

fn main() {
    let a = Value::new(1.0);
    let b = Value::new(2.0);
    let c = &a + &b;
    let d = &(&a * &b) + &c;
    let result = back_propagate(d);
    println!("{:#?}", result);
}

use microgradrs::{Operation, Value};

fn main() {
    let mut a = Value::new(1.0);
    println!("{:?}", a);
    let mut b = Value::new(2.0);
    let mut c = a + b;
    print!("{:?}", c);
}

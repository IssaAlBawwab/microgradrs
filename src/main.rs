use microgradrs::{Value, back_propogate, topo_sort};

fn main() {
    let a = Value::new(1.0);
    let b = Value::new(2.0);
    let c = &a + &b;
    let d = &(&a * &b) + &c;
    let mut sorted = topo_sort(d);
    back_propogate(&mut sorted);
    println!("{:#?}", sorted);
}

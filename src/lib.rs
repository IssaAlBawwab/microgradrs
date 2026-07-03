use std::cell::{Ref, RefCell};
use std::ops::Add;
use std::rc::Rc;
#[derive(Debug)]
pub struct Value {
    pub data: f32,
    pub op: Operation,
    pub gradient: f32,
    pub parents: Vec<Rc<RefCell<Value>>>,
}

impl Value {
    //pub fn backward(dyn  ) ->
    pub fn new(data: f32) -> Value {
        Self {
            data,
            op: Operation::None,
            gradient: 0.0,
            parents: vec![],
        }
    }
}
impl Add for Value {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self {
            data: self.data + other.data,
            op: Operation::Add,
            gradient: 0.0,
            parents: vec![Rc::new(RefCell::new(self)), Rc::new(RefCell::new(other))],
        }
    }
}
#[derive(Debug)]
pub enum Operation {
    Add,
    Mul,
    None,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut a = Value::new(1.0);
        print!("{:?}", a);
        let mut b = Value::new(2.0);
        let mut c = a + b;

        assert_eq!(c.data, 3.0);
    }
}

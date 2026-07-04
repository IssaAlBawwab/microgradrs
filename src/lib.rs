use std::cell::{Ref, RefCell};
use std::ops::Add;
use std::rc::Rc;
#[derive(Debug)]
pub struct ValueData {
    pub data: f32,
    pub op: Operation,
    pub gradient: f32,
    pub parents: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct Value(Rc<RefCell<ValueData>>);

impl Value {
    //pub fn backward(dyn  ) ->
    pub fn new(data: f32) -> Value {
        Value(Rc::new(RefCell::new(ValueData {
            data,
            op: Operation::None,
            gradient: 0.0,
            parents: vec![],
        })))
    }
    pub fn data(&self) -> f32 {
        self.0.borrow().data
    }
    pub fn gradient(&self) -> f32 {
        self.0.borrow().gradient
    }
}
impl<'a, 'b> Add<&'b Value> for &'a Value {
    type Output = Value;
    fn add(self, other: &'b Value) -> Value {
        let data = self.data() + other.data();
        Value(Rc::new(RefCell::new(ValueData {
            data,
            op: Operation::Add,
            gradient: 0.0,
            parents: vec![self.clone(), other.clone()],
        })))
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

        assert_eq!(c.data(), 3.0);
    }
}

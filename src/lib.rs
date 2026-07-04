use std::cell::{Ref, RefCell};
use std::ops::{Add, Mul};
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
    pub fn op(&self) -> Operation {
        self.0.borrow().op
    }
    pub fn backward(&mut self) {
        match self.op() {
            Operation::Add => {
                for parent in &self.0.borrow().parents {
                    parent.0.borrow_mut().gradient += self.gradient() * 1.0;
                }
            }
            Operation::Mul => {
                let data = self.0.borrow();
                let p1 = &data.parents[0];
                let p2 = &data.parents[1];
                let p1_data = p1.data();
                let p2_data = p2.data();
                p1.0.borrow_mut().gradient += self.gradient() * p2_data;
                p2.0.borrow_mut().gradient += self.gradient() * p1_data;
            }
            Operation::None => {}
        }
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

impl<'a, 'b> Mul<&'b Value> for &'a Value {
    type Output = Value;
    fn mul(self, rhs: &'b Value) -> Self::Output {
        let data = self.data() * rhs.data();
        Value(Rc::new(RefCell::new(ValueData {
            data,
            op: Operation::Mul,
            gradient: 0.0,
            parents: vec![self.clone(), rhs.clone()],
        })))
    }
}

#[derive(Debug, Copy, Clone)]
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
        let mut c = &a + &b;

        assert_eq!(c.data(), 3.0);
    }
}

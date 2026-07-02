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
    pub fn new(data: f32) -> Rc<RefCell<Value>> {
        Rc::new(RefCell::new(Self {
            data,
            op: Operation::None,
            gradient: 0.0,
            parents: vec![],
        }))
    }
}
impl Add for Rc<RefCell<Value>> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {}
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
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

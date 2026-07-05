use std::cell::{Ref, RefCell};
use std::ops::{Add, Mul};
use std::rc::Rc;
#[derive(Debug, PartialEq)]
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

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Operation {
    Add,
    Mul,
    None,
}

pub fn topo_sort(last: Value) -> Vec<Value> {
    let mut sorted: Vec<Value> = Vec::new();
    let mut visited: Vec<Value> = vec![last.clone()];
    let mut stack: Vec<Value> = vec![last.clone()];
    while !stack.is_empty() {
        if let Some(last) = stack.last().cloned() {
            let mut unvisited_flag = false;
            for parent in &last.0.borrow().parents {
                if !visited.contains(parent) {
                    visited.push(parent.clone());
                    stack.push(parent.clone());
                    unvisited_flag = true;
                } else {
                    continue;
                }
            }
            if !unvisited_flag && let Some(last_item) = stack.pop() {
                sorted.push(last_item);
            }
        }
    }
    sorted.reverse();
    sorted[0].0.borrow_mut().gradient = 1.0;
    sorted
}

pub fn back_propogate(list: &mut Vec<Value>) {
    for node in list {
        node.backward();
    }
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

use std::cell::RefCell;
use std::fmt::Write;
use std::ops::{Add, AddAssign, Mul};
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
            Operation::Relu => {
                let grad = if self.data() > 1e-9 {
                    self.gradient()
                } else {
                    0.0
                };
                self.0.borrow_mut().parents[0].0.borrow_mut().gradient += grad;
            }
        }
    }
    pub fn relu(&self) -> Value {
        let val = self.data();
        Value(Rc::new(RefCell::new(ValueData {
            data: val.max(0.0),
            op: Operation::Relu,
            gradient: 0.0,
            parents: vec![self.clone()],
        })))
    }
    pub fn mse(&self, truth: f32) -> f32 {
        0.5 * (self.data() - truth).powi(2)
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

impl<'a> AddAssign<&'a Value> for Value {
    fn add_assign(&mut self, rhs: &'a Value) {
        *self = &*self + rhs;
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Operation {
    Add,
    Mul,
    None,
    Relu,
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
    sorted
}

pub fn back_propagate(last: Value) -> Vec<Value> {
    let mut list = topo_sort(last);
    list.iter_mut().for_each(|node| node.backward());
    list
}

/// render with: `dot -Tsvg graph.dot -o graph.svg`
pub fn to_dot(root: &Value) -> String {
    let mut dot = String::new();
    writeln!(dot, "digraph {{").unwrap();
    writeln!(dot, "    rankdir=LR;").unwrap();
    writeln!(dot, "    node [shape=record];").unwrap();

    let mut visited: Vec<Value> = Vec::new();
    let mut stack: Vec<Value> = vec![root.clone()];

    while let Some(node) = stack.pop() {
        if visited.contains(&node) {
            continue;
        }
        visited.push(node.clone());

        let id = node_id(&node);
        let data = node.data();
        let grad = node.gradient();
        let op = node.op();

        writeln!(
            dot,
            "    {id} [label=\"{{ data: {data:.4} | grad: {grad:.4} }}\"];",
        )
        .unwrap();

        if op != Operation::None {
            let op_id = format!("{id}_op");
            let op_label = match op {
                Operation::Add => "+",
                Operation::Mul => "*",
                Operation::Relu => "ReLU",
                Operation::None => unreachable!(),
            };
            writeln!(dot, "    {op_id} [label=\"{op_label}\", shape=circle];").unwrap();
            writeln!(dot, "    {op_id} -> {id};").unwrap();

            let borrowed = node.0.borrow();
            for parent in &borrowed.parents {
                let parent_id = node_id(parent);
                writeln!(dot, "    {parent_id} -> {op_id};").unwrap();
                stack.push(parent.clone());
            }
        }
    }

    writeln!(dot, "}}").unwrap();
    dot
}

fn node_id(value: &Value) -> String {
    let ptr = Rc::as_ptr(&value.0) as usize;
    format!("node_{ptr:x}")
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

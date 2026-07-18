use ndarray::{Array, Array2, ArrayRef2, ArrayView2, Axis, Zip};
use std::cell::{Ref, RefCell};
use std::fmt::Write;
use std::ops::{Add, AddAssign, Mul, Sub};
use std::rc::Rc;
#[derive(Debug, PartialEq)]
pub struct ValueData {
    pub data: Array2<f32>,
    pub op: Operation,
    pub gradient: Array2<f32>,
    pub parents: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct Value(Rc<RefCell<ValueData>>);

impl Value {
    pub fn new(data: Array2<f32>) -> Value {
        Value(Rc::new(RefCell::new(ValueData {
            gradient: Array::zeros(data.raw_dim()),
            data,
            op: Operation::None,
            parents: vec![],
        })))
    }
    pub fn data<'a>(&'a self) -> Ref<'_, Array2<f32>> {
        Ref::map(self.0.borrow(), |value_data| &value_data.data)
    }
    pub fn gradient<'a>(&'a self) -> Ref<'_, Array2<f32>> {
        Ref::map(self.0.borrow(), |value_data| &value_data.gradient)
    }
    pub fn op(&self) -> Operation {
        self.0.borrow().op
    }
    pub fn update_gradient(&self, value: Array2<f32>) {
        self.0.borrow_mut().gradient = value;
    }
    pub fn subtract_value(&self, value: Array2<f32>) {
        self.0.borrow_mut().data -= &value;
    }
    pub fn zero_gradient(&self) {
        self.0.borrow_mut().gradient.fill(0.0);
    }
    pub fn backward(&mut self) {
        match self.op() {
            Operation::Add => {
                for parent in &self.0.borrow().parents {
                    let g = self.gradient();
                    let parent_shape = parent.data().shape().to_vec();
                    parent.0.borrow_mut().gradient += &unbroadcast(&g, &parent_shape)
                }
            }
            Operation::Sub => {
                let g = self.gradient();
                let shape = self.data().shape().to_vec();
                self.0.borrow().parents[0].0.borrow_mut().gradient += &unbroadcast(&g, &shape);
                self.0.borrow().parents[1].0.borrow_mut().gradient -= &unbroadcast(&g, &shape);
            }
            Operation::Mul => {
                let data = self.0.borrow();
                let p1 = &data.parents[0];
                let p2 = &data.parents[1];
                let p1_shape = p1.data().shape().to_vec();
                let p2_shape = p2.data().shape().to_vec();
                let p2_mul = &(&*self.gradient() * &*p2.data());
                let p1_mul = &(&*self.gradient() * &*p1.data());
                p1.0.borrow_mut().gradient += &unbroadcast(p2_mul, &p1_shape);
                p2.0.borrow_mut().gradient += &unbroadcast(p1_mul, &p2_shape);
            }
            Operation::MatMul => {
                let data = self.0.borrow();
                let p1 = &data.parents[0];
                let p2 = &data.parents[1];
                p1.0.borrow_mut().gradient += &(self.gradient().dot(&p2.data().t()));
                p2.0.borrow_mut().gradient += &(p1.data().t().dot(&*self.gradient()));
            }
            Operation::None => {}
            Operation::Relu => {
                let grad = Zip::from(&*self.data())
                    .and(&*self.gradient())
                    .map_collect(|d, g| if *d > 0.0 { *g } else { 0.0 });
                self.0.borrow_mut().parents[0].0.borrow_mut().gradient += &grad;
            }
        }
    }
    pub fn relu(&self) -> Value {
        let val = self.data();
        Value(Rc::new(RefCell::new(ValueData {
            data: val.mapv(|element| if element > 0.0 { element } else { 0.0 }),
            op: Operation::Relu,
            gradient: Array::zeros(val.raw_dim()),
            parents: vec![self.clone()],
        })))
    }
    pub fn mse(&self, truth: &Array2<f32>) -> Array2<f32> {
        0.5 * (&*self.data() - truth).powi(2)
    }
    pub fn matmul(&self, rhs: &Self) -> Self {
        assert!(
            self.data().shape()[1] == rhs.data().shape()[0],
            "Matmul dims not correct: lhs: {:?}, rhs: {:?}",
            self.data().shape(),
            rhs.data().shape()
        );
        let data = self.data().dot(&*rhs.data());
        Value(Rc::new(RefCell::new(ValueData {
            gradient: Array::zeros(data.raw_dim()),
            data,
            op: Operation::MatMul,
            parents: vec![self.clone(), rhs.clone()],
        })))
    }
}
fn unbroadcast(g: &Array2<f32>, target: &[usize]) -> Array2<f32> {
    let mut g = g.clone();
    if target[0] == 1 && g.shape()[0] > 1 {
        g = g.sum_axis(Axis(0)).insert_axis(Axis(0));
    }
    if target[1] == 1 && g.shape()[1] > 1 {
        g = g.sum_axis(Axis(1)).insert_axis(Axis(1));
    }
    g
}
impl<'a, 'b> Add<&'b Value> for &'a Value {
    type Output = Value;
    fn add(self, other: &'b Value) -> Value {
        let data = &*self.data() + &*other.data();
        Value(Rc::new(RefCell::new(ValueData {
            gradient: Array::zeros(data.raw_dim()),
            data,
            op: Operation::Add,
            parents: vec![self.clone(), other.clone()],
        })))
    }
}

impl<'a, 'b> Mul<&'b Value> for &'a Value {
    type Output = Value;
    fn mul(self, rhs: &'b Value) -> Self::Output {
        let data = &*self.data() * &*rhs.data();
        Value(Rc::new(RefCell::new(ValueData {
            gradient: Array::zeros(data.raw_dim()),
            data,
            op: Operation::Mul,
            parents: vec![self.clone(), rhs.clone()],
        })))
    }
}

impl<'a, 'b> Sub<&'b Value> for &'a Value {
    type Output = Value;
    fn sub(self, rhs: &'b Value) -> Self::Output {
        let data = &*self.data() - &*rhs.data();
        Value(Rc::new(RefCell::new(ValueData {
            gradient: Array::zeros(data.raw_dim()),
            data,
            op: Operation::Sub,
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
    Sub,
    None,
    Relu,
    MatMul,
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
            "    {id} [label=\"{{ data: {data:?} | grad: {grad:?} }}\"];",
        )
        .unwrap();

        if op != Operation::None {
            let op_id = format!("{id}_op");
            let op_label = match op {
                Operation::Add => "+",
                Operation::Mul => "*",
                Operation::MatMul => "@",
                Operation::Relu => "ReLU",
                Operation::Sub => "-",
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
        let a = Value::new(Array2::from_elem((2, 2), 1.0));
        let b = Value::new(Array2::from_elem((2, 2), 2.0));
        let c = &a + &b;

        assert_eq!(*c.data(), Array2::from_elem((2, 2), 3.0));
    }
}

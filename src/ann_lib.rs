use std::f64::consts::E;
use rand::prelude::*;

#[derive(Clone)]
pub enum Activation {
    Relu,
    Sigmoid
}

#[derive(Clone)]
pub enum Node {
    Neurone(Neurone),
    F64(f64)
}

#[derive(Clone)]
pub struct Neurone {
    weights: Vec<f64>,
    bias: f64,
    activation: Activation,
    pub output: f64,
}

impl Neurone {
    pub fn new(input_size: usize, activation: Activation) -> Self {
        let mut weights: Vec<f64> = vec![];
        for _ in 0..input_size {
            weights.push(f64_random());
        }
        Self { weights, bias: f64_random()*10.0, output: 0.0, activation }
    }
    pub fn calculate(&mut self, previous: Vec<Node>) {
        self.output = 0.0;
        for n in 0..previous.len() {
            match &previous[n] {
                Node::F64(x) => self.output += x * self.weights[n],
                Node::Neurone(x) => self.output += x.output * self.weights[n]
            }
        }
        self.output = self.output + self.bias;
        // if self.output > 1.0 || self.output < 0.0 {          // Used for Sigmoid activation
        //     panic!("boundary exceeded for some reason: {}", self.output);
        // }
    }
    pub fn activate(&mut self) {
        match self.activation {
            Activation::Relu => self.output = relu(self.output),
            Activation::Sigmoid => self.output = sigmoid(self.output)
        }
    }
}

fn relu(n:f64) -> f64 {
    n.max(0.0)
}
fn sigmoid(n:f64) -> f64 {
    1.0/(1.0+E.powf(-n))
}

pub fn f64_random() -> f64 {
    let mut rng = rand::thread_rng();
    rng.gen::<f64>() *2.0 - 1.0
}

pub struct ANN {
    net : Vec<Vec<Node>>
}

#[allow(dead_code)]
impl ANN {
    pub fn new(input_size: usize, hidden_layer_sizes: Vec<usize>, output_layer_size: usize) -> ANN {
        let mut q: Vec<Vec<Node>> = vec![];
        q.push(create_first_layer(input_size));
        for i in 0..hidden_layer_sizes.len() {
            q.push(create_layer(hidden_layer_sizes[i], q[i].len(), Activation::Relu));
        }
        q.push(create_layer(output_layer_size, q[q.len()-1].len(), Activation::Sigmoid));
        Self { net: q }
    }
    pub fn calculate(&mut self) {
        let mut net_clone = self.net.clone();
        for i in 1..self.net.len() {
            self.net[i].iter_mut().for_each(|n| {
                
                match n {
                    Node::Neurone(x) => {
                        x.calculate(net_clone[i-1].clone());
                        x.activate();
                    },
                    _ => {},
                }
                
            });
            net_clone = self.net.clone();
        }
    }
    pub fn get_out_layer(&self) -> Vec<Node> {
        self.net[self.net.len()-1].clone()
    }
    pub fn get_out(&self) -> Vec<f64> {
        let mut q: Vec<f64> = vec![];
        for n in self.get_out_layer().into_iter() {
            match n {
                Node::Neurone(x) => q.push(x.output),
                Node::F64(x) => q.push(x)
            }
        }
        q
    }
    pub fn set_ins(&mut self, ins: Vec<f64>) {
        if ins.len() != self.net[0].len() {
            panic!("Length of new input layer does not match length of old input layer. {} vs {}", ins.len(), self.net[0].len());
        }
        let mut new_in_layer: Vec<Node> = vec![];
        for v in ins {
            new_in_layer.push(Node::F64(v));
        }
        self.net[0] = new_in_layer;
    }
    pub fn calculate_error(&self, expected: Vec<f64>) -> f64 {
        let mut q = 0.0;
        let comparison_values: Vec<f64> = self.get_out();
        for i in 0..expected.len() {
            q += (comparison_values[i] - expected[i]).powi(2);
        }
        q
    }
}

fn create_layer(size: usize, previous_size: usize, activation: Activation) -> Vec<Node> {
    (0..size).map(|_| Neurone::new(previous_size, activation.clone())).map(Node::Neurone).collect()
}
fn create_first_layer(size: usize) -> Vec<Node> {
    (0..size).map(|_| f64_random()).map(Node::F64).collect()
}

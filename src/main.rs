mod ann_lib;
use ann_lib::ANN;


fn main() {
    // Setup neural network
    let mut ann: ANN = ANN::new(2, vec![10, 5], 2);

    ann.set_ins(vec![1.0_f64, 0.0_f64]);

    // Process data
    ann.calculate();

    // Display outputs of the output layer
    println!("Output values:\n{:#?}", ann.get_out());
    // Display error of Network
    println!("Error Value:\n{}", ann.calculate_error(vec![0.0_f64, 1.0_f64]));
}



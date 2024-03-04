#![allow(dead_code)]
#![allow(unused)]

use orc::io::read_mesh;
use orc::solver::build_solution_matrices;
use orc::solver::{MomentumDiscretization, PressureInterpolation};

fn main() {
    env_logger::init();
    // Interface: allow user to choose from
    // 1. Read mesh
    // 2. Read data
    // 3. Read settings
    // 4. Write mesh
    // 5. Write data
    // 6. Write settings
    // 7. Run solver
    println!("Starting.");
    let mut mesh = orc::io::read_mesh("./examples/3x3_cube.msh");
    let linear_system = build_solution_matrices(&mut mesh, MomentumDiscretization::UD, PressureInterpolation::Linear, 1000., 0.001);
    println!("Complete.");
}

#![allow(dead_code)]
#![allow(unused)]

use orc::io::read_mesh;
use orc::solver::*;
use orc::mesh::*;

fn main() {
    env_logger::init();
    // Interface: allow user to choose from
    // 1. Read mesh
    println!("Starting.");
    println!("Reading mesh.");
    let mut mesh = orc::io::read_mesh("./examples/2D_2x4.msh");
    mesh.get_face_zone("INLET").zone_type = BoundaryConditionTypes::PressureInlet;
    mesh.get_face_zone("OUTLET").zone_type = BoundaryConditionTypes::PressureOutlet;
    mesh.get_face_zone("PERIODIC_-Z").zone_type = BoundaryConditionTypes::Wall;
    mesh.get_face_zone("PERIODIC_+Z").zone_type = BoundaryConditionTypes::Wall;

    println!("Building solution matrices.");
    let linear_system = build_solution_matrices(
        &mut mesh,
        MomentumDiscretization::UD,
        PressureInterpolation::Linear,
        1000.,
        0.001,
    );
    // 2. Read data
    // 3. Read settings
    // 4. Write mesh
    // 5. Write data
    // 6. Write settings
    // 7. Run solver
    println!("Complete.");
}

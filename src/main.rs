// #![allow(dead_code)]
// #![allow(unused)]

use orc::io::read_mesh;
use orc::mesh::*;
use orc::solver::*;

fn test_2d() -> Mesh {
    let domain_height = 1.;
    let domain_length = 2.;
    let cell_height = domain_height / 3.;
    let cell_width = domain_length / 6.;
    let cell_volume = cell_width * cell_height;
    let face_min = f32::min(cell_width, cell_height);
    let face_max = f32::max(cell_width, cell_height);
    let mut mesh = orc::io::read_mesh("./examples/2D_3x6.msh");

    mesh.get_face_zone("INLET").zone_type = FaceConditionTypes::PressureInlet;
    mesh.get_face_zone("OUTLET").zone_type = FaceConditionTypes::PressureOutlet;
    mesh.get_face_zone("BOTTOM").zone_type = FaceConditionTypes::Wall;
    mesh.get_face_zone("TOP").zone_type = FaceConditionTypes::Wall;

    let (face_min_actual, face_max_actual) =
        mesh.faces.iter().fold((face_max, face_min), |acc, (i, f)| {
            (f32::min(acc.0, f.area), f32::max(acc.1, f.area))
        });
    let area_tolerance = 0.001;
    if (face_min_actual + area_tolerance < face_min) {
        panic!("face calculated as too small");
    }
    if (face_max_actual - area_tolerance > face_max) {
        panic!("face calculated as too large");
    }
    let cell_tolerance = 0.0001;
    for (i, cell) in &mesh.cells {
        if f32::abs(cell.volume - cell_volume) > cell_tolerance {
            println!("Volume should be: {cell_volume}");
            println!("Volume is: {}", cell.volume);
            panic!("cell volume wrong");
        }
    }

    mesh
}

fn test_3d() -> Mesh {
    let cell_size = 1. / 3.;
    let face_area = cell_size * cell_size;
    let cell_volume = face_area * cell_size;
    let mut mesh = orc::io::read_mesh("./examples/3x3_cube.msh");

    mesh.get_face_zone("INLET").zone_type = FaceConditionTypes::PressureInlet;
    mesh.get_face_zone("OUTLET").zone_type = FaceConditionTypes::PressureOutlet;
    mesh.get_face_zone("PERIODIC_-Z").zone_type = FaceConditionTypes::Wall;
    mesh.get_face_zone("PERIODIC_+Z").zone_type = FaceConditionTypes::Wall;

    let (face_min_actual, face_max_actual) =
        mesh.faces.iter().fold((face_area, face_area), |acc, (i, f)| {
            (f32::min(acc.0, f.area), f32::max(acc.1, f.area))
        });
    let area_tolerance = 0.001;
    if (face_min_actual + area_tolerance < face_area) {
        panic!("face calculated as too small");
    }
    if (face_max_actual - area_tolerance > face_area) {
        panic!("face calculated as too large");
    }
    let cell_tolerance = 0.0001;
    for (i, cell) in &mesh.cells {
        if f32::abs(cell.volume - cell_volume) > cell_tolerance {
            println!("Volume should be: {cell_volume}");
            println!("Volume is: {}", cell.volume);
            panic!("cell volume wrong");
        }
    }

    mesh
}

fn main() {
    env_logger::init();
    // Interface: allow user to choose from
    // 1. Read mesh
    println!("Starting.");
    println!("Reading mesh.");
    let mut mesh = test_2d();
    println!("Building solution matrices.");
    let linear_system = build_discretized_momentum_matrices(
        &mut mesh,
        MomentumDiscretization::UD,
        PressureInterpolation::Linear,
        VelocityInterpolation::Linear,
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

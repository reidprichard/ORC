#![allow(dead_code)]
#![allow(unused)]

use orc::common::Float;
use orc::io::read_mesh;
use orc::io::write_data;
use orc::mesh::*;
use orc::solver::*;
use orc::common::Vector;
use sprs::{CsMat, TriMat};

fn test_gauss_seidel() {
    println!("*** Testing Gauss-Seidel for correctness. ***");
    const TOL: Float = 1e-6;
    // | 2 0 1 |   | 3 |
    // | 0 3 2 | = | 2 |
    // | 2 0 4 |   | 1 |
    //
    // | 1 0 0 |   | 11/6 |   | 1.833 |
    // | 0 1 0 |   | 10/9 |   | 1.111 |
    // | 0 0 1 | = | -2/3 | = | -0.67 |
    let mut a_tri: TriMat<Float> = TriMat::new((3, 3));
    a_tri.add_triplet(0, 0, 2.);
    a_tri.add_triplet(0, 2, 1.);

    a_tri.add_triplet(1, 1, 3.);
    a_tri.add_triplet(1, 2, 2.);

    a_tri.add_triplet(2, 0, 2.);
    a_tri.add_triplet(2, 2, 4.);

    let a = a_tri.to_csr();
    // let mut a = CsMat::new((3, 3), vec![2., 0., 1.], vec![0., 3., 2.], vec![2., 0., 4.]);
    let b = vec![3., 2., 1.];

    let mut x = vec![0., 0., 0.];

    solve_linear_system(&a, &b, &mut x, 20, SolutionMethod::GaussSeidel);

    for row_num in 0..a.rows() {
        assert!(
            Float::abs(
                a.get(row_num, 0).unwrap_or(&0.) * x[0]
                    + a.get(row_num, 1).unwrap_or(&0.) * x[1]
                    + a.get(row_num, 2).unwrap_or(&0.) * x[2]
                    - b[row_num]
            ) < TOL
        );
    }

    println!("x = {x:?}");
    println!("*** Gauss-Seidel test passed. ***");
}

fn test_2d() {
    let domain_height = 1.;
    let domain_length = 2.;
    let cell_height = domain_height / 3.;
    let cell_width = domain_length / 6.;
    let cell_volume = cell_width * cell_height;
    let face_min = Float::min(cell_width, cell_height);
    let face_max = Float::max(cell_width, cell_height);
    let mut mesh = read_mesh("./examples/2D_3x6.msh");

    mesh.get_face_zone("INLET").zone_type = FaceConditionTypes::PressureInlet;
    mesh.get_face_zone("INLET").scalar_value = 100.;

    mesh.get_face_zone("OUTLET").zone_type = FaceConditionTypes::PressureOutlet;
    mesh.get_face_zone("OUTLET").scalar_value = 0.;

    mesh.get_face_zone("BOTTOM").zone_type = FaceConditionTypes::Wall;

    mesh.get_face_zone("TOP").zone_type = FaceConditionTypes::Wall;

    let (face_min_actual, face_max_actual) =
        mesh.faces.iter().fold((face_max, face_min), |acc, (_, f)| {
            (Float::min(acc.0, f.area), Float::max(acc.1, f.area))
        });
    const AREA_TOL: Float = 0.001;
    if face_min_actual + AREA_TOL < face_min {
        panic!("face calculated as too small");
    }
    if face_max_actual - AREA_TOL > face_max {
        panic!("face calculated as too large");
    }
    const VOLUME_TOL: Float = 0.0001;
    for (_, cell) in &mesh.cells {
        if Float::abs(cell.volume - cell_volume) > VOLUME_TOL {
            println!("Volume should be: {cell_volume}");
            println!("Volume is: {}", cell.volume);
            panic!("cell volume wrong");
        }
    }

    solve_steady(
        &mut mesh,
        PressureVelocityCoupling::SIMPLE,
        MomentumDiscretization::UD,
        PressureInterpolation::Linear,
        VelocityInterpolation::Linear,
        1000.,
        0.001,
        10,
    );
    write_data(&mesh, "2d_test_case.csv".into());
}

fn test_3d_1x3() {
    let cell_length = 1.;
    let face_area = Float::powi(cell_length, 2);
    let cell_volume = Float::powi(cell_length, 3);
    let mut mesh = orc::io::read_mesh("./examples/3d_1x3.msh");

    // mesh.get_face_zone("INLET").zone_type = FaceConditionTypes::VelocityInlet;
    // mesh.get_face_zone("INLET").vector_value = Vector {x: 0.1, y: 0., z: 0.};
    mesh.get_face_zone("INLET").zone_type = FaceConditionTypes::PressureInlet;
    mesh.get_face_zone("INLET").scalar_value = 3.; // 1 Pa/m = 1 Pa/cell

    mesh.get_face_zone("OUTLET").zone_type = FaceConditionTypes::PressureOutlet;
    mesh.get_face_zone("OUTLET").scalar_value = 0.;

    mesh.get_face_zone("WALL").zone_type = FaceConditionTypes::Wall;

    let (face_min_actual, face_max_actual) = mesh
        .faces
        .iter()
        .fold((face_area, face_area), |acc, (_, f)| {
            (Float::min(acc.0, f.area), Float::max(acc.1, f.area))
        });

    const AREA_TOL: Float = 0.001;
    if face_min_actual + AREA_TOL < face_area {
        panic!("face calculated as too small");
    }
    if face_max_actual - AREA_TOL > face_area {
        panic!("face calculated as too large");
    }
    const VOLUME_TOL: Float = 0.0001;
    for (_, cell) in &mesh.cells {
        if Float::abs(cell.volume - cell_volume) > VOLUME_TOL {
            println!("Volume should be: {cell_volume}");
            println!("Volume is: {}", cell.volume);
            panic!("cell volume wrong");
        }
    }

    solve_steady(
        &mut mesh,
        PressureVelocityCoupling::SIMPLE,
        MomentumDiscretization::CD,
        PressureInterpolation::Linear,
        VelocityInterpolation::Linear,
        1000.,
        10.,
        10,
    )
}

fn test_3d_3x3() {
    let cell_length = 1. / 3.;
    let face_area = Float::powi(cell_length, 2);
    let cell_volume = Float::powi(cell_length, 3);
    let mut mesh = orc::io::read_mesh("./examples/3x3_cube.msh");

    mesh.get_face_zone("INLET").zone_type = FaceConditionTypes::PressureInlet;
    mesh.get_face_zone("INLET").scalar_value = 1.;

    mesh.get_face_zone("OUTLET").zone_type = FaceConditionTypes::PressureOutlet;
    mesh.get_face_zone("OUTLET").scalar_value = 0.;

    mesh.get_face_zone("PERIODIC_-Z").zone_type = FaceConditionTypes::Wall;
    mesh.get_face_zone("PERIODIC_+Z").zone_type = FaceConditionTypes::Wall;

    let (face_min_actual, face_max_actual) = mesh
        .faces
        .iter()
        .fold((face_area, face_area), |acc, (_, f)| {
            (Float::min(acc.0, f.area), Float::max(acc.1, f.area))
        });
    const AREA_TOL: Float = 0.001;
    if face_min_actual + AREA_TOL < face_area {
        panic!("face calculated as too small");
    }
    if face_max_actual - AREA_TOL > face_area {
        panic!("face calculated as too large");
    }
    const VOLUME_TOL: Float = 0.0001;
    for (_, cell) in &mesh.cells {
        if Float::abs(cell.volume - cell_volume) > VOLUME_TOL {
            println!("Volume should be: {cell_volume}");
            println!("Volume is: {}", cell.volume);
            panic!("cell volume wrong");
        }
    }

    solve_steady(
        &mut mesh,
        PressureVelocityCoupling::SIMPLE,
        MomentumDiscretization::CD,
        PressureInterpolation::Linear,
        VelocityInterpolation::Linear,
        1000.,
        100.,
        100,
    )
}

fn main() {
    env_logger::init();
    test_gauss_seidel();
    // Interface: allow user to choose from
    // 1. Read mesh
    // test_2d();
    test_3d_1x3();
    // test_3d_3x3();
    // test_3d();
    // 2. Read data
    // 3. Read settings
    // 4. Write mesh
    // 5. Write data
    // 6. Write settings
    // 7. Run solver
    println!("Complete.");
}

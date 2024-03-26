#![allow(dead_code)]
#![allow(unused)]

use orc::common::Float;
use orc::common::{Uint, Vector};
use orc::io::read_mesh;
use orc::io::write_data;
use orc::mesh::*;
use orc::solver::*;
use sprs::{CsMat, CsVec, TriMat};
use std::env;

const PRESSURE_RELAXATION: Float = 0.4;
const MOMENTUM_RELAXATION: Float = 1.0;

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
    let b = CsVec::new(3, vec![0,1,2], vec![3., 2., 1.]);

    let mut x = CsVec::new(3, vec![0,1,2], vec![0., 0., 0.]);

    solve_linear_system(&a, &b, &mut x, 20, SolutionMethod::GaussSeidel, 1.0);

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

fn test_2d(iteration_count: Uint) {
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
        iteration_count,
        MOMENTUM_RELAXATION,
        PRESSURE_RELAXATION,
    );
}

fn test_3d_1x3(iteration_count: Uint, momentum_relaxation: Float, pressure_relaxation: Float) {
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

    // High viscosity needed to keep Peclet under control
    let (u, v, w, p) = solve_steady(
        &mut mesh,
        PressureVelocityCoupling::SIMPLE,
        MomentumDiscretization::CD,
        PressureInterpolation::Linear,
        VelocityInterpolation::Linear,
        1000.,
        100.,
        iteration_count,
        momentum_relaxation,
        pressure_relaxation,
    );

    for cell_number in 0..mesh.cells.len() {
        let cell_velocity = Vector {
            x: u[cell_number],
            y: v[cell_number],
            z: w[cell_number],
        };
        assert!(cell_velocity.approx_equals(
            &Vector {
                x: 0.005,
                y: 0.,
                z: 0.
            },
            1e-6
        ));
    }
}

fn test_3d_3x3(iteration_count: Uint, momentum_relaxation: Float, pressure_relaxation: Float) {
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

    let (u, v, w, p) = solve_steady(
        &mut mesh,
        PressureVelocityCoupling::SIMPLE,
        MomentumDiscretization::CD,
        PressureInterpolation::Linear,
        VelocityInterpolation::Linear,
        1000.,
        100.,
        iteration_count,
        momentum_relaxation,
        pressure_relaxation,
    );

    let mut avg_velocity = Vector::zero();
    for cell_number in 0..mesh.cells.len() {
        let cell_velocity = Vector {
            x: u[cell_number],
            y: v[cell_number],
            z: w[cell_number],
        };
        // assert!(cell_velocity.approx_equals(
        //     &Vector {
        //         x: -7.54e-2,
        //         y: 0.,
        //         z: 0.
        //     },
        //     1e-2
        // ));
        avg_velocity += cell_velocity;
    }
    avg_velocity /= mesh.cells.len();
    // assert!(avg_velocity.approx_equals(
    //     &Vector {
    //         x: -7.54e-2,
    //         y: 0.,
    //         z: 0.
    //     },
    //     1e-3
    // ));
    // write_data(&mesh, &u, &v, &w, &p, "./examples/3d_3x3.csv".into());
}

fn couette(iteration_count: Uint, momentum_relaxation: Float, pressure_relaxation: Float) {
    let mut mesh = orc::io::read_mesh("./examples/coutte_flow.msh");

    mesh.get_face_zone("INLET").zone_type = FaceConditionTypes::PressureInlet;
    mesh.get_face_zone("INLET").scalar_value = 0.001;

    mesh.get_face_zone("OUTLET").zone_type = FaceConditionTypes::PressureOutlet;
    mesh.get_face_zone("OUTLET").scalar_value = 0.;

    mesh.get_face_zone("PERIODIC_-Z").zone_type = FaceConditionTypes::Symmetry;
    mesh.get_face_zone("PERIODIC_+Z").zone_type = FaceConditionTypes::Symmetry;

    let (u, v, w, p) = solve_steady(
        &mut mesh,
        PressureVelocityCoupling::SIMPLE,
        MomentumDiscretization::CD,
        PressureInterpolation::Linear,
        VelocityInterpolation::Linear,
        1000.,
        0.001,
        iteration_count,
        momentum_relaxation,
        pressure_relaxation,
    );
    // V = U_top / 2 - a^2 / 12mu * dp/dx
    // a = 0.001; dx = 0.01; mu = 0.001; dp = -0.01
    // V = 0.0000833333 = 8.33e-5

    // write_data(&mesh, &u, &v, &w, &p, "./examples/couette.csv".into());
}

fn main() {
    env_logger::init();
    let args: Vec<String> = env::args().collect();
    let iteration_count: Uint = args
        .get(1)
        .unwrap_or(&"10".to_string())
        .parse()
        .expect("arg 1 should be an integer");
    // test_gauss_seidel();
    // test_2d();
    test_3d_1x3(1000, 1.0, 0.4);
    // test_3d_3x3(iteration_count, 1.0, 0.4);
    // test_3d();
    // couette(iteration_count, 0.5, 0.2);

    // Interface: allow user to choose from
    // 1. Read mesh
    // 2. Read data
    // 3. Read settings
    // 4. Write mesh
    // 5. Write data
    // 6. Write settings
    // 7. Run solver
    println!("Complete.");
}

#![allow(dead_code, unused)]

use log::log_enabled;
use nalgebra::DVector;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use orc::common::{Float, Tensor3};
use orc::common::{Uint, Vector3};
use orc::io::write_data;
use orc::io::{read_data, read_mesh};
use orc::mesh::*;
use orc::solver::*;
use rolling_file::{BasicRollingFileAppender, RollingConditionBasic};
use std::env;
use std::time::Instant;
use tracing::{info, Level};
use tracing_appender::rolling::{self, RollingFileAppender};
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::prelude::*;

// TODO: Set this with command line arg
const DEBUG: bool = false;

fn validate_solvers() {
    const TOL: Float = 1e-6;

    println!("*** Testing Jacobi solver for correctness. ***");
    // | 2 0 1 |   | 3 |
    // | 0 3 2 | = | 2 |
    // | 2 0 4 |   | 1 |
    //
    // | 1 0 0 |   | 11/6 |   | 1.833 |
    // | 0 1 0 | = | 10/9 | = | 1.111 |
    // | 0 0 1 |   | -2/3 |   | -0.67 |
    let mut a_coo: CooMatrix<Float> = CooMatrix::new(3, 3);
    a_coo.push(0, 0, 2.);
    a_coo.push(0, 2, 1.);

    a_coo.push(1, 1, 3.);
    a_coo.push(1, 2, 2.);

    a_coo.push(2, 0, 2.);
    a_coo.push(2, 2, 4.);

    let a = CsrMatrix::from(&a_coo);
    let b = DVector::from_column_slice(&[3., 2., 1.]);
    let mut x = DVector::from_column_slice(&[0., 0., 0.]);

    iterative_solve(
        &a,
        &b,
        &mut x,
        10000,
        SolutionMethod::Jacobi,
        0.5,
        TOL / 10.,
        PreconditionMethod::Jacobi,
    );

    for row_num in 0..a.nrows() {
        assert!(
            Float::abs(
                a.get_entry(row_num, 0).unwrap().into_value() * x[0]
                    + a.get_entry(row_num, 1).unwrap().into_value() * x[1]
                    + a.get_entry(row_num, 2).unwrap().into_value() * x[2]
                    - b[row_num]
            ) < TOL
        );
    }

    println!("x = {x:?}");
    println!("*** Jacobi solver validated. ***");

    // println!("*** Testing Gauss-Seidel solver for correctness. ***");
    // // | 2 0 1 |   | 3 |
    // // | 0 3 2 | = | 2 |
    // // | 2 0 4 |   | 1 |
    // //
    // // | 1 0 0 |   | 11/6 |   | 1.833 |
    // // | 0 1 0 |   | 10/9 |   | 1.111 |
    // // | 0 0 1 | = | -2/3 | = | -0.67 |
    // let mut a_coo: CooMatrix<Float> = CooMatrix::new(3, 3);
    // a_coo.push(0, 0, 2.);
    // a_coo.push(0, 2, 1.);
    //
    // a_coo.push(1, 1, 3.);
    // a_coo.push(1, 2, 2.);
    //
    // a_coo.push(2, 0, 2.);
    // a_coo.push(2, 2, 4.);
    //
    // let a = CsrMatrix::from(&a_coo);
    // // let mut a = CsMat::new((3, 3), vec![2., 0., 1.], vec![0., 3., 2.], vec![2., 0., 4.]);
    // let b = DVector::from_column_slice(&vec![3., 2., 1.]);
    // let mut x = DVector::from_column_slice(&vec![0., 0., 0.]);
    //
    // iterative_solve(&a, &b, &mut x, 100, SolutionMethod::GaussSeidel, 0.7);
    //
    // for row_num in 0..a.nrows() {
    //     assert!(
    //         Float::abs(
    //             a.get_entry(row_num, 0).unwrap().into_value() * x[0]
    //                 + a.get_entry(row_num, 1).unwrap().into_value() * x[1]
    //                 + a.get_entry(row_num, 2).unwrap().into_value() * x[2]
    //                 - b[row_num]
    //         ) < TOL
    //     );
    // }
    //
    // println!("x = {x:?}");
    // println!("*** Gauss-Seidel solver validated. ***");
    // TODO: Validate multigrid
}

fn channel_flow(iteration_count: Uint, reporting_interval: Uint) {
    // ************ Constants ********
    let channel_height = 0.001;
    let mu = 0.1;
    let rho = 1000.;
    let dp = -10.;
    let dx = 0.002;

    // *********** Read mesh ************
    let mut mesh = orc::io::read_mesh("./examples/couette_flow.msh");

    // ************ Set boundary conditions **********
    mesh.get_face_zone("WALL").zone_type = FaceConditionTypes::Wall;

    mesh.get_face_zone("INLET").zone_type = FaceConditionTypes::PressureInlet;
    mesh.get_face_zone("INLET").scalar_value = -dp;

    mesh.get_face_zone("OUTLET").zone_type = FaceConditionTypes::PressureOutlet;
    mesh.get_face_zone("OUTLET").scalar_value = 0.;

    mesh.get_face_zone("PERIODIC_-Z").zone_type = FaceConditionTypes::Symmetry;
    mesh.get_face_zone("PERIODIC_+Z").zone_type = FaceConditionTypes::Symmetry;

    // ************* Set numerical methods ***************
    let settings = NumericalSettings {
        momentum_relaxation: 0.8,
        // This needs to be EXTREMELY low (~0.01)
        // What is causing the solution to oscillate?
        pressure_relaxation: 0.02,
        matrix_solver: MatrixSolverSettings {
            solver_type: SolutionMethod::Multigrid,
            iterations: 50,
            // ~0.5 seems like roughly upper limit for Jacobi; does nothing for BiCGSTAB
            relaxation: 0.5,
            relative_convergence_threshold: 1e-3,
            preconditioner: PreconditionMethod::Jacobi,
        },
        momentum: TVD_QUICK,
        pressure_interpolation: PressureInterpolation::SecondOrder,
        velocity_interpolation: VelocityInterpolation::LinearWeighted,
        ..NumericalSettings::default()
    };

    // ************ Solve **************
    let (mut u, mut v, mut w, mut p) = read_data("./examples/channel_flow.csv")
        .unwrap_or_else(|_| initialize(&mesh, mu, rho, 1000));
    solve_steady(
        &mut mesh,
        &mut u,
        &mut v,
        &mut w,
        &mut p,
        &settings,
        rho,
        mu,
        iteration_count,
        Uint::max(reporting_interval, 1),
    );
    write_data(
        &mesh,
        &u,
        &v,
        &w,
        &p,
        "./examples/channel_flow.csv".into(),
        8,
    );

    let u_avg = u.iter().sum::<Float>() / (u.len() as Float);
    let u_max = u.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let u_avg_analytical = -(Float::powi(channel_height, 2) / (12. * mu)) * (dp / dx);
    let bulk_velocity_correct =
        Float::max(u_avg, u_avg_analytical) / Float::min(u_avg, u_avg_analytical) < 1.01;
    let max_velocity_correct = Float::abs(u_max / u_avg - 1.5) < 0.01;
    if bulk_velocity_correct && max_velocity_correct {
        print!("channel_flow flow validation passed.");
    } else {
        print!("channel_flow flow validation failed.");
    }
    println!(" U_mean = {u_avg:.2e}; U_mean_analytical = {u_avg_analytical:.2e}");
    println!(" U_max = {:.2e} = {:.1e}U_mean", u_max, u_max / u_avg);
}

fn main() {
    if DEBUG {
        let file_appender = BasicRollingFileAppender::new(
            "./logs/orc.log",
            RollingConditionBasic::new().max_size(2_u64.pow(26)),
            10,
        )
        .unwrap();
        // Start chatgippity
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(non_blocking)
                    .with_span_events(FmtSpan::CLOSE),
            )
            .init();
        // end chatgippity
    }

    let start = Instant::now();
    let args: Vec<String> = env::args().collect();
    let iteration_count: Uint = args
        .get(1)
        .unwrap_or(&"10".to_string())
        .parse()
        .expect("arg 1 should be an integer");
    let reporting_interval: Uint = args
        .get(2)
        .unwrap_or(&"0".to_string())
        .parse()
        .expect("arg 2 should be an integer");
    validate_solvers();
    channel_flow(iteration_count, reporting_interval);
    // test_3d_1x3(iteration_count);
    // Interface: allow user to choose from
    // 1. Read mesh
    // 2. Read data
    // 3. Read settings
    // 4. Write mesh
    // 5. Write data
    // 6. Write settings
    // 7. Run solver
    println!("Complete in {}s.", start.elapsed().as_secs());
}

// ************** old cases *************

fn test_2d(iteration_count: Uint) {
    let rho = 1000.;
    let mu = 0.001;
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

    let settings = NumericalSettings::default();

    let (mut u, mut v, mut w, mut p) = initialize(&mesh, mu, rho, 200);
    solve_steady(
        &mut mesh,
        &mut u,
        &mut v,
        &mut w,
        &mut p,
        &settings,
        rho,
        mu,
        iteration_count,
        Uint::max(iteration_count / 1000, 1),
    );
}

fn test_3d_1x3(iteration_count: Uint) {
    let rho = 1000.;
    let mu = 10.;
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

    let settings = NumericalSettings {
        momentum_relaxation: 0.8,
        // This needs to be EXTREMELY low (~0.01)
        // What is causing the solution to oscillate?
        pressure_relaxation: 0.02,
        matrix_solver: MatrixSolverSettings {
            solver_type: SolutionMethod::Jacobi,
            iterations: 100,
            relaxation: 0.2, // ~0.5 seems like roughly upper limit for Jacobi; does nothing for
            relative_convergence_threshold: 1e-3,
            preconditioner: PreconditionMethod::Jacobi,
        },
        momentum: MomentumDiscretization::UD,
        pressure_interpolation: PressureInterpolation::SecondOrder,
        velocity_interpolation: VelocityInterpolation::LinearWeighted,
        ..NumericalSettings::default()
    };

    let (mut u, mut v, mut w, mut p) = initialize(&mesh, mu, rho, 200);
    solve_steady(
        &mut mesh,
        &mut u,
        &mut v,
        &mut w,
        &mut p,
        &settings,
        rho,
        mu,
        iteration_count,
        Uint::max(iteration_count / 1000, 1),
    );

    for cell_number in 0..mesh.cells.len() {
        let cell_velocity = Vector3 {
            x: u[cell_number],
            y: v[cell_number],
            z: w[cell_number],
        };
        assert!(cell_velocity.approx_equals(
            &Vector3 {
                x: 0.005,
                y: 0.,
                z: 0.
            },
            1e-6
        ));
    }
}

fn test_3d_3x3(iteration_count: Uint) {
    let mu = 100.;
    let rho = 1000.;
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

    let settings = NumericalSettings::default();
    let (mut u, mut v, mut w, mut p) = initialize(&mesh, mu, rho, 200);
    solve_steady(
        &mut mesh,
        &mut u,
        &mut v,
        &mut w,
        &mut p,
        &settings,
        rho,
        mu,
        iteration_count,
        Uint::max(iteration_count / 1000, 1),
    );

    let mut avg_velocity = Vector3::zero();
    for cell_number in 0..mesh.cells.len() {
        let cell_velocity = Vector3 {
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

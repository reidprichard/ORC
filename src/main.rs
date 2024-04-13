#![allow(dead_code, unused)]

use orc::discretization::{
    build_momentum_advection_matrices, build_momentum_diffusion_matrix, initialize_momentum_matrix,
};
use orc::io::{print_vec_scientific, read_data, read_mesh, write_data, write_gradients};
use orc::mesh::*;
use orc::numerical_types::{Float, Tensor, Uint, Vector};
use orc::settings::*;
use orc::solver::*;
use orc::tests::channel_flow::*;

use log::log_enabled;
use nalgebra::DVector;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use rolling_file::{BasicRollingFileAppender, RollingConditionBasic};
use tracing::{info, Level};
use tracing_appender::rolling::{self, RollingFileAppender};
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::prelude::*;

use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

// TODO: Set this with command line arg
const DEBUG: bool = false;

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

    // solve_channel_flow(
    //     iteration_count,
    //     reporting_interval,
    //     ChannelFlowParameters {
    //         top_wall_velocity: 0.,
    //         dp_dx: 5.,
    //         mu: 0.001,
    //         rho: 1000.,
    //     },
    //     NumericalSettings {
    //         momentum: TVD_UMIST,
    //         pressure_interpolation: PressureInterpolation::SecondOrder,
    //         velocity_interpolation: VelocityInterpolation::RhieChow,
    //         ..NumericalSettings::default()
    //     },
    //     "channel_flow",
    //     0.1, // NOTE: Generous 10% validation threshold
    // );

    solve_channel_flow(
        iteration_count,
        reporting_interval,
        ChannelFlowParameters {
            top_wall_velocity: 5e-4,
            dp_dx: 5.,
            mu: 0.001,
            rho: 1000.,
        },
        NumericalSettings {
            momentum: TVD_UMIST,
            pressure_interpolation: PressureInterpolation::SecondOrder,
            velocity_interpolation: VelocityInterpolation::RhieChow,
            ..NumericalSettings::default()
        },
        "couette_flow",
        0.1, // NOTE: Generous 10% validation threshold
    );

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

    let (face_min_actual, face_max_actual) = mesh
        .faces
        .iter()
        .enumerate()
        .fold((face_max, face_min), |acc, (_, f)| {
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
    for cell_index in 0..mesh.cells.len() {
        let cell = &mesh.cells[cell_index];
        if Float::abs(cell.volume - cell_volume) > VOLUME_TOL {
            println!("Volume should be: {cell_volume}");
            println!("Volume is: {}", cell.volume);
            panic!("cell volume wrong");
        }
    }

    let settings = NumericalSettings::default();

    let (mut u, mut v, mut w, mut p) = initialize_flow(&mesh, mu, rho, 200);
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
        .enumerate()
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
    for cell_index in 0..mesh.cells.len() {
        let cell = &mesh.cells[cell_index];
        if Float::abs(cell.volume - cell_volume) > VOLUME_TOL {
            println!("Volume should be: {cell_volume}");
            println!("Volume is: {}", cell.volume);
            panic!("cell volume wrong");
        }
    }

    let settings = NumericalSettings::default();

    let (mut u, mut v, mut w, mut p) = initialize_flow(&mesh, mu, rho, 200);
    print_vec_scientific(&p);

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

    for cell_index in 0..mesh.cells.len() {
        let cell_velocity = Vector {
            x: u[cell_index],
            y: v[cell_index],
            z: w[cell_index],
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
        .enumerate()
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
    for cell_index in 0..mesh.cells.len() {
        let cell = &mesh.cells[cell_index];
        if Float::abs(cell.volume - cell_volume) > VOLUME_TOL {
            println!("Volume should be: {cell_volume}");
            println!("Volume is: {}", cell.volume);
            panic!("cell volume wrong");
        }
    }

    let settings = NumericalSettings::default();
    let (mut u, mut v, mut w, mut p) = initialize_flow(&mesh, mu, rho, 200);
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
    write_data(&mesh, &u, &v, &w, &p, "./examples/3d_3x3.csv".into());
}

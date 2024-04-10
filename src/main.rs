#![allow(dead_code, unused)]

use log::log_enabled;
use nalgebra::DVector;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use orc::discretization::{
    build_momentum_advection_matrices, build_momentum_diffusion_matrix, initialize_momentum_matrix,
};
use orc::io::{print_vec_scientific, read_data, read_mesh};
use orc::io::{write_data, write_gradients};
use orc::linear_algebra::iterative_solve;
use orc::mesh::*;
use orc::numerical_types::{Float, Tensor3};
use orc::numerical_types::{Uint, Vector3};
use orc::settings::*;
use orc::solver::*;
use rolling_file::{BasicRollingFileAppender, RollingConditionBasic};
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use tracing::{info, Level};
use tracing_appender::rolling::{self, RollingFileAppender};
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::prelude::*;

// TODO: Set this with command line arg
const DEBUG: bool = false;

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
        momentum_relaxation: 0.5,
        // This needs to be EXTREMELY low (~0.01)
        // What is causing the solution to oscillate?
        pressure_relaxation: 0.02,
        matrix_solver: MatrixSolverSettings::default(),
        momentum: MomentumDiscretization::UD,
        pressure_interpolation: PressureInterpolation::SecondOrder,
        velocity_interpolation: VelocityInterpolation::Linear,
        gradient_reconstruction: GradientReconstructionMethods::GreenGauss(
            GreenGaussVariants::CellBased,
        ),
        ..NumericalSettings::default()
    };

    // ************ Solve **************
    let (mut u, mut v, mut w, mut p) = read_data("./examples/channel_flow.csv")
        .unwrap_or_else(|_| initialize_flow(&mesh, mu, rho, 1000));
    solve_steady(
        &mut mesh,
        &mut u,
        &mut v,
        &mut w,
        &mut p,
        rho,
        mu,
        iteration_count,
        Uint::max(reporting_interval, 1),
    );
    write_data(&mesh, &u, &v, &w, &p, "./examples/channel_flow.csv");
    write_gradients(
        &mesh,
        &u,
        &v,
        &w,
        &p,
        "./examples/channel_flow_gradients.csv",
        7,
        settings.gradient_reconstruction,
    );

    let mut a_u = initialize_momentum_matrix(&mesh);
    let mut a_v = initialize_momentum_matrix(&mesh);
    let mut a_w = initialize_momentum_matrix(&mesh);
    let a_di = build_momentum_diffusion_matrix(&mesh, mu);
    let cell_count: usize = mesh.cells.len();
    macro_rules! dvector_zeros {
        ($n:expr) => {
            DVector::from_column_slice(&vec![0.; $n])
        };
    }
    let mut b_u = dvector_zeros!(cell_count);
    let mut b_v = dvector_zeros!(cell_count);
    let mut b_w = dvector_zeros!(cell_count);
    build_momentum_advection_matrices(
        &mut a_u, &mut a_v, &mut a_w, &mut b_u, &mut b_v, &mut b_w, &a_di, &mesh, &u, &v, &w, &p,
        rho,
    );
    let filename = "face_velocities";
    let mut file = File::create(format!("./examples/{filename}.csv")).unwrap();
    for face_index in 0..mesh.faces.len() {
        let face_vel = get_face_velocity(&mesh, &u, &v, &w, face_index);
        let cell_index = mesh.faces[face_index].cell_indices[0];
        writeln!(
            file,
            "{}\t{}\t{}",
            face_index, &mesh.faces[face_index].centroid, face_vel
        )
        .unwrap();
    }

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
    println!(" U_max/U_mean = {:.2e}", u_max / u_avg);
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
    // validate_solvers(); // TODO: Get this back up
    channel_flow(iteration_count, reporting_interval);
    // couette_flow(iteration_count, reporting_interval);
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

use orc::mesh::*;
use orc::numerical_types::Float;
use orc::numerical_types::Vector3;
use orc::settings::*;
use orc::solver::*;
use rolling_file::{BasicRollingFileAppender, RollingConditionBasic};
// use tracing_appender::rolling::{self, RollingFileAppender};
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::prelude::*;

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

    // ************ Constants ********
    let mu = 0.1;
    let rho = 1000.;
    let dp = -10.;
    let top_wall_velocity = 0.;

    // *********** Read mesh ************
    let mut mesh = orc::io::read_mesh("./examples/couette_flow_128x64x1.msh");

    // ************ Set boundary conditions **********
    mesh.get_face_zone("TOP_WALL").zone_type = FaceConditionTypes::Wall;
    mesh.get_face_zone("TOP_WALL").vector_value = Vector3 {
        x: top_wall_velocity,
        y: 0.,
        z: 0.,
    };

    mesh.get_face_zone("BOTTOM_WALL").zone_type = FaceConditionTypes::Wall;

    mesh.get_face_zone("INLET").zone_type = FaceConditionTypes::PressureInlet;
    mesh.get_face_zone("INLET").scalar_value = -dp;

    mesh.get_face_zone("OUTLET").zone_type = FaceConditionTypes::PressureOutlet;
    mesh.get_face_zone("OUTLET").scalar_value = 0.;

    mesh.get_face_zone("PERIODIC_-Z").zone_type = FaceConditionTypes::Symmetry;
    mesh.get_face_zone("PERIODIC_+Z").zone_type = FaceConditionTypes::Symmetry;

    // ************* Set numerical methods ***************
    let settings = NumericalSettings::default();

    // ************ Solve **************
    let (mut u, mut v, mut w, mut p) = initialize_flow(&mesh, mu, rho, 1000);
    solve_steady(
        &mut mesh,
        &mut u,
        &mut v,
        &mut w,
        &mut p,
        &settings,
        rho,
        mu,
        1000,
        100,
    );
    let u_avg = u.iter().sum::<Float>() / (u.len() as Float);
    let u_max = u.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    println!(" U_mean = {u_avg:.2e}, U_max = {u_max:.2e}");
}

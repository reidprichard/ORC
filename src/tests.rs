pub mod channel_flow {
    use crate::io::{read_data, read_mesh, write_data, write_gradients};
    use crate::mesh::FaceConditionTypes;
    use crate::numerical_types::{Float, Uint, Vector};
    use crate::settings::*;
    use crate::solver::{initialize_flow, initialize_flow_new, solve_steady};

    use std::fs::File;
    use std::io::Write;

    pub struct ChannelFlowParameters {
        pub top_wall_velocity: Float,
        pub dp_dx: Float,
        pub mu: Float,
        pub rho: Float,
    }

    fn write_couette_flow_analytical_profile(
        output_path: &str,
        parameters: ChannelFlowParameters,
        channel_height: Float,
    ) -> (Float, Float, Float) {
        let mut file = File::create(output_path).unwrap();
        const N: u16 = 128;
        for i in 0..N {
            // 1 / (2 * MU) * DP / DX * (y_linear**2 - a * y_linear)
            let y = (i as Float) / (N as Float) * channel_height;
            let u = parameters.top_wall_velocity * y / channel_height
                + 1. / (2. * parameters.mu) * parameters.dp_dx * (y.powi(2) - channel_height * y);
            writeln!(file, "{y:.3e},{u:.3e}").unwrap();
        }

        let u_extremum: Float = -(2. * parameters.mu * parameters.top_wall_velocity
            - channel_height.powi(2) * parameters.dp_dx)
            .powi(2)
            / (8. * channel_height.powi(2) * parameters.dp_dx * parameters.mu);
        let u_avg = parameters.top_wall_velocity / 2.
            - channel_height.powi(2) / (12. * parameters.mu) * parameters.dp_dx;
        let u_max = Float::max(Float::max(parameters.top_wall_velocity, 0.), u_extremum);
        let u_min = Float::min(Float::min(parameters.top_wall_velocity, 0.), u_extremum);
        (u_avg, u_min, u_max)
    }

    pub fn solve_channel_flow(
        iteration_count: Uint,
        reporting_interval: Uint,
        flow_parameters: ChannelFlowParameters,
        numerics: NumericalSettings,
        name: &str,
        validation_threshold: Float,
    ) {
        // ************ Constants ********
        const CHANNEL_HEIGHT: Float = 0.001;
        const DX: Float = 0.002;

        // *********** Read mesh ************
        let mut mesh = read_mesh("./examples/couette_flow_128x64x1.msh");

        // ************ Set boundary conditions **********
        mesh.get_face_zone("TOP_WALL").zone_type = FaceConditionTypes::Wall;
        mesh.get_face_zone("TOP_WALL").vector_value = Vector {
            x: flow_parameters.top_wall_velocity,
            y: 0.,
            z: 0.,
        };

        mesh.get_face_zone("BOTTOM_WALL").zone_type = FaceConditionTypes::Wall;

        mesh.get_face_zone("INLET").zone_type = FaceConditionTypes::PressureInlet;
        mesh.get_face_zone("INLET").scalar_value = -flow_parameters.dp_dx * DX;

        mesh.get_face_zone("OUTLET").zone_type = FaceConditionTypes::PressureOutlet;
        mesh.get_face_zone("OUTLET").scalar_value = 0.;

        mesh.get_face_zone("PERIODIC_-Z").zone_type = FaceConditionTypes::Symmetry;
        mesh.get_face_zone("PERIODIC_+Z").zone_type = FaceConditionTypes::Symmetry;

        // TODO: Clean this up
        let data_path: &str = &format!("./examples/{name}.csv")[..];
        let analytical_path: &str = &format!("./examples/{name}_analytical.csv")[..];
        let gradients_path: &str = &format!("./examples/{name}_gradients.csv")[..];

        // ************ Solve **************
        let (mut u, mut v, mut w, mut p) = read_data(data_path).unwrap_or_else(|_| {
            initialize_flow(&mesh, flow_parameters.mu, flow_parameters.rho, 1000)
        });
        solve_steady(
            &mut mesh,
            &mut u,
            &mut v,
            &mut w,
            &mut p,
            &numerics,
            flow_parameters.rho,
            flow_parameters.mu,
            iteration_count,
            Uint::max(reporting_interval, 1),
        );
        write_data(&mesh, &u, &v, &w, &p, data_path);
        write_gradients(
            &mesh,
            &u,
            &v,
            &w,
            &p,
            gradients_path,
            7,
            numerics.gradient_reconstruction,
        );

        let (u_mean_analytical, u_min_analytical, u_max_analytical) =
            write_couette_flow_analytical_profile(analytical_path, flow_parameters, CHANNEL_HEIGHT);

        let u_mean = u.iter().sum::<Float>() / (u.len() as Float);
        let u_min = *u.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
        let u_max = *u.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

        fn compare(value_1: Float, value_2: Float, tolerance: Float) -> bool {
            Float::max(value_1, value_2) / Float::min(value_1, value_2) - 1. < tolerance
        }

        let bulk_velocity_correct = compare(u_mean, u_mean_analytical, validation_threshold);
        let max_velocity_correct = compare(u_max, u_max_analytical, validation_threshold);
        let min_velocity_correct = compare(u_min, u_min_analytical, validation_threshold);
        const FIELD_WIDTH: usize = 8;
        println!(
            " U_mean:\tCFD = {:>FIELD_WIDTH$.2e}; Analytical = {:>FIELD_WIDTH$.2e}; Error = {:>6.1}%",
            u_mean, u_mean_analytical, (u_mean/u_mean_analytical - 1.)*100.
        );
        if !bulk_velocity_correct {
            println!("**FAIL**");
        }
        println!(
            " U_min: \tCFD = {:>FIELD_WIDTH$.2e}; Analytical = {:>FIELD_WIDTH$.2e}; Error = {:>6.1}%",
            u_min, u_min_analytical, (u_min/u_min_analytical - 1.)*100.
        );
        if !min_velocity_correct {
            println!("**FAIL**");
        }
        println!(
            " U_max: \tCFD = {:>FIELD_WIDTH$.2e}; Analytical = {:>FIELD_WIDTH$.2e}; Error = {:>6.1}%",
            u_max, u_max_analytical, (u_max/u_max_analytical - 1.)*100.
        );
        if !max_velocity_correct {
            println!("**FAIL**");
        }
        if bulk_velocity_correct && min_velocity_correct && max_velocity_correct {
            println!("{name} validation passed.");
        } else {
            println!("{name} validation failed.");
        }
    }

    pub fn solve_channel_flow_velocity_inlet(
        iteration_count: Uint,
        reporting_interval: Uint,
        numerics: NumericalSettings,
        name: &str,
        top_wall_velocity: Float,
        inlet_velocity: Float,
        mu: Float,
        rho: Float,
    ) {
        // *********** Read mesh ************
        let mut mesh = read_mesh("./examples/couette_flow_128x64x1.msh");

        // ************ Set boundary conditions **********
        mesh.get_face_zone("TOP_WALL").zone_type = FaceConditionTypes::Wall;
        mesh.get_face_zone("TOP_WALL").vector_value = Vector {
            x: top_wall_velocity,
            y: 0.,
            z: 0.,
        };

        mesh.get_face_zone("BOTTOM_WALL").zone_type = FaceConditionTypes::Wall;

        mesh.get_face_zone("INLET").zone_type = FaceConditionTypes::VelocityInlet;
        mesh.get_face_zone("INLET").vector_value = Vector {
            x: inlet_velocity,
            y: 0.,
            z: 0.,
        };

        mesh.get_face_zone("OUTLET").zone_type = FaceConditionTypes::PressureOutlet;
        mesh.get_face_zone("OUTLET").scalar_value = 0.;

        mesh.get_face_zone("PERIODIC_-Z").zone_type = FaceConditionTypes::Symmetry;
        mesh.get_face_zone("PERIODIC_+Z").zone_type = FaceConditionTypes::Symmetry;

        // TODO: Clean this up
        let data_path: &str = &format!("./examples/{name}.csv")[..];
        let gradients_path: &str = &format!("./examples/{name}_gradients.csv")[..];

        // ************ Solve **************
        let (mut u, mut v, mut w, mut p) = read_data(data_path).unwrap_or_else(|_| {
            initialize_flow_new(&mesh, mu, rho, 1000)
        });
        solve_steady(
            &mut mesh,
            &mut u,
            &mut v,
            &mut w,
            &mut p,
            &numerics,
            rho,
            mu,
            iteration_count,
            Uint::max(reporting_interval, 1),
        );
        write_data(&mesh, &u, &v, &w, &p, data_path);
        write_gradients(
            &mesh,
            &u,
            &v,
            &w,
            &p,
            gradients_path,
            7,
            numerics.gradient_reconstruction,
        );

        let u_mean = u.iter().sum::<Float>() / (u.len() as Float);
        let u_min = *u.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
        let u_max = *u.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

        println!(
            " U_mean:\tCFD = {u_mean:>5.2e}"
        );
        println!(
            " U_min: \tCFD = {u_min:>5.2e}"
        );
        println!(
            " U_max: \tCFD = {u_max:>5.2e}"
        );
    }
}

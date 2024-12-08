#![allow(unreachable_patterns)]

use crate::discretization::*;
use crate::io::{
    linear_system_to_string, matrix_to_string, print_linear_system, print_matrix,
    print_vec_scientific, write_data, write_gradients,
};
use crate::linear_algebra::*;
use crate::mesh::*;
use crate::nalgebra::GetEntry;
use crate::numerical_types::*;
use crate::settings::*;
use log::{debug, log_enabled, trace};
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use std::time::Instant;

const MAX_PRINT_ROWS: usize = 64;

// TODO: Normalize mesh lengths to reduce roundoff error
// TODO: Measure impact of logging on performance
// TODO: Rename all the `n`s to cell_count (or whatever is appropriate)

// TODO: Make struct to encapsulate all settings so I don't have a million args
#[allow(clippy::too_many_arguments)]
pub fn solve_steady(
    mesh: &mut Mesh,
    u: &mut DVector<Float>,
    v: &mut DVector<Float>,
    w: &mut DVector<Float>,
    p: &mut DVector<Float>,
    numerical_settings: &NumericalSettings,
    rho: Float,
    mu: Float,
    iteration_count: Uint,
    reporting_interval: Uint,
) {
    println!("Solving...");
    let cell_count: usize = mesh.cells.len();

    let (a_di, b_u_di, b_v_di, b_w_di) =
        build_momentum_diffusion_matrix(mesh, numerical_settings.diffusion, mu);
    let mut a_u = initialize_momentum_matrix(mesh);
    let mut a_v = initialize_momentum_matrix(mesh);
    let mut a_w = initialize_momentum_matrix(mesh);
    let mut b_u = DVector::zeros(cell_count);
    let mut b_v = DVector::zeros(cell_count);
    let mut b_w = DVector::zeros(cell_count);
    let mut p_prime = DVector::zeros(cell_count);

    if log_enabled!(log::Level::Debug) && a_di.nrows() < MAX_PRINT_ROWS {
        println!("\nMomentum diffusion:");
        print_matrix(&a_di);
    } else {
        debug!("{}", matrix_to_string(&a_di));
    }
    let mut start = Instant::now();
    match numerical_settings.pressure_velocity_coupling {
        PressureVelocityCoupling::SIMPLE => {
            for iter_number in 1..=iteration_count {
                let (peclet_avg, peclet_min, peclet_max) = build_momentum_advection_matrices(
                    &mut a_u,
                    &mut a_v,
                    &mut a_w,
                    &mut b_u,
                    &mut b_v,
                    &mut b_w,
                    &a_di,
                    mesh,
                    u,
                    v,
                    w,
                    p,
                    numerical_settings.momentum,
                    numerical_settings.velocity_interpolation,
                    numerical_settings.pressure_interpolation,
                    numerical_settings.gradient_reconstruction,
                    rho,
                );
                b_u += &b_u_di;
                b_v += &b_v_di;
                b_w += &b_w_di;

                if log_enabled!(log::Level::Debug) && a_di.nrows() < MAX_PRINT_ROWS {
                    println!("\nMomentum:");
                    println!("u:");
                    print_linear_system(&a_u, &b_u);
                    println!("v:");
                    print_linear_system(&a_v, &b_v);
                    println!("w:");
                    print_linear_system(&a_w, &b_w);
                } else {
                    debug!("{}", linear_system_to_string(&a_u, &b_u));
                    debug!("{}", linear_system_to_string(&a_v, &b_v));
                    debug!("{}", linear_system_to_string(&a_w, &b_w));
                }

                trace!("solving u");
                iterative_solve(
                    &a_u,
                    &b_u,
                    u,
                    numerical_settings.matrix_solver.iterations,
                    numerical_settings.matrix_solver.solver_type,
                    numerical_settings.matrix_solver.relaxation,
                    numerical_settings
                        .matrix_solver
                        .relative_convergence_threshold,
                    numerical_settings.matrix_solver.preconditioner,
                );
                trace!("solving v");
                iterative_solve(
                    &a_v,
                    &b_v,
                    v,
                    numerical_settings.matrix_solver.iterations,
                    numerical_settings.matrix_solver.solver_type,
                    numerical_settings.matrix_solver.relaxation,
                    numerical_settings
                        .matrix_solver
                        .relative_convergence_threshold,
                    numerical_settings.matrix_solver.preconditioner,
                );
                trace!("solving w");
                iterative_solve(
                    &a_w,
                    &b_w,
                    w,
                    numerical_settings.matrix_solver.iterations,
                    numerical_settings.matrix_solver.solver_type,
                    numerical_settings.matrix_solver.relaxation,
                    numerical_settings
                        .matrix_solver
                        .relative_convergence_threshold,
                    numerical_settings.matrix_solver.preconditioner,
                );
                let pressure_correction_matrices = build_pressure_correction_matrices(
                    mesh,
                    u,
                    v,
                    w,
                    p,
                    &a_u,
                    &a_v,
                    &a_w,
                    numerical_settings,
                    rho,
                );
                if log_enabled!(log::Level::Debug) && a_di.nrows() < MAX_PRINT_ROWS {
                    println!("\nPressure:");
                    print_linear_system(
                        &pressure_correction_matrices.a,
                        &pressure_correction_matrices.b,
                    );
                } else {
                    debug!(
                        "{}",
                        linear_system_to_string(
                            &pressure_correction_matrices.a,
                            &pressure_correction_matrices.b
                        )
                    );
                }

                trace!("solving p");
                // Zero the pressure correction for a reasonable initial guess
                p_prime *= 0.;
                iterative_solve(
                    &pressure_correction_matrices.a,
                    &pressure_correction_matrices.b,
                    &mut p_prime,
                    numerical_settings.matrix_solver.iterations,
                    numerical_settings.matrix_solver.solver_type,
                    numerical_settings.matrix_solver.relaxation,
                    numerical_settings
                        .matrix_solver
                        .relative_convergence_threshold,
                    numerical_settings.matrix_solver.preconditioner,
                );

                if log_enabled!(log::Level::Info) && u.nrows() < 64 {
                    print!("u:  ");
                    print_vec_scientific(u);
                    print!("v:  ");
                    print_vec_scientific(v);
                    print!("w:  ");
                    print_vec_scientific(w);
                    print!("p:  ");
                    print_vec_scientific(p);
                    print!("p': ");
                    print_vec_scientific(&p_prime);
                }
                let (avg_pressure_correction, avg_vel_correction) = apply_pressure_correction(
                    mesh,
                    &a_u,
                    &a_v,
                    &a_w,
                    &p_prime,
                    u,
                    v,
                    w,
                    p,
                    numerical_settings,
                );

                let u_avg = u.iter().sum::<Float>() / (cell_count as Float);
                let v_avg = v.iter().sum::<Float>() / (cell_count as Float);
                let w_avg = w.iter().sum::<Float>() / (cell_count as Float);
                if (iter_number) % reporting_interval == 0 {
                    let elapsed = (Instant::now() - start).as_millis();
                    let millis_per_iter = elapsed / (reporting_interval as u128);
                    start = Instant::now();
                    print!("Iteration {iter_number}: avg velocity = ({u_avg:.2e}, {v_avg:.2e}, {w_avg:.2e})\t");
                    print!("avg peclet = {peclet_avg:.1e}\tmin peclet = {peclet_min:.1e}\tmax peclet = {peclet_max:.1e}\t");
                    print!("velocity correction: {avg_vel_correction:.2e}\tpressure correction: {avg_pressure_correction:.2e}\tms/iter: {millis_per_iter:.1e}\n")
                }
                if Float::is_nan(u_avg) || Float::is_nan(v_avg) || Float::is_nan(w_avg) {
                    // TODO: Some undefined behavior seems to be causing this to trigger
                    // intermittently
                    panic!("solution diverged");
                }
            }
        }
        _ => panic!("unsupported pressure-velocity coupling"),
    }

    let mut pressure_grad_mean = Vector::zero();
    let mut velocity_grad_mean = Tensor::zero();
    for i in 0..mesh.cells.len() {
        let pressure_gradient =
            calculate_pressure_gradient(mesh, p, i, numerical_settings.gradient_reconstruction);
        let velocity_gradient = calculate_velocity_gradient(
            mesh,
            u,
            v,
            w,
            i,
            numerical_settings.gradient_reconstruction,
        );
        pressure_grad_mean += pressure_gradient.abs();
        velocity_grad_mean = velocity_grad_mean + velocity_gradient.abs();
    }
    println!("Done solving.");
}

pub fn initialize_flow(
    mesh: &Mesh,
    mu: Float,
    rho: Float,
    iteration_count: Uint,
) -> (
    DVector<Float>,
    DVector<Float>,
    DVector<Float>,
    DVector<Float>,
) {
    // This needs a few different strategies:
    // 1. All pressure / stationary wall BCs
    // 2. All velocity / stationary wall / moving wall BCs
    // 3. Mixture of 1 and 2
    //
    // In case 1, we can solve Laplace's equation for pressure with the conditions that
    // boundary-normal pressure gradients are zero. This will produce a smooth pressure field that
    // can be fed into the momentum equation. Here we will want to use a very stable iterative
    // solver.
    // * Is the assumption reasonable that wall-normal pressure gradient is zero?
    //
    // In case 2, we can solve Laplace's equation for velocity as above.
    //
    // Case 3 is more challenging.

    check_boundary_conditions(mesh);
    let n = mesh.cells.len();
    let mut u = DVector::zeros(n);
    let mut v = DVector::zeros(n);
    let mut w = DVector::zeros(n);
    let mut p = DVector::zeros(n);
    let (a_di, b_u_di, b_v_di, b_w_di) =
        build_momentum_diffusion_matrix(mesh, DiffusionScheme::CD, mu);
    let mut a_u = initialize_momentum_matrix(mesh);
    let mut a_v = initialize_momentum_matrix(mesh);
    let mut a_w = initialize_momentum_matrix(mesh);
    let mut b_u: DVector<Float> = DVector::zeros(n);
    let mut b_v: DVector<Float> = DVector::zeros(n);
    let mut b_w: DVector<Float> = DVector::zeros(n);

    initialize_pressure_field(mesh, &mut p, iteration_count);
    build_momentum_advection_matrices(
        &mut a_u,
        &mut a_v,
        &mut a_w,
        &mut b_u,
        &mut b_v,
        &mut b_w,
        &a_di,
        mesh,
        &u,
        &v,
        &w,
        &p,
        MomentumDiscretization::UD,
        VelocityInterpolation::LinearWeighted,
        PressureInterpolation::LinearWeighted,
        GradientReconstructionMethods::GreenGauss(GreenGaussVariants::CellBased),
        rho,
    );
    b_u += b_u_di;
    b_v += b_v_di;
    b_w += b_w_di;
    // TODO: Consider initializing velocity field by the following:
    // Solve with only diffusive term
    // Slowly ramp up to diffusive + advective

    println!("Initializing velocity field...");

    let mut diffusion_fraction = 1.;
    while diffusion_fraction >= 0. {
        iterative_solve(
            &(&a_u * (1. - diffusion_fraction) + &a_di * diffusion_fraction),
            &b_u,
            &mut u,
            iteration_count,
            SolutionMethod::BiCGSTAB,
            0.5,
            1e-6,
            PreconditionMethod::Jacobi,
        );
        iterative_solve(
            &(&a_v * (1. - diffusion_fraction) + &a_di * diffusion_fraction),
            &b_v,
            &mut v,
            iteration_count,
            SolutionMethod::BiCGSTAB,
            0.5,
            1e-6,
            PreconditionMethod::Jacobi,
        );
        iterative_solve(
            &(&a_w * (1. - diffusion_fraction) + &a_di * diffusion_fraction),
            &b_w,
            &mut w,
            iteration_count,
            SolutionMethod::BiCGSTAB,
            0.5,
            1e-6,
            PreconditionMethod::Jacobi,
        );
        diffusion_fraction -= 0.2;
    }
    println!("Done!");
    (u, v, w, p)
}

pub fn initialize_flow_new(
    mesh: &Mesh,
    mu: Float,
    rho: Float,
    iteration_count: Uint,
) -> (
    DVector<Float>,
    DVector<Float>,
    DVector<Float>,
    DVector<Float>,
) {
    println!("Initializing flow...");
    // This needs a few different strategies:
    // 1. All pressure / stationary wall BCs
    // 2. All velocity / stationary wall / moving wall BCs
    // 3. Mixture of 1 and 2
    //
    // In case 1, we can solve Laplace's equation for pressure with the conditions that
    // boundary-normal pressure gradients are zero. This will produce a smooth pressure field that
    // can be fed into the momentum equation. Here we will want to use a very stable iterative
    // solver.
    // * Is the assumption reasonable that wall-normal pressure gradient is zero?
    //
    // In case 2, we can solve Laplace's equation for velocity as above.
    //
    // Case 3 is more challenging.

    let n = mesh.cells.len();
    let mut u = DVector::zeros(n);
    let mut v = DVector::zeros(n);
    let mut w = DVector::zeros(n);
    let mut p = DVector::zeros(n);
    // let (a_di, b_u_di, b_v_di, b_w_di) =
    //     build_momentum_diffusion_matrix(mesh, DiffusionScheme::CD, mu);
    // let mut a_u = initialize_momentum_matrix(mesh);
    // let mut a_v = initialize_momentum_matrix(mesh);
    // let mut a_w = initialize_momentum_matrix(mesh);
    // let mut b_u = DVector::zeros(n);
    // let mut b_v = DVector::zeros(n);
    // let mut b_w = DVector::zeros(n);

    let constraint_type = check_boundary_conditions(mesh);
    println!("{constraint_type:?}");

    match constraint_type {
        SystemConstraintType::PressureOnly | SystemConstraintType::Hybrid => {
            // Do pressure stuff
            initialize_pressure_field(mesh, &mut p, iteration_count);
        }
        SystemConstraintType::VelocityOnly | SystemConstraintType::Hybrid => {
            initialize_velocity_field(mesh, &mut u, &mut v, &mut w, iteration_count);
        }
    }

    println!("Done!");
    (u, v, w, p)
}

// TODO: Find a way to reduce the pressure/velocity calculation repetition. For example,
// `initialize_pressure_field()` + `initialize_velocity_field()` -> `initialize_scalar_field()`
fn initialize_pressure_field(mesh: &Mesh, p: &mut DVector<Float>, _iteration_count: Uint) {
    println!("Initializing pressure field...");
    // Solve laplace's equation (nabla^2 P = dP^2/dx^2 + dP^2/dy^2 + dP^2/dz^2 = 0) based on BCs:
    // - Wall: d/dn (P) = 0
    // - Pressure BC: P = P_boundary (duh)
    // - Other BCs: ???
    // For each cell, compute pressure gradient at face as:
    // dP/dx = (P_cell - P_neighbor)/(x_cell - x_neighbor) (and so on for other axes)
    // Therefore: grad P    = (P[cell] - P[neighbor]) / (cell_centroid - neighbor_centroid)
    //                      = (P[neighbor] - P[cell]) / (neighbor_centroid - cell_centroid)
    //                      (reformulated so that a_nb is obvious)
    // Knowing grad P at the face, we can use Green-Gauss to get second derivative at cell center:
    // (d^2/dx^2)[P] = sum(face_outward_normal_vector.x * dP/dx * face_area) / cell_volume
    // (d^2/dy^2)[P] = sum(face_outward_normal_vector.y * dP/dy * face_area) / cell_volume
    // (d^2/dz^2)[P] = sum(face_outward_normal_vector.z * dP/dz * face_area) / cell_volume
    // laplacian P  = (d^2/dx^2 + d^2/dy^2 + d^2/dz^2)[P]
    //              = sum(face_outward_normal_vector.dot(grad P) * face_area) / cell_volume
    // So, for each face and each axis:
    // a_nb = face_outward_normal_vector.dot(1/(x_neighbor - x_cell)) * (face_area / cell_volume)
    // And for each cell:
    // a_p = sum(-a_nb)
    // At boundary cells, we will have to estimate dP/dn
    let cell_count = mesh.cells.len();
    let mut a_coo = CooMatrix::<Float>::new(cell_count, cell_count);
    let mut b = DVector::zeros(a_coo.nrows());

    for cell_index in 0..cell_count {
        let cell = &mesh.cells[cell_index];
        let mut a_p: Float = 0.;
        for face in cell
            .face_indices
            .iter()
            .map(|face_index| &mesh.faces[*face_index])
        {
            let outward_normal_vector = get_outward_face_normal(face, cell_index);
            let face_zone = &mesh.face_zones[&face.zone];
            let (a_nb, source, neighbor_cell_index) = match face_zone.zone_type {
                FaceConditionTypes::Interior => {
                    let neighbor_cell_index = if face.cell_indices[0] == cell_index {
                        face.cell_indices[1]
                    } else {
                        face.cell_indices[0]
                    };
                    // TODO: Skewness correction here
                    (
                        (cell.centroid - mesh.cells[neighbor_cell_index].centroid)
                            .reciprocal()
                            .dot(&outward_normal_vector)
                            * (face.area / cell.volume),
                        0.,
                        neighbor_cell_index,
                    )
                }
                FaceConditionTypes::PressureInlet | FaceConditionTypes::PressureOutlet => {
                    // grad P   = (P[cell] - P_bc) / (cell_centroid - face_centroid)
                    let a_nb = (cell.centroid - face.centroid)
                        .reciprocal()
                        .dot(&outward_normal_vector)
                        * (face.area / cell.volume);
                    (a_nb, a_nb * face_zone.scalar_value, usize::MAX)
                }
                FaceConditionTypes::Symmetry | FaceConditionTypes::Wall => {
                    // we're saying wall-normal pressure gradient is zero, so all terms here
                    // are zero
                    (0., 0., usize::MAX)
                }
                _ => {
                    // println!("Skipping zone type: {:?}", face_zone.zone_type);
                    (0., 0., usize::MAX)
                }
            };
            if neighbor_cell_index != usize::MAX {
                a_coo.push(cell_index, neighbor_cell_index, -a_nb);
            }
            b[cell_index] += source;
            a_p += a_nb;
        }
        a_coo.push(cell_index, cell_index, a_p);
    }
    let a = &CsrMatrix::from(&a_coo);
    // print_linear_system(&a, &b);
    // let a_dense = DMatrix::from(a);
    // *p = a_dense.try_inverse().unwrap() * &b;

    iterative_solve(
        a,
        &b,
        p,
        10,
        SolutionMethod::Jacobi,
        0.1,
        1e-6,
        PreconditionMethod::Jacobi,
    );
    println!("Done!");
}

fn initialize_velocity_field(
    mesh: &Mesh,
    u: &mut DVector<Float>,
    v: &mut DVector<Float>,
    w: &mut DVector<Float>,
    iteration_count: Uint,
) {
    println!("Initializing velocity field...");
    let cell_count = mesh.cells.len();
    let mut a_coo = CooMatrix::<Float>::new(cell_count, cell_count);
    let mut b = DVector::zeros(a_coo.nrows());
    let mut psi = DVector::zeros(a_coo.nrows());

    for cell_index in 0..cell_count {
        let cell = &mesh.cells[cell_index];
        let mut a_p: Float = 0.;
        for face in cell
            .face_indices
            .iter()
            .map(|face_index| &mesh.faces[*face_index])
        {
            let outward_normal_vector = get_outward_face_normal(face, cell_index);
            let face_zone = &mesh.face_zones[&face.zone];
            let (a_nb, source, neighbor_cell_index) = match face_zone.zone_type {
                FaceConditionTypes::Interior => {
                    let neighbor_cell_index = if face.cell_indices[0] == cell_index {
                        face.cell_indices[1]
                    } else {
                        face.cell_indices[0]
                    };
                    // TODO: Skewness correction here
                    (
                        (cell.centroid - mesh.cells[neighbor_cell_index].centroid)
                            .reciprocal()
                            .dot(&outward_normal_vector)
                            * (face.area / cell.volume),
                        0.,
                        neighbor_cell_index,
                    )
                }
                FaceConditionTypes::VelocityInlet => {
                    // grad psi = face velocity
                    (
                        0.,
                        -face_zone.vector_value.dot(&outward_normal_vector),
                        usize::MAX,
                    )
                }
                FaceConditionTypes::Symmetry | FaceConditionTypes::Wall => {
                    // Velocity normal to these boundaries is zero, therefore d/dn (psi) = 0
                    (0., 0., usize::MAX)
                }
                FaceConditionTypes::PressureOutlet => {
                    // Here we define psi to be 0, so the gradient will be
                    // (cell_value - 0.)/(x_cell - x_face)
                    (
                        (cell.centroid - face.centroid)
                            .reciprocal()
                            .dot(&outward_normal_vector),
                        0.,
                        usize::MAX,
                    )
                }
                _ => {
                    // println!("Skipping zone type: {:?}", face_zone.zone_type);
                    (0., 0., usize::MAX)
                }
            };
            if neighbor_cell_index != usize::MAX {
                a_coo.push(cell_index, neighbor_cell_index, -a_nb);
            }
            b[cell_index] += source;
            a_p += a_nb;
        }
        a_coo.push(cell_index, cell_index, a_p);
    }
    let a = &CsrMatrix::from(&a_coo);
    // print_linear_system(&a, &b);
    // let a_dense = DMatrix::from(a);
    // psi = a_dense.try_inverse().unwrap() * &b;

    iterative_solve(
        a,
        &b,
        &mut psi,
        10,
        SolutionMethod::BiCGSTAB,
        0.1,
        1e-6,
        PreconditionMethod::Jacobi,
    );
    // print_linear_system(a, &b);
    // print_vec_scientific(&psi);

    write_data(mesh, u, v, w, &psi, "./examples/psi.csv");
    write_gradients(
        mesh,
        u,
        v,
        w,
        &psi,
        "./examples/psi_gradients.csv",
        3,
        GradientReconstructionMethods::GreenGauss(GreenGaussVariants::CellBased),
    );

    // Take the gradient of psi with least-squares
    // Using least-squares since we can't get boundary values of psi I think?
    for (cell_index, cell) in mesh.cells.iter().enumerate() {
        // This might reserve more mem than we need since boundary faces aren't considered,
        // but I think it's worth it still.
        let mut a_data: Vec<Float> = Vec::with_capacity(3 * cell.face_indices.len());
        let mut b_data: Vec<Float> = Vec::with_capacity(cell.face_indices.len());
        let mut neighbor_count: usize = 0;
        // Iterate over neighbor cells - maybe worth abstracting this?
        cell.face_indices
            .iter()
            .map(|&fi| &mesh.faces[fi])
            .filter(|f| f.cell_indices.len() == 2)
            .for_each(|f| {
                let neighbor_index = if f.cell_indices[0] != cell_index {
                    f.cell_indices[0]
                } else {
                    f.cell_indices[1]
                };
                let neighbor_cell = &mesh.cells[neighbor_index];
                let delta_x = neighbor_cell.centroid - cell.centroid;
                let delta_psi = psi[neighbor_index] - psi[cell_index];
                a_data.push(delta_x.x);
                a_data.push(delta_x.y);
                a_data.push(delta_x.z);
                b_data.push(delta_psi);
                neighbor_count += 1;
            });

        let mut a: DMatrix<Float> = DMatrix::from_row_slice(neighbor_count, 3, &a_data[..]);

        let mut nonzero_columns: Vec<usize> = Vec::new();
        a.column_iter().enumerate().for_each(|(i, col)| {
            if col.min() != 0. || col.max() != 0. {
                nonzero_columns.push(i);
            }
        });
        a = a.select_columns(nonzero_columns.iter());

        let mut b: DVector<Float> = DVector::from_vec(b_data);

        // println!("{a}");

        b = &a.transpose() * &b;
        a = &a.transpose() * &a;

        // println!("{a}");
        let a_inv = a.clone().try_inverse();
        let mut cell_velocity: DVector<Float> = DVector::zeros(3);
        match a_inv {
            Some(a_inv) => {
                cell_velocity = a_inv * b;
            }
            None => {
                println!("Could not invert. Skipping.");
            }
        }
        let u_i: Float = if nonzero_columns.contains(&0) {
            cell_velocity[0]
        } else {
            0.
        };
        let v_i: Float = if nonzero_columns.contains(&1) {
            cell_velocity[nonzero_columns.iter().position(|&i| i == 1).unwrap()]
        } else {
            0.
        };
        let w_i: Float = if nonzero_columns.contains(&2) {
            cell_velocity[nonzero_columns.iter().position(|&i| i == 2).unwrap()]
        } else {
            0.
        };

        u[cell_index] = if u_i.is_nan() { 0. } else { u_i };
        v[cell_index] = if v_i.is_nan() { 0. } else { v_i };
        w[cell_index] = if w_i.is_nan() { 0. } else { w_i };
    }

    println!("Done!");
}

pub fn get_momentum_source_term(_location: Vector) -> Vector {
    // TODO
    Vector::zero()
}

#[derive(Debug)]
enum SystemConstraintType {
    PressureOnly,
    VelocityOnly,
    Hybrid,
}

fn check_boundary_conditions(mesh: &Mesh) -> SystemConstraintType {
    const PI: Float = std::f32::consts::PI as Float;
    // 5 degrees
    const TOL: Float = 5. * 180. / PI;

    let mut pressure_bc_count: u16 = 0;
    let mut velocity_bc_count: u16 = 0;

    for face_zone in mesh.face_zones.values() {
        match face_zone.zone_type {
            FaceConditionTypes::Wall => {
                if face_zone.vector_value.norm() > 0. {
                    for face_index in 0..mesh.faces.len() {
                        velocity_bc_count += 1;
                        let face = &mesh.faces[face_index];
                        // println!("Angle = {}", vector_angle(&face.normal, &face_zone.vector_value));
                        if PI / 2. - Float::abs(vector_angle(&face.normal, &face_zone.vector_value))
                            > TOL
                        {
                            panic!("Wall velocity must be tangent to faces in zone.");
                        }
                    }
                }
            }
            FaceConditionTypes::VelocityInlet => {
                velocity_bc_count += 1;
                for face_index in 0..mesh.faces.len() {
                    let face = &mesh.faces[face_index];
                    if Float::abs(vector_angle(&face.normal, &face_zone.vector_value)) > TOL {
                        panic!("VelocityInlet velocity must not be tangent to faces in zone.");
                    }
                }
            }

            FaceConditionTypes::PressureInlet | FaceConditionTypes::PressureOutlet => {
                pressure_bc_count += 1;
            }
            _ => {
                // TODO: Handle other zone types. Could check if scalar_value/vector_value are
                // set/not set appropriately, could check if system is overdefined/underdefined,
                // etc.
            }
        }
    }

    if velocity_bc_count > 0 {
        if pressure_bc_count > 1 {
            SystemConstraintType::Hybrid
        } else {
            if pressure_bc_count == 0 {
                println!("Warning - only velocity boundary conditions exist. System may be overconstrained.");
            }
            SystemConstraintType::VelocityOnly
        }
    } else if pressure_bc_count > 0 {
        // Should I throw an error if there's only a single pressure BC?
        SystemConstraintType::PressureOnly
    } else {
        panic!("You must set boundary conditions.");
    }
}

// TODO: Find a clean way to integrate all of the `fn ____velocity____` and `fn ____pressure_____`
// into general `fn ____scalar____` functions
pub fn calculate_velocity_gradient(
    mesh: &Mesh,
    u: &DVector<Float>,
    v: &DVector<Float>,
    w: &DVector<Float>,
    cell_index: usize,
    gradient_scheme: GradientReconstructionMethods,
) -> Tensor {
    let cell = &mesh.cells[cell_index];
    match gradient_scheme {
        GradientReconstructionMethods::GreenGauss(_) => {
            cell.face_indices
                .iter()
                .fold(Tensor::zero(), |acc, face_index| {
                    let face = &mesh.faces[*face_index];
                    let face_value: Vector = get_face_velocity(
                        mesh,
                        u,
                        v,
                        w,
                        *face_index,
                        // NOTE: Is LinearWeighted sufficient?
                        VelocityInterpolation::Linear,
                    );
                    acc + face_value.outer(
                        &(get_outward_face_normal(face, cell_index) * (face.area / cell.volume)),
                    )
                })
        }
        GradientReconstructionMethods::LeastSquares => {
            let n = cell.face_indices.len();
            let mut a_data: Vec<Float> = Vec::with_capacity(3 * n);
            let mut b_u_data: Vec<Float> = Vec::with_capacity(n);
            let mut b_v_data: Vec<Float> = Vec::with_capacity(n);
            let mut b_w_data: Vec<Float> = Vec::with_capacity(n);

            let cell_centroid = mesh.cells[cell_index].centroid;

            cell.face_indices.iter().for_each(|f| {
                let face = &mesh.faces[*f];
                let zone = &mesh.face_zones[&face.zone];
                let (x, u) = match zone.zone_type {
                    FaceConditionTypes::Interior => {
                        let mut neighbor_cell_index = face.cell_indices[0];
                        if neighbor_cell_index == cell_index {
                            neighbor_cell_index = face.cell_indices[1];
                        }
                        (
                            mesh.cells[neighbor_cell_index].centroid - cell_centroid,
                            (
                                u[neighbor_cell_index] - u[cell_index],
                                v[neighbor_cell_index] - v[cell_index],
                                w[neighbor_cell_index] - w[cell_index],
                            ),
                        )
                    }
                    _ => {
                        let face_velocity =
                            get_face_velocity(mesh, u, v, w, *f, VelocityInterpolation::None);
                        (
                            face.centroid - cell_centroid,
                            (face_velocity.x, face_velocity.y, face_velocity.z),
                        )
                    }
                };
                a_data.push(x.x);
                a_data.push(x.y);
                a_data.push(x.z);
                b_u_data.push(u.0);
                b_v_data.push(u.1);
                b_w_data.push(u.2);
            });
            let mut a: DMatrix<Float> = DMatrix::from_row_slice(n, 3, &a_data[..]);
            let mut b_u: DVector<Float> = DVector::from_vec(b_u_data);
            let mut b_v: DVector<Float> = DVector::from_vec(b_v_data);
            let mut b_w: DVector<Float> = DVector::from_vec(b_w_data);

            b_u = &a.transpose() * &b_u;
            b_v = &a.transpose() * &b_v;
            b_w = &a.transpose() * &b_w;
            a = &a.transpose() * &a;

            let a_inv = a.try_inverse().unwrap();

            let grad_u = &a_inv * b_u;
            let grad_v = &a_inv * b_v;
            let grad_w = &a_inv * b_w;

            // let grad_u = iterative_solve(&a, b, solution_vector, iteration_count, method, relaxation_factor, convergence_threshold, preconditioner)

            Tensor {
                x: Vector::from_dvector(&grad_u),
                y: Vector::from_dvector(&grad_v),
                z: Vector::from_dvector(&grad_w),
            }
        }
        _ => panic!("unsupported gradient scheme"),
    }
}

pub fn calculate_pressure_gradient(
    mesh: &Mesh,
    p: &DVector<Float>,
    cell_index: usize,
    gradient_scheme: GradientReconstructionMethods,
) -> Vector {
    let cell = &mesh.cells[cell_index];
    match gradient_scheme {
        GradientReconstructionMethods::GreenGauss(variant) => match variant {
            GreenGaussVariants::CellBased => cell
                .face_indices
                .iter()
                .map(|face_index| {
                    let face = &mesh.faces[*face_index];
                    let face_value: Float = get_face_pressure(
                        mesh,
                        p,
                        *face_index,
                        // TODO: Evaluate if this is sufficient
                        PressureInterpolation::Linear,
                        gradient_scheme,
                    );
                    face_value
                        * (face.area / cell.volume)
                        * get_outward_face_normal(face, cell_index)
                })
                .fold(Vector::zero(), |acc, v| acc + v),
            _ => panic!("unsupported Green-Gauss scheme"),
        },
        GradientReconstructionMethods::LeastSquares => {
            let n = cell.face_indices.len();
            let mut a_data: Vec<Float> = Vec::with_capacity(3 * n);
            let mut b_data: Vec<Float> = Vec::with_capacity(n);
            let cell_centroid = mesh.cells[cell_index].centroid;

            cell.face_indices.iter().for_each(|f| {
                let face = &mesh.faces[*f];
                let zone = &mesh.face_zones[&face.zone];
                let (x, p) = match zone.zone_type {
                    FaceConditionTypes::Interior => {
                        let mut neighbor_cell_index = face.cell_indices[0];
                        if neighbor_cell_index == cell_index {
                            neighbor_cell_index = face.cell_indices[1];
                        }
                        (
                            mesh.cells[neighbor_cell_index].centroid - cell_centroid,
                            p[neighbor_cell_index] - p[cell_index],
                        )
                    }
                    _ => (
                        face.centroid - cell_centroid,
                        get_face_pressure(
                            mesh,
                            p,
                            *f,
                            PressureInterpolation::None,
                            GradientReconstructionMethods::None,
                        ),
                    ),
                };
                a_data.push(x.x);
                a_data.push(x.y);
                a_data.push(x.z);
                b_data.push(p);
            });
            let mut a: DMatrix<Float> = DMatrix::from_row_slice(n, 3, &a_data[..]);
            let mut b: DVector<Float> = DVector::from_vec(b_data);

            b = &a.transpose() * &b;
            a = &a.transpose() * &a;

            let a_inv = a.try_inverse().unwrap();
            Vector::from_dvector(&(a_inv * b))
        }
        _ => panic!("unsupported gradient scheme"),
    }
}

pub fn get_face_velocity(
    mesh: &Mesh,
    u: &DVector<Float>,
    v: &DVector<Float>,
    w: &DVector<Float>,
    face_index: usize,
    interpolation_scheme: VelocityInterpolation,
) -> Vector {
    // ****** TODO: Add skewness corrections!!! ********
    let face = &mesh.faces[face_index];
    let face_zone = &mesh.face_zones[&face.zone];
    let cell_index = face.cell_indices[0];
    match face_zone.zone_type {
        FaceConditionTypes::Wall => face_zone.vector_value,
        FaceConditionTypes::VelocityInlet => face_zone.vector_value,
        FaceConditionTypes::PressureInlet
        | FaceConditionTypes::PressureOutlet
        | FaceConditionTypes::Symmetry => Vector {
            x: u[cell_index],
            y: v[cell_index],
            z: w[cell_index],
        },
        FaceConditionTypes::Interior => {
            let neighbor_index = face.cell_indices[1];
            let vel0 = Vector {
                x: u[cell_index],
                y: v[cell_index],
                z: w[cell_index],
            };
            let vel1 = Vector {
                x: u[neighbor_index],
                y: v[neighbor_index],
                z: w[neighbor_index],
            };
            match interpolation_scheme {
                VelocityInterpolation::Linear => (vel0 + vel1) / 2.,
                VelocityInterpolation::LinearWeighted => {
                    let dx0 = (mesh.cells[cell_index].centroid - face.centroid).norm();
                    let dx1 = (mesh.cells[neighbor_index].centroid - face.centroid).norm();
                    vel0 + (vel1 - vel0) * dx0 / (dx0 + dx1)
                }
                VelocityInterpolation::RhieChow => {
                    panic!("unsupported");
                }
                VelocityInterpolation::None => {
                    panic!("`None` VelocityInterpolation cannot be sed for interior faces")
                }
            }
        }
        _ => panic!("unsupported face zone type"),
    }
}

// TODO: Move this to discretization? Or a new interpolation module?
#[allow(clippy::too_many_arguments)]
pub fn get_face_flux(
    mesh: &Mesh,
    u: &DVector<Float>,
    v: &DVector<Float>,
    w: &DVector<Float>,
    p: &DVector<Float>,
    face_index: usize,
    cell_index: usize,
    interpolation_scheme: VelocityInterpolation,
    gradient_scheme: GradientReconstructionMethods,
    a_u: &CsrMatrix<Float>,
    a_v: &CsrMatrix<Float>,
    a_w: &CsrMatrix<Float>,
) -> Float {
    // ****** TODO: Add skewness corrections!!! ********
    let face = &mesh.faces[face_index];
    let outward_face_normal = get_outward_face_normal(face, cell_index);
    let face_zone = &mesh.face_zones[&face.zone];
    match face_zone.zone_type {
        FaceConditionTypes::Wall | FaceConditionTypes::Symmetry => 0.,
        FaceConditionTypes::VelocityInlet
        | FaceConditionTypes::PressureInlet
        | FaceConditionTypes::PressureOutlet => {
            outward_face_normal.dot(&get_face_velocity(
                mesh,
                u,
                v,
                w,
                face_index,
                VelocityInterpolation::None, // If it's a boundary face, no interpolation will be
                                             // needed
            ))
        }
        FaceConditionTypes::Interior => match interpolation_scheme {
            VelocityInterpolation::Linear | VelocityInterpolation::LinearWeighted => {
                outward_face_normal.dot(&get_face_velocity(
                    mesh,
                    u,
                    v,
                    w,
                    face_index,
                    interpolation_scheme,
                ))
            }
            VelocityInterpolation::RhieChow => {
                let mut neighbor_cell_index = face.cell_indices[0];
                if neighbor_cell_index == cell_index {
                    neighbor_cell_index = face.cell_indices[1];
                }
                let vel_i = Vector {
                    x: u[cell_index],
                    y: v[cell_index],
                    z: w[cell_index],
                };
                let vel_j = Vector {
                    x: u[neighbor_cell_index],
                    y: v[neighbor_cell_index],
                    z: w[neighbor_cell_index],
                };
                let cell_centroid_vector =
                    mesh.cells[neighbor_cell_index].centroid - mesh.cells[cell_index].centroid;
                let a_i = get_normal_momentum_coefficient!(
                    cell_index,
                    a_u,
                    a_v,
                    a_w,
                    &outward_face_normal
                );
                let a_j = get_normal_momentum_coefficient!(
                    neighbor_cell_index,
                    a_u,
                    a_v,
                    a_w,
                    &outward_face_normal
                );
                let p_grad_i = calculate_pressure_gradient(mesh, p, cell_index, gradient_scheme);
                let p_grad_j =
                    calculate_pressure_gradient(mesh, p, neighbor_cell_index, gradient_scheme);
                let cell_vol_i = mesh.cells[cell_index].volume;
                let cell_vol_j = mesh.cells[neighbor_cell_index].volume;

                let term_1 = (vel_i + vel_j).dot(&outward_face_normal);
                let term_2 = (cell_vol_i / a_i + cell_vol_j / a_j)
                    * (p[cell_index] - p[neighbor_cell_index])
                    / cell_centroid_vector.norm();
                let term_3 = (cell_vol_i / a_i * p_grad_i + cell_vol_j / a_j * p_grad_j)
                    .dot(&cell_centroid_vector.unit());
                0.5 * (term_1 + term_2 - term_3)
            }
            VelocityInterpolation::None => {
                panic!("`None` VelocityInterpolation cannot be used for interior faces")
            }
        },
        _ => panic!("unsupported face zone type"),
    }
}

pub fn get_face_pressure(
    mesh: &Mesh,
    p: &DVector<Float>,
    face_index: usize,
    interpolation_scheme: PressureInterpolation,
    gradient_scheme: GradientReconstructionMethods,
) -> Float {
    // ****** TODO: Add skewness corrections!!! ********
    let face = &mesh.faces[face_index];
    let face_zone = &mesh.face_zones[&face.zone];
    match face_zone.zone_type {
        FaceConditionTypes::Symmetry
        | FaceConditionTypes::Wall
        | FaceConditionTypes::VelocityInlet => {
            // du/dn = 0, so we need the projection of cell center velocity onto the face's plane
            p[face.cell_indices[0]]
        }
        FaceConditionTypes::PressureInlet | FaceConditionTypes::PressureOutlet => {
            face_zone.scalar_value
        }
        FaceConditionTypes::Interior => {
            let c0 = face.cell_indices[0];
            let c1 = face.cell_indices[1];
            match interpolation_scheme {
                PressureInterpolation::Linear => (p[c0] + p[c1]) * 0.5,
                PressureInterpolation::LinearWeighted => {
                    let x0 = (mesh.cells[c0].centroid - face.centroid).norm();
                    let x1 = (mesh.cells[c1].centroid - face.centroid).norm();
                    p[c0] + (p[c1] - p[c0]) * x0 / (x0 + x1)
                }
                PressureInterpolation::Standard => {
                    // requires momentum matrix coeffs
                    panic!("`standard` pressure interpolation unsupported");
                }
                PressureInterpolation::SecondOrder => {
                    let c0_grad = calculate_pressure_gradient(mesh, p, c0, gradient_scheme);
                    let c1_grad = calculate_pressure_gradient(mesh, p, c1, gradient_scheme);
                    let r_c0 = face.centroid - mesh.cells[c0].centroid;
                    let r_c1 = face.centroid - mesh.cells[c1].centroid;
                    0.5 * ((p[c0] + p[c1]) + (c0_grad.dot(&r_c0) + c1_grad.dot(&r_c1)))
                }
                _ => panic!("unsupported pressure interpolation"),
            }
        }
        _ => panic!("unsupported face zone type"),
    }
}

// fn build_k_epsilon_matrices(
//     a_k: &mut CsrMatrix<Float>,
//     a_eps: &mut CsrMatrix<Float>,
//     b_k: &mut DVector<Float>,
//     b_eps: &mut DVector<Float>,
//     mesh: &Mesh,
//     u: &DVector<Float>,
//     v: &DVector<Float>,
//     w: &DVector<Float>,
//     p: &DVector<Float>,
//     momentum_matrices: &CsrMatrix<Float>,
//     numerical_settings: &NumericalSettings,
//     rho: Float,
// ) {
//
// }

#[allow(clippy::too_many_arguments)]
fn apply_pressure_correction(
    mesh: &mut Mesh,
    a_u: &CsrMatrix<Float>, // NOTE: I really only need the diagonal
    a_v: &CsrMatrix<Float>, // NOTE: I really only need the diagonal
    a_w: &CsrMatrix<Float>, // NOTE: I really only need the diagonal
    p_prime: &DVector<Float>,
    u: &mut DVector<Float>,
    v: &mut DVector<Float>,
    w: &mut DVector<Float>,
    p: &mut DVector<Float>,
    numerical_settings: &NumericalSettings,
) -> (Float, Float) {
    let mut velocity_corr_sum: Float = 0.;
    for cell_index in 0..mesh.cells.len() {
        let cell = &mesh.cells.get_mut(cell_index).unwrap();
        let pressure_correction = *p_prime.get(cell_index).unwrap_or(&0.);
        p[cell_index] += numerical_settings.pressure_relaxation * pressure_correction;
        let velocity_correction =
            cell.face_indices
                .iter()
                .fold(Vector::zero(), |acc, face_index| {
                    let face = &mesh.faces[*face_index];
                    let face_zone = &mesh.face_zones[&face.zone];
                    let outward_face_normal = get_outward_face_normal(face, cell_index);
                    let p_prime_neighbor = match face_zone.zone_type {
                        FaceConditionTypes::Wall
                        | FaceConditionTypes::Symmetry
                        | FaceConditionTypes::VelocityInlet => p_prime[cell_index],
                        FaceConditionTypes::PressureInlet | FaceConditionTypes::PressureOutlet => {
                            0.
                        }
                        FaceConditionTypes::Interior => {
                            let neighbor_cell_index = if face.cell_indices[0] == cell_index {
                                face.cell_indices[1]
                            } else {
                                face.cell_indices[0]
                            };
                            p_prime[neighbor_cell_index]
                        }
                        unsupported_zone_type => {
                            println!("*** {} ***", unsupported_zone_type);
                            panic!("BC not supported");
                        }
                    };
                    let scaled_normal = Vector {
                        x: outward_face_normal.x / a_u.get(cell_index, cell_index),
                        y: outward_face_normal.y / a_v.get(cell_index, cell_index),
                        z: outward_face_normal.z / a_w.get(cell_index, cell_index),
                    };
                    acc + scaled_normal * (p_prime[cell_index] - p_prime_neighbor) * face.area
                });
        u[cell_index] += velocity_correction.x * numerical_settings.momentum_relaxation;
        v[cell_index] += velocity_correction.y * numerical_settings.momentum_relaxation;
        w[cell_index] += velocity_correction.z * numerical_settings.momentum_relaxation;
        velocity_corr_sum += velocity_correction.norm().powi(2);
    }
    (p_prime.norm(), velocity_corr_sum.sqrt())
}

// fn compute_courant_number(mesh: &Mesh, u: &DVector<Float>, v: &DVector<Float>, w: &DVector<Float>) -> (Float, Float, Float) {
//     let mut min_courant_number: Float = Float::INFINITY;
//     let mut max_courant_number: Float = Float::NEG_INFINITY;
//     let mut avg_courant_number: Float = 0.;
//
//     for cell_index in 0..mesh.cells.len() {
//         let velocity = Vector{x: u[cell_index], y: v[cell_index], z: w[cell_index]};
//     }
//
//     (avg_courant_number, min_courant_number, max_courant_number)
// }

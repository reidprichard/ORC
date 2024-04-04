#![allow(unreachable_patterns, unused)]

use crate::io::{linear_system_to_string, matrix_to_string, print_linear_system, print_matrix, print_vec_scientific};
use crate::mesh::*;
use crate::discretization::*;
use crate::linear_algebra::*;
use crate::nalgebra::{GetEntry, dvector_zeros};
use crate::numerical_types::*;
use crate::settings::*;
use log::{debug, info, log_enabled, trace};
use nalgebra::DVector;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix, SparseEntryMut::*};
use std::collections::HashSet;
// use rayon::prelude::*;
use std::thread;
use std::time::Instant;

const MAX_PRINT_ROWS: usize = 64;

// TODO: Normalize mesh lengths to reduce roundoff error
// TODO: Measure impact of logging on performance



// TODO: Make struct to encapsulate all settings so I don't have a million args
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
    let cell_count: usize = mesh.cells.len();

    let a_di = build_momentum_diffusion_matrix(mesh, numerical_settings.diffusion, mu);
    let mut a_u = initialize_momentum_matrix(mesh);
    let mut a_v = initialize_momentum_matrix(mesh);
    let mut a_w = initialize_momentum_matrix(mesh);
    let mut b_u = dvector_zeros!(cell_count);
    let mut b_v = dvector_zeros!(cell_count);
    let mut b_w = dvector_zeros!(cell_count);
    let mut p_prime = dvector_zeros!(cell_count);

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
                build_momentum_matrices(
                    &mut a_u,
                    &mut a_v,
                    &mut a_w,
                    &mut b_u,
                    &mut b_v,
                    &mut b_w,
                    mesh,
                    &u,
                    &v,
                    &w,
                    &p,
                    numerical_settings.momentum,
                    numerical_settings.velocity_interpolation,
                    numerical_settings.pressure_interpolation,
                    numerical_settings.gradient_reconstruction,
                    rho,
                );

                a_u = &a_u + &a_di;
                a_v = &a_v + &a_di;
                a_w = &a_w + &a_di;

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

                if !log_enabled!(log::Level::Trace) {
                    thread::scope(|s| {
                        s.spawn(|| {
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
                        });
                        s.spawn(|| {
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
                        });
                        s.spawn(|| {
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
                        });
                    });
                } else {
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
                }
                let pressure_correction_matrices = build_pressure_correction_matrices(
                    mesh,
                    &u,
                    &v,
                    &w,
                    &p,
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
                    print_vec_scientific(&u);
                    print!("v:  ");
                    print_vec_scientific(&v);
                    print!("w:  ");
                    print_vec_scientific(&w);
                    print!("p:  ");
                    print_vec_scientific(&p);
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
                    println!(
                        "Iteration {}: avg velocity = ({:.2e}, {:.2e}, {:.2e})\tvelocity correction: {:.2e}\tpressure correction: {:.2e}\tms/iter: {:.1e}",
                        iter_number, u_avg, v_avg, w_avg, avg_vel_correction, avg_pressure_correction, millis_per_iter
                    );
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

    let mut pressure_grad_mean = Vector3::zero();
    let mut velocity_grad_mean = Tensor3::zero();
    for i in 0..mesh.cells.len() {
        let pressure_gradient =
            calculate_pressure_gradient(&mesh, &p, i, numerical_settings.gradient_reconstruction);
        let velocity_gradient = calculate_velocity_gradient(
            mesh,
            &u,
            &v,
            &w,
            i,
            numerical_settings.gradient_reconstruction,
        );
        pressure_grad_mean += pressure_gradient.abs();
        velocity_grad_mean = velocity_grad_mean + velocity_gradient.abs();
    }
    // println!(
    //     "MEAN\n{}\t{}\n",
    //     velocity_grad_mean / mesh.cells.len(),
    //     pressure_grad_mean / mesh.cells.len()
    // );
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
    let n = mesh.cells.len();
    let mut u = dvector_zeros!(n);
    let mut v = dvector_zeros!(n);
    let mut w = dvector_zeros!(n);
    let mut p = dvector_zeros!(n);
    let a_di = build_momentum_diffusion_matrix(mesh, DiffusionScheme::CD, mu);
    let mut a_u = initialize_momentum_matrix(mesh);
    let mut a_v = initialize_momentum_matrix(mesh);
    let mut a_w = initialize_momentum_matrix(mesh);
    let mut b_u = dvector_zeros!(n);
    let mut b_v = dvector_zeros!(n);
    let mut b_w = dvector_zeros!(n);

    initialize_pressure_field(mesh, &mut p, iteration_count);
    build_momentum_matrices(
        &mut a_u,
        &mut a_v,
        &mut a_w,
        &mut b_u,
        &mut b_v,
        &mut b_w,
        &mesh,
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

    println!("Initializing velocity field...");
    // NOTE: Initializing with multigrid was diverging, so using BiCGSTAB
    iterative_solve(
        &(&a_u + &a_di),
        &b_u,
        &mut u,
        iteration_count,
        SolutionMethod::BiCGSTAB,
        0.5,
        1e-6,
        PreconditionMethod::Jacobi,
    );
    iterative_solve(
        &(&a_v + &a_di),
        &b_v,
        &mut v,
        iteration_count,
        SolutionMethod::BiCGSTAB,
        0.5,
        1e-6,
        PreconditionMethod::Jacobi,
    );
    iterative_solve(
        &(&a_w + &a_di),
        &b_w,
        &mut w,
        iteration_count,
        SolutionMethod::BiCGSTAB,
        0.5,
        1e-6,
        PreconditionMethod::Jacobi,
    );
    println!("Done!");
    (u, v, w, p)
}

fn initialize_pressure_field(mesh: &Mesh, p: &mut DVector<Float>, iteration_count: Uint) {
    println!("Initializing pressure field...");
    // TODO
    // Solve laplace's equation (nabla^2 psi = 0) based on BCs:
    // - Wall: d/dn (psi) = 0
    // - Inlet: d/dn (psi) = V
    // - Outlet: psi = 0
    let n = mesh.cells.len();
    let mut a_coo = CooMatrix::<Float>::new(n, n);
    let mut b = dvector_zeros!(a_coo.nrows());

    for (i, cell) in &mesh.cells {
        let mut a_ii = 0.;
        for face in cell
            .face_indices
            .iter()
            .map(|face_index| &mesh.faces[face_index])
        {
            match mesh.face_zones[&face.zone].zone_type {
                FaceConditionTypes::Interior => {
                    let neighbor_cell_index = if face.cell_indices[0] == *i {
                        face.cell_indices[1]
                    } else {
                        face.cell_indices[0]
                    };
                    a_coo.push(*i, neighbor_cell_index, 1.0);
                    a_ii -= 1.0;
                }
                FaceConditionTypes::PressureInlet | FaceConditionTypes::PressureOutlet => {
                    a_ii -= 1e2;
                    b[*i] -= mesh.face_zones[&face.zone].scalar_value * 1e2;
                }
                _ => (),
            }
        }
        a_coo.push(*i, *i, a_ii);
    }
    let a = &CsrMatrix::from(&a_coo);
    iterative_solve(
        a,
        &b,
        p,
        iteration_count,
        SolutionMethod::BiCGSTAB,
        0.5,
        1e-6,
        PreconditionMethod::Jacobi,
    );
    println!("Done!");
}

pub fn get_velocity_source_term(_location: Vector3) -> Vector3 {
    // TODO!!!
    Vector3::zero()
}

pub fn calculate_velocity_gradient(
    mesh: &Mesh,
    u: &DVector<Float>,
    v: &DVector<Float>,
    w: &DVector<Float>,
    cell_index: usize,
    gradient_scheme: GradientReconstructionMethods,
) -> Tensor3 {
    let cell = &mesh.cells[&cell_index];
    match gradient_scheme {
        GradientReconstructionMethods::GreenGauss(_) => {
            cell.face_indices
                .iter()
                .fold(Tensor3::zero(), |acc, face_index| {
                    let face = &mesh.faces[&face_index];
                    let face_value: Vector3 = get_face_velocity(
                        mesh,
                        u,
                        v,
                        w,
                        *face_index,
                        VelocityInterpolation::LinearWeighted,
                    );
                    acc + face_value.outer(
                        &(get_outward_face_normal(face, cell_index) * (face.area / cell.volume)),
                    )
                })
        }
        _ => panic!("unsupported gradient scheme"),
    }
}

pub fn calculate_pressure_gradient(
    mesh: &Mesh,
    p: &DVector<Float>,
    cell_index: usize,
    gradient_scheme: GradientReconstructionMethods,
) -> Vector3 {
    let cell = &mesh.cells[&cell_index];
    match gradient_scheme {
        GradientReconstructionMethods::GreenGauss(variant) => match variant {
            GreenGaussVariants::CellBased => cell
                .face_indices
                .iter()
                .map(|face_index| {
                    let face = &mesh.faces[&face_index];
                    let face_value: Float = get_face_pressure(
                        mesh,
                        p,
                        *face_index,
                        PressureInterpolation::LinearWeighted,
                        gradient_scheme,
                    );
                    face_value
                        * (face.area / cell.volume)
                        * get_outward_face_normal(face, cell_index)
                })
                .fold(Vector3::zero(), |acc, v| acc + v),
            _ => panic!("unsupported Green-Gauss scheme"),
        },
        _ => panic!("unsupported gradient scheme"),
    }
}

fn get_face_velocity(
    mesh: &Mesh,
    u: &DVector<Float>,
    v: &DVector<Float>,
    w: &DVector<Float>,
    face_index: usize,
    interpolation_scheme: VelocityInterpolation,
) -> Vector3 {
    // ****** TODO: Add skewness corrections!!! ********
    let face = &mesh.faces[&face_index];
    let face_zone = &mesh.face_zones[&face.zone];
    let cell_index = face.cell_indices[0];
    match face_zone.zone_type {
        FaceConditionTypes::Wall => Vector3::zero(),
        FaceConditionTypes::VelocityInlet => face_zone.vector_value,
        FaceConditionTypes::PressureInlet
        | FaceConditionTypes::PressureOutlet
        | FaceConditionTypes::Symmetry => Vector3 {
            x: u[cell_index],
            y: v[cell_index],
            z: w[cell_index],
        },
        FaceConditionTypes::Interior => {
            let neighbor_index = face.cell_indices[1];
            let vel0 = Vector3 {
                x: u[cell_index],
                y: v[cell_index],
                z: w[cell_index],
            };
            let vel1 = Vector3 {
                x: u[neighbor_index],
                y: v[neighbor_index],
                z: w[neighbor_index],
            };
            match interpolation_scheme {
                VelocityInterpolation::LinearWeighted => {
                    let dx0 = (mesh.cells[&cell_index].centroid - face.centroid).norm();
                    let dx1 = (mesh.cells[&neighbor_index].centroid - face.centroid).norm();
                    vel0 + (vel1 - vel0) * dx0 / (dx0 + dx1)
                }
                VelocityInterpolation::RhieChow => {
                    panic!("unsupported");
                }
            }
        }
        _ => panic!("unsupported face zone type"),
    }
}

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
    let face = &mesh.faces[&face_index];
    let outward_face_normal = get_outward_face_normal(face, cell_index);
    let face_zone = &mesh.face_zones[&face.zone];
    match face_zone.zone_type {
        FaceConditionTypes::Wall | FaceConditionTypes::Symmetry => 0.,
        FaceConditionTypes::VelocityInlet
        | FaceConditionTypes::PressureInlet
        | FaceConditionTypes::PressureOutlet => {
            // TODO: optimize
            outward_face_normal.dot(&get_face_velocity(
                mesh,
                u,
                v,
                w,
                face_index,
                VelocityInterpolation::LinearWeighted, // Don't think it makes sense to need
                                                       // Rhie-Chow here
            ))
        }
        FaceConditionTypes::Interior => {
            match interpolation_scheme {
                VelocityInterpolation::LinearWeighted => outward_face_normal.dot(
                    &get_face_velocity(mesh, u, v, w, face_index, interpolation_scheme),
                ),
                VelocityInterpolation::RhieChow => {
                    let mut neighbor_index = face.cell_indices[0];
                    if neighbor_index == cell_index {
                        neighbor_index = face.cell_indices[1];
                    }
                    let vel0 = Vector3 {
                        x: u[cell_index],
                        y: v[cell_index],
                        z: w[cell_index],
                    };
                    let vel1 = Vector3 {
                        x: u[neighbor_index],
                        y: v[neighbor_index],
                        z: w[neighbor_index],
                    };
                    // WARNING: Something is wrong here
                    let xi = (mesh.cells[&neighbor_index].centroid
                        - mesh.cells[&cell_index].centroid)
                        .unit();
                    let a0 = a_u.get(cell_index, cell_index);
                    let a1 = a_u.get(neighbor_index, neighbor_index);
                    let p_grad_0 =
                        calculate_pressure_gradient(mesh, p, cell_index, gradient_scheme);
                    let p_grad_1 =
                        calculate_pressure_gradient(mesh, p, neighbor_index, gradient_scheme);
                    let v0 = mesh.cells[&cell_index].volume;
                    let v1 = mesh.cells[&neighbor_index].volume;
                    0.5 * (outward_face_normal.dot(&(vel0 + vel1))
                        + (v0 / a0 + v1 / a1) * (&p[0] - &p[1]) / xi.norm()
                        - (v0 / a0 * p_grad_0 + v1 / a1 * p_grad_1).dot(&xi))
                }
            }
        }
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
    let face = &mesh.faces[&face_index];
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
                PressureInterpolation::Linear => (&p[c0] + &p[c1]) * 0.5,
                PressureInterpolation::LinearWeighted => {
                    let x0 = (mesh.cells[&c0].centroid - face.centroid).norm();
                    let x1 = (mesh.cells[&c1].centroid - face.centroid).norm();
                    &p[c0] + (&p[c1] - &p[c0]) * x0 / (x0 + x1)
                }
                PressureInterpolation::Standard => {
                    // requires momentum matrix coeffs; seems expensive to have to pass that whole
                    // matrix each time
                    panic!("`standard` pressure interpolation unsupported");
                }
                PressureInterpolation::SecondOrder => {
                    let c0_grad = calculate_pressure_gradient(mesh, p, c0, gradient_scheme);
                    // println!("Pressure gradient: {c0_grad}");
                    // println!("Cell volume: {}", &mesh.cells[&c0].volume);
                    let c1_grad = calculate_pressure_gradient(mesh, p, c1, gradient_scheme);
                    let r_c0 = face.centroid - mesh.cells[&c0].centroid;
                    let r_c1 = face.centroid - mesh.cells[&c1].centroid;
                    0.5 * ((&p[c0] + &p[c1]) + (c0_grad.dot(&r_c0) + c1_grad.dot(&r_c1)))
                }
                _ => panic!("unsupported pressure interpolation"),
            }
        }
        _ => panic!("unsupported face zone type"),
    }
}

fn initialize_momentum_matrix(mesh: &Mesh) -> CsrMatrix<Float> {
    let cell_count = mesh.cells.len();
    let mut a: CooMatrix<Float> = CooMatrix::new(cell_count, cell_count);
    for (cell_index, cell) in &mesh.cells {
        a.push(*cell_index, *cell_index, 1.);
        for face_index in &cell.face_indices {
            let face = &mesh.faces[face_index];
            if face.cell_indices.len() == 2 {
                let neighbor_cell_index = if face.cell_indices[0] == *cell_index {
                    face.cell_indices[1]
                } else {
                    face.cell_indices[0]
                };
                a.push(*cell_index, neighbor_cell_index, 0.);
            }
        }
    }
    CsrMatrix::from(&a)
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
    let mut velocity_corr_sum = 0.;
    for (cell_index, cell) in &mut mesh.cells {
        let pressure_correction = *p_prime.get(*cell_index).unwrap_or(&0.);
        p[*cell_index] =
            &p[*cell_index] + numerical_settings.pressure_relaxation * pressure_correction;
        let velocity_correction =
            cell.face_indices
                .iter()
                .fold(Vector3::zero(), |acc, face_index| {
                    let face = &mesh.faces[face_index];
                    let face_zone = &mesh.face_zones[&face.zone];
                    let outward_face_normal = get_outward_face_normal(face, *cell_index);
                    let p_prime_neighbor = match face_zone.zone_type {
                        FaceConditionTypes::Wall
                        | FaceConditionTypes::Symmetry
                        | FaceConditionTypes::VelocityInlet => p_prime[*cell_index],
                        FaceConditionTypes::PressureInlet | FaceConditionTypes::PressureOutlet => {
                            0.
                        }
                        FaceConditionTypes::Interior => {
                            let neighbor_cell_index = if face.cell_indices[0] == *cell_index {
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
                    let scaled_normal = Vector3 {
                        x: outward_face_normal.x / a_u.get(*cell_index, *cell_index),
                        y: outward_face_normal.y / a_v.get(*cell_index, *cell_index),
                        z: outward_face_normal.z / a_w.get(*cell_index, *cell_index),
                    };
                    acc + scaled_normal * (&p_prime[*cell_index] - p_prime_neighbor) * face.area
                });
        u[*cell_index] =
            &u[*cell_index] + velocity_correction.x * numerical_settings.momentum_relaxation;
        v[*cell_index] =
            &v[*cell_index] + velocity_correction.y * numerical_settings.momentum_relaxation;
        w[*cell_index] =
            &w[*cell_index] + velocity_correction.z * numerical_settings.momentum_relaxation;
        velocity_corr_sum += velocity_correction.norm().powi(2);
    }
    (p_prime.norm(), velocity_corr_sum.sqrt())
}

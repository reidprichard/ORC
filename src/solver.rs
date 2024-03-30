#![allow(unreachable_patterns)]

use crate::io::{dvector_to_str, print_linear_system, print_matrix};
use crate::mesh::*;
use crate::GetEntry;
use crate::{common::*, io::print_vec_scientific};
use log::{log_enabled, trace};
use nalgebra::DVector;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix, SparseEntryMut::*};
use std::collections::HashSet;
// use rayon::prelude::*;
use std::thread;

// TODO: make this hierarchical
pub struct NumericalSettings {
    pub pressure_velocity_coupling: PressureVelocityCoupling,
    // Setting to 1.0 seems to cause problems
    pub momentum: MomentumDiscretization,
    pub diffusion: DiffusionScheme,
    // LinearWeighted or SecondOrder recommended; LinearWeighted is more stable but less accurate
    pub pressure_interpolation: PressureInterpolation,
    // LinearWeighted recommended; RhieChow is extremely unstable
    pub velocity_interpolation: VelocityInterpolation,
    pub gradient_reconstruction: GradientReconstructionMethods,
    pub pressure_relaxation: Float,
    pub momentum_relaxation: Float,
    pub matrix_solver: MatrixSolverSettings,
}

impl Default for NumericalSettings {
    fn default() -> Self {
        NumericalSettings {
            pressure_velocity_coupling: PressureVelocityCoupling::SIMPLE,
            momentum: MomentumDiscretization::CD1,
            diffusion: DiffusionScheme::CD,
            pressure_interpolation: PressureInterpolation::LinearWeighted,
            velocity_interpolation: VelocityInterpolation::LinearWeighted,
            gradient_reconstruction: GradientReconstructionMethods::GreenGauss(
                GreenGaussVariants::CellBased,
            ),
            pressure_relaxation: 0.25,
            momentum_relaxation: 0.5,
            matrix_solver: MatrixSolverSettings::default(),
        }
    }
}

pub struct MatrixSolverSettings {
    pub solver_type: SolutionMethod,
    // It seems like too many iterations can cause pv coupling to go crazy
    pub iterations: Uint,
    pub relaxation: Float,
    pub relative_convergence_threshold: Float,
}
impl Default for MatrixSolverSettings {
    fn default() -> Self {
        MatrixSolverSettings {
            solver_type: SolutionMethod::Multigrid,
            iterations: 50,
            relaxation: 0.5,
            relative_convergence_threshold: 0.001,
        }
    }
}

#[derive(Copy, Clone)]
pub enum SolutionMethod {
    GaussSeidel,
    Jacobi,
    Multigrid,
    BiCGSTAB,
}

#[derive(Copy, Clone)]
pub enum PressureVelocityCoupling {
    SIMPLE,
}

#[derive(Copy, Clone)]
pub enum MomentumDiscretization {
    // First-order upwind
    UD,
    // Basic CD; first-order on arbitrary grid and second-order with even spacing
    CD1,
    // CD incorporating cell gradients; second-order on arbitrary grid
    CD2,
}

#[derive(Copy, Clone)]
pub enum DiffusionScheme {
    CD,
}

#[derive(Copy, Clone)]
pub enum PressureInterpolation {
    Linear,
    LinearWeighted,
    Standard,
    SecondOrder,
}

#[derive(Copy, Clone)]
pub enum VelocityInterpolation {
    LinearWeighted,
    // Rhie-Chow is expensive! And it seems to be causing bad unphysical oscillations.
    RhieChow,
}

#[derive(Copy, Clone)]
pub enum GreenGaussVariants {
    CellBased,
    NodeBased,
}

#[derive(Copy, Clone)]
pub enum GradientReconstructionMethods {
    GreenGauss(GreenGaussVariants),
    LeastSquares,
}

pub struct LinearSystem {
    a: CsrMatrix<Float>,
    b: DVector<Float>,
}

macro_rules! initialize_DVector {
    ($n:expr) => {
        DVector::from_column_slice(&vec![0.; $n])
    };
}

// TODO: Make struct to encapsulate all settings so I don't have a million args
pub fn solve_steady(
    mesh: &mut Mesh,
    numerical_settings: &NumericalSettings,
    rho: Float,
    mu: Float,
    iteration_count: Uint,
    reporting_interval: Uint,
) -> (
    DVector<Float>,
    DVector<Float>,
    DVector<Float>,
    DVector<Float>,
) {
    let cell_count: usize = mesh.cells.len();
    let mut u = initialize_DVector!(cell_count);
    let mut v = initialize_DVector!(cell_count);
    let mut w = initialize_DVector!(cell_count);
    let mut p = initialize_DVector!(cell_count);
    let mut p_prime = initialize_DVector!(cell_count);

    initialize_pressure_field(mesh, &mut p, 10000);

    let a_di = build_momentum_diffusion_matrix(&mesh, numerical_settings.diffusion, mu);
    let mut a = initialize_momentum_matrix(&mesh);
    let mut b_u = initialize_DVector!(cell_count);
    let mut b_v = initialize_DVector!(cell_count);
    let mut b_w = initialize_DVector!(cell_count);

    if log_enabled!(log::Level::Debug) && a_di.ncols() < 256 {
        println!("\nMomentum diffusion:");
        print_matrix(&a_di);
    }
    match numerical_settings.pressure_velocity_coupling {
        PressureVelocityCoupling::SIMPLE => {
            for iter_number in 1..=iteration_count {
                build_momentum_matrices(
                    &mut a,
                    &mut b_u,
                    &mut b_v,
                    &mut b_w,
                    mesh,
                    &u,
                    &v,
                    &w,
                    &p,
                    numerical_settings.momentum,
                    if iteration_count > 1 {
                        numerical_settings.velocity_interpolation
                    } else {
                        VelocityInterpolation::LinearWeighted
                    },
                    if iteration_count > 1 {
                        numerical_settings.pressure_interpolation
                    } else {
                        PressureInterpolation::LinearWeighted
                    },
                    numerical_settings.gradient_reconstruction,
                    rho,
                );
                a = &a + &a_di;
                if log_enabled!(log::Level::Debug) && a.nrows() < 256 {
                    println!("\nMomentum:");
                    print_linear_system(&a, &b_u);
                }

                if !log_enabled!(log::Level::Trace) {
                    thread::scope(|s| {
                        s.spawn(|| {
                            trace!("solving u");
                            iterative_solve(
                                &a,
                                &b_u,
                                &mut u,
                                numerical_settings.matrix_solver.iterations,
                                numerical_settings.matrix_solver.solver_type,
                                numerical_settings.matrix_solver.relaxation,
                                numerical_settings
                                    .matrix_solver
                                    .relative_convergence_threshold,
                            );
                        });
                        s.spawn(|| {
                            trace!("solving v");
                            iterative_solve(
                                &a,
                                &b_v,
                                &mut v,
                                numerical_settings.matrix_solver.iterations,
                                numerical_settings.matrix_solver.solver_type,
                                numerical_settings.matrix_solver.relaxation,
                                numerical_settings
                                    .matrix_solver
                                    .relative_convergence_threshold,
                            );
                        });
                        s.spawn(|| {
                            trace!("solving w");
                            iterative_solve(
                                &a,
                                &b_w,
                                &mut w,
                                numerical_settings.matrix_solver.iterations,
                                numerical_settings.matrix_solver.solver_type,
                                numerical_settings.matrix_solver.relaxation,
                                numerical_settings
                                    .matrix_solver
                                    .relative_convergence_threshold,
                            );
                        });
                    });
                } else {
                    trace!("solving u");
                    iterative_solve(
                        &a,
                        &b_u,
                        &mut u,
                        numerical_settings.matrix_solver.iterations,
                        numerical_settings.matrix_solver.solver_type,
                        numerical_settings.matrix_solver.relaxation,
                        numerical_settings
                            .matrix_solver
                            .relative_convergence_threshold,
                    );
                    trace!("solving v");
                    iterative_solve(
                        &a,
                        &b_v,
                        &mut v,
                        numerical_settings.matrix_solver.iterations,
                        numerical_settings.matrix_solver.solver_type,
                        numerical_settings.matrix_solver.relaxation,
                        numerical_settings
                            .matrix_solver
                            .relative_convergence_threshold,
                    );
                    trace!("solving w");
                    iterative_solve(
                        &a,
                        &b_w,
                        &mut w,
                        numerical_settings.matrix_solver.iterations,
                        numerical_settings.matrix_solver.solver_type,
                        numerical_settings.matrix_solver.relaxation,
                        numerical_settings
                            .matrix_solver
                            .relative_convergence_threshold,
                    );
                }
                let pressure_correction_matrices = build_pressure_correction_matrices(
                    mesh,
                    &u,
                    &v,
                    &w,
                    &p,
                    &a,
                    numerical_settings,
                    rho,
                );
                if log_enabled!(log::Level::Debug) && a.nrows() < 256 {
                    println!("\nPressure:");
                    print_linear_system(
                        &pressure_correction_matrices.a,
                        &pressure_correction_matrices.b,
                    );
                }

                trace!("solving p");
                // Zero the pressure correction for a reasonable initial guess
                p_prime = p_prime * 0.;
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
                    &a,
                    &p_prime,
                    &mut u,
                    &mut v,
                    &mut w,
                    &mut p,
                    &numerical_settings,
                );

                let u_avg = u.iter().sum::<Float>() / (cell_count as Float);
                let v_avg = v.iter().sum::<Float>() / (cell_count as Float);
                let w_avg = w.iter().sum::<Float>() / (cell_count as Float);
                if (iter_number) % reporting_interval == 0 {
                    println!(
                        "Iteration {}: avg velocity = ({:.2e}, {:.2e}, {:.2e})\tvelocity correction: {:.2e}\tpressure correction: {:.2e}",
                        iter_number, u_avg, v_avg, w_avg, avg_vel_correction, avg_pressure_correction
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
    (u, v, w, p)
}

fn initialize_pressure_field(mesh: &mut Mesh, p: &mut DVector<Float>, iteration_count: Uint) {
    println!("Initializing pressure field...");
    // TODO
    // Solve laplace's equation (nabla^2 psi = 0) based on BCs:
    // - Wall: d/dn (psi) = 0
    // - Inlet: d/dn (psi) = V
    // - Outlet: psi = 0
    let n = mesh.cells.len();
    let mut a_coo = CooMatrix::<Float>::new(n, n);
    let mut b = initialize_DVector!(a_coo.nrows());

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
        &a,
        &b,
        p,
        iteration_count,
        SolutionMethod::Jacobi,
        0.5,
        1e-6,
    );
    println!("Done!");
}

fn get_velocity_source_term(_location: Vector3) -> Vector3 {
    Vector3::zero()
}

fn calculate_pressure_gradient(
    mesh: &Mesh,
    p: &DVector<Float>,
    cell_index: usize,
    gradient_scheme: GradientReconstructionMethods,
) -> Vector3 {
    let cell = &mesh.cells[&cell_index];
    match gradient_scheme {
        GradientReconstructionMethods::GreenGauss(variant) => match variant {
            GreenGaussVariants::CellBased => {
                cell.face_indices
                    .iter()
                    .map(|face_index| {
                        let face = &mesh.faces[&face_index];
                        let face_value: Float = get_face_pressure(
                            &mesh,
                            &p,
                            *face_index,
                            PressureInterpolation::Linear,
                            gradient_scheme,
                        );
                        face_value * face.area * get_outward_face_normal(face, cell_index)
                    })
                    .fold(Vector3::zero(), |acc, v| acc + v)
                    / cell.volume
            }
            _ => panic!("unsupported Green-Gauss scheme"),
        },
        _ => panic!("unsupported gradient scheme"),
    }
}

fn get_face_flux(
    mesh: &Mesh,
    u: &DVector<Float>,
    v: &DVector<Float>,
    w: &DVector<Float>,
    p: &DVector<Float>,
    face_index: usize,
    cell_index: usize,
    interpolation_scheme: VelocityInterpolation,
    gradient_scheme: GradientReconstructionMethods,
    momentum_matrices: &CsrMatrix<Float>,
) -> Float {
    // ****** TODO: Add skewness corrections!!! ********
    let face = &mesh.faces[&face_index];
    let outward_face_normal = get_outward_face_normal(face, cell_index);
    let face_zone = &mesh.face_zones[&face.zone];
    match face_zone.zone_type {
        FaceConditionTypes::Wall | FaceConditionTypes::Symmetry => 0.,
        FaceConditionTypes::VelocityInlet => face_zone.vector_value.dot(&outward_face_normal),
        FaceConditionTypes::PressureInlet | FaceConditionTypes::PressureOutlet => {
            // TODO: optimize
            outward_face_normal.dot(&Vector3 {
                x: u[cell_index],
                y: v[cell_index],
                z: w[cell_index],
            })
        }
        FaceConditionTypes::Interior => {
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
            match interpolation_scheme {
                VelocityInterpolation::LinearWeighted => {
                    let dx0 = (mesh.cells[&cell_index].centroid - face.centroid).norm();
                    let dx1 = (mesh.cells[&neighbor_index].centroid - face.centroid).norm();
                    outward_face_normal.dot(&(vel0 + (vel1 - vel0) * dx0 / (dx0 + dx1)))
                }
                VelocityInterpolation::RhieChow => {
                    // WARNING: Something is wrong here
                    let xi = (mesh.cells[&neighbor_index].centroid
                        - mesh.cells[&cell_index].centroid)
                        .unit();
                    let a0 = momentum_matrices.get(cell_index, cell_index);
                    let a1 = momentum_matrices.get(neighbor_index, neighbor_index);
                    let p_grad_0 =
                        calculate_pressure_gradient(&mesh, &p, cell_index, gradient_scheme);
                    let p_grad_1 =
                        calculate_pressure_gradient(&mesh, &p, neighbor_index, gradient_scheme);
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

fn get_face_pressure(
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
        FaceConditionTypes::Symmetry | FaceConditionTypes::Wall => {
            // du/dn = 0, so we need the projection of cell center velocity onto the face's plane
            p[face.cell_indices[0]]
        }
        FaceConditionTypes::VelocityInlet => p[face.cell_indices[0]],
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
                    let c0_grad = calculate_pressure_gradient(&mesh, &p, c0, gradient_scheme);
                    // println!("Pressure gradient: {c0_grad}");
                    // println!("Cell volume: {}", &mesh.cells[&c0].volume);
                    let c1_grad = calculate_pressure_gradient(&mesh, &p, c1, gradient_scheme);
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

#[derive(Copy, Clone)]
enum RestrictionMethods {
    Injection,
    Strongest,
}

fn build_restriction_matrix(a: &CsrMatrix<Float>, method: RestrictionMethods) -> CsrMatrix<Float> {
    let n = a.ncols() / 2 + a.ncols() % 2;
    let mut restriction_matrix_coo = CooMatrix::<Float>::new(n, a.ncols());
    match method {
        RestrictionMethods::Injection => {
            for row_num in 0..n - 1 {
                restriction_matrix_coo.push(row_num, 2 * row_num, 1.);
                restriction_matrix_coo.push(row_num, 2 * row_num + 1, 1.);
            }
            // Must treat the last row separately since a.ncols() could be odd
            restriction_matrix_coo.push(n - 1, 2 * (n - 1), 1.);
            if 2 * (n - 1) + 1 < a.ncols() {
                restriction_matrix_coo.push(n - 1, 2 * (n - 1) + 1, 1.);
            }
        }
        RestrictionMethods::Strongest => {
            // For each row, find the largest off-diagonal value
            // If that cell hasn't already been combined, combine it with diagonal
            // If it *has* been combined, find the next largest and so on.
            let mut combined_cells = HashSet::<usize>::new();
            a.row_iter().enumerate().for_each(|(i, row)| {
                let mut largest_coefficient: Float = Float::MAX;
                let strongest_unmerged_neighbor =
                    row.col_indices().iter().fold(usize::MAX, |acc, j| {
                        // not efficient to look this up each time
                        if combined_cells.contains(j) || i == *j {
                            acc
                        } else {
                            let entry = a.get(i, *j);
                            if entry < largest_coefficient {
                                largest_coefficient = entry;
                                *j
                            } else {
                                acc
                            }
                        }
                    });
                if strongest_unmerged_neighbor != usize::MAX {
                    // panic!("Multigrid failed. It is likely the solution has diverged.");
                    combined_cells.insert(strongest_unmerged_neighbor);
                    restriction_matrix_coo.push(i / 2, i, 1.0);
                    restriction_matrix_coo.push(i / 2, strongest_unmerged_neighbor, 1.0);
                }
            });
        }
    }
    CsrMatrix::from(&restriction_matrix_coo)
}

fn multigrid_solve(
    a: &CsrMatrix<Float>,
    r: &DVector<Float>,
    multigrid_level: Uint,
    max_levels: Uint,
    smooth_method: SolutionMethod,
    smooth_iter_count: Uint,
    smooth_relaxation: Float,
    smooth_convergence_threshold: Float,
    restriction_method: RestrictionMethods,
) -> DVector<Float> {
    // 1. Build restriction matrix R
    // 5. If multigrid_level < max_level, recurse
    let restriction_matrix = build_restriction_matrix(&a, restriction_method);
    // 2. Restrict r to r'
    let r_prime = &restriction_matrix * r;
    // 3. Create a' = R * a * R.⊺
    let a_prime = &restriction_matrix * a * &restriction_matrix.transpose();
    // 4. Solve a' * e' = r'
    let mut e_prime: DVector<Float> = initialize_DVector!(a_prime.ncols());
    iterative_solve(
        &a_prime,
        &r_prime,
        &mut e_prime,
        smooth_iter_count,
        smooth_method,
        smooth_relaxation,
        smooth_convergence_threshold,
    );
    if log_enabled!(log::Level::Trace) {
        println!(
            "Multigrid level {}: |e| = {:.2e}",
            multigrid_level,
            (&r_prime - &a_prime * &e_prime).norm()
        );
    }

    // 5. Recurse to max desired coarsening level
    // Only coarsen if the matrix is bigger than 16x16 because come on
    if multigrid_level < max_levels && a_prime.nrows() > 16 {
        e_prime += multigrid_solve(
            &a_prime,
            &r_prime,
            multigrid_level + 1,
            max_levels,
            smooth_method,
            smooth_iter_count,
            smooth_relaxation,
            smooth_convergence_threshold,
            restriction_method,
        );
        // Smooth after incorporating corrections from coarser grid
        iterative_solve(
            &a_prime,
            &r_prime,
            &mut e_prime,
            smooth_iter_count,
            smooth_method,
            smooth_relaxation,
            smooth_convergence_threshold / 10.,
        );
        if log_enabled!(log::Level::Trace) {
            println!(
                "Multigrid level {}: |e| = {:.2e}",
                multigrid_level,
                (&r_prime - &a_prime * &e_prime).norm() // e_prime.norm()
            );
        }
    }
    // Prolong correction vector
    &restriction_matrix.transpose() * e_prime
}

pub fn iterative_solve(
    a: &CsrMatrix<Float>,
    b: &DVector<Float>,
    solution_vector: &mut DVector<Float>,
    iteration_count: Uint,
    method: SolutionMethod,
    relaxation_factor: Float,
    convergence_threshold: Float,
) {
    let mut initial_residual: Float = 0.;
    match method {
        SolutionMethod::Jacobi => {
            let mut a_prime = a.clone();
            a_prime
                .triplet_iter_mut()
                .for_each(|(i, j, v)| *v = if i == j { 0. } else { *v / a.get(i, i) });
            let b_prime: DVector<Float> = DVector::from_iterator(
                b.nrows(),
                b.iter().enumerate().map(|(i, v)| *v / a.get(i, i)),
            );
            for iter_num in 0..iteration_count {
                // It seems like there must be a way to avoid cloning solution_vector, even if that
                // turns this into Gauss-Seidel
                *solution_vector = relaxation_factor * (&b_prime - &a_prime * &*solution_vector)
                    + &*solution_vector * (1. - relaxation_factor);

                let r = (b - a * &*solution_vector).norm();
                if iter_num == 1 {
                    initial_residual = r;
                } else if r / initial_residual < convergence_threshold {
                    trace!("Converged in {} iters", iter_num);
                    break;
                }
            }
        }
        SolutionMethod::GaussSeidel => {
            for _iter_num in 0..iteration_count {
                // if log_enabled!(log::Level::Trace) {
                //     println!("Gauss-Seidel iteration {iter_num} = {solution_vector:?}");
                // }
                for i in 0..a.nrows() {
                    solution_vector[i] = solution_vector[i] * (1. - relaxation_factor)
                        + relaxation_factor
                            * (b[i]
                                - solution_vector
                                    .iter() // par_iter is slower here with 1k cells; might be worth it with more cells
                                    .enumerate()
                                    .map(|(j, x)| if i != j { a.get(i, j) * x } else { 0. })
                                    .sum::<Float>())
                            / a.get(i, i);
                    if solution_vector[i].is_nan() {
                        panic!("****** Solution diverged ******");
                    }
                }
            }
            panic!("Gauss-Seidel out for maintenance :)");
        }
        SolutionMethod::BiCGSTAB => {
            // TODO: Optimize this
            let mut r = b - a * &*solution_vector;
            let r_hat_0 = DVector::from_column_slice(&vec![1.; r.nrows()]);
            let mut rho = r.dot(&r_hat_0);
            let mut p = r.clone();
            for _iter_num in 0..iteration_count {
                let nu: DVector<Float> = a * &p;
                let alpha: Float = rho / &r_hat_0.dot(&nu);
                let h: DVector<Float> = &*solution_vector + alpha * &p;
                let s: DVector<Float> = &r - alpha * &nu;
                let t: DVector<Float> = a * &s;
                let omega: Float = &t.dot(&s) / &t.dot(&t);
                *solution_vector = &h + omega * &s;
                r = &s - omega * &t;
                let rho_prev: Float = rho;
                rho = r_hat_0.dot(&r);
                let beta: Float = rho / rho_prev * alpha / omega;
                p = &r + beta * (p - omega * &nu);
            }
        }
        SolutionMethod::Multigrid => {
            // It seems that too many coarsening levels can cause stability issues.
            // I wonder if this is why Fluent has more complex AMG cycles.
            const COARSENING_LEVELS: Uint = 4;
            iterative_solve(
                a,
                b,
                solution_vector,
                iteration_count,
                SolutionMethod::BiCGSTAB,
                relaxation_factor,
                convergence_threshold,
            );
            // TODO: find a way not to have to clone
            let r = b - a * &*solution_vector;
            *solution_vector += multigrid_solve(
                &a,
                &r,
                1,
                COARSENING_LEVELS,
                SolutionMethod::Jacobi,
                iteration_count,
                relaxation_factor,
                convergence_threshold,
                RestrictionMethods::Strongest,
            );
        }
        _ => panic!("unsupported solution method"),
    }
}

fn build_momentum_diffusion_matrix(
    mesh: &Mesh,
    diffusion_scheme: DiffusionScheme,
    mu: Float,
) -> CsrMatrix<Float> {
    if !matches!(diffusion_scheme, DiffusionScheme::CD) {
        panic!("unsupported diffusion scheme");
    }

    let cell_count = mesh.cells.len();
    let mut a = CooMatrix::<Float>::new(cell_count, cell_count);

    // Iterate over all cells in the mesh
    for (cell_index, cell) in &mesh.cells {
        // The current cell's coefficients (matrix diagonal)
        let mut a_p = 0.;

        // Iterate over this cell's faces
        for face_index in &cell.face_indices {
            let face = &mesh.faces[face_index];
            let face_bc = &mesh.face_zones[&face.zone];
            let (d_i, neighbor_cell_index) = match face_bc.zone_type {
                FaceConditionTypes::Wall => {
                    let d_i = mu * face.area / (face.centroid - cell.centroid).norm();
                    (d_i, usize::MAX)
                }
                FaceConditionTypes::PressureInlet
                | FaceConditionTypes::PressureOutlet
                | FaceConditionTypes::Symmetry => {
                    // NOTE: Assumes zero boundary-normal velocity gradient
                    (
                        0., // no diffusion since face velocity == cell velocity
                        usize::MAX,
                    )
                }
                FaceConditionTypes::VelocityInlet => {
                    // Diffusion coefficient
                    // TODO: Add source term contribution to diffusion since there's no cell on the other side
                    let d_i: Float = mu * face.area / (face.centroid - cell.centroid).norm();
                    (d_i, usize::MAX);
                    panic!("untested");
                }
                FaceConditionTypes::Interior => {
                    let mut neighbor_cell_index: usize = face.cell_indices[0];
                    if neighbor_cell_index == *cell_index {
                        neighbor_cell_index = *face
                            .cell_indices
                            .get(1)
                            .expect("interior faces should have two neighbors");
                    }

                    // Cell centroids vector
                    let e_xi: Vector3 = mesh.cells[&neighbor_cell_index].centroid - cell.centroid;
                    // Diffusion coefficient
                    let d_i: Float = mu * face.area / e_xi.norm();

                    // Skipping cross diffusion for now TODO
                    // let d_i: Float = mu
                    //     * (outward_face_normal.dot(&outward_face_normal)
                    //         / outward_face_normal.dot(&e_xi))
                    //     * face.area;
                    // Face tangent vector -- this feels inefficient
                    // let e_nu: Vector = outward_face_normal.cross(&e_xi.cross(&outward_face_normal)).unit();
                    // Cross diffusion source term
                    // let s_cross_diffusion = -mu *
                    (d_i, neighbor_cell_index)
                }
                unsupported_zone_type => {
                    println!("*** {} ***", unsupported_zone_type);
                    panic!("BC not supported");
                }
            }; // end BC match
               // if log_enabled!(log::Level::Trace) {
               //     println!(
               //         "cell {: >3}, face {: >3}: D_i = {: >9}",
               //         cell_index, face_index, d_i
               //     );
               // }

            let face_contribution = d_i;
            a_p += face_contribution;
            // If it's MAX, that means it's a boundary face
            if neighbor_cell_index != usize::MAX {
                // negate a_nb to move to LHS of equation
                a.push(*cell_index, neighbor_cell_index, -d_i);
            }
        } // end face loop
        a.push(*cell_index, *cell_index, a_p);
    } // end cell loop
    CsrMatrix::from(&a)
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

fn build_momentum_matrices(
    a: &mut CsrMatrix<Float>,
    b_u: &mut DVector<Float>,
    b_v: &mut DVector<Float>,
    b_w: &mut DVector<Float>,
    mesh: &Mesh,
    u: &DVector<Float>,
    v: &DVector<Float>,
    w: &DVector<Float>,
    p: &DVector<Float>,
    momentum_discretization: MomentumDiscretization,
    velocity_interpolation: VelocityInterpolation,
    pressure_interpolation: PressureInterpolation,
    gradient_reconstruction: GradientReconstructionMethods,
    rho: Float,
) {
    // Iterate over all cells in the mesh
    for (cell_index, cell) in &mesh.cells {
        // Diffusion of scalar phi from neighbor into this cell
        // = <face area> * <diffusivity> * <face-normal gradient of phi>
        // = A * nu * d/dn(phi)
        // = A * nu * (phi_nb - phi_c) / |n|
        // = -D_i * (phi_c - phi_nb)
        // Advection of scalar phi from neighbor into this cell
        // = <face velocity> dot <face inward normal> * <face area> * <face value of phi>
        // = ~F_i * (phi_c + phi_nb) / 2 (central differencing)
        //
        // <sum of convection/diffusion of momentum into cell> = <momentum source>

        // let this_cell_velocity_gradient = &mesh.calculate_velocity_gradient(*cell_number);
        let mut s_u = get_velocity_source_term(cell.centroid); // general source term
        let s_u_dc = Vector3::zero(); // deferred correction source term TODO
        let s_d_cross = Vector3::zero(); // cross diffusion source term TODO

        // The current cell's coefficients (matrix diagonal)
        let mut a_p = 0.;

        // Iterate over this cell's faces
        for face_index in &cell.face_indices {
            let face = &mesh.faces[face_index];
            let face_flux = get_face_flux(
                &mesh,
                &u,
                &v,
                &w,
                &p,
                *face_index,
                *cell_index,
                velocity_interpolation,
                gradient_reconstruction,
                &a,
            );
            // println!("cell {cell_number}, face {face_number}: velocity = {face_velocity:?}");
            // TODO: Consider flipping convention of face normal direction and/or potentially
            // make it an area vector
            let outward_face_normal = get_outward_face_normal(&face, *cell_index);
            // Mass flow rate out of this cell through this face
            let f_i = face_flux * face.area * rho;
            let face_pressure = get_face_pressure(
                mesh,
                &p,
                *face_index,
                pressure_interpolation,
                gradient_reconstruction,
            );
            let neighbor_cell_index = if face.cell_indices.len() == 1 {
                usize::MAX
            } else if face.cell_indices[0] == *cell_index {
                // face normal points to cell 0 by default, so we need to flip it
                *face
                    .cell_indices
                    .get(1)
                    .expect("interior faces should have two neighbors")
            } else {
                face.cell_indices[0]
            };

            // if log_enabled!(log::Level::Trace) {
            //     println!(
            //         "cell {: >3}, face {: >3}: F_i = {: >9} {}",
            //         cell_index,
            //         face_index,
            //         f_i,
            //         if neighbor_cell_index == usize::MAX {
            //             "(boundary)"
            //         } else {
            //             ""
            //         }
            //     );
            // }

            let a_nb: Float = match momentum_discretization {
                MomentumDiscretization::UD => {
                    // Neighbor only affects this cell if flux is into this
                    // cell => f_i < 0. Therefore, if f_i > 0, we set it to 0.
                    Float::min(f_i, 0.)
                }
                MomentumDiscretization::CD1 => f_i / 2.,
                // MomentumDiscretization::CD2 => {
                //
                // }
                _ => panic!("unsupported momentum scheme"),
            };

            let face_contribution = -a_nb + f_i; // sign of s_p?
            a_p += face_contribution;
            s_u += (-outward_face_normal) * face_pressure * face.area;

            // If it's MAX, that means it's a boundary face
            if neighbor_cell_index != usize::MAX {
                // negate a_nb to move to LHS of equation
                match a.get_entry_mut(*cell_index, neighbor_cell_index).unwrap() {
                    NonZero(v) => *v = a_nb,
                    Zero => panic!(),
                }
            }
        } // end face loop
        let source_total = s_u + s_u_dc + s_d_cross;
        b_u[*cell_index] = source_total.x;
        b_v[*cell_index] = source_total.y;
        b_w[*cell_index] = source_total.z;

        match a.get_entry_mut(*cell_index, *cell_index).unwrap() {
            NonZero(v) => *v = a_p,
            Zero => panic!(),
        }
    } // end cell loop
      // NOTE: I *think* all diagonal terms in `a` should be positive and all off-diagonal terms
      // negative. It may be worth adding assertions to validate this.
}

fn build_pressure_correction_matrices(
    mesh: &Mesh,
    u: &DVector<Float>,
    v: &DVector<Float>,
    w: &DVector<Float>,
    p: &DVector<Float>,
    momentum_matrices: &CsrMatrix<Float>,
    numerical_settings: &NumericalSettings,
    rho: Float,
) -> LinearSystem {
    let cell_count = mesh.cells.len();
    // The coefficients of the pressure correction matrix
    let mut a = CooMatrix::<Float>::new(cell_count, cell_count);
    // This is the net mass flow rate into each cell
    let mut b = initialize_DVector!(cell_count);

    for (cell_index, cell) in &mesh.cells {
        let mut a_p: Float = 0.;
        let mut b_p: Float = 0.;
        for face_index in &cell.face_indices {
            let face = &mesh.faces[face_index];
            let face_flux = get_face_flux(
                mesh,
                &u,
                &v,
                &w,
                &p,
                *face_index,
                *cell_index,
                numerical_settings.velocity_interpolation,
                numerical_settings.gradient_reconstruction,
                &momentum_matrices,
            );
            // The net mass flow rate through this face into this cell
            b_p += rho * -face_flux * face.area;

            let a_ii = momentum_matrices.get(*cell_index, *cell_index);

            if face.cell_indices.len() > 1 {
                let neighbor_cell_index = if face.cell_indices[0] != *cell_index {
                    face.cell_indices[0]
                } else {
                    face.cell_indices[1]
                };
                // NOTE: It would be more rigorous to recalculate the advective coefficients,
                // but I think this should be sufficient for now.
                let a_ij = momentum_matrices.get(*cell_index, neighbor_cell_index);

                let a_interpolated = (a_ij + a_ii) / 2.;

                if a_interpolated != 0. {
                    let a_nb = rho * Float::powi(face.area, 2) / a_interpolated;
                    a.push(*cell_index, neighbor_cell_index, -a_nb);
                    a_p += a_nb;
                }
            } else {
                let a_nb = rho * Float::powi(face.area, 2) / a_ii;
                a_p += a_nb / 2.; // NOTE: I'm not sure if dividing by two is correct here?
            }
        }
        a.push(*cell_index, *cell_index, a_p);
        b[*cell_index] = b_p;
    }

    LinearSystem {
        a: CsrMatrix::from(&a),
        b,
    }
}

fn apply_pressure_correction(
    mesh: &mut Mesh,
    momentum_matrices: &CsrMatrix<Float>, // NOTE: I really only need the diagonal
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
        let velocity_correction = cell
            .face_indices
            .iter()
            .fold(Vector3::zero(), |acc, face_index| {
                let face = &mesh.faces[face_index];
                let face_zone = &mesh.face_zones[&face.zone];
                let outward_face_normal = get_outward_face_normal(&face, *cell_index);
                let p_prime_neighbor = match face_zone.zone_type {
                    FaceConditionTypes::Wall | FaceConditionTypes::Symmetry => {
                        p_prime[*cell_index]
                    }
                    FaceConditionTypes::PressureInlet | FaceConditionTypes::PressureOutlet => {
                        0.
                    }
                    FaceConditionTypes::VelocityInlet => {
                        p_prime[*cell_index]
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
                acc + outward_face_normal * (&p_prime[*cell_index] - p_prime_neighbor) * face.area / momentum_matrices.get_entry(*cell_index, *cell_index).expect("momentum matrix should have nonzero coeffs relating each cell to its neighbors")
.into_value()
            });
        u[*cell_index] =
            &u[*cell_index] + velocity_correction.x * numerical_settings.momentum_relaxation;
        v[*cell_index] =
            &v[*cell_index] + velocity_correction.y * numerical_settings.momentum_relaxation;
        w[*cell_index] =
            &w[*cell_index] + velocity_correction.z * numerical_settings.momentum_relaxation;
        velocity_corr_sum += velocity_correction.norm().powi(2);
    }
    (
        p_prime.norm(),
        velocity_corr_sum.sqrt(),
    )
}

use crate::io::{dvector_to_str, print_linear_system, print_matrix};
use crate::mesh::*;
use crate::{common::*, io::print_vec_scientific};
use itertools::izip;
use log::{info, log_enabled};
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use rayon::prelude::*;
// use sprs::{CsMat, TriMat};
use std::thread;

// TODO: Change to SOA format (separate u, v, w, p arrays rather than being stored in cell objs)
// TODO: Change cell/face/node numbers to `usize`

const MATRIX_SOLVER_RELAXATION: Float = 0.33;
const MATRIX_SOLVER_ITERS: Uint = 20;
const PARALLELIZE_U_V_W: bool = true;

#[derive(Copy, Clone)]
pub enum SolutionMethod {
    GaussSeidel,
    Jacobi,
}

#[derive(Copy, Clone)]
pub enum PressureVelocityCoupling {
    SIMPLE,
}

#[derive(Copy, Clone)]
pub enum MomentumDiscretization {
    UD,
    CD,
}

#[derive(Copy, Clone)]
pub enum PressureInterpolation {
    Linear,
    Standard,
    SecondOrder,
}

#[derive(Copy, Clone)]
pub enum VelocityInterpolation {
    Linear,
    RhieChow2,
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

pub fn solve_steady(
    mesh: &mut Mesh,
    pressure_velocity_coupling: PressureVelocityCoupling,
    momentum_scheme: MomentumDiscretization,
    pressure_interpolation_scheme: PressureInterpolation,
    velocity_interpolation_scheme: VelocityInterpolation,
    rho: Float,
    mu: Float,
    iteration_count: Uint,
    momentum_relaxation_factor: Float,
    pressure_relaxation_factor: Float, // 0.4 seems to be the upper limit for stability
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
    // initialize_pressure_field(mesh, &mut p, 1000);
    let a_di = build_momentum_diffusion_matrix(
        mesh,
        &u,
        &v,
        &w,
        &p,
        pressure_interpolation_scheme,
        velocity_interpolation_scheme,
        rho,
        mu,
    );
    if log_enabled!(log::Level::Debug) {
        println!("\nMomentum diffusion:");
        print_matrix(&a_di);
    }
    match pressure_velocity_coupling {
        PressureVelocityCoupling::SIMPLE => {
            for iter_number in 1..=iteration_count {
                let (mut a, b_u, b_v, b_w) = build_momentum_matrices(
                    mesh,
                    &u,
                    &v,
                    &w,
                    &p,
                    momentum_scheme,
                    pressure_interpolation_scheme,
                    velocity_interpolation_scheme,
                    rho,
                    mu,
                );
                a = &a + &a_di;
                if log_enabled!(log::Level::Debug) {
                    println!("\nMomentum:");
                    print_linear_system(&a, &b_u);
                }

                if PARALLELIZE_U_V_W {
                    thread::scope(|s| {
                        s.spawn(|| {
                            solve_linear_system(
                                &a,
                                &b_u,
                                &mut u,
                                MATRIX_SOLVER_ITERS,
                                SolutionMethod::Jacobi,
                                MATRIX_SOLVER_RELAXATION,
                            );
                        });
                        s.spawn(|| {
                            solve_linear_system(
                                &a,
                                &b_v,
                                &mut v,
                                MATRIX_SOLVER_ITERS,
                                SolutionMethod::Jacobi,
                                MATRIX_SOLVER_RELAXATION,
                            );
                        });
                        s.spawn(|| {
                            solve_linear_system(
                                &a,
                                &b_w,
                                &mut w,
                                MATRIX_SOLVER_ITERS,
                                SolutionMethod::Jacobi,
                                MATRIX_SOLVER_RELAXATION,
                            );
                        });
                    });
                } else {
                    solve_linear_system(
                        &a,
                        &b_u,
                        &mut u,
                        MATRIX_SOLVER_ITERS,
                        SolutionMethod::Jacobi,
                        MATRIX_SOLVER_RELAXATION,
                    );
                    solve_linear_system(
                        &a,
                        &b_v,
                        &mut v,
                        MATRIX_SOLVER_ITERS,
                        SolutionMethod::Jacobi,
                        MATRIX_SOLVER_RELAXATION,
                    );
                    solve_linear_system(
                        &a,
                        &b_w,
                        &mut w,
                        MATRIX_SOLVER_ITERS,
                        SolutionMethod::Jacobi,
                        MATRIX_SOLVER_RELAXATION,
                    );
                }
                let pressure_correction_matrices = build_pressure_correction_matrices(
                    mesh,
                    &u,
                    &v,
                    &w,
                    &p,
                    &a,
                    velocity_interpolation_scheme,
                    rho,
                );
                if log_enabled!(log::Level::Debug) {
                    println!("\nPressure:");
                    print_linear_system(
                        &pressure_correction_matrices.a,
                        &pressure_correction_matrices.b,
                    );
                }

                // I think what's happening here is that, if Pe < 1 in all cells, a_nb will all
                // have the same sign in each row of the pressure correction system. This means
                // that sum(abs(a_nb)) == a_p for each row, whereas the boundedness criterion
                // requires that sum(abs(a_nb)) < a_p for at least one row.
                // What's the solution? Skip pressure correction if max Pe < 1?
                // Reducing the iteration count (10k -> 10) seems to have fixed the issue for now.
                solve_linear_system(
                    &pressure_correction_matrices.a,
                    &pressure_correction_matrices.b,
                    &mut p_prime,
                    MATRIX_SOLVER_ITERS,
                    SolutionMethod::Jacobi,
                    MATRIX_SOLVER_RELAXATION,
                );

                if log_enabled!(log::Level::Info) {
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
                apply_pressure_correction(
                    mesh,
                    &a,
                    &p_prime,
                    &mut u,
                    &mut v,
                    &mut w,
                    &mut p,
                    pressure_relaxation_factor,
                    momentum_relaxation_factor,
                );

                println!(
                    "Iteration {}: avg velocity = ({:.2e}, {:.2e}, {:.2e})",
                    iter_number,
                    u.iter().sum::<Float>() / (cell_count as Float),
                    v.iter().sum::<Float>() / (cell_count as Float),
                    w.iter().sum::<Float>() / (cell_count as Float),
                );
            }
        } // _ => panic!("unsupported pressure-velocity coupling"),
    }
    (u, v, w, p)
}

fn initialize_pressure_field(mesh: &mut Mesh, p: &mut DVector<Float>, iteration_count: Uint) {
    // TODO
    // Solve laplace's equation (nabla^2 psi = 0) based on BCs:
    // - Wall: d/dn (psi) = 0
    // - Inlet: d/dn (psi) = V
    // - Outlet: psi = 0
    for _ in 0..iteration_count {
        for cell_index in 0..mesh.cells.len() {
            let mut p_i = 0.;
            let cell = &mesh.cells[&cell_index];
            for face_index in &cell.face_indices {
                // TODO: rewrite face_zone nonsense
                let face_zone = &mesh.faces[&face_index].zone;
                p_i += match &mesh.face_zones[face_zone].zone_type {
                    FaceConditionTypes::Wall | FaceConditionTypes::Symmetry => p[cell_index],
                    FaceConditionTypes::PressureInlet | FaceConditionTypes::PressureOutlet => {
                        mesh.face_zones[face_zone].scalar_value
                    }
                    FaceConditionTypes::Interior => {
                        mesh.faces[&face_index]
                            .cell_indices
                            .iter()
                            .map(|neighbor_cell_index| p[*neighbor_cell_index])
                            .sum::<Float>()
                            / 2.
                    }
                    _ => panic!("unsupported face zone type for initialization"),
                }
            }
            let cell = mesh.cells.get_mut(&cell_index).unwrap();
            p[cell_index] = p_i / (cell.face_indices.len() as Float);
        }
    }
    // print!("\n\n");
    // for cell_number in 1..=mesh.cells.len() {
    //     let cell = &mesh.cells[&cell_number];
    //     println!("{}, {}", cell.centroid, cell.pressure);
    // }
    // print!("\n\n");
}

fn get_velocity_source_term(_location: Vector3) -> Vector3 {
    Vector3::zero()
}

fn calculate_face_velocity(
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
    match face_zone.zone_type {
        FaceConditionTypes::Wall => Vector3::zero(),
        FaceConditionTypes::Symmetry => {
            // du/dn = 0, so we need the projection of cell center velocity onto the face's plane
            let cell_velocity = Vector3 {
                x: u[face.cell_indices[0]],
                y: v[face.cell_indices[0]],
                z: w[face.cell_indices[0]],
            };
            cell_velocity - cell_velocity.dot(&face.normal)
        }
        FaceConditionTypes::VelocityInlet => face_zone.vector_value,
        FaceConditionTypes::PressureInlet | FaceConditionTypes::PressureOutlet => Vector3 {
            x: u[face.cell_indices[0]],
            y: v[face.cell_indices[0]],
            z: w[face.cell_indices[0]],
        },
        FaceConditionTypes::Interior => match interpolation_scheme {
            VelocityInterpolation::Linear => {
                let c0 = face.cell_indices[0];
                let c1 = face.cell_indices[1];
                let x0 = (mesh.cells[&c0].centroid - face.centroid).norm();
                let x1 = (mesh.cells[&c1].centroid - face.centroid).norm();
                Vector3 {
                    x: &u[c0] + (&u[c1] - &u[c0]) * x0 / (x0 + x1),
                    y: &v[c0] + (&v[c1] - &v[c0]) * x0 / (x0 + x1),
                    z: &w[c0] + (&w[c1] - &w[c0]) * x0 / (x0 + x1),
                }
            }
            VelocityInterpolation::RhieChow2 => {
                // see Versteeg & Malalasekara p340-341
                panic!("unsupported");
            }
        },
        _ => panic!("unsupported face zone type"),
    }
}

fn calculate_face_pressure(
    mesh: &Mesh,
    p: &DVector<Float>,
    face_index: usize,
    interpolation_scheme: PressureInterpolation,
) -> Float {
    // ****** TODO: Add skewness corrections!!! ********
    let face = &mesh.faces[&face_index];
    let face_zone = &mesh.face_zones[&face.zone];
    match face_zone.zone_type {
        FaceConditionTypes::Symmetry | FaceConditionTypes::Wall => {
            // du/dn = 0, so we need the projection of cell center velocity onto the face's plane
            // TODO: Off by one here?
            p[face.cell_indices[0]]
        }
        FaceConditionTypes::VelocityInlet => p[face.cell_indices[0]],
        FaceConditionTypes::PressureInlet | FaceConditionTypes::PressureOutlet => {
            face_zone.scalar_value
        }
        FaceConditionTypes::Interior => match interpolation_scheme {
            PressureInterpolation::Linear => {
                let c0 = face.cell_indices[0];
                let c1 = face.cell_indices[1];
                let x0 = (mesh.cells[&c0].centroid - face.centroid).norm();
                let x1 = (mesh.cells[&c1].centroid - face.centroid).norm();
                &p[c0] + (&p[c1] - &p[c0]) * x0 / (x0 + x1)
            }
            _ => panic!("unsupported pressure interpolation"),
        },
        _ => panic!("unsupported face zone type"),
    }
}

pub fn solve_linear_system(
    a: &CsrMatrix<Float>,
    b: &DVector<Float>,
    solution_vector: &mut DVector<Float>,
    iteration_count: Uint,
    method: SolutionMethod,
    relaxation_factor: Float,
) {
    match method {
        SolutionMethod::Jacobi => {
            let a_new = a - a.diagonal_as_csr();
            for iter_num in 0..iteration_count {
                if log_enabled!(log::Level::Trace) {
                    println!(
                        "[{:?}] Jacobi iteration {iter_num} = {}",
                        std::thread::current().id(),
                        dvector_to_str(&solution_vector)
                    );
                }
                let prev_guess = solution_vector.clone();
                *solution_vector = relaxation_factor * (b - &a_new * &prev_guess);
                // idk a better way to do this
                for i in 0..solution_vector.nrows() {
                    solution_vector[i] /= a.get_entry(i as usize, i as usize).unwrap().into_value();
                }
                *solution_vector += prev_guess * (1. - relaxation_factor);
            }
        }
        SolutionMethod::GaussSeidel => {
            'iter_loop: for iter_num in 0..iteration_count {
                if log_enabled!(log::Level::Trace) {
                    println!("Gauss-Seidel iteration {iter_num} = {solution_vector:?}");
                }
                'row_loop: for i in 0..a.nrows() {
                    solution_vector[i] = solution_vector[i] * (1. - relaxation_factor)
                        + relaxation_factor
                            * (b[i]
                                - solution_vector
                                    .iter() // par_iter is slower here with 1k cells; might be worth it with more cells
                                    .enumerate()
                                    .map(|(j, x)| {
                                        if i != j {
                                            a.get_entry(i, j).unwrap().into_value() * x
                                        } else {
                                            0.
                                        }
                                    })
                                    .sum::<Float>())
                            / a.get_entry(i, i).unwrap().into_value();
                    if solution_vector[i].is_nan() {
                        panic!("****** Solution diverged ******");
                    }
                }
            }
        }
        _ => panic!("unsupported solution method"),
    }
}

fn build_momentum_diffusion_matrix(
    mesh: &Mesh,
    u: &DVector<Float>,
    v: &DVector<Float>,
    w: &DVector<Float>,
    p: &DVector<Float>,
    pressure_interpolation_scheme: PressureInterpolation,
    velocity_interpolation_scheme: VelocityInterpolation,
    rho: Float,
    mu: Float,
) -> CsrMatrix<Float> {
    // TODO: split calculation of D_i into different function so it only needs to be done once
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
                    panic!("untested");
                    // Diffusion coefficient
                    // TODO: Add source term contribution to diffusion since there's no cell on the other side
                    let d_i: Float = mu * face.area / (face.centroid - cell.centroid).norm();
                    (d_i, usize::MAX)
                }
                FaceConditionTypes::Interior => {
                    let neighbor_cell_index: usize;
                    if face.cell_indices[0] == *cell_index {
                        // face normal points to cell 0 by default, so we need to flip it
                        neighbor_cell_index = *face
                            .cell_indices
                            .get(1)
                            .expect("interior faces should have two neighbors");
                    } else {
                        // this cell must be cell 1,
                        neighbor_cell_index = face.cell_indices[0];
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
            if log_enabled!(log::Level::Trace) {
                println!(
                    "cell {: >3}, face {: >3}: D_i = {: >9}",
                    cell_index, face_index, d_i
                );
            }

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

fn build_momentum_matrices(
    mesh: &Mesh,
    u: &DVector<Float>,
    v: &DVector<Float>,
    w: &DVector<Float>,
    p: &DVector<Float>,
    momentum_scheme: MomentumDiscretization,
    pressure_interpolation_scheme: PressureInterpolation,
    velocity_interpolation_scheme: VelocityInterpolation,
    rho: Float,
    mu: Float,
) -> (
    CsrMatrix<Float>,
    DVector<Float>,
    DVector<Float>,
    DVector<Float>,
) {
    // TODO: split calculation of D_i into different function so it only needs to be done once
    let cell_count = mesh.cells.len();
    let mut a = CooMatrix::<Float>::new(cell_count, cell_count);
    let mut u_source = initialize_DVector!(cell_count);
    let mut v_source = initialize_DVector!(cell_count);
    let mut w_source = initialize_DVector!(cell_count);

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
        let mut s_p = 0.; // proportional source term TODO (don't think I need this since I include
                          // BCs in a_p directly)
        let mut s_u = get_velocity_source_term(cell.centroid); // general source term
        let mut s_u_dc = Vector3::zero(); // deferred correction source term TODO
        let mut s_d_cross = Vector3::zero(); // cross diffusion source term TODO

        // The current cell's coefficients (matrix diagonal)
        let mut a_p = 0.;

        // Iterate over this cell's faces
        for face_index in &cell.face_indices {
            let face = &mesh.faces[face_index];
            let face_bc = &mesh.face_zones[&face.zone];
            let face_velocity = calculate_face_velocity(
                &mesh,
                &u,
                &v,
                &w,
                *face_index,
                velocity_interpolation_scheme,
            );
            // println!("cell {cell_number}, face {face_number}: velocity = {face_velocity:?}");
            // TODO: Consider flipping convention of face normal direction and/or potentially
            // make it an area vector
            let outward_face_normal = get_outward_face_normal(&face, *cell_index);
            // Mass flow rate out of this cell through this face
            let f_i = face_velocity.dot(&outward_face_normal) * face.area * rho;
            let face_pressure =
                calculate_face_pressure(mesh, &p, *face_index, pressure_interpolation_scheme);
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

            if log_enabled!(log::Level::Trace) {
                println!(
                    "cell {: >3}, face {: >3}: F_i = {: >9} {}",
                    cell_index,
                    face_index,
                    f_i,
                    if neighbor_cell_index == usize::MAX {
                        "(boundary)"
                    } else {
                        ""
                    }
                );
            }

            let a_nb: Float = match momentum_scheme {
                MomentumDiscretization::UD => {
                    // Neighbor only affects this cell if flux is into this
                    // cell => f_i < 0. Therefore, if f_i > 0, we set it to 0.
                    Float::min(f_i, 0.)
                }
                MomentumDiscretization::CD => f_i / 2.,
                _ => panic!("unsupported momentum scheme"),
            };

            let face_contribution = -a_nb + f_i + s_p; // sign of s_p?
            a_p += face_contribution;
            s_u += (-outward_face_normal) * face_pressure * face.area;

            // If it's MAX, that means it's a boundary face
            if neighbor_cell_index != usize::MAX {
                // negate a_nb to move to LHS of equation
                a.push(*cell_index, neighbor_cell_index, a_nb);
            }
        } // end face loop
        let source_total = s_u + s_u_dc + s_d_cross;
        u_source[*cell_index] = source_total.x;
        v_source[*cell_index] = source_total.y;
        w_source[*cell_index] = source_total.z;

        a.push(*cell_index, *cell_index, a_p + s_p);
    } // end cell loop
      // NOTE: I *think* all diagonal terms in `a` should be positive and all off-diagonal terms
      // negative. It may be worth adding assertions to validate this.
    (CsrMatrix::from(&a), u_source, v_source, w_source)
}

fn build_pressure_correction_matrices(
    mesh: &Mesh,
    u: &DVector<Float>,
    v: &DVector<Float>,
    w: &DVector<Float>,
    p: &DVector<Float>,
    momentum_matrices: &CsrMatrix<Float>,
    velocity_interpolation_scheme: VelocityInterpolation,
    rho: Float,
) -> LinearSystem {
    // TODO: ignore boundary cells
    let cell_count = mesh.cells.len();
    // The coefficients of the pressure correction matrix
    let mut a = CooMatrix::new(cell_count, cell_count);
    // This is the net mass flow rate into each cell
    let mut b = initialize_DVector!(cell_count);

    for (cell_index, cell) in &mesh.cells {
        let mut a_p: Float = 0.;
        let mut b_p: Float = 0.;
        for face_index in &cell.face_indices {
            let face = &mesh.faces[face_index];
            let face_velocity = calculate_face_velocity(
                mesh,
                &u,
                &v,
                &w,
                *face_index,
                velocity_interpolation_scheme,
            );
            let inward_face_normal = get_inward_face_normal(&face, *cell_index);
            // The net mass flow rate through this face into this cell
            b_p += rho * face_velocity.dot(&inward_face_normal) * face.area;

            let a_ii = momentum_matrices
                .get_entry(*cell_index, *cell_index)
                .unwrap()
                .into_value();

            if face.cell_indices.len() > 1 {
                let neighbor_cell_index = if face.cell_indices[0] != *cell_index {
                    face.cell_indices[0]
                } else {
                    face.cell_indices[1]
                };
                // NOTE: It would be more rigorous to recalculate the advective coefficients,
                // but I think this should be sufficient for now.
                let a_ij = momentum_matrices
                    .get_entry(*cell_index, neighbor_cell_index)
                    .unwrap()
                    .into_value();

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
    pressure_relaxation_factor: Float,
    momentum_relaxation_factor: Float,
) {
    for (cell_index, cell) in &mut mesh.cells {
        p[*cell_index] = &p[*cell_index]
            + pressure_relaxation_factor * (*p_prime.get(*cell_index).unwrap_or(&0.));
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
            &u[*cell_index] * (1. - momentum_relaxation_factor) + velocity_correction.x;
        v[*cell_index] =
            &v[*cell_index] * (1. - momentum_relaxation_factor) + velocity_correction.y;
        w[*cell_index] =
            &w[*cell_index] * (1. - momentum_relaxation_factor) + velocity_correction.z;
    }
}

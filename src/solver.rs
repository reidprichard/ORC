use crate::common::*;
use crate::io::print_linear_system;
use crate::mesh::*;
use itertools::izip;
use sprs::{CsMat, CsVec, TriMat};
// use std::collections::HashMap;

#[derive(Copy, Clone)]
pub enum SolutionMethod {
    GaussSeidel,
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
    a: CsMat<Float>,
    b: Vec<Float>,
}

pub struct MomentumMatrices {
    u: LinearSystem,
    v: LinearSystem,
    w: LinearSystem,
}

pub struct MomentumSolutionVectors {
    u: Vec<Float>,
    v: Vec<Float>,
    w: Vec<Float>,
}

fn get_velocity_source_term(location: Vector) -> Vector {
    Vector::zero()
}

fn interpolate_face_velocity(
    mesh: &Mesh,
    face_number: Uint,
    interpolation_scheme: VelocityInterpolation,
) -> Vector {
    // ****** TODO: Add skewness corrections!!! ********
    let face = &mesh.faces[&face_number];
    match interpolation_scheme {
        VelocityInterpolation::Linear => {
            let mut divisor: Float = 0.;
            face.cell_numbers
                .iter()
                .map(|c| {
                    (
                        &mesh.cells[c].velocity,
                        (face.centroid - mesh.cells[c].centroid).norm(),
                    )
                })
                .fold(Vector::zero(), |acc, (v, x)| {
                    divisor += x;
                    acc + (*v) * x
                })
                / divisor
        }
        VelocityInterpolation::RhieChow2 => {
            // see Versteeg & Malalasekara p340-341
            panic!("unsupported");
        }
    }
}

fn interpolate_face_pressure(
    mesh: &Mesh,
    face_number: Uint,
    interpolation_scheme: PressureInterpolation,
) -> Float {
    let face = &mesh.faces[&face_number];
    match interpolation_scheme {
        PressureInterpolation::Linear => {
            let mut divisor: Float = 0.;
            face.cell_numbers
                .iter()
                .map(|c| {
                    (
                        mesh.cells[c].pressure,
                        (face.centroid - mesh.cells[c].centroid).norm(),
                    )
                })
                .fold(0., |acc, (p, x)| {
                    divisor += x;
                    acc + p / x
                })
                / divisor
        }
        _ => panic!("not supported"),
    }
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
) {
    initialize_pressure_field(mesh);
    match pressure_velocity_coupling {
        PressureVelocityCoupling::SIMPLE => {
            for _ in 0..iteration_count {
                let mut u: Vec<Float> = mesh.cells.iter().map(|(i, c)| c.velocity.x).collect();
                let mut v: Vec<Float> = mesh.cells.iter().map(|(i, c)| c.velocity.y).collect();
                let mut w: Vec<Float> = mesh.cells.iter().map(|(i, c)| c.velocity.z).collect();
                let mut p: Vec<Float> = mesh.cells.iter().map(|(i, c)| c.pressure).collect();
                let (a, b_u, b_v, b_w) = build_discretized_momentum_matrices(
                    mesh,
                    momentum_scheme,
                    pressure_interpolation_scheme,
                    velocity_interpolation_scheme,
                    rho,
                    mu,
                );
                print_linear_system(&a, &b_u);
                print!("\n\n");
                solve_linear_system(&a, &b_u, &mut u, 20, SolutionMethod::GaussSeidel);
                solve_linear_system(&a, &b_v, &mut v, 20, SolutionMethod::GaussSeidel);
                solve_linear_system(&a, &b_w, &mut w, 20, SolutionMethod::GaussSeidel);

                for (i, (u_i, v_i, w_i)) in izip!(u, v, w).enumerate() {
                    mesh.cells
                        .get_mut(&(i + 1).try_into().unwrap())
                        .unwrap()
                        .velocity = Vector {
                        x: u_i,
                        y: v_i,
                        z: w_i,
                    };
                }

                let pressure_correction_matrices = build_pressure_correction_matrices(
                    mesh,
                    &a,
                    velocity_interpolation_scheme,
                    rho,
                );
                let mut p_prime: Vec<Float> = vec![0.; mesh.cells.len()];
                solve_linear_system(
                    &pressure_correction_matrices.a,
                    &pressure_correction_matrices.b,
                    &mut p_prime,
                    20,
                    SolutionMethod::GaussSeidel,
                );
            }
        }
        _ => panic!("unsupported pressure-velocity coupling"),
    }
}

pub fn solve_linear_system(
    a: &CsMat<Float>,
    b: &Vec<Float>,
    solution_vector: &mut Vec<Float>,
    iteration_count: Uint,
    method: SolutionMethod,
) {
    // TODO: implement
    match method {
        SolutionMethod::GaussSeidel => {
            'iter_loop: for _ in 0..iteration_count {
                'row_loop: for i in 0..solution_vector.len() {
                    solution_vector[i] = (
                        b[i]
                        - solution_vector
                            .iter()
                            .enumerate()
                            .fold(0., |acc, (j, x)| {
                                acc + a.get(i, j).unwrap_or(&0.) * x * Float::from(i != j)
                            })
                    ) / a.get(i, i)
                        .expect("matrix A should have a (nonzero) diagonal element for each element of solution vector")
                }
            }
        }
        _ => panic!("unsupported solution method"),
    }
}

pub fn build_pressure_correction_matrices(
    mesh: &Mesh,
    momentum_matrices: &CsMat<Float>,
    velocity_interpolation_scheme: VelocityInterpolation,
    rho: Float,
) -> LinearSystem {
    // TODO: ignore boundary cells
    let cell_count = mesh.cells.len();
    let mut a = TriMat::new((cell_count, cell_count));
    let mut b: Vec<Float> = vec![0.; cell_count];

    for (cell_number, cell) in &mesh.cells {
        let mut a_p: Float = 0.;
        let mut b_p: Float = 0.;
        for face_number in &cell.face_numbers {
            let face = &mesh.faces[face_number];
            if face.cell_numbers.len() == 1 {
                continue;
            }

            let face_velocity =
                interpolate_face_velocity(mesh, *face_number, velocity_interpolation_scheme);
            let outward_face_normal = mesh.get_outward_face_normal(*face_number, *cell_number);
            b_p += -rho * face_velocity.dot(&outward_face_normal) * face.area;

            let neighbor_cell_number: Uint = if face.cell_numbers[0] != *cell_number {
                face.cell_numbers[0]
            } else {
                face.cell_numbers[1]
            };

            let a_nb = rho * Float::powi(face.area, 2)
                / momentum_matrices
                    .get(
                        (*cell_number - 1) as usize,
                        (neighbor_cell_number - 1) as usize,
                    )
                    .unwrap();

            a.add_triplet(
                (cell_number - 1) as usize,
                (neighbor_cell_number - 1) as usize,
                -a_nb,
            );
            a_p += a_nb;
        }
        a.add_triplet((cell_number - 1) as usize, (cell_number - 1) as usize, a_p);
        b.push(b_p);
    }

    LinearSystem { a: a.to_csr(), b }
}

pub fn build_discretized_momentum_matrices(
    mesh: &Mesh,
    momentum_scheme: MomentumDiscretization,
    pressure_interpolation_scheme: PressureInterpolation,
    velocity_interpolation_scheme: VelocityInterpolation,
    rho: Float,
    mu: Float,
) -> (CsMat<Float>, Vec<Float>, Vec<Float>, Vec<Float>) {
    // TODO: Ignore boundary cells
    let cell_count = mesh.cells.len();
    let mut a = TriMat::new((cell_count, cell_count));
    let mut u_source: Vec<Float> = vec![0.; cell_count];
    let mut v_source: Vec<Float> = vec![0.; cell_count];
    let mut w_source: Vec<Float> = vec![0.; cell_count];

    match momentum_scheme {
        MomentumDiscretization::UD => {}
        _ => panic!("Invalid momentum scheme."),
    }

    // Iterate over all cells in the mesh
    for (cell_number, cell) in &mesh.cells {
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
        let mut s_p = 0.; // proportional source term TODO
        let mut s_u = get_velocity_source_term(cell.centroid); // general source term
        let mut s_u_dc = Vector::zero(); // deferred correction source term TODO
        let mut s_d_cross = Vector::zero(); // cross diffusion source term TODO

        // The current cell's coefficients (matrix diagonal)
        let mut a_p = s_p;

        // Iterate over this cell's faces
        for face_number in &cell.face_numbers {
            let face = &mesh.faces[face_number];
            let face_bc = &mesh.face_zones[&face.zone];
            let (f_i, d_i, source_term, neighbor_cell_number) = match face_bc.zone_type {
                FaceConditionTypes::Wall => {
                    // Do I need to do anything here?
                    // The advective and diffusive fluxes through this face are zero, and there's
                    // no source here (right?), so I think not.
                    // The normal points to cell 0 (this cell), which is the direction the pressure
                    // force acts
                    // NOTE: Assumes zero wall-normal pressure gradient
                    (0., 0., face.normal * face.area * cell.pressure, 0)
                }
                FaceConditionTypes::VelocityInlet => {
                    // By default, face normals point to cell 0.
                    // If a face only has one neighbor, that neighbor will be cell 0.
                    // Therefore, if we reverse this normal, it will be facing outward.
                    // TODO: Consider flipping convention of face normal direction
                    // Advection (flux) coefficient
                    let f_i: Float = -face_bc.vector_value.dot(&face.normal) * face.area * rho;
                    // Diffusion coefficient
                    let d_i: Float = mu * face.area;

                    // Again, face normal points toward this cell which is what we want
                    // NOTE: Assumes zero streamwise pressure gradient
                    (f_i, d_i, face.normal * face.area * cell.pressure, 0)
                }
                FaceConditionTypes::PressureInlet | FaceConditionTypes::PressureOutlet => {
                    (0., 0., face.normal * face.area * face_bc.scalar_value, 0)
                }
                FaceConditionTypes::Interior => {
                    let face_velocity = interpolate_face_velocity(
                        &mesh,
                        *face_number,
                        velocity_interpolation_scheme,
                    );

                    let mut neighbor_cell_number: Uint = 0;
                    let outward_face_normal: Vector =
                        mesh.get_outward_face_normal(*face_number, *cell_number);
                    if face.cell_numbers[0] == *cell_number {
                        // face normal points to cell 0 by default, so we need to flip it
                        neighbor_cell_number = *face
                            .cell_numbers
                            .get(1)
                            .expect("interior faces should have two neighbors");
                    } else {
                        // this cell must be cell 1,
                        neighbor_cell_number = face.cell_numbers[0];
                    }

                    // TODO: Consider changing face unit normal to face area vector
                    // Potentially inward area vector?
                    // Advection (flux) coefficient = advective mass flow rate into this face
                    let f_i: Float = -face_velocity.dot(&outward_face_normal) * face.area * rho;
                    // Cell centroids vector
                    let e_xi: Vector = mesh.cells[&neighbor_cell_number].centroid - cell.centroid;
                    // Diffusion coefficient
                    let d_i: Float = mu * face.area * e_xi.norm();

                    // Skipping cross diffusion for now
                    // let d_i: Float = mu
                    //     * (outward_face_normal.dot(&outward_face_normal)
                    //         / outward_face_normal.dot(&e_xi))
                    //     * face.area;
                    // Face tangent vector -- this feels inefficient
                    // let e_nu: Vector = outward_face_normal.cross(&e_xi.cross(&outward_face_normal)).unit();
                    // Cross diffusion source term
                    // let s_cross_diffusion = -mu *
                    let face_pressure = interpolate_face_pressure(
                        &mesh,
                        *face_number,
                        pressure_interpolation_scheme,
                    );
                    (
                        f_i,
                        d_i,
                        -outward_face_normal * face_pressure,
                        neighbor_cell_number,
                    )
                }
                _ => {
                    println!("*** {} ***", face_bc.zone_type);
                    panic!("BC not supported");
                }
            }; // end BC match
            let a_nb: Float = (d_i
                + match momentum_scheme {
                    MomentumDiscretization::UD => {
                        // Neighbor only affects this cell if flux is into this
                        // cell => f_i < 0. Therefore, if f_i > 0, we set it to 0.
                        Float::min(f_i, 0.)
                    }
                    MomentumDiscretization::CD => f_i / 2.,
                    _ => panic!("unsupported momentum scheme"),
                });
            a_p += a_nb - f_i + s_p; // sign of f_i?
            s_u += source_term;

            // If it's zero, that means it's a boundary face
            if neighbor_cell_number > 0 {
                a.add_triplet(
                    (*cell_number - 1) as usize,
                    (neighbor_cell_number - 1) as usize,
                    -a_nb,
                );
            }
        } // end face loop
        let source_total = s_u + s_u_dc + s_d_cross;
        u_source[(cell_number - 1) as usize] = source_total.x;
        v_source[(cell_number - 1) as usize] = source_total.y;
        w_source[(cell_number - 1) as usize] = source_total.z;

        a.add_triplet(
            (*cell_number - 1).try_into().unwrap(),
            (*cell_number - 1).try_into().unwrap(),
            a_p + s_p,
        );
    } // end cell loop
    (a.to_csr(), u_source, v_source, w_source)
}

fn initialize_pressure_field(mesh: &mut Mesh) {
    // TODO
    // Solve laplace's equation (nabla^2 psi = 0) based on BCs:
    // - Wall: d/dn (psi) = 0
    // - Inlet: d/dn (psi) = V
    // - Outlet: psi = 0
}

fn iterate_steady(iteration_count: u32) {
    // 1. Guess the
}

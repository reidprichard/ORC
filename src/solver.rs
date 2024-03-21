use crate::common::*;
use crate::mesh::*;
use sprs::{CsMat, CsVec, TriMat};
use std::collections::HashMap;

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

pub struct SolutionMatrices {
    u: LinearSystem,
    v: LinearSystem,
    w: LinearSystem,
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

pub fn build_solution_matrices(
    mesh: &mut Mesh,
    momentum_scheme: MomentumDiscretization,
    pressure_interpolation_scheme: PressureInterpolation,
    velocity_interpolation_scheme: VelocityInterpolation,
    rho: Float,
    mu: Float,
) -> SolutionMatrices {
    use std::collections::HashSet;

    let cell_count = mesh.cells.len();
    let face_count = mesh.faces.len();
    let mut u_matrix = TriMat::new((cell_count, cell_count));
    let mut v_matrix = TriMat::new((cell_count, cell_count));
    let mut w_matrix = TriMat::new((cell_count, cell_count));
    let mut u_source: Vec<Float> = vec![0.; cell_count];
    let mut v_source: Vec<Float> = vec![0.; cell_count];
    let mut w_source: Vec<Float> = vec![0.; cell_count];

    match momentum_scheme {
        MomentumDiscretization::UD => {}
        _ => panic!("Invalid momentum scheme."),
    }

    // Iterate over all cells in the mesh
    for (cell_number, cell) in &mesh.cells {
        let this_cell_velocity_gradient = &mesh.calculate_velocity_gradient(*cell_number);
        // TODO: Implement S_p
        let s_p = Vector::zero(); // proportional source term
        let mut s_u = get_velocity_source_term(cell.centroid); // general source term
        let s_u_dc = Vector::zero(); // deferred correction source term
                                     // TODO: Implement cross-diffusion
        let s_d_cross = Vector::zero(); // cross diffusion source term

        // The current cell's coefficients (matrix diagonal)
        let mut a_p = Vector::ones() * s_p;

        // Iterate over this cell's faces
        for face_number in &cell.face_numbers {
            let face = &mesh.faces[face_number];
            match face.cell_numbers.len() {
                1 => {
                    let face_bc = &mesh.face_zones[&face.zone];
                    match face_bc.zone_type {
                        BoundaryConditionTypes::Interior => panic!("Interior one-sided face."),
                        BoundaryConditionTypes::Wall => (), // already zero
                        BoundaryConditionTypes::VelocityInlet => {
                            // face_velocity = face.normal * face_bc.value;
                        }
                        BoundaryConditionTypes::PressureInlet => {
                            // TODO: Write something proper!!
                            // face_velocity = cell.velocity;
                        }
                        BoundaryConditionTypes::PressureOutlet => {
                            // TODO: Write something proper!!
                            // face_velocity = cell.velocity;
                        }
                        _ => {
                            println!("*** {} ***", face_bc.zone_type);
                            panic!("BC not supported");
                        }
                    }
                    // Maybe a face's boundary status should be encoded in a bool?
                }
                2 => {
                    let face_velocity = interpolate_face_velocity(
                        &mesh,
                        *face_number,
                        velocity_interpolation_scheme,
                    );

                    let mut neighbor_cell_number: Uint = 0;
                    let mut outward_face_normal: Vector = face.normal;
                    if face.cell_numbers[0] == *cell_number {
                        // face normal points to cell 0 by default
                        neighbor_cell_number = face.cell_numbers[1];
                    } else {
                        outward_face_normal *= -1;
                        neighbor_cell_number = face.cell_numbers[0];
                    }

                    // TODO: set coefficient convention to be out = positive
                    // Advection (flux) coefficient
                    let f_i: Float = face_velocity.dot(&outward_face_normal) * face.area * rho;
                    // Cell centroids vector
                    let e_xi: Vector = mesh.cells[&neighbor_cell_number].centroid - cell.centroid;
                    // Diffusion coefficient
                    let d_i: Float = mu
                        * (outward_face_normal.dot(&outward_face_normal)
                            / outward_face_normal.dot(&e_xi))
                        * face.area;

                    // Skipping cross diffusion for now
                    // Face tangent vector -- this feels inefficient
                    // let e_nu: Vector = outward_face_normal.cross(&e_xi.cross(&outward_face_normal)).unit();
                    // Cross diffusion source term
                    // let s_cross_diffusion = -mu *

                    let a_nb: Vector = Vector::ones()
                        * (d_i
                            + match momentum_scheme {
                                MomentumDiscretization::UD => {
                                    if f_i > 0. {
                                        f_i
                                    } else {
                                        0.
                                    }
                                }
                                MomentumDiscretization::CD => f_i / 2.,
                                _ => panic!("unsupported momentum scheme"),
                            });
                    a_p = a_p - a_nb - f_i + s_p;

                    // TODO: DRY
                    u_matrix.add_triplet(
                        (*cell_number - 1).try_into().unwrap(),
                        (neighbor_cell_number - 1).try_into().unwrap(),
                        a_nb.x,
                    );
                    v_matrix.add_triplet(
                        (*cell_number - 1).try_into().unwrap(),
                        (neighbor_cell_number - 1).try_into().unwrap(),
                        a_nb.y,
                    );
                    w_matrix.add_triplet(
                        (*cell_number - 1).try_into().unwrap(),
                        (neighbor_cell_number - 1).try_into().unwrap(),
                        a_nb.z,
                    );

                    // let face_pressure:Float = interpolate_face_velocity
                }
                _ => panic!("faces must have 1 or 2 neighbors"),
            }
            u_source.push(s_u.x + s_u_dc.x + s_d_cross.x);
            v_source.push(s_u.y + s_u_dc.y + s_d_cross.y);
            w_source.push(s_u.z + s_u_dc.z + s_d_cross.z);

            u_matrix.add_triplet(
                (*cell_number - 1).try_into().unwrap(),
                (*cell_number - 1).try_into().unwrap(),
                a_p.x + s_p.x,
            );
            v_matrix.add_triplet(
                (*cell_number - 1).try_into().unwrap(),
                (*cell_number - 1).try_into().unwrap(),
                a_p.y + s_p.y,
            );
            w_matrix.add_triplet(
                (*cell_number - 1).try_into().unwrap(),
                (*cell_number - 1).try_into().unwrap(),
                a_p.z + s_p.z,
            );
        }
    }

    SolutionMatrices {
        u: LinearSystem {
            a: u_matrix.to_csr(),
            b: u_source,
        },
        v: LinearSystem {
            a: v_matrix.to_csr(),
            b: v_source,
        },
        w: LinearSystem {
            a: w_matrix.to_csr(),
            b: w_source,
        },
    }
}

fn initialize_flow(mesh: Mesh) {
    // Solve laplace's equation (nabla^2 psi = 0) based on BCs:
    // - Wall: d/dn (psi) = 0
    // - Inlet: d/dn (psi) = V
    // - Outlet: psi = 0
}

fn iterate_steady(iteration_count: u32) {
    // 1. Guess the
}

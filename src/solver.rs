use crate::common::*;
use crate::mesh::*;
use sprs::{CsMat, CsVec, TriMat};
use std::collections::HashMap;

pub enum PressureVelocityCoupling {
    SIMPLE,
}

pub enum MomentumDiscretization {
    UD,
    CD,
}

pub enum PressureInterpolation {
    Linear,
    Standard,
    SecondOrder,
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

fn interpolate_face_velocity(mesh: &Mesh, face_number: &Uint) -> Vector {
    // ****** TODO: Add skewness corrections!!! ********
    let face = &mesh.faces[face_number];
    face.cell_indices
        .iter()
        .map(|c| &mesh.cells[c].velocity)
        .fold(Vector::zero(), |acc, v| acc + *v)
        / (face.cell_indices.len() as Uint)
}

pub fn build_solution_matrices(
    mesh: &mut Mesh,
    momentum_scheme: MomentumDiscretization,
    pressure_scheme: PressureInterpolation,
    rho: Float,
    nu: Float,
) -> SolutionMatrices {
    use std::collections::HashSet;

    let cell_count = mesh.cells.len();
    let face_count = mesh.faces.len();
    let mut u_source: Vec<Float> = vec![0.; cell_count];
    let mut u_matrix = TriMat::new((cell_count, cell_count));
    let mut v_matrix = TriMat::new((cell_count, cell_count));
    let mut w_matrix = TriMat::new((cell_count, cell_count));
    let mut v_source: Vec<Float> = vec![0.; cell_count];
    let mut w_source: Vec<Float> = vec![0.; cell_count];

    match momentum_scheme {
        MomentumDiscretization::UD => {}
        _ => panic!("Invalid momentum scheme."),
    }

    let mut warned: bool = false;

    for (cell_number, cell) in &mesh.cells {
        // &cell.face_indices.iter().map(|f| mesh.faces.get_mut(f).expect("face exists").velocity = Vector::zero());

        // Transient term

        // TODO: Rewrite this as a 3x3 tensor
        let mut grad_u = Vector::zero();
        let mut grad_v = Vector::zero();
        let mut grad_w = Vector::zero();
        let mut interior_face_count: Uint = 0;

        for face_number in &cell.face_indices {
            let mut face_velocity: Vector = Vector::zero();
            let face = &mesh.faces[face_number];
            if face.cell_indices.len() < 2 {
                let face_bc = &mesh.face_zones[&face.zone];
                match face_bc.zone_type {
                    BoundaryConditionTypes::Interior => panic!("Interior one-sided face."),
                    BoundaryConditionTypes::Wall => (), // already zero
                    BoundaryConditionTypes::VelocityInlet => {
                        face_velocity = face.normal * face_bc.value;
                    }
                    BoundaryConditionTypes::PressureInlet => {
                        // TODO: Write something proper!!
                        face_velocity = cell.velocity;
                    }
                    BoundaryConditionTypes::PressureOutlet => {
                        // TODO: Write something proper!!
                        face_velocity = cell.velocity;
                    }
                    _ => {
                        println!("*** {} ***", face_bc.zone_type);
                        panic!("BC not supported");
                    }
                }
            } else {
                face_velocity = interpolate_face_velocity(&mesh, face_number);
            }
            let face_vector = mesh.faces[face_number].centroid - cell.centroid;
            grad_u += face_vector * face_velocity.x;
            grad_v += face_vector * face_velocity.y;
            grad_w += face_vector * face_velocity.z;
        }

        // Convection term
        for face_number in &cell.face_indices {
            let face_velocity = interpolate_face_velocity(&mesh, face_number);

            let face = &mut mesh.faces.get_mut(face_number).expect("face exists");
            if face.cell_indices.len() < 2 {
                // TODO: Handle BCs
                continue;
            }
            let mut upwind_cell_number = face.cell_indices[0];
            let neighbor_cell_number = *face
                .cell_indices
                .iter()
                .filter(|c| *c != cell_number)
                .collect::<Vec<&Uint>>()
                .get(0)
                .unwrap();
            // By default, face.normal points toward cell 0
            // We need it facing outward
            let mut outward_face_normal = -face.normal;
            // If cell 1 exists and is upwind, use that
            if face.normal.dot(&cell.velocity) < 0. {
                upwind_cell_number = face.cell_indices[1];
                outward_face_normal = -outward_face_normal;
            }

            let advective_term =
                face_velocity * rho * outward_face_normal.dot(&face_velocity) * face.area;
            
            let diffusive_term = Vector {
                x: face.normal.dot(&grad_u),
                y: face.normal.dot(&grad_v),
                z: face.normal.dot(&grad_w),
            } * nu;

            u_matrix.add_triplet(
                *cell_number as usize - 1,
                *neighbor_cell_number as usize - 1,
                advective_term.x + diffusive_term.x,
            );
            v_matrix.add_triplet(
                *cell_number as usize - 1,
                *neighbor_cell_number as usize - 1,
                advective_term.y + diffusive_term.y,
            );
            w_matrix.add_triplet(
                *cell_number as usize - 1,
                *neighbor_cell_number as usize - 1,
                advective_term.z + diffusive_term.z,
            );
        }

        // Diffusion term
        grad_u /= cell.volume;
        grad_v /= cell.volume;
        grad_w /= cell.volume;

        for face_number in &cell.face_indices {}

        let mut cell_faces: Vec<&Face> = cell
            .face_indices
            .iter()
            .map(|face_number| &mesh.faces[face_number])
            .collect();
    }

    // let p_matrix: CsMat<Float> = CsMat::zero((face_count, face_count));
    // let p_source: Vec<Float> = Vec::new();

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

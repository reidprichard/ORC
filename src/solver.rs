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
    Vector::default()
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

    for (cell_index, cell) in &mesh.cells {
        // &cell.face_indices.iter().map(|f| mesh.faces.get_mut(f).expect("face exists").velocity = Vector::default());

        // Transient term

        // Convection term
        for face_index in &cell.face_indices {
            let face = &mut mesh.faces.get_mut(face_index).expect("face exists");
            if face.cell_indices.len() < 2 {
                continue; // assume boundary cells have already been treated
            }
            // ****** TODO: Add skewness corrections!!! ********
            let face_velocity = face
                .cell_indices
                .iter()
                .map(|c| &mesh.cells[c].velocity)
                .fold(Vector::default(), |acc, v| acc + *v)
                / (face.cell_indices.len() as Uint);
            let mut upwind_cell_index = face.cell_indices[0];
            let neighbor_cell_index = *face
                .cell_indices
                .iter()
                .filter(|c| *c != cell_index)
                .collect::<Vec<&Uint>>()
                .get(0)
                .unwrap();
            // By default, face.normal points toward cell 0
            // We need it facing outward
            let mut face_normal = -face.normal;
            // If cell 1 exists and is upwind, use that
            if face.normal.dot(&cell.velocity) < 0. {
                upwind_cell_index = face.cell_indices[1];
                face_normal = -face_normal;
            }

            let flux = face_velocity * rho * face_normal.dot(&face_velocity);
            u_matrix.add_triplet(
                *cell_index as usize - 1,
                *neighbor_cell_index as usize - 1,
                flux.x,
            );
            v_matrix.add_triplet(
                *cell_index as usize - 1,
                *neighbor_cell_index as usize - 1,
                flux.y,
            );
            w_matrix.add_triplet(
                *cell_index as usize - 1,
                *neighbor_cell_index as usize - 1,
                flux.z,
            );
        }

        // Diffusion term

        let mut cell_faces: Vec<&Face> = cell
            .face_indices
            .iter()
            .map(|face_index| &mesh.faces[face_index])
            .collect();

        // cell_faces.get(0).unwrap().velocity = Vector::default();
        // let neighbor_cells: Vec<&Cell> = cell_faces
        //     .iter()
        //     .map(|face| &face.cell_indices)
        //     .flatten()
        //     .filter(|c| *c != cell_index)
        //     .map(|cell| &mesh.cells[cell])
        //     .collect();

        // 1. Diffusion term

        // 2. Advection term
        // for face in cell_faces {
        //     (*face).velocity = Vector::default();//face.cell_indices.iter().map(|i| &mesh.cells[i].velocity).fold(Vector::default(), |acc, v| acc + *v) / (face.cell_indices.len() as Uint);
        // }
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

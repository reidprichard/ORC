use crate::common::*;
use crate::mesh::*;
use sprs::{CsMat, CsVec};
use std::collections::HashMap;

pub enum PressureVelocityCoupling {
    SIMPLE,
}

pub enum MomentumDiscretization {
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

fn build_solution_matrices(
    mesh: Mesh,
    momentum_scheme: MomentumDiscretization,
    pressure_scheme: PressureInterpolation,
) -> SolutionMatrices {
    use std::collections::HashSet;
    let cell_count = mesh.cells.len();
    let face_count = mesh.faces.len();
    let u_matrix: CsMat<Float> = CsMat::zero((cell_count, cell_count));
    let u_source: Vec<Float> = vec![0.; cell_count];
    let v_matrix: CsMat<Float> = CsMat::zero((cell_count, cell_count));
    let v_source: Vec<Float> = vec![0.; cell_count];
    let w_matrix: CsMat<Float> = CsMat::zero((cell_count, cell_count));
    let w_source: Vec<Float> = vec![0.; cell_count];

    match momentum_scheme {
        MomentumDiscretization::CD => {}
        _ => panic!("Invalid momentum scheme."),
    }

    for (cell_index, cell) in &mesh.cells {
        let neighbor_indices: HashSet<&Uint> = cell
            .face_indices
            .iter()
            .map(|face_index| &mesh.faces[face_index])
            .map(|face| &face.cell_indices)
            .flatten()
            .filter(|c| *c != cell_index)
            .collect();
        // 1. Diffusion term
        // 2. Advection term
    }

    // let p_matrix: CsMat<Float> = CsMat::zero((face_count, face_count));
    // let p_source: Vec<Float> = Vec::new();

    SolutionMatrices {
        u: LinearSystem {
            a: u_matrix,
            b: u_source,
        },
        v: LinearSystem {
            a: v_matrix,
            b: v_source,
        },
        w: LinearSystem {
            a: w_matrix,
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

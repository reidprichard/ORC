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

fn interpolate_face_velocity(mesh: &Mesh, face_number: Uint) -> Vector {
    // ****** TODO: Add skewness corrections!!! ********
    let face = &mesh.faces[&face_number];
    face.cell_numbers
        .iter()
        .map(|c| &mesh.cells[c].velocity)
        .fold(Vector::zero(), |acc, v| acc + *v)
        / (face.cell_numbers.len() as Uint)
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

    // This is the function psi(r) in TVD (in vector form)
    // I think it's inefficient to hide constants behind a closure,
    // but I'm doing it this way for clarity and generality
    let tvd_psi = |r: Vector| -> Vector {
        match momentum_scheme {
            MomentumDiscretization::UD => Vector::zero(),
            MomentumDiscretization::CD => Vector::one(),
            _ => panic!("unsupported momentum scheme"),
        }
    };

    for (cell_number, cell) in &mesh.cells {
        let this_cell_velocity_gradient = &mesh.calculate_velocity_gradient(*cell_number);
        let mut a_p = 0.;
        for face_number in &cell.face_numbers {
            let face = &mesh.faces[face_number];
            let f_i = face.velocity.dot(&face.normal) * rho; // Advection (flux) coefficient
            let d_i = 0.; // Diffusion coefficient
            let face_velocity = interpolate_face_velocity(&mesh, *face_number);
            let neighbor_count = face.cell_numbers.len();

            match neighbor_count {
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
                    let (upwind_cell_number, downwind_cell_number): (Uint, Uint) = {
                        if face_velocity.dot(&face.normal) > 0. {
                            // Face normal points toward cell 0, so velocity must be coming from cell 1
                            (
                                if (neighbor_count > 1) {
                                    face.cell_numbers[1]
                                } else {
                                    0
                                },
                                face.cell_numbers[0],
                            )
                        } else {
                            // Face normal and velocity are pointing different directions so cell 0 upwind
                            (
                                face.cell_numbers[0],
                                if (neighbor_count > 1) {
                                    face.cell_numbers[1]
                                } else {
                                    0
                                },
                            )
                        }
                    };

                    let (upwind_velocity, downwind_velocity) = (
                        mesh.cells[&upwind_cell_number].velocity,
                        mesh.cells[&downwind_cell_number].velocity,
                    );

                    let neighbor_cell_number = if upwind_cell_number != *cell_number {
                        upwind_cell_number
                    } else {
                        downwind_cell_number
                    };

                    let neighbor_cell_velocity_gradient =
                        &mesh.calculate_velocity_gradient(neighbor_cell_number);

                    // Either upwind or downwind can be equivalently used in the numerator
                    // per Versteeg & Malalasekara (note that the numerator cell's velocity
                    // must come second in the denominator)
                    let tvd_r: Vector = this_cell_velocity_gradient
                        .dot(&(mesh.cells[&neighbor_cell_number].centroid - cell.centroid)) // r_pa
                        * 2.
                        / (downwind_velocity - upwind_velocity)
                        - 1.;

                    let psi_r = tvd_psi(tvd_r);

                    let upwind_advective_coefficients = (-psi_r + 1.) * f_i / 2;
                    let downwind_advective_coefficients = psi_r * f_i / 2;

                    let mut this_cell_coefficients = Vector::zero();
                    let mut neighbor_cell_coefficients = Vector::zero();

                    if upwind_cell_number == *cell_number {
                        this_cell_coefficients += upwind_advective_coefficients;
                        neighbor_cell_coefficients += downwind_advective_coefficients;
                    } else {
                        neighbor_cell_coefficients += upwind_advective_coefficients;
                        this_cell_coefficients += downwind_advective_coefficients;
                    }

                    // TODO: DRY
                    u_matrix.add_triplet(
                        (*cell_number-1).try_into().unwrap(),
                        (*cell_number-1).try_into().unwrap(),
                        this_cell_coefficients.x,
                    );
                    v_matrix.add_triplet(
                        (*cell_number-1).try_into().unwrap(),
                        (*cell_number-1).try_into().unwrap(),
                        this_cell_coefficients.y,
                    );
                    w_matrix.add_triplet(
                        (*cell_number-1).try_into().unwrap(),
                        (*cell_number-1).try_into().unwrap(),
                        this_cell_coefficients.z,
                    );
                    u_matrix.add_triplet(
                        (*cell_number-1).try_into().unwrap(),
                        (neighbor_cell_number-1).try_into().unwrap(),
                        this_cell_coefficients.x,
                    );
                    v_matrix.add_triplet(
                        (*cell_number-1).try_into().unwrap(),
                        (neighbor_cell_number-1).try_into().unwrap(),
                        this_cell_coefficients.y,
                    );
                    w_matrix.add_triplet(
                        (*cell_number-1).try_into().unwrap(),
                        (neighbor_cell_number-1).try_into().unwrap(),
                        this_cell_coefficients.z,
                    );
                }
                _ => panic!("faces must have 1 or 2 neighbors"),
            }
        }
    }

    // for (cell_number, cell) in &mesh.cells {
    //     // &cell.face_indices.iter().map(|f| mesh.faces.get_mut(f).expect("face exists").velocity = Vector::zero());
    //
    //     // **** Step 1: calculate gradients at cell centers **** //
    //     // TODO: Rewrite this as a 3x3 tensor
    //     let mut grad_u = Vector::zero();
    //     let mut grad_v = Vector::zero();
    //     let mut grad_w = Vector::zero();
    //     let mut interior_face_count: Uint = 0;
    //
    //     for face_number in &cell.face_numbers {
    //         let mut face_velocity: Vector = Vector::zero();
    //         let face = &mesh.faces[face_number];
    //
    //         // Boundary cell
    //         if face.cell_numbers.len() < 2 {
    //             let face_bc = &mesh.face_zones[&face.zone];
    //             match face_bc.zone_type {
    //                 BoundaryConditionTypes::Interior => panic!("Interior one-sided face."),
    //                 BoundaryConditionTypes::Wall => (), // already zero
    //                 BoundaryConditionTypes::VelocityInlet => {
    //                     face_velocity = face.normal * face_bc.value;
    //                 }
    //                 BoundaryConditionTypes::PressureInlet => {
    //                     // TODO: Write something proper!!
    //                     face_velocity = cell.velocity;
    //                 }
    //                 BoundaryConditionTypes::PressureOutlet => {
    //                     // TODO: Write something proper!!
    //                     face_velocity = cell.velocity;
    //                 }
    //                 _ => {
    //                     println!("*** {} ***", face_bc.zone_type);
    //                     panic!("BC not supported");
    //                 }
    //             }
    //         } else {
    //             // Interior cell
    //             face_velocity = interpolate_face_velocity(&mesh, *face_number);
    //         }
    //         let face_vector = mesh.faces[face_number].centroid - cell.centroid;
    //         grad_u += face_vector * face_velocity.x;
    //         grad_v += face_vector * face_velocity.y;
    //         grad_w += face_vector * face_velocity.z;
    //     }
    //
    //     // **** Step 2: Calculate diffusive terms **** //
    //     for face_number in &cell.face_numbers {
    //         let mut diffusive_term = Vector::zero();
    //
    //         let face_velocity = interpolate_face_velocity(&mesh, *face_number);
    //         let face = &mut mesh.faces.get_mut(face_number).expect("face exists");
    //         // By default, face.normal points toward cell 0
    //         // We need it facing outward
    //         let mut outward_face_normal: Vector = face.normal
    //             * (if face.cell_numbers.get(0).unwrap() == cell_number {
    //                 -1.
    //             } else {
    //                 1.
    //             });
    //
    //         let mut neighbor_cell_number = 1;
    //         let advective_term =
    //             face_velocity * rho * outward_face_normal.dot(&face_velocity) * face.area;
    //
    //         if face.cell_numbers.len() < 2 {
    //             let diffusive_term = Vector {
    //                 x: 0.,
    //                 y: 0.,
    //                 z: 0.,
    //             };
    //             // TODO: Handle BCs
    //         } else {
    //             let mut upwind_cell_number = face.cell_numbers[0];
    //             let neighbor_cell_number = *face
    //                 .cell_numbers
    //                 .iter()
    //                 .filter(|c| *c != cell_number)
    //                 .collect::<Vec<&Uint>>()
    //                 .get(0)
    //                 .unwrap();
    //             // If cell 1 exists and is upwind, use that
    //             if face.normal.dot(&cell.velocity) < 0. {
    //                 upwind_cell_number = face.cell_numbers[1];
    //                 outward_face_normal = -outward_face_normal;
    //             }
    //
    //             let diffusive_term = Vector {
    //                 x: outward_face_normal.dot(&grad_u),
    //                 y: outward_face_normal.dot(&grad_v),
    //                 z: outward_face_normal.dot(&grad_w),
    //             } * nu;
    //         }
    //
    //         u_matrix.add_triplet(
    //             *cell_number as usize - 1,
    //             neighbor_cell_number as usize - 1,
    //             advective_term.x + diffusive_term.x,
    //         );
    //         v_matrix.add_triplet(
    //             *cell_number as usize - 1,
    //             neighbor_cell_number as usize - 1,
    //             advective_term.y + diffusive_term.y,
    //         );
    //         w_matrix.add_triplet(
    //             *cell_number as usize - 1,
    //             neighbor_cell_number as usize - 1,
    //             advective_term.z + diffusive_term.z,
    //         );
    //     }
    //
    //     grad_u /= cell.volume;
    //     grad_v /= cell.volume;
    //     grad_w /= cell.volume;
    //
    //     for face_number in &cell.face_numbers {}
    //
    //     let mut cell_faces: Vec<&Face> = cell
    //         .face_numbers
    //         .iter()
    //         .map(|face_number| &mesh.faces[face_number])
    //         .collect();
    // }

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

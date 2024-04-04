use crate::mesh::*;
use crate::nalgebra::{dvector_zeros, GetEntry};
use crate::numerical_types::*;
use crate::settings::*;
use crate::solver::{
    calculate_velocity_gradient, get_face_flux, get_face_pressure, get_velocity_source_term,
};
use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CsrMatrix, SparseEntryMut::*};

pub fn build_momentum_diffusion_matrix(
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

#[allow(clippy::too_many_arguments)]
pub fn build_momentum_matrices(
    a_u: &mut CsrMatrix<Float>,
    a_v: &mut CsrMatrix<Float>,
    a_w: &mut CsrMatrix<Float>,
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
    gradient_scheme: GradientReconstructionMethods,
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
        let mut a_p = Vector3::zero();
        let cell_velocity_gradient =
            calculate_velocity_gradient(mesh, u, v, w, *cell_index, gradient_scheme);
        // println!("\n{cell_velocity_gradient}");
        // Iterate over this cell's faces
        for face_index in &cell.face_indices {
            let face = &mesh.faces[face_index];
            let face_flux = get_face_flux(
                mesh,
                u,
                v,
                w,
                p,
                *face_index,
                *cell_index,
                velocity_interpolation,
                gradient_scheme,
                a_u,
                a_v,
                a_w,
            );
            // TODO: Consider flipping convention of face normal direction and/or potentially
            // make it an area vector
            let outward_face_normal = get_outward_face_normal(face, *cell_index);
            // Mass flow rate out of this cell through this face
            let f_i = face_flux * face.area * rho;
            let face_pressure = get_face_pressure(
                mesh,
                p,
                *face_index,
                pressure_interpolation,
                gradient_scheme,
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

            let a_nb: Vector3 = match momentum_discretization {
                MomentumDiscretization::UD => {
                    // Neighbor only affects this cell if flux is into this
                    // cell => f_i < 0. Therefore, if f_i > 0, we set it to 0.
                    let a_nb = Float::min(f_i, 0.);
                    Vector3 {
                        x: a_nb,
                        y: a_nb,
                        z: a_nb,
                    }
                }
                MomentumDiscretization::TVD(psi) => {
                    if neighbor_cell_index == usize::MAX {
                        let a_nb = Float::min(f_i, 0.);
                        Vector3 {
                            x: a_nb,
                            y: a_nb,
                            z: a_nb,
                        }
                    } else {
                        let downstream_cell = if f_i > 0. {
                            neighbor_cell_index
                        } else {
                            *cell_index
                        };
                        let downstream_velocity = Vector3 {
                            x: u[downstream_cell],
                            y: v[downstream_cell],
                            z: w[downstream_cell],
                        };
                        let velocity = Vector3 {
                            x: u[*cell_index],
                            y: v[*cell_index],
                            z: w[*cell_index],
                        };
                        if (downstream_velocity - velocity).norm() == 0. {
                            let a_nb = Float::min(f_i, 0.);
                            Vector3 {
                                x: a_nb,
                                y: a_nb,
                                z: a_nb,
                            }
                        } else {
                            let r_pa = mesh.cells[&neighbor_cell_index].centroid
                                - mesh.cells[cell_index].centroid;
                            let r = 2. * cell_velocity_gradient.inner(&r_pa)
                                / (downstream_velocity - velocity)
                                - 1.;
                            f_i * Vector3 {
                                x: psi(r.x),
                                y: psi(r.y),
                                z: psi(r.z),
                            } / 2.
                        }
                    }
                }
                _ => panic!("unsupported momentum scheme"),
            };

            a_p += -a_nb + f_i;
            s_u += (-outward_face_normal) * face_pressure * face.area;

            // If it's MAX, that means it's a boundary face
            if neighbor_cell_index != usize::MAX {
                // negate a_nb to move to LHS of equation
                match a_u.get_entry_mut(*cell_index, neighbor_cell_index).unwrap() {
                    NonZero(a_ij) => *a_ij = a_nb.x,
                    Zero => panic!(),
                }
                match a_v.get_entry_mut(*cell_index, neighbor_cell_index).unwrap() {
                    NonZero(a_ij) => *a_ij = a_nb.y,
                    Zero => panic!(),
                }
                match a_w.get_entry_mut(*cell_index, neighbor_cell_index).unwrap() {
                    NonZero(a_ij) => *a_ij = a_nb.z,
                    Zero => panic!(),
                }
            }
        } // end face loop
        let source_total = s_u + s_u_dc + s_d_cross;
        b_u[*cell_index] = source_total.x;
        b_v[*cell_index] = source_total.y;
        b_w[*cell_index] = source_total.z;

        match a_u.get_entry_mut(*cell_index, *cell_index).unwrap() {
            NonZero(v) => *v = a_p.x,
            Zero => panic!(),
        }
        match a_v.get_entry_mut(*cell_index, *cell_index).unwrap() {
            NonZero(v) => *v = a_p.y,
            Zero => panic!(),
        }
        match a_w.get_entry_mut(*cell_index, *cell_index).unwrap() {
            NonZero(v) => *v = a_p.z,
            Zero => panic!(),
        }
    } // end cell loop
      // NOTE: I *think* all diagonal terms in `a` should be positive and all off-diagonal terms
      // negative. It may be worth adding assertions to validate this.
}

#[allow(clippy::too_many_arguments)]
pub fn build_pressure_correction_matrices(
    mesh: &Mesh,
    u: &DVector<Float>,
    v: &DVector<Float>,
    w: &DVector<Float>,
    p: &DVector<Float>,
    a_u: &CsrMatrix<Float>,
    a_v: &CsrMatrix<Float>,
    a_w: &CsrMatrix<Float>,
    numerical_settings: &NumericalSettings,
    rho: Float,
) -> LinearSystem {
    let cell_count = mesh.cells.len();
    // The coefficients of the pressure correction matrix
    let mut a = CooMatrix::<Float>::new(cell_count, cell_count);
    // This is the net mass flow rate into each cell
    let mut b = dvector_zeros!(cell_count);

    for (cell_index, cell) in &mesh.cells {
        let mut a_p: Float = 0.;
        let mut b_p: Float = 0.;
        for face_index in &cell.face_indices {
            let face = &mesh.faces[face_index];
            let outward_face_flux = get_face_flux(
                mesh,
                u,
                v,
                w,
                p,
                *face_index,
                *cell_index,
                numerical_settings.velocity_interpolation,
                numerical_settings.gradient_reconstruction,
                a_u,
                a_v,
                a_w,
            );
            let inward_face_normal = get_inward_face_normal(face, *cell_index);
            // The net mass flow rate through this face into this cell
            b_p += rho * (-outward_face_flux) * face.area;

            let a_ii = Vector3 {
                x: a_u.get(*cell_index, *cell_index) * Float::abs(inward_face_normal.x),
                y: a_v.get(*cell_index, *cell_index) * Float::abs(inward_face_normal.y),
                z: a_w.get(*cell_index, *cell_index) * Float::abs(inward_face_normal.z),
            };

            if face.cell_indices.len() > 1 {
                let neighbor_cell_index = if face.cell_indices[0] != *cell_index {
                    face.cell_indices[0]
                } else {
                    face.cell_indices[1]
                };
                // This relates the pressure drop across this face to its velocity
                // NOTE: It would be more rigorous to recalculate the advective coefficients,
                // but I think this should be sufficient for now.
                let a_ij = Vector3 {
                    x: (a_u.get(neighbor_cell_index, neighbor_cell_index))
                        * Float::abs(inward_face_normal.x),
                    y: (a_v.get(neighbor_cell_index, neighbor_cell_index))
                        * Float::abs(inward_face_normal.y),
                    z: (a_w.get(neighbor_cell_index, neighbor_cell_index))
                        * Float::abs(inward_face_normal.z),
                };
                let a_interpolated = (a_ii + a_ij) / 2.;

                // NOTE: Unsure on how I'm combining the x/y/z components of `a` here.
                // TODO: Test with mesh that doesn't align with XY grid to make sure this is
                // correct
                let a_nb = rho * Float::powi(face.area, 2) / a_interpolated.norm();
                a.push(*cell_index, neighbor_cell_index, -a_nb);
                a_p += a_nb;
            } else {
                // TODO: Handle boundary types here (e.g. wall should add zero)
                let a_nb = rho * Float::powi(face.area, 2) / a_ii.norm();
                a_p += a_nb / 2.; // NOTE: I'm not sure if dividing by two is correct here?
            }
        }
        a.push(*cell_index, *cell_index, a_p);
        b[*cell_index] = b_p;
    }

    // TODO: Consider passing a and b as mut args rather than returning
    LinearSystem {
        a: CsrMatrix::from(&a),
        b,
    }
}

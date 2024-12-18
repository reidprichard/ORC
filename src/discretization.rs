use crate::mesh::*;
use crate::nalgebra::GetEntry;
use crate::numerical_types::*;
use crate::settings::*;
use crate::solver::{
    calculate_velocity_gradient, get_face_flux, get_face_pressure, get_momentum_source_term,
};

use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CsrMatrix, SparseEntryMut::*};

// TODO: Find, list, and justify areas where user-specified settings are overridden

macro_rules! get_normal_momentum_coefficient {
    ($i: expr, $a_u: expr, $a_v: expr, $a_w: expr, $n: expr) => {
        Vector {
            x: $a_u.get($i, $i) * $n.x,
            y: $a_v.get($i, $i) * $n.y,
            z: $a_w.get($i, $i) * $n.z,
        }
        .norm()
    };
}

macro_rules! get_face_normal_momentum_coefficient {
    ($i: expr, $j: expr, $a_u: expr, $a_v: expr, $a_w: expr, $n: expr) => {
        0.5 * Vector {
            x: ($a_u.get($i, $i) + $a_u.get($j, $j)) * $n.x,
            y: ($a_v.get($i, $i) + $a_v.get($j, $j)) * $n.y,
            z: ($a_w.get($i, $i) + $a_w.get($j, $j)) * $n.z,
        }
        .norm()
    };
}

pub(crate) use get_face_normal_momentum_coefficient;
pub(crate) use get_normal_momentum_coefficient;

pub fn build_momentum_diffusion_matrix(
    mesh: &Mesh,
    diffusion_scheme: DiffusionScheme,
    mu: Float,
) -> (
    CsrMatrix<Float>,
    DVector<Float>,
    DVector<Float>,
    DVector<Float>,
) {
    if !matches!(diffusion_scheme, DiffusionScheme::CD) {
        panic!("unsupported diffusion scheme");
    }

    let cell_count = mesh.cells.len();
    let mut a = CooMatrix::<Float>::new(cell_count, cell_count);
    let mut b_u: DVector<Float> = DVector::zeros(cell_count);
    let mut b_v: DVector<Float> = DVector::zeros(cell_count);
    let mut b_w: DVector<Float> = DVector::zeros(cell_count);

    // Iterate over all cells in the mesh
    for cell_index in 0..mesh.cells.len() {
        let cell = &mesh.cells[cell_index];
        // The current cell's coefficients (matrix diagonal)
        let mut a_p = 0.;

        // Iterate over this cell's faces
        for face_index in &cell.face_indices {
            let face = &mesh.faces[*face_index];
            let face_bc = &mesh.face_zones[&face.zone];
            let (d_i, neighbor_cell_index) = match face_bc.zone_type {
                FaceConditionTypes::Wall | FaceConditionTypes::VelocityInlet => {
                    // NOTE: These boundaries need to have a source term contribution since there
                    // is no cell on the other side.
                    let d_i = mu * face.area / (face.centroid - cell.centroid).norm();
                    let source_contribution = face_bc.vector_value * d_i;
                    b_u[cell_index] += source_contribution.x;
                    b_v[cell_index] += source_contribution.y;
                    b_w[cell_index] += source_contribution.z;
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
                FaceConditionTypes::Interior => {
                    let mut neighbor_cell_index: usize = face.cell_indices[0];
                    if neighbor_cell_index == cell_index {
                        neighbor_cell_index = *face
                            .cell_indices
                            .get(1)
                            .expect("interior faces should have two neighbors");
                    }

                    // Cell centroids vector
                    let e_xi: Vector = mesh.cells[neighbor_cell_index].centroid - cell.centroid;
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
                a.push(cell_index, neighbor_cell_index, -d_i);
            }
        } // end face loop
        a.push(cell_index, cell_index, a_p);
    } // end cell loop
    (CsrMatrix::from(&a), b_u, b_v, b_w)
}

#[allow(clippy::too_many_arguments)]
pub fn build_momentum_advection_matrices(
    a_u: &mut CsrMatrix<Float>,
    a_v: &mut CsrMatrix<Float>,
    a_w: &mut CsrMatrix<Float>,
    b_u: &mut DVector<Float>,
    b_v: &mut DVector<Float>,
    b_w: &mut DVector<Float>,
    a_di: &CsrMatrix<Float>,
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
) -> (Float, Float, Float) {
    let mut min_peclet_number = Float::INFINITY;
    let mut max_peclet_number = Float::NEG_INFINITY;
    let mut avg_peclet_number: Float = 0.;
    // Iterate over all cells in the mesh
    for cell_index in 0..mesh.cells.len() {
        let cell = &mesh.cells[cell_index];
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
        let mut s_u = get_momentum_source_term(cell.centroid); // general source term
        let s_u_dc = Vector::zero(); // deferred correction source term TODO
        let s_d_cross = Vector::zero(); // cross diffusion source term TODO

        // The current cell's coefficients (matrix diagonal)
        let a_ii_di = a_di.get(cell_index, cell_index);
        let mut a_p = Vector::zero();
        // println!("\n{cell_velocity_gradient}");
        // Iterate over this cell's faces
        for face_index in &cell.face_indices {
            let face = &mesh.faces[*face_index];
            // WARNING: Things can get tricky here from the recursion: you're calculating flux
            // based on a_u/v/w, but you're doing so to calculate the coefficients in a_u/v/w
            let face_flux = get_face_flux(
                mesh,
                u,
                v,
                w,
                p,
                *face_index,
                cell_index,
                velocity_interpolation,
                gradient_scheme,
                a_u,
                a_v,
                a_w,
            );
            // TODO: Consider flipping convention of face normal direction and/or potentially
            // make it an area vector
            let outward_face_normal = get_outward_face_normal(face, cell_index);
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
            } else if face.cell_indices[0] == cell_index {
                // face normal points to cell 0 by default, so we need to flip it
                *face
                    .cell_indices
                    .get(1)
                    .expect("interior faces should have two neighbors")
            } else {
                face.cell_indices[0]
            };

            let a_nb: Vector = match momentum_discretization {
                MomentumDiscretization::UD => {
                    // Neighbor only affects this cell if flux is into this
                    // cell => f_i < 0. Therefore, if f_i > 0, we set it to 0.
                    Float::min(f_i, 0.) * Vector::ones()
                }
                MomentumDiscretization::CD1 => {
                    // Neighbor only affects this cell if flux is into this
                    // cell => f_i < 0. Therefore, if f_i > 0, we set it to 0.
                    f_i * Vector::ones() / 2.
                }
                MomentumDiscretization::TVD(psi) => {
                    // TODO: Gradient limiter
                    if neighbor_cell_index == usize::MAX {
                        // NOTE: On boundary faces, use UD
                        // TODO: Consider strategy here
                        Float::min(f_i, 0.) * Vector::ones()
                    } else {
                        // TODO: Consider directly representing gradient term in solution matrices
                        // Green-Gauss face values are a function of neighboring cell values, so they can be
                        // represented in the matrix I think?

                        let downstream_cell = if f_i > 0. {
                            neighbor_cell_index
                        } else {
                            cell_index
                        };
                        let downstream_velocity = Vector {
                            x: u[downstream_cell],
                            y: v[downstream_cell],
                            z: w[downstream_cell],
                        };
                        let velocity = Vector {
                            x: u[cell_index],
                            y: v[cell_index],
                            z: w[cell_index],
                        };
                        if (downstream_velocity - velocity).norm() == 0. {
                            // If the velocities are equal, the scheme we use doesn't matter, as
                            // face velocity = cell 1 velocity = cell 2 velocity
                            // NOTE: Technically this isn't true, but odds of vels being exactly
                            // equal are nil.
                            f_i * Vector::ones() / 2.
                        } else {
                            let cell_velocity_gradient = calculate_velocity_gradient(
                                mesh,
                                u,
                                v,
                                w,
                                cell_index,
                                gradient_scheme,
                            );
                            let r_pa = mesh.cells[neighbor_cell_index].centroid
                                - mesh.cells[cell_index].centroid;
                            let r = 2. * cell_velocity_gradient.inner(&r_pa)
                                / (downstream_velocity - velocity)
                                - 1.;
                            f_i * Vector {
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
            if neighbor_cell_index == usize::MAX {
                let face_zone = &mesh.face_zones[&face.zone];
                s_u += match face_zone.zone_type {
                    FaceConditionTypes::Wall | FaceConditionTypes::VelocityInlet => {
                        // TODO: Get this working
                        Vector {
                            x: (a_nb.x - f_i) * face_zone.vector_value.x,
                            y: (a_nb.y - f_i) * face_zone.vector_value.y,
                            z: (a_nb.z - f_i) * face_zone.vector_value.z,
                        }
                    }
                    _ => Vector::zero(), // Do nothing for other BC types
                }
                // Add source term contributions for velocity BCs
            } else {
                let a_ij_di = a_di.get(cell_index, neighbor_cell_index);
                // negate a_nb to move to LHS of equation
                // can I clean this up with .get_entry_mut().unwrap().into_value()?
                match a_u.get_entry_mut(cell_index, neighbor_cell_index).unwrap() {
                    NonZero(a_ij) => *a_ij = a_nb.x + a_ij_di,
                    Zero => panic!(),
                }
                match a_v.get_entry_mut(cell_index, neighbor_cell_index).unwrap() {
                    NonZero(a_ij) => *a_ij = a_nb.y + a_ij_di,
                    Zero => panic!(),
                }
                match a_w.get_entry_mut(cell_index, neighbor_cell_index).unwrap() {
                    NonZero(a_ij) => *a_ij = a_nb.z + a_ij_di,
                    Zero => panic!(),
                }
            }
        } // end face loop
        let source_total = s_u + s_u_dc + s_d_cross;
        b_u[cell_index] = source_total.x;
        b_v[cell_index] = source_total.y;
        b_w[cell_index] = source_total.z;

        let peclet_x = a_p.x / a_ii_di;
        let peclet_y = a_p.y / a_ii_di;
        let peclet_z = a_p.z / a_ii_di;

        // TODO: Consider these values being based on absolute values
        max_peclet_number = *[max_peclet_number, peclet_x, peclet_y, peclet_z].iter().max_by(|a,b| a.total_cmp(b)).unwrap();
        min_peclet_number = *[min_peclet_number, peclet_x, peclet_y, peclet_z].iter().min_by(|a,b| a.total_cmp(b)).unwrap();
        avg_peclet_number += [peclet_x, peclet_y, peclet_z].iter().fold(0., |acc, p| acc + p) / 3.;

        match a_u.get_entry_mut(cell_index, cell_index).unwrap() {
            NonZero(v) => *v = a_p.x + a_ii_di,
            Zero => panic!(),
        }
        match a_v.get_entry_mut(cell_index, cell_index).unwrap() {
            NonZero(v) => *v = a_p.y + a_ii_di,
            Zero => panic!(),
        }
        match a_w.get_entry_mut(cell_index, cell_index).unwrap() {
            NonZero(v) => *v = a_p.z + a_ii_di,
            Zero => panic!(),
        }
    } // end cell loop
      // NOTE: I *think* all diagonal terms in `a` should be positive and all off-diagonal terms
      // negative. It may be worth adding assertions to validate this.
    (avg_peclet_number / (mesh.cells.len() as Float), min_peclet_number, max_peclet_number)
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
    let mut b = DVector::zeros(cell_count);

    for cell_index in 0..mesh.cells.len() {
        let cell = &mesh.cells[cell_index];
        let mut a_p: Float = 0.;
        let mut b_p: Float = 0.;
        for face_index in &cell.face_indices {
            let face = &mesh.faces[*face_index];
            let outward_face_flux = get_face_flux(
                mesh,
                u,
                v,
                w,
                p,
                *face_index,
                cell_index,
                numerical_settings.velocity_interpolation,
                numerical_settings.gradient_reconstruction,
                a_u,
                a_v,
                a_w,
            );
            let inward_face_normal = get_inward_face_normal(face, cell_index);
            // The net mass flow rate through this face into this cell
            b_p += rho * (-outward_face_flux) * face.area;

            if face.cell_indices.len() > 1 {
                let neighbor_cell_index = if face.cell_indices[0] != cell_index {
                    face.cell_indices[0]
                } else {
                    face.cell_indices[1]
                };
                // This relates the pressure drop across this face to its velocity
                // NOTE: It would be more rigorous to recalculate the advective coefficients,
                // but I think this should be sufficient for now.
                let a_interpolated_magnitude = get_face_normal_momentum_coefficient!(
                    cell_index,
                    neighbor_cell_index,
                    a_u,
                    a_v,
                    a_w,
                    inward_face_normal
                );

                // NOTE: Unsure on how I'm combining the x/y/z components of `a` here.
                // TODO: Test with mesh that doesn't align with XY grid to make sure this is
                // correct
                let a_nb = rho * Float::powi(face.area, 2) / a_interpolated_magnitude;
                a.push(cell_index, neighbor_cell_index, -a_nb);
                a_p += a_nb;
            } else {
                // TODO: Handle boundary types here (e.g. wall should add zero)
                let a_ii_norm = get_normal_momentum_coefficient!(
                    cell_index,
                    a_u,
                    a_v,
                    a_w,
                    &inward_face_normal
                );
                let a_nb = rho * Float::powi(face.area, 2) / a_ii_norm;
                a_p += a_nb / 2.; // NOTE: I'm not sure if dividing by two is correct here?
            }
        }
        a.push(cell_index, cell_index, a_p);
        b[cell_index] = b_p;
    }
    // panic!("done");

    // TODO: Consider passing a and b as mut args rather than returning
    LinearSystem {
        a: CsrMatrix::from(&a),
        b,
    }
}

pub fn initialize_momentum_matrix(mesh: &Mesh) -> CsrMatrix<Float> {
    let cell_count = mesh.cells.len();
    let mut a: CooMatrix<Float> = CooMatrix::new(cell_count, cell_count);
    for cell_index in 0..mesh.cells.len() {
        let cell = &mesh.cells[cell_index];
        a.push(cell_index, cell_index, 1.);
        let n = cell.face_indices.len() as Float;
        for face_index in &cell.face_indices {
            let face = &mesh.faces[*face_index];
            if face.cell_indices.len() == 2 {
                let neighbor_cell_index = if face.cell_indices[0] == cell_index {
                    face.cell_indices[1]
                } else {
                    face.cell_indices[0]
                };
                // NOTE: Not sure if it's necessary to put a nonzero value, but this could save
                // some headaches. However, `n` isn't necessarily correct I don't think.
                a.push(cell_index, neighbor_cell_index, -1. / n);
            }
        }
    }
    CsrMatrix::from(&a)
}

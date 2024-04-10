use crate::nalgebra::GetEntry;
use crate::numerical_types::*;
use crate::settings::*;
use ahash::{HashSet, RandomState};
use log::{info, trace};
use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CsrMatrix};

const MULTIGRID_SMOOTHER: SolutionMethod = SolutionMethod::BiCGSTAB;
const MULTIGRID_COARSENING_LEVELS: Uint = 3;

fn build_restriction_matrix(a: &CsrMatrix<Float>, method: RestrictionMethods) -> CsrMatrix<Float> {
    let n = a.ncols() / 2 + a.ncols() % 2; // half rounded up
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
        // TODO: Rethink this restriction strategy. I suspect it produces very long combined cells
        // along directions of streamlines, which may not be the best strategy to get error to
        // propagate.
        RestrictionMethods::Strongest => {
            // For each row, find the most negative off-diagonal value
            // If that cell hasn't already been combined, combine it with diagonal
            // If it *has* been combined, find the next largest and so on.
            let mut combined_cells = HashSet::<usize>::with_capacity_and_hasher(n, RandomState::with_seeds(3, 1, 4, 1));
            a.row_iter().enumerate().for_each(|(i, row)| {
                let mut strongest_coeff: Float = Float::MAX;
                let strongest_unmerged_neighbor =
                    row.col_indices().iter().fold(usize::MAX, |acc, j| {
                        // not efficient to look this up each time
                        if combined_cells.contains(j) || i == *j {
                            acc
                        } else {
                            let coeff = a.get(i, *j);
                            if coeff < strongest_coeff {
                                strongest_coeff = coeff;
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


#[allow(clippy::too_many_arguments, unreachable_patterns)]
pub fn iterative_solve(
    a: &CsrMatrix<Float>,
    b: &DVector<Float>,
    solution_vector: &mut DVector<Float>,
) {
    let method = SolutionMethod::BiCGSTAB;
    let relaxation_factor = 0.5;
    match method {
        SolutionMethod::Jacobi => {
            let mut a_prime = a.clone();
            a_prime.triplet_iter_mut().for_each(|(i, j, v)| {
                *v = if i == j {
                    0.
                } else {
                    *v / a.get(i, i)
                }
            });
            let b_prime: DVector<Float> = DVector::from_iterator(
                b.nrows(),
                b
                    .iter()
                    .enumerate()
                    .map(|(i, v)| *v / a.get(i, i)),
            );
            for iter_num in 0..50 {
                // if log_enabled!(log::Level::Trace) {
                //     trace!("\nJacobi iteration {iter_num} = {solution_vector:?}");
                // }
                for v in solution_vector.iter() {
                    if v.is_nan() {
                        panic!("diverged");
                    }
                }
                // It seems like there must be a way to avoid cloning solution_vector, even if that
                // turns this into Gauss-Seidel
                *solution_vector = relaxation_factor * (&b_prime - &a_prime * &*solution_vector)
                    + &*solution_vector * (1. - relaxation_factor);
            }
        }
        SolutionMethod::BiCGSTAB => {
            // TODO: Optimize this
            let mut r = b - a * &*solution_vector;
            // TODO: Set search direction more intelligently
            let r_hat_0 = DVector::from_column_slice(&vec![1.; r.nrows()]);
            let mut rho = r.dot(&r_hat_0);
            let mut p = r.clone();
            for _iter_num in 0..50 {
                let nu: DVector<Float> = a * &p;
                let alpha: Float = rho / r_hat_0.dot(&nu);
                let h: DVector<Float> = &*solution_vector + alpha * &p;
                let s: DVector<Float> = &r - alpha * &nu;
                let t: DVector<Float> = a * &s;
                let omega: Float = t.dot(&s) / t.dot(&t);
                *solution_vector = &h + omega * &s;
                r = &s - omega * &t;
                let rho_prev: Float = rho;
                rho = r_hat_0.dot(&r);
                let beta: Float = rho / rho_prev * alpha / omega;
                p = &r + beta * (p - omega * &nu);
            }
        }
        _ => panic!("unsupported solution method"),
    }
}

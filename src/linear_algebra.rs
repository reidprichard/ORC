use crate::nalgebra::{dvector_zeros, GetEntry};
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
            let mut combined_cells =
                HashSet::<usize>::with_capacity_and_hasher(n, RandomState::with_seeds(3, 1, 4, 1));
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

#[allow(clippy::too_many_arguments)]
fn multigrid_solve(
    a: &CsrMatrix<Float>,
    r: &DVector<Float>,
    multigrid_level: Uint,
    max_levels: Uint,
    smooth_method: SolutionMethod,
    smooth_iter_count: Uint,
    smooth_relaxation: Float,
    smooth_convergence_threshold: Float,
    restriction_method: RestrictionMethods,
    preconditioner: PreconditionMethod,
) -> DVector<Float> {
    // 1. Build restriction matrix R
    // 5. If multigrid_level < max_level, recurse
    let restriction_matrix = build_restriction_matrix(a, restriction_method);
    // 2. Restrict r to r'
    let r_prime = &restriction_matrix * r;
    // 3. Create a' = R * a * R.⊺
    let a_prime = &restriction_matrix * a * &restriction_matrix.transpose();
    // 4. Solve a' * e' = r'
    let mut e_prime: DVector<Float> = dvector_zeros!(a_prime.ncols());
    iterative_solve(
        &a_prime,
        &r_prime,
        &mut e_prime,
        smooth_iter_count,
        smooth_method,
        smooth_relaxation,
        smooth_convergence_threshold,
        preconditioner,
    );
    let error_magnitude = (&r_prime - &a_prime * &e_prime).norm();
    trace!(
        "Multigrid level {}: |e| = {:.2e}",
        multigrid_level,
        error_magnitude
    );
    if error_magnitude.is_nan() {
        panic!("Multigrid diverged");
    }

    // 5. Recurse to max desired coarsening level
    // Only coarsen if the matrix is bigger than 16x16 because come on
    if multigrid_level < max_levels && a_prime.nrows() > 16 {
        e_prime += multigrid_solve(
            &a_prime,
            &r_prime,
            multigrid_level + 1,
            max_levels,
            smooth_method,
            smooth_iter_count,
            smooth_relaxation,
            smooth_convergence_threshold,
            restriction_method,
            preconditioner,
        );
        // Smooth after incorporating corrections from coarser grid
        iterative_solve(
            &a_prime,
            &r_prime,
            &mut e_prime,
            smooth_iter_count,
            smooth_method,
            smooth_relaxation,
            smooth_convergence_threshold / 10.,
            preconditioner,
        );
        trace!(
            "Multigrid level {}: |e| = {:.2e}",
            multigrid_level,
            (&r_prime - &a_prime * &e_prime).norm() // e_prime.norm()
        );
    }
    // Prolong correction vector
    &restriction_matrix.transpose() * e_prime
}

#[allow(clippy::too_many_arguments, unreachable_patterns)]
pub fn iterative_solve(
    a: &CsrMatrix<Float>,
    b: &DVector<Float>,
    solution_vector: &mut DVector<Float>,
    iteration_count: Uint,
    method: SolutionMethod,
    relaxation_factor: Float,
    convergence_threshold: Float,
    preconditioner: PreconditionMethod,
) {
    // Surely there is a better way to do this...
    let a_tmp: CsrMatrix<Float>;
    let b_tmp: DVector<Float>;
    let (a_preconditioned, b_preconditioned) = match preconditioner {
        PreconditionMethod::None => (a, b),
        PreconditionMethod::Jacobi => {
            let mut p_inv = a.diagonal_as_csr();
            p_inv
                .triplet_iter_mut()
                .for_each(|(_i, _j, v)| *v = 1. / *v);
            a_tmp = &p_inv * a;
            b_tmp = &p_inv * b;
            (&a_tmp, &b_tmp)
        }
    };

    let mut initial_residual: Float = 0.;
    match method {
        SolutionMethod::Jacobi => {
            let mut a_prime = a_preconditioned.clone();
            a_prime.triplet_iter_mut().for_each(|(i, j, v)| {
                *v = if i == j {
                    0.
                } else {
                    *v / a_preconditioned.get(i, i)
                }
            });
            let b_prime: DVector<Float> = DVector::from_iterator(
                b_preconditioned.nrows(),
                b_preconditioned
                    .iter()
                    .enumerate()
                    .map(|(i, v)| *v / a_preconditioned.get(i, i)),
            );
            for iter_num in 0..iteration_count {
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

                let r = (b_preconditioned - a_preconditioned * &*solution_vector).norm();
                if iter_num == 1 {
                    initial_residual = r;
                } else if r / initial_residual < convergence_threshold {
                    info!("Converged in {} iters", iter_num);
                    break;
                }
            }
        }
        SolutionMethod::GaussSeidel => {
            for _iter_num in 0..iteration_count {
                // if log_enabled!(log::Level::Trace) {
                //     println!("Gauss-Seidel iteration {iter_num} = {solution_vector:?}");
                // }
                for i in 0..a_preconditioned.nrows() {
                    solution_vector[i] = solution_vector[i] * (1. - relaxation_factor)
                        + relaxation_factor
                            * (b_preconditioned[i]
                                - solution_vector
                                    .iter() // par_iter is slower here with 1k cells; might be worth it with more cells
                                    .enumerate()
                                    .map(|(j, x)| {
                                        if i != j {
                                            a_preconditioned.get(i, j) * x
                                        } else {
                                            0.
                                        }
                                    })
                                    .sum::<Float>())
                            / a_preconditioned.get(i, i);
                    if solution_vector[i].is_nan() {
                        panic!("****** Solution diverged ******");
                    }
                }
            }
            panic!("Gauss-Seidel out for maintenance :)");
        }
        SolutionMethod::BiCGSTAB => {
            // TODO: Precondition properly: https://doc.comsol.com/5.5/doc/com.comsol.help.comsol/comsol_ref_solver.27.123.html
            // TODO: Optimize this
            let mut r = b_preconditioned - a_preconditioned * &*solution_vector;
            // TODO: Set search direction more intelligently
            let r_hat_0 = DVector::from_column_slice(&vec![1.; r.nrows()]);
            let mut rho = r.dot(&r_hat_0);
            let mut p = r.clone();
            for _iter_num in 0..iteration_count {
                let nu: DVector<Float> = a_preconditioned * &p;
                let alpha: Float = rho / r_hat_0.dot(&nu);
                let h: DVector<Float> = &*solution_vector + alpha * &p;
                let s: DVector<Float> = &r - alpha * &nu;
                let t: DVector<Float> = a_preconditioned * &s;
                let omega: Float = t.dot(&s) / t.dot(&t);
                *solution_vector = &h + omega * &s;
                r = &s - omega * &t;
                let rho_prev: Float = rho;
                rho = r_hat_0.dot(&r);
                let beta: Float = rho / rho_prev * alpha / omega;
                p = &r + beta * (p - omega * &nu);
            }
        }
        SolutionMethod::Multigrid => {
            // It seems that too many coarsening levels can cause stability issues.
            // I wonder if this is why Fluent has more complex AMG cycles.
            iterative_solve(
                a_preconditioned,
                b_preconditioned,
                solution_vector,
                iteration_count,
                MULTIGRID_SMOOTHER,
                relaxation_factor,
                convergence_threshold,
                preconditioner,
            );
            let r = b_preconditioned - a_preconditioned * &*solution_vector;
            *solution_vector += multigrid_solve(
                a_preconditioned,
                &r,
                1,
                MULTIGRID_COARSENING_LEVELS,
                MULTIGRID_SMOOTHER,
                iteration_count,
                relaxation_factor,
                convergence_threshold,
                RestrictionMethods::Strongest,
                preconditioner,
            );
        }
        _ => panic!("unsupported solution method"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::io::print_vec_scientific;

    use std::time::Instant;

    #[test]
    fn validate_iterative_solvers() {
        const TOL: Float = 1e-3;

        const N: usize = 100;

        let mut a_coo: CooMatrix<Float> = CooMatrix::new(N, N);
        let mut b_vec: Vec<Float> = Vec::with_capacity(N);
        let solution: Vec<Float> = (0..N).map(|x_i| 2. * (x_i as Float)).collect();

        for i in 0..N {
            let mut b_value = 0.;
            for j in 0..N {
                let mut value = 0.;
                if i == j {
                    value = 1.;
                } else if j != 0 && j != N - 1 && i.abs_diff(j) == 1 {
                    value = -1. / 4.;
                }
                if value != 0. {
                    a_coo.push(i, j, value);
                    b_value += value * solution[j];
                }
            }
            b_vec.push(b_value);
        }

        let a = CsrMatrix::from(&a_coo);
        let b = DVector::from_vec(b_vec);

        let n = b.len();
        let mut x = b.map(|_| 0.);
        for (solution_method, name) in [
            (SolutionMethod::Jacobi, "Jacobi"),
            (SolutionMethod::BiCGSTAB, "BiCGSTAB"),
            // TODO: Figure out why Multigrid won't pass this test
            // (SolutionMethod::Multigrid, "Multigrid"),
        ]
        .iter()
        {
            println!("*** Testing {name} solver for correctness. ***");

            let start = Instant::now();
            iterative_solve(
                &a,
                &b,
                &mut x,
                50,
                *solution_method,
                0.5,
                TOL / (n.pow(3) as Float),
                PreconditionMethod::Jacobi,
            );

            let r = &a * &x - &b;
            if n < 10 {
                print!("x = ");
                print_vec_scientific(&x);
                print!("r = ");
                print_vec_scientific(&r);
            }
            println!(
                "|r| = {:.1e}, Δt = {}µs",
                r.norm(),
                (Instant::now() - start).as_micros()
            );
            assert!(r.norm() < TOL);
            println!("*** {name} solver validated. ***");
        }
    }
}

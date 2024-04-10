use crate::nalgebra::GetEntry;
use crate::numerical_types::*;
use crate::settings::*;
use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;

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
            a_prime
                .triplet_iter_mut()
                .for_each(|(i, j, v)| *v = if i == j { 0. } else { *v / a.get(i, i) });
            let b_prime: DVector<Float> = DVector::from_iterator(
                b.nrows(),
                b.iter().enumerate().map(|(i, v)| *v / a.get(i, i)),
            );
            for _ in 0..50 {
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

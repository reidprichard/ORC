use crate::io::print_vec_scientific;
use crate::linear_algebra::iterative_solve;
use crate::numerical_types::Float;
use crate::settings::{PreconditionMethod, SolutionMethod};

use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CsrMatrix};

use std::time::Instant;

pub fn validate_solvers() {
    const TOL: Float = 1e-3;

    const N: usize = 20;

    let mut a_coo: CooMatrix<Float> = CooMatrix::new(N, N);
    let mut b_vec: Vec<Float> = Vec::with_capacity(N);

    let a = CsrMatrix::from(&a_coo);
    let b = DVector::from_column_slice(&[
        -0.5, 2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0, 90.0, 110.0, 132.0, 156.0, 182.0,
        210.0, 240.0, 272.0, 306.0, 342.0, 580.0,
    ]);

    let n = b.len();
    let mut x = b.map(|_| 0.);
    for (solution_method, name) in [
        (SolutionMethod::Jacobi, "Jacobi"),
        (SolutionMethod::BiCGSTAB, "BiCGSTAB"),
        (SolutionMethod::Multigrid, "Multigrid"),
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

use crate::io::print_vec_scientific;
use crate::linear_algebra::iterative_solve;
use crate::numerical_types::Float;
use crate::settings::{PreconditionMethod, SolutionMethod};

use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CsrMatrix};

use std::time::Instant;

pub fn validate_solvers() {
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

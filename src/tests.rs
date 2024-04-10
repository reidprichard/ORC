fn validate_solvers() {
    // TODO: Move to tests.rs
    // TODO: Feed this a much larger linear system representative of CFD
    const TOL: Float = 1e-6;

    // | 2 0 1 |   | 3 |
    // | 0 3 2 | = | 2 |
    // | 2 0 4 |   | 1 |
    //
    // | 1 0 0 |   | 11/6 |   | 1.833 |
    // | 0 1 0 | = | 10/9 | = | 1.111 |
    // | 0 0 1 |   | -2/3 |   | -0.67 |
    let mut a_coo: CooMatrix<Float> = CooMatrix::new(3, 3);
    a_coo.push(0, 0, 2.);
    a_coo.push(0, 2, 1.);

    a_coo.push(1, 1, 3.);
    a_coo.push(1, 2, 2.);

    a_coo.push(2, 0, 2.);
    a_coo.push(2, 2, 4.);

    let a = CsrMatrix::from(&a_coo);
    let b = DVector::from_column_slice(&[3., 2., 1.]);

    for (solution_method, name) in [
        (SolutionMethod::Jacobi, "Jacobi"),
        (SolutionMethod::Multigrid, "Multigrid"),
        (SolutionMethod::BiCGSTAB, "BiCGSTAB"),
    ]
    .iter()
    {
        println!("*** Testing {name} solver for correctness. ***");
        let mut x = DVector::from_column_slice(&[0., 0., 0.]);

        iterative_solve(
            &a,
            &b,
            &mut x,
            10000,
            *solution_method,
            0.5,
            TOL / 10.,
            PreconditionMethod::Jacobi,
        );

        print!("x = ");
        print_vec_scientific(&x);
        for row_num in 0..a.nrows() {
            assert!(
                Float::abs(
                    a.get_entry(row_num, 0).unwrap().into_value() * x[0]
                        + a.get_entry(row_num, 1).unwrap().into_value() * x[1]
                        + a.get_entry(row_num, 2).unwrap().into_value() * x[2]
                        - b[row_num]
                ) < TOL
            );
        }
        println!("*** {name} solver validated. ***");
    }
}

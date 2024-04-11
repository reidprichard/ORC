pub mod discretization;
pub mod io;
pub mod linear_algebra;
pub mod mesh;
pub mod solver;
pub mod tests;

pub mod settings {
    use crate::numerical_types::*;
    use nalgebra::DVector;
    use nalgebra_sparse::CsrMatrix;

    // ****** Solver numerical settings structs ******
    pub struct NumericalSettings {
        // SIMPLE is the only option for now
        pub pressure_velocity_coupling: PressureVelocityCoupling,
        // Pick a suitable option for your flow. In theory, TVD schemes should be more expensive.
        pub momentum: MomentumDiscretization,
        // CD the only option for now
        pub diffusion: DiffusionScheme,
        // SecondOrder is more rigorous but costs ~15% performance. LinearWeighted should be
        // sufficient in most cases.
        pub pressure_interpolation: PressureInterpolation,
        // RhieChow is more rigorous and helps avoid checkerboarding but induces a ~25% performance
        // hit
        pub velocity_interpolation: VelocityInterpolation,
        // Green-Gauss cell-based only option for now
        pub gradient_reconstruction: GradientReconstructionMethods,
        // Large values are generally stable (~0.1 - 1.)
        pub momentum_relaxation: Float,
        // Must be extremely small (<<0.1)
        pub pressure_relaxation: Float,
        // Multigrid is by far the fastest option; BiCGSTAB is the most stable
        pub matrix_solver: MatrixSolverSettings,
    }

    pub struct TurbulenceModelSettings {}

    pub struct MatrixSolverSettings {
        // The type of matrix solver that will be employed. Multigrid is almost always the best
        // choice.
        pub solver_type: SolutionMethod,
        // Solver iteration count. I've found stability issues with fewer than ~50.
        // For multigrid, this is the number of iterations per level.
        pub iterations: Uint,
        // Reducing this value increases the solver stability but reduces convergence rate.
        pub relaxation: Float,
        // When the norm of the error vector becomes this proportion of its norm after 1 iteration,
        // the matrix solver will terminate. In practice, it seems this criterion is rarely met
        // except possibly with large thresholds (>>0.001).
        pub relative_convergence_threshold: Float,
        // Using a preconditioner should make the solver more stable. There is a ~20% performance
        // hit for the Jacobi preconditioner, but it is almost always worth it, allowing more
        // aggressive numerics elsewhere (e.g. fewer iterations)
        pub preconditioner: PreconditionMethod,
    }

    impl Default for NumericalSettings {
        fn default() -> Self {
            NumericalSettings {
                pressure_velocity_coupling: PressureVelocityCoupling::SIMPLE,
                momentum: MomentumDiscretization::CD1,
                diffusion: DiffusionScheme::CD,
                pressure_interpolation: PressureInterpolation::SecondOrder,
                velocity_interpolation: VelocityInterpolation::RhieChow,
                gradient_reconstruction: GradientReconstructionMethods::GreenGauss(
                    GreenGaussVariants::CellBased,
                ),
                pressure_relaxation: 0.01,
                momentum_relaxation: 0.5,
                matrix_solver: MatrixSolverSettings::default(),
            }
        }
    }

    impl Default for MatrixSolverSettings {
        fn default() -> Self {
            MatrixSolverSettings {
                solver_type: SolutionMethod::Multigrid,
                iterations: 50,
                relaxation: 0.5,
                relative_convergence_threshold: 1e-3,
                preconditioner: PreconditionMethod::Jacobi,
            }
        }
    }

    // ****** Solver numerical settings enums ******
    #[derive(Copy, Clone)]
    pub enum PressureVelocityCoupling {
        SIMPLE,
    }

    // TODO: Add flux limiters to higher-order methods
    #[derive(Copy, Clone)]
    pub enum MomentumDiscretization {
        // First-order upwind
        UD,
        // Basic CD; first-order on arbitrary grid and second-order with even spacing
        CD1,
        // CD incorporating cell gradients; second-order on arbitrary grid
        CD2,
        // Function specifies psi(r)
        TVD(fn(Float) -> Float),
    }

    const _TVD_UD: MomentumDiscretization = MomentumDiscretization::TVD(|_r| 0.);
    const _TVD_CD1: MomentumDiscretization = MomentumDiscretization::TVD(|_r| 1.);
    pub const TVD_LUD: MomentumDiscretization = MomentumDiscretization::TVD(|r| r);
    pub const TVD_QUICK: MomentumDiscretization = MomentumDiscretization::TVD(|r| (3. + r) / 4.);
    pub const TVD_UMIST: MomentumDiscretization = MomentumDiscretization::TVD(|r| {
        Float::max(
            0.,
            [2. * r, (1. + 3. * r) / 4., (3. + r) / 4., 2.]
                .iter()
                .fold(Float::INFINITY, |acc, &n| acc.min(n)),
        )
    });

    #[derive(Copy, Clone)]
    pub enum DiffusionScheme {
        CD,
    }

    #[derive(Copy, Clone)]
    pub enum PressureInterpolation {
        Linear,
        LinearWeighted,
        Standard,
        SecondOrder,
        // For internal use only! Designates a function call where it is known that no interpolation will be needed, e.g. getting flux through a boundary face
        None,
    }

    #[derive(Copy, Clone)]
    pub enum VelocityInterpolation {
        // Arithmetic mean
        Linear,
        // Distance-weighted average
        LinearWeighted,
        // Rhie-Chow is expensive! And it seems to be causing bad unphysical oscillations.
        // TODO: Rhie-Chow is causing crazy boundary effects
        RhieChow,
        // For internal use only! Designates a function call where it is known that no interpolation will be needed, e.g. getting flux through a boundary face
        None,
    }

    #[derive(Copy, Clone)]
    pub enum GreenGaussVariants {
        CellBased,
        NodeBased,
    }

    #[derive(Copy, Clone)]
    pub enum GradientReconstructionMethods {
        GreenGauss(GreenGaussVariants),
        // WARNING: Seems to be causing some unphysical pressure oscillations near the wall
        LeastSquares,
        // For internal use only! Designates a function call where it is known that gradient
        // reconstruction will not be needed
        None,
    }

    #[derive(Copy, Clone)]
    pub enum TurbulenceModel {
        None,
        StandardKEpsilon,
    }

    // TODO: GMRES, ILU
    #[derive(Copy, Clone)]
    pub enum SolutionMethod {
        GaussSeidel, // TODO: add backward sweep
        Jacobi,
        Multigrid,
        // Biconjugate gradient stabilized method. WARNING: Too many matrix solver iterations may
        // cause divergence. TODO: Investigate this behavior.
        BiCGSTAB,
    }

    #[derive(Copy, Clone)]
    pub enum PreconditionMethod {
        None,
        Jacobi,
    }

    // TODO: Do I need this?
    pub struct LinearSystem {
        pub a: CsrMatrix<Float>,
        pub b: DVector<Float>,
    }

    pub struct MultigridSettings {
        pub smoother: SolutionMethod,
    }

    #[derive(Copy, Clone)]
    pub enum RestrictionMethods {
        Injection,
        Strongest,
    }
}

pub mod numerical_types {
    use std::{
        fmt,
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub},
    };

    use nalgebra::DVector;

    pub type Int = i64;
    pub type Float = f64;
    pub type Uint = u64;

    #[derive(Copy, Clone, Debug)]
    pub struct Vector {
        pub x: Float,
        pub y: Float,
        pub z: Float,
    }

    impl Vector {
        pub fn zero() -> Vector {
            Vector {
                x: 0.,
                y: 0.,
                z: 0.,
            }
        }

        pub fn ones() -> Vector {
            Vector {
                x: 1.,
                y: 1.,
                z: 1.,
            }
        }

        pub fn dot(&self, other: &Vector) -> Float {
            self.x * other.x + self.y * other.y + self.z * other.z
        }

        pub fn cross(&self, other: &Vector) -> Vector {
            Vector {
                x: self.y * other.z - self.z * other.y,
                y: self.z * other.x - self.x * other.z,
                z: self.x * other.y - self.y * other.x,
            }
        }

        pub fn norm(&self) -> Float {
            (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
        }

        pub fn unit(&self) -> Vector {
            let len: Float = self.norm();
            Vector {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
            }
        }

        pub fn outer(&self, other: &Self) -> Tensor {
            Tensor {
                x: Vector {
                    x: self.x * other.x,
                    y: self.x * other.y,
                    z: self.x * other.z,
                },
                y: Vector {
                    x: self.y * other.x,
                    y: self.y * other.y,
                    z: self.y * other.z,
                },
                z: Vector {
                    x: self.z * other.x,
                    y: self.z * other.y,
                    z: self.z * other.z,
                },
            }
        }

        pub fn infinity_norm(&self) -> Float {
            // this is gross, must be a better way
            Float::max(
                Float::abs(self.x),
                Float::max(Float::abs(self.y), Float::abs(self.z)),
            )
        }

        pub fn approx_equals(&self, other: &Self, tol: Float) -> bool {
            (*self - *other).infinity_norm() < tol
        }

        pub fn abs(&self) -> Self {
            Vector {
                x: Float::abs(self.x),
                y: Float::abs(self.y),
                z: Float::abs(self.z),
            }
        }

        pub fn parse(s: &str) -> Self {
            let xyz = s
                .strip_prefix('(')
                .unwrap()
                .strip_suffix(')')
                .unwrap()
                .splitn(3, ", ")
                .map(|s_i| s_i.parse::<Float>().unwrap())
                .collect::<Vec<Float>>();
            Vector {
                x: xyz[0],
                y: xyz[1],
                z: xyz[2],
            }
        }

        pub fn from_dvector(v: &DVector<Float>) -> Self {
            Vector {
                x: v[0],
                y: v[1],
                z: v[2],
            }
        }

        pub fn from_vec(v: &Vec<Float>) -> Self {
            Vector {
                x: v[0],
                y: v[1],
                z: v[2],
            }
        }

        pub fn to_vec(&self) -> Vec<Float> {
            vec![self.x, self.y, self.z]
        }
    }

    impl Add<Float> for Vector {
        type Output = Self;

        fn add(self, other: Float) -> Self {
            Self {
                x: self.x + other,
                y: self.y + other,
                z: self.z + other,
            }
        }
    }

    impl Add<Self> for Vector {
        type Output = Self;

        fn add(self, other: Self) -> Self {
            Self {
                x: self.x + other.x,
                y: self.y + other.y,
                z: self.z + other.z,
            }
        }
    }

    impl AddAssign for Vector {
        fn add_assign(&mut self, other: Self) {
            *self = Self {
                x: self.x + other.x,
                y: self.y + other.y,
                z: self.z + other.z,
            }
        }
    }

    impl Sub<Float> for Vector {
        type Output = Self;

        fn sub(self, other: Float) -> Self {
            Self {
                x: self.x - other,
                y: self.y - other,
                z: self.z - other,
            }
        }
    }

    impl Sub<Self> for Vector {
        type Output = Self;

        fn sub(self, other: Self) -> Self {
            Self {
                x: self.x - other.x,
                y: self.y - other.y,
                z: self.z - other.z,
            }
        }
    }

    // impl Sum<Self> for Vector3 {
    //     fn sum<I>(iter: I) -> Self {
    //         iter.fold(Vector3::zero(), |acc, v| acc + v)
    //     }
    // }

    macro_rules! vector_div {
        ($T: ty) => {
            impl Div<$T> for Vector {
                type Output = Self;

                fn div(self, rhs: $T) -> Self {
                    Self {
                        x: self.x / (rhs as Float),
                        y: self.y / (rhs as Float),
                        z: self.z / (rhs as Float),
                    }
                }
            }
        };
    }

    vector_div!(usize);
    vector_div!(Uint);
    vector_div!(Float);

    // Element-wise division
    impl Div<Self> for Vector {
        type Output = Self;
        fn div(self, rhs: Vector) -> Self {
            Vector {
                x: self.x / rhs.x,
                y: self.y / rhs.y,
                z: self.z / rhs.z,
            }
        }
    }

    // Is this really the best way to do this...?
    macro_rules! vector_scalar_div_assign {
        ($T: ty) => {
            impl DivAssign<$T> for Vector {
                fn div_assign(&mut self, rhs: $T) {
                    *self = Self {
                        x: self.x / (rhs as Float),
                        y: self.y / (rhs as Float),
                        z: self.z / (rhs as Float),
                    }
                }
            }
        };
    }
    vector_scalar_div_assign!(usize);
    vector_scalar_div_assign!(Uint);
    vector_scalar_div_assign!(Float);

    macro_rules! vector_scalar_mul {
        ($T: ty) => {
            impl Mul<$T> for Vector {
                type Output = Self;
                fn mul(self, rhs: $T) -> Self {
                    Self {
                        x: self.x * (rhs as Float),
                        y: self.y * (rhs as Float),
                        z: self.z * (rhs as Float),
                    }
                }
            }
        };
    }

    // Element-wise multiply
    impl Mul<Self> for Vector {
        type Output = Self;
        fn mul(self, rhs: Vector) -> Self {
            Vector {
                x: self.x * rhs.x,
                y: self.y * rhs.y,
                z: self.z * rhs.z,
            }
        }
    }

    vector_scalar_mul!(usize);
    vector_scalar_mul!(Uint);
    vector_scalar_mul!(Float);

    macro_rules! vector_scalar_mul_assign {
        ($T: ty) => {
            impl MulAssign<$T> for Vector {
                fn mul_assign(&mut self, rhs: $T) {
                    *self = Self {
                        x: self.x * (rhs as Float),
                        y: self.y * (rhs as Float),
                        z: self.z * (rhs as Float),
                    }
                }
            }
        };
    }

    vector_scalar_mul_assign!(usize);
    vector_scalar_mul_assign!(i32);
    vector_scalar_mul_assign!(i64);
    vector_scalar_mul_assign!(Float);

    impl Neg for Vector {
        type Output = Self;
        fn neg(self) -> Self {
            Vector {
                x: -self.x,
                y: -self.y,
                z: -self.z,
            }
        }
    }

    impl Mul<Vector> for Float {
        type Output = Vector;
        fn mul(self, rhs: Vector) -> Vector {
            Vector {
                x: rhs.x * self,
                y: rhs.y * self,
                z: rhs.y * self,
            }
        }
    }

    impl fmt::Display for Vector {
        // TODO: Switch between regular and scientific fmt based on magnitude
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "({:.2e}, {:.2e}, {:.2e})", self.x, self.y, self.z)
        }
    }

    impl Default for Vector {
        fn default() -> Vector {
            Vector {
                x: 0.,
                y: 0.,
                z: 0.,
            }
        }
    }

    #[derive(Copy, Clone, Debug)]
    pub struct Tensor {
        pub x: Vector,
        pub y: Vector,
        pub z: Vector,
    }
    impl Tensor {
        pub fn zero() -> Self {
            Tensor {
                x: Vector::zero(),
                y: Vector::zero(),
                z: Vector::zero(),
            }
        }

        // TODO: pass by value?
        pub fn inner(&self, v: &Vector) -> Vector {
            Vector {
                x: self.x.dot(v),
                y: self.y.dot(v),
                z: self.z.dot(v),
            }
        }

        pub fn abs(&self) -> Self {
            Self {
                x: self.x.abs(),
                y: self.y.abs(),
                z: self.z.abs(),
            }
        }

        pub fn flatten(&self) -> Vec<Float> {
            let row_1 = self.x.to_vec();
            let row_2 = self.y.to_vec();
            let row_3 = self.z.to_vec();
            [row_1, row_2, row_3].concat()
        }
    }

    impl Add<Self> for Tensor {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self {
                x: self.x + rhs.x,
                y: self.y + rhs.y,
                z: self.z + rhs.z,
            }
        }
    }

    macro_rules! tensor_div {
        ($T: ty) => {
            impl Div<$T> for Tensor {
                type Output = Self;
                fn div(self, rhs: $T) -> Self {
                    Self {
                        x: self.x / (rhs as Float),
                        y: self.y / (rhs as Float),
                        z: self.z / (rhs as Float),
                    }
                }
            }
        };
    }

    tensor_div!(Uint);
    tensor_div!(usize);
    tensor_div!(Float);

    impl fmt::Display for Tensor {
        // TODO: Switch between regular and scientific fmt based on magnitude
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{}\n{}\n{}", self.x, self.y, self.z,)
        }
    }
}

pub fn vector_angle(vector_1: &numerical_types::Vector, vector_2: &numerical_types::Vector) -> numerical_types::Float {
    numerical_types::Float::acos(&vector_1.dot(&vector_2) / (vector_1.norm() * vector_2.norm()))
}

pub mod nalgebra {
    use crate::numerical_types::Float;
    use nalgebra_sparse::CsrMatrix;
    pub trait GetEntry<T> {
        fn get(&self, i: usize, j: usize) -> T;
    }

    impl GetEntry<Float> for CsrMatrix<Float> {
        fn get(&self, i: usize, j: usize) -> Float {
            match self.get_entry(i, j).unwrap() {
                nalgebra_sparse::SparseEntry::NonZero(v) => *v,
                nalgebra_sparse::SparseEntry::Zero => {
                    panic!("Tried to access CsrMatrix element that hasn't been stored yet.")
                }
            }
        }
    }
    macro_rules! dvector_zeros {
        ($n:expr) => {
            DVector::from_column_slice(&vec![0.; $n])
        };
    }
    pub(crate) use dvector_zeros;
}

pub mod discretization;
pub mod io;
pub mod linear_algebra;
pub mod mesh;
pub mod solver;

pub mod settings {
    use crate::numerical_types::*;
    use nalgebra::DVector;
    use nalgebra_sparse::CsrMatrix;
    // ****** Solver numerical settings structs ******
    // TODO: make this hierarchical
    pub struct NumericalSettings {
        pub pressure_velocity_coupling: PressureVelocityCoupling,
        pub momentum: MomentumDiscretization,
        pub diffusion: DiffusionScheme,
        // SecondOrder recommended; LinearWeighted is more stable but less accurate
        pub pressure_interpolation: PressureInterpolation,
        // LinearWeighted recommended; RhieChow is extremely unstable
        pub velocity_interpolation: VelocityInterpolation,
        pub gradient_reconstruction: GradientReconstructionMethods,
        pub pressure_relaxation: Float,
        pub momentum_relaxation: Float,
        pub matrix_solver: MatrixSolverSettings,
    }

    pub struct TurbulenceModelSettings {}

    pub struct MatrixSolverSettings {
        pub solver_type: SolutionMethod,
        // It seems like too many iterations can cause pv coupling to go crazy
        pub iterations: Uint,
        pub relaxation: Float,
        pub relative_convergence_threshold: Float,
        pub preconditioner: PreconditionMethod,
    }

    impl Default for NumericalSettings {
        fn default() -> Self {
            NumericalSettings {
                pressure_velocity_coupling: PressureVelocityCoupling::SIMPLE,
                momentum: MomentumDiscretization::CD1,
                diffusion: DiffusionScheme::CD,
                pressure_interpolation: PressureInterpolation::LinearWeighted,
                velocity_interpolation: VelocityInterpolation::LinearWeighted,
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
                relative_convergence_threshold: 0.001,
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

    pub const TVD_UD: MomentumDiscretization = MomentumDiscretization::TVD(|_r| 0.);
    pub const TVD_LUD: MomentumDiscretization = MomentumDiscretization::TVD(|r| r);
    pub const TVD_CD1: MomentumDiscretization = MomentumDiscretization::TVD(|_r| 1.);
    pub const TVD_QUICK: MomentumDiscretization = MomentumDiscretization::TVD(|r| (3. + r) / 4.);

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
    }

    #[derive(Copy, Clone)]
    pub enum VelocityInterpolation {
        LinearWeighted,
        // Rhie-Chow is expensive! And it seems to be causing bad unphysical oscillations.
        RhieChow,
    }

    #[derive(Copy, Clone)]
    pub enum GreenGaussVariants {
        CellBased,
        NodeBased,
    }

    #[derive(Copy, Clone)]
    pub enum GradientReconstructionMethods {
        GreenGauss(GreenGaussVariants),
        LeastSquares,
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
        // HashSet.
        Multigrid,
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

    pub type Int = i64;
    pub type Float = f64;
    pub type Uint = u64;

    #[derive(Copy, Clone, Debug)]
    pub struct Vector3 {
        pub x: Float,
        pub y: Float,
        pub z: Float,
    }

    impl Vector3 {
        pub fn zero() -> Vector3 {
            Vector3 {
                x: 0.,
                y: 0.,
                z: 0.,
            }
        }

        pub fn ones() -> Vector3 {
            Vector3 {
                x: 1.,
                y: 1.,
                z: 1.,
            }
        }

        pub fn dot(&self, other: &Vector3) -> Float {
            self.x * other.x + self.y * other.y + self.z * other.z
        }

        pub fn cross(&self, other: &Vector3) -> Vector3 {
            Vector3 {
                x: self.y * other.z - self.z * other.y,
                y: self.z * other.x - self.x * other.z,
                z: self.x * other.y - self.y * other.x,
            }
        }

        pub fn norm(&self) -> Float {
            (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
        }

        pub fn unit(&self) -> Vector3 {
            let len: Float = self.norm();
            Vector3 {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
            }
        }

        pub fn outer(&self, other: &Self) -> Tensor3 {
            Tensor3 {
                x: Vector3 {
                    x: self.x * other.x,
                    y: self.x * other.y,
                    z: self.x * other.z,
                },
                y: Vector3 {
                    x: self.y * other.x,
                    y: self.y * other.y,
                    z: self.y * other.z,
                },
                z: Vector3 {
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
            Vector3 {
                x: Float::abs(self.x),
                y: Float::abs(self.y),
                z: Float::abs(self.z),
            }
        }

        pub fn from_str(s: &str) -> Self {
            let xyz = s
                .strip_prefix("(")
                .unwrap()
                .strip_suffix(")")
                .unwrap()
                .splitn(3, ", ")
                .map(|s_i| s_i.parse::<Float>().unwrap())
                .collect::<Vec<Float>>();
            Vector3 {
                x: xyz[0],
                y: xyz[1],
                z: xyz[2],
            }
        }

        pub fn to_vec(&self) -> Vec<Float> {
            vec![self.x, self.y, self.z]
        }
    }

    impl Add<Float> for Vector3 {
        type Output = Self;

        fn add(self, other: Float) -> Self {
            Self {
                x: self.x + other,
                y: self.y + other,
                z: self.z + other,
            }
        }
    }

    impl Add<Self> for Vector3 {
        type Output = Self;

        fn add(self, other: Self) -> Self {
            Self {
                x: self.x + other.x,
                y: self.y + other.y,
                z: self.z + other.z,
            }
        }
    }

    impl AddAssign for Vector3 {
        fn add_assign(&mut self, other: Self) {
            *self = Self {
                x: self.x + other.x,
                y: self.y + other.y,
                z: self.z + other.z,
            }
        }
    }

    impl Sub<Float> for Vector3 {
        type Output = Self;

        fn sub(self, other: Float) -> Self {
            Self {
                x: self.x - other,
                y: self.y - other,
                z: self.z - other,
            }
        }
    }

    impl Sub<Self> for Vector3 {
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
            impl Div<$T> for Vector3 {
                type Output = Self;

                fn div(self, rhs: $T) -> Self {
                    Vector3 {
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
    impl Div<Self> for Vector3 {
        type Output = Self;
        fn div(self, rhs: Vector3) -> Self {
            Vector3 {
                x: self.x / rhs.x,
                y: self.y / rhs.y,
                z: self.z / rhs.z,
            }
        }
    }

    // Is this really the best way to do this...?
    macro_rules! vector_scalar_div_assign {
        ($T: ty) => {
            impl DivAssign<$T> for Vector3 {
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
            impl Mul<$T> for Vector3 {
                type Output = Self;
                fn mul(self, rhs: $T) -> Self {
                    Vector3 {
                        x: self.x * (rhs as Float),
                        y: self.y * (rhs as Float),
                        z: self.z * (rhs as Float),
                    }
                }
            }
        };
    }

    // Element-wise multiply
    impl Mul<Self> for Vector3 {
        type Output = Self;
        fn mul(self, rhs: Vector3) -> Self {
            Vector3 {
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
            impl MulAssign<$T> for Vector3 {
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

    impl Neg for Vector3 {
        type Output = Self;
        fn neg(self) -> Self {
            Vector3 {
                x: -self.x,
                y: -self.y,
                z: -self.z,
            }
        }
    }

    impl Mul<Vector3> for Float {
        type Output = Vector3;
        fn mul(self, rhs: Vector3) -> Vector3 {
            Vector3 {
                x: rhs.x * self,
                y: rhs.y * self,
                z: rhs.y * self,
            }
        }
    }

    impl fmt::Display for Vector3 {
        // TODO: Switch between regular and scientific fmt based on magnitude
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "({:.2e}, {:.2e}, {:.2e})", self.x, self.y, self.z)
        }
    }

    impl Default for Vector3 {
        fn default() -> Vector3 {
            Vector3 {
                x: 0.,
                y: 0.,
                z: 0.,
            }
        }
    }

    #[derive(Copy, Clone, Debug)]
    pub struct Tensor3 {
        pub x: Vector3,
        pub y: Vector3,
        pub z: Vector3,
    }
    impl Tensor3 {
        pub fn zero() -> Self {
            Tensor3 {
                x: Vector3::zero(),
                y: Vector3::zero(),
                z: Vector3::zero(),
            }
        }

        // TODO: pass by value?
        pub fn inner(&self, v: &Vector3) -> Vector3 {
            Vector3 {
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
            vec![row_1, row_2, row_3].concat()
        }
    }

    impl Add<Self> for Tensor3 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self {
                x: self.x.clone() + rhs.x.clone(),
                y: self.y.clone() + rhs.y.clone(),
                z: self.z.clone() + rhs.z.clone(),
            }
        }
    }

    macro_rules! tensor_div {
        ($T: ty) => {
            impl Div<$T> for Tensor3 {
                type Output = Self;
                fn div(self, rhs: $T) -> Self {
                    Tensor3 {
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

    impl fmt::Display for Tensor3 {
        // TODO: Switch between regular and scientific fmt based on magnitude
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{}\n{}\n{}", self.x, self.y, self.z,)
        }
    }
}

pub mod nalgebra {
    use crate::numerical_types::Float;
    use nalgebra_sparse::CsrMatrix;
    pub trait GetEntry<T> {
        fn get(&self, i: usize, j: usize) -> T;
    }

    impl GetEntry<Float> for CsrMatrix<Float> {
        fn get(&self, i: usize, j: usize) -> Float {
            self.get_entry(i, j).unwrap().into_value()
        }
    }
    macro_rules! dvector_zeros {
        ($n:expr) => {
            DVector::from_column_slice(&vec![0.; $n])
        };
    }
    pub(crate) use dvector_zeros;
}

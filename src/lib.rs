#![allow(dead_code)]
#![allow(unused)]

// I need to store a list of cells
// Each cell has an associated list of faces & nodes
// Each cell has scalars & vectors

pub mod io;
pub mod mesh;
pub mod solver;

pub mod common {
    use std::{
        fmt,
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub},
        iter::Sum
    };

    pub type Int = i32;
    pub type Float = f32;
    pub type Uint = u32;

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
            (self.x.powf(2 as Float) + self.y.powf(2 as Float) + self.z.powf(2 as Float)).sqrt()
        }

        pub fn unit(&self) -> Vector3 {
            let len: Float = self.norm();
            Vector3 {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
            }
        }

        pub fn outer(&self, other: &Self) -> Tensor {
            Tensor {
                x: Vector3 {
                    x: self.x * other.x,
                    y: self.x * other.y,
                    z: self.z * other.z,
                },
                y: Vector3 {
                    x: self.y * other.x,
                    y: self.y * other.y,
                    z: self.z * other.z,
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

    pub struct Tensor {
        pub x: Vector3,
        pub y: Vector3,
        pub z: Vector3,
    }
    impl Tensor {
        pub fn zero() -> Self {
            Tensor {
                x: Vector3::zero(),
                y: Vector3::zero(),
                z: Vector3::zero(),
            }
        }

        // TODO: pass by value?
        pub fn dot(&self, vector: &Vector3) -> Vector3 {
            Vector3 {
                x: self.x.dot(&vector),
                y: self.x.dot(&vector),
                z: self.z.dot(&vector),
            }
        }
    }

    impl Add<Self> for Tensor {
        type Output = Self;
        fn add(self, rhs: Tensor) -> Tensor {
            Tensor {
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
                    Tensor {
                        x: self.x * (rhs as Float),
                        y: self.y * (rhs as Float),
                        z: self.z * (rhs as Float),
                    }
                }
            }
        };
    }

    tensor_div!(Uint);
    tensor_div!(Float);
}

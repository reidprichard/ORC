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
        ops::{Add, AddAssign, DivAssign, Neg, Sub},
    };

    pub type Int = i32;
    pub type Float = f32;
    pub type Uint = u32;

    #[derive(Copy, Clone)]
    pub struct Vector {
        pub x: Float,
        pub y: Float,
        pub z: Float,
    }
    impl Vector {
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
            (self.x.powf(2 as Float) + self.y.powf(2 as Float) + self.z.powf(2 as Float)).sqrt()
        }
        pub fn unit(&self) -> Vector {
            let len: Float = self.norm();
            Vector {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
            }
        }
    }

    impl Add for Vector {
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

    impl Sub for Vector {
        type Output = Self;

        fn sub(self, other: Self) -> Self {
            Self {
                x: self.x - other.x,
                y: self.y - other.y,
                z: self.z - other.z,
            }
        }
    }

    // Is this really the best way to do this...?
    macro_rules! div_assign {
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
    div_assign!(usize);
    div_assign!(Uint);
    div_assign!(Float);

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
}

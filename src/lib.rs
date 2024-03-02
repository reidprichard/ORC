#![allow(dead_code)]
#![allow(unused_imports)]

// I need to store a list of cells
// Each cell has an associated list of faces & nodes
// Each cell has scalars & vectors

pub mod common {
    use std::{fmt, ops::{Add, AddAssign, DivAssign}};

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
        fn scalar_divf(&self, divisor: Float) -> Vector {
            Vector {
                x: self.x / divisor,
                y: self.y / divisor,
                z: self.z / divisor,
            }
        }

        fn dot(&self, other: &Vector) -> Float {
            self.x * other.x + self.y * other.y + self.z * other.z
        }

        fn norm(&self) -> Float {
            (self.x.powf(2 as Float) + self.y.powf(2 as Float) + self.z.powf(2 as Float)).sqrt()
        }
        fn unit(&self) -> Vector {
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

    impl fmt::Display for Vector {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "({}, {}, {})", self.x, self.y, self.z)
        }
    }
}

pub mod mesh {
    use std::collections::HashMap;

    use crate::common::*;

    pub enum GreenGaussVariants {
        CellBased,
        NodeBased,
    }

    pub enum GradientReconstructionMethods {
        GreenGauss(GreenGaussVariants),
        LeastSquares,
    }

    pub struct Cell {
        pub face_indices: Vec<Uint>,
        pub centroid: Vector,
        pub velocity: Vector,
        pub pressure: Float,
    }
    impl Cell {}
    impl Default for Cell {
        fn default() -> Cell {
            Cell {
                face_indices: Vec::new(),
                centroid: Vector {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                velocity: Vector {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                pressure: 0.,
            }
        }
    }

    pub struct Node {
        pub cell_indices: Vec<Uint>,
        pub position: Vector,
        pub velocity: Vector,
        pub pressure: Float,
    }
    impl Node {
        fn interpolate_velocity(&mut self) {}

        fn interpolate_pressure(&mut self, method: GreenGaussVariants) {
            match method {
                _ => (),
            }
        }
    }
    impl Default for Node {
        fn default() -> Node {
            Node {
                cell_indices: Vec::new(),
                position: Vector {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                velocity: Vector {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                pressure: 0.,
            }
        }
    }

    pub struct Face {
        pub cell_indices: Vec<Uint>,
        pub node_indices: Vec<Uint>,
        pub centroid: Vector,
        pub velocity: Vector,
        pub pressure: Float,
    }
    impl Face {}
    impl Default for Face {
        fn default() -> Face {
            Face {
                cell_indices: Vec::new(),
                node_indices: Vec::new(),
                centroid: Vector {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                velocity: Vector {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                pressure: 0.,
            }
        }
    }

    pub struct Mesh {
        pub nodes: HashMap<Uint, Node>,
        pub faces: HashMap<Uint, Face>,
        pub cells: HashMap<Uint, Cell>,
    }
    impl Mesh {}
}

#![allow(dead_code)]
#![allow(unused_imports)]

// I need to store a list of cells
// Each cell has an associated list of faces & nodes
// Each cell has scalars & vectors

pub mod common {
    pub type Int = i32;
    pub type Float = f32;
    pub type Uint = u32;
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
}


pub mod mesh {
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

    pub struct Face {
        pub cell_indices: Vec<Uint>,
        pub node_indices: Vec<Uint>,
        pub centroid: Vector,
        pub pressure: Float,
        pub velocity: Vector,
    }
    impl Face {
    }

    pub struct Mesh {
        pub nodes: Vec<Node>,
        pub faces: Vec<Face>,
        pub cells: Vec<Cell>,
    }
    impl Mesh {
    }
}

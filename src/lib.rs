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

    // struct VectorGradient {
    //
    // }

    pub trait Cell {
        fn calculate_pressure_gradient(
            &mut self,
            reconstruction_method: GradientReconstructionMethods,
        );
        // Might be cleaner to just have one get() and one set() with u,v,w,p enum values
        fn get_u(&self) -> Float;
        fn get_v(&self) -> Float;
        fn get_w(&self) -> Float;
        fn get_p(&self) -> Float;
        fn set_u(&mut self, u: Float);
        fn set_v(&mut self, v: Float);
        fn set_w(&mut self, w: Float);
        fn set_p(&mut self, p: Float);
    }

    macro_rules! cell_type {
        ($name: ident, $node_count: literal, $face_count: literal) => {
            struct $name {
                pub nodes: [&'static Node; $node_count],
                pub faces: [&'static Face; $face_count],
                pub centroid: Vector,
                pub velocity: Vector,
                pub pressure: Float,
                pub pressure_gradient: Vector,
            }

            impl Cell for $name {
                fn get_u(&self) -> Float { self.velocity.x }
                fn get_v(&self) -> Float { self.velocity.y }
                fn get_w(&self) -> Float { self.velocity.z }
                fn get_p(&self) -> Float { self.pressure }
                fn set_u(&mut self, u: Float) {self.velocity.x = u; }
                fn set_v(&mut self, v: Float) {self.velocity.y = v; }
                fn set_w(&mut self, w: Float) {self.velocity.z = w; }
                fn set_p(&mut self, p: Float) {self.pressure = p; }
                fn calculate_pressure_gradient(
                    &mut self,
                    reconstruction_method: GradientReconstructionMethods,
                ) {
                    use GradientReconstructionMethods::*;
                    use GreenGaussVariants::*;
                    match reconstruction_method {
                        GreenGauss(variant) => {
                            // 1. Iterate over faces in cell
                            // 2. For each face, calculate face value
                            // 3. Perform Green Gauss
                            match variant {
                                CellBased => {}
                                NodeBased => {}
                            }
                            self.pressure_gradient = Vector {
                                x: 0 as Float,
                                y: 0 as Float,
                                z: 0 as Float,
                            };
                        }
                        _ => {}
                    }
                    // iterate over faces and interpolate face values
                    // perform green gauss
                    //
                }
            }
        };
    }

    cell_type!(Tet, 4, 4);
    cell_type!(Pyramid, 5, 5);
    cell_type!(Prism, 6, 5);
    cell_type!(Hex, 8, 6);
    cell_type!(Poly, 0, 0); // Unimplemented

    pub struct Node {
        cells: Vec<Box<dyn Cell>>,
        position: Vector,
        pressure: Float,
        velocity: Vector,
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
        cells: Vec<Box<dyn Cell>>,
        centroid: Vector,
        pressure: Float,
        velocity: Vector,
    }
    impl Face {
    }

    pub struct Mesh {
        nodes: Vec<Node>,
        faces: Vec<Face>,
        cells: Vec<Box<dyn Cell>>,
    }
}

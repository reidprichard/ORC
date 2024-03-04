#![allow(dead_code)]
#![allow(unused_imports)]

// I need to store a list of cells
// Each cell has an associated list of faces & nodes
// Each cell has scalars & vectors

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

pub mod mesh {
    use crate::common::*;
    use core::fmt;
    use std::collections::HashMap;

    pub fn get_cell_zone_types() -> HashMap<Uint, &'static str> {
        HashMap::from([(0, "dead zone"), (1, "fluid zone"), (17, "solid zone")])
    }

    // TODO: Move BCs somewhere more suitable
    pub enum BoundaryConditionTypes {
        Interior,
        Wall,
        PressureInlet,
        PressureOutlet,
        Symmetry,
        PeriodicShadow,
        PressureFarField,
        VelocityInlet,
        Periodic,
        PorousJump,
        MassFlowInlet,
        Interface,
        Parent,
        Outflow,
        Axis,
    }
    impl fmt::Display for BoundaryConditionTypes {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "{}",
                match &self {
                    BoundaryConditionTypes::Interior => "Interior",
                    BoundaryConditionTypes::Wall => "Wall",
                    BoundaryConditionTypes::PressureInlet => "PressureInlet",
                    BoundaryConditionTypes::PressureOutlet => "PressureOutlet",
                    BoundaryConditionTypes::Symmetry => "Symmetry",
                    BoundaryConditionTypes::PeriodicShadow => "PeriodicShadow",
                    BoundaryConditionTypes::PressureFarField => "PressureFarField",
                    BoundaryConditionTypes::VelocityInlet => "VelocityInlet",
                    BoundaryConditionTypes::Periodic => "Periodic",
                    BoundaryConditionTypes::PorousJump => "PorousJump",
                    BoundaryConditionTypes::MassFlowInlet => "MassFlowInlet",
                    BoundaryConditionTypes::Interface => "Interface",
                    BoundaryConditionTypes::Parent => "Parent",
                    BoundaryConditionTypes::Outflow => "Outflow",
                    BoundaryConditionTypes::Axis => "Axis",
                }
            )
        }
    }
    // (2, "interior"),
    // (3, "wall"),
    // (4, "pressure-inlet, inlet-vent, intake-fan"),
    // (5, "pressure-outlet, exhaust-fan, outlet-vent"),
    // (7, "symmetry"),
    // (8, "periodic-shadow"),
    // (9, "pressure-far-field"),
    // (10, "velocity-inlet"),
    // (12, "periodic"),
    // (14, "fan, porous-jump, radiator"),
    // (20, "mass-flow-inlet"),
    // (24, "interface"),
    // (31, "parent (hanging node)"),
    // (36, "outflow"),
    // (37, "axis"),

    pub fn get_boundary_condition_types() -> HashMap<Uint, BoundaryConditionTypes> {
        HashMap::from([
            (2, BoundaryConditionTypes::Interior),
            (3, BoundaryConditionTypes::Wall),
            (4, BoundaryConditionTypes::PressureInlet),
            (5, BoundaryConditionTypes::PressureOutlet),
            (7, BoundaryConditionTypes::Symmetry),
            (8, BoundaryConditionTypes::PeriodicShadow),
            (9, BoundaryConditionTypes::PressureFarField),
            (10, BoundaryConditionTypes::VelocityInlet),
            (12, BoundaryConditionTypes::Periodic),
            (14, BoundaryConditionTypes::PorousJump),
            (20, BoundaryConditionTypes::MassFlowInlet),
            (24, BoundaryConditionTypes::Interface),
            (31, BoundaryConditionTypes::Parent),
            (36, BoundaryConditionTypes::Outflow),
            (37, BoundaryConditionTypes::Axis),
        ])
    }

    // pub fn get_face_types() -> HashMap<Uint, &'static str> {
    // (0, "?"),
    // (2, "linear face (2 nodes)"),
    // (3, "triangular face (3 nodes)"),
    // (4, "quadrilateral face (4 nodes)"),
    // (5, "polygonal (N nodes)"),
    // }

    pub enum GreenGaussVariants {
        CellBased,
        NodeBased,
    }

    pub enum GradientReconstructionMethods {
        GreenGauss(GreenGaussVariants),
        LeastSquares,
    }

    pub struct Cell {
        pub zone_number: Uint,
        pub face_indices: Vec<Uint>,
        pub volume: Float,
        pub centroid: Vector,
        pub velocity: Vector,
        pub pressure: Float,
    }
    impl Cell {}
    impl Default for Cell {
        fn default() -> Cell {
            Cell {
                zone_number: 0,
                face_indices: Vec::new(),
                volume: 0.,
                centroid: Vector::default(),
                velocity: Vector::default(),
                pressure: 0.,
            }
        }
    }

    // #[derive(Copy, Clone)]
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
                position: Vector::default(),
                velocity: Vector::default(),
                pressure: 0.,
            }
        }
    }

    pub struct Face {
        pub zone_number: Uint,
        pub cell_indices: Vec<Uint>,
        pub node_indices: Vec<Uint>,
        pub area: Float,
        pub centroid: Vector,
        pub normal: Vector,
        pub velocity: Vector,
        pub pressure: Float,
    }
    impl Face {}
    impl Default for Face {
        fn default() -> Face {
            Face {
                zone_number: 0,
                cell_indices: Vec::new(),
                node_indices: Vec::new(),
                area: 0.,
                centroid: Vector::default(),
                normal: Vector::default(), // points toward cell 0!
                velocity: Vector::default(),
                pressure: 0.,
            }
        }
    }

    pub struct Mesh {
        pub nodes: HashMap<Uint, Node>,
        pub faces: HashMap<Uint, Face>,
        pub cells: HashMap<Uint, Cell>,
        pub face_zones: HashMap<Uint, Uint>,
        pub cell_zones: HashMap<Uint, Uint>,
    }
    impl Mesh {}
}

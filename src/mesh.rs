use crate::common::*;
use core::fmt;
use std::collections::HashMap;

pub fn get_cell_zone_types() -> HashMap<Uint, &'static str> {
    HashMap::from([(0, "dead zone"), (1, "fluid zone"), (17, "solid zone")])
}

// TODO: Allow BC vector for velocity BC non-normal to boundary
pub struct FaceZone {
    pub zone_type: BoundaryConditionTypes,
    pub scalar_value: Float,
    pub vector_value: Vector,
    pub name: String,
}

pub struct CellZone {
    pub zone_type: Uint,
    // pub name: String,
}

// TODO: Move BCs somewhere more suitable
#[derive(Debug)]
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

macro_rules! bc_types_from {
    ($T: ty) => {
        impl TryFrom<$T> for BoundaryConditionTypes {
            type Error = &'static str;

            fn try_from(value: $T) -> Result<Self, Self::Error> {
                match value {
                    2 => Ok(BoundaryConditionTypes::Interior),
                    3 => Ok(BoundaryConditionTypes::Wall),
                    4 => Ok(BoundaryConditionTypes::PressureInlet),
                    5 => Ok(BoundaryConditionTypes::PressureOutlet),
                    7 => Ok(BoundaryConditionTypes::Symmetry),
                    8 => Ok(BoundaryConditionTypes::PeriodicShadow),
                    9 => Ok(BoundaryConditionTypes::PressureFarField),
                    10 => Ok(BoundaryConditionTypes::VelocityInlet),
                    12 => Ok(BoundaryConditionTypes::Periodic),
                    14 => Ok(BoundaryConditionTypes::PorousJump),
                    20 => Ok(BoundaryConditionTypes::MassFlowInlet),
                    24 => Ok(BoundaryConditionTypes::Interface),
                    31 => Ok(BoundaryConditionTypes::Parent),
                    36 => Ok(BoundaryConditionTypes::Outflow),
                    37 => Ok(BoundaryConditionTypes::Axis),
                    _ => Err("Invalid boundary condition value."),
                }
            }
        }
    };
}

bc_types_from!(u32);
bc_types_from!(u16);
bc_types_from!(u8);

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
    pub face_numbers: Vec<Uint>,
    pub volume: Float,
    pub centroid: Vector,
    pub velocity: Vector,
    pub pressure: Float,
}
impl Default for Cell {
    fn default() -> Cell {
        Cell {
            zone_number: 0,
            face_numbers: Vec::new(),
            volume: 0.,
            centroid: Vector::zero(),
            velocity: Vector::zero(),
            pressure: 0.,
        }
    }
}

// #[derive(Copy, Clone)]
pub struct Node {
    pub cell_numbers: Vec<Uint>,
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
            cell_numbers: Vec::new(),
            position: Vector::zero(),
            velocity: Vector::zero(),
            pressure: 0.,
        }
    }
}

pub struct Face {
    pub zone: Uint,
    // TODO: Make this an array
    pub cell_numbers: Vec<Uint>,
    pub node_numbers: Vec<Uint>,
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
            zone: 0,
            cell_numbers: Vec::new(),
            node_numbers: Vec::new(),
            area: 0.,
            centroid: Vector::zero(),
            normal: Vector::zero(), // points toward cell 0!
            velocity: Vector::zero(),
            pressure: 0.,
        }
    }
}

pub struct Mesh {
    pub nodes: HashMap<Uint, Node>,
    pub faces: HashMap<Uint, Face>,
    pub cells: HashMap<Uint, Cell>,
    pub face_zones: HashMap<Uint, FaceZone>,
    pub cell_zones: HashMap<Uint, CellZone>,
}
impl Mesh {
    pub fn get_face_zone(&mut self, zone_name: &str) -> &mut FaceZone {
        self.face_zones
            .iter_mut()
            .map(|(zone_num, fz)| fz)
            .filter(|fz| fz.name == zone_name)
            .next()
            .expect(&format!("face zone '{zone_name}' exists in mesh"))
    }
    pub fn calculate_velocity_gradient(&self, cell_number: Uint) -> Tensor {
        let cell = self.cells.get(&cell_number).expect("valid cell number");
        cell.face_numbers.iter().map(|face_number| {
            let face = &self.faces[face_number];
            let mut neighbor_count = 0;
            face
                .cell_numbers
                .iter()
                .filter(|c| **c != 0)
                .inspect(|_| neighbor_count += 1)
                .map(|c| self.cells[c].velocity)
                .fold(Vector::zero(), |acc, v| acc + v)
                .outer(&face.centroid) / neighbor_count
        }).fold(Tensor::zero(), |acc, t| acc + t)
    }
}

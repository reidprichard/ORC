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

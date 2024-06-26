use crate::numerical_types::*;
use ahash::RandomState;
use core::fmt;
use std::collections::HashMap;
// TODO: Can I use aHash's HashMap?

pub fn get_cell_zone_types() -> HashMap<Uint, &'static str> {
    HashMap::from([(0, "dead zone"), (1, "fluid zone"), (17, "solid zone")])
}

// TODO: Allow BC vector for velocity BC non-normal to boundary
pub struct FaceZone {
    pub zone_type: FaceConditionTypes,
    pub scalar_value: Float,
    pub vector_value: Vector,
    pub name: String,
}

pub struct CellZone {
    pub zone_type: Uint,
    // pub name: String,
}

// TODO: Move BCs somewhere more suitable
#[derive(Clone, Copy, Debug)]
pub enum FaceConditionTypes {
    Interior,
    Wall,
    PressureInlet,
    PressureOutlet, // TODO: Consider merging PressureInlet/PressureOutlet
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
        impl TryFrom<$T> for FaceConditionTypes {
            type Error = &'static str;

            fn try_from(value: $T) -> Result<Self, Self::Error> {
                match value {
                    2 => Ok(FaceConditionTypes::Interior),
                    3 => Ok(FaceConditionTypes::Wall),
                    4 => Ok(FaceConditionTypes::PressureInlet),
                    5 => Ok(FaceConditionTypes::PressureOutlet),
                    7 => Ok(FaceConditionTypes::Symmetry),
                    8 => Ok(FaceConditionTypes::PeriodicShadow),
                    9 => Ok(FaceConditionTypes::PressureFarField),
                    10 => Ok(FaceConditionTypes::VelocityInlet),
                    12 => Ok(FaceConditionTypes::Periodic),
                    14 => Ok(FaceConditionTypes::PorousJump),
                    20 => Ok(FaceConditionTypes::MassFlowInlet),
                    24 => Ok(FaceConditionTypes::Interface),
                    31 => Ok(FaceConditionTypes::Parent),
                    36 => Ok(FaceConditionTypes::Outflow),
                    37 => Ok(FaceConditionTypes::Axis),
                    _ => Err("Invalid boundary condition value."),
                }
            }
        }
    };
}

bc_types_from!(u64);
bc_types_from!(u32);
bc_types_from!(u16);
bc_types_from!(u8);

impl fmt::Display for FaceConditionTypes {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                FaceConditionTypes::Interior => "Interior",
                FaceConditionTypes::Wall => "Wall",
                FaceConditionTypes::PressureInlet => "PressureInlet",
                FaceConditionTypes::PressureOutlet => "PressureOutlet",
                FaceConditionTypes::Symmetry => "Symmetry",
                FaceConditionTypes::PeriodicShadow => "PeriodicShadow",
                FaceConditionTypes::PressureFarField => "PressureFarField",
                FaceConditionTypes::VelocityInlet => "VelocityInlet",
                FaceConditionTypes::Periodic => "Periodic",
                FaceConditionTypes::PorousJump => "PorousJump",
                FaceConditionTypes::MassFlowInlet => "MassFlowInlet",
                FaceConditionTypes::Interface => "Interface",
                FaceConditionTypes::Parent => "Parent",
                FaceConditionTypes::Outflow => "Outflow",
                FaceConditionTypes::Axis => "Axis",
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

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: Vector,
}
impl Default for Vertex {
    fn default() -> Vertex {
        Vertex {
            position: Vector::zero(),
        }
    }
}

#[derive(Clone)]
pub struct Face {
    pub zone: Uint,
    // TODO: Make this an array
    pub cell_indices: Vec<usize>,
    pub node_indices: Vec<usize>,
    pub area: Float,
    pub centroid: Vector,
    // TODO: rename to unit_normal?
    pub normal: Vector,
}
impl Default for Face {
    fn default() -> Face {
        Face {
            zone: 0,
            cell_indices: Vec::new(),
            node_indices: Vec::new(),
            area: 0.,
            centroid: Vector::zero(),
            normal: Vector::zero(), // points toward cell 0!
        }
    }
}

#[derive(Clone)]
pub struct Cell {
    pub zone_number: Uint,
    pub face_indices: Vec<usize>,
    pub volume: Float,
    pub centroid: Vector,
}
impl Default for Cell {
    fn default() -> Cell {
        Cell {
            zone_number: 0,
            face_indices: Vec::new(),
            volume: 0.,
            centroid: Vector::zero(),
        }
    }
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub faces: Vec<Face>,
    pub cells: Vec<Cell>,
    pub face_zones: HashMap<Uint, FaceZone, RandomState>,
    pub cell_zones: HashMap<Uint, CellZone, RandomState>,
}
impl Mesh {
    pub fn get_face_zone(&mut self, zone_name: &str) -> &mut FaceZone {
        self.face_zones
            .iter_mut()
            .map(|(_zone_num, fz)| fz)
            .find(|fz| fz.name == zone_name)
            .unwrap_or_else(|| panic!("face zone '{zone_name}' should exist in mesh"))
    }
    // pub fn calculate_velocity_gradient(&self, cell_number: usize) -> Tensor {
    //     let cell = self.cells.get(&cell_number).expect("valid cell number");
    //     cell.face_numbers
    //         .iter()
    //         .map(|face_number| {
    //             let face = &self.faces[face_number];
    //             let mut neighbor_count = 0;
    //             face.cell_numbers
    //                 .iter()
    //                 .filter(|c| **c != 0)
    //                 .inspect(|_| neighbor_count += 1)
    //                 .map(|c| self.cells[c].velocity)
    //                 .fold(Vector::zero(), |acc, v| acc + v)
    //                 .outer(&face.centroid)
    //                 / neighbor_count
    //         })
    //         .fold(Tensor::zero(), |acc, t| acc + t)
    // }
}

pub fn get_outward_face_normal(face: &Face, cell_index: usize) -> Vector {
    if cell_index == face.cell_indices[0] {
        face.normal
    } else {
        -face.normal
    }
}

pub fn get_inward_face_normal(face: &Face, cell_index: usize) -> Vector {
    get_outward_face_normal(face, cell_index) * -1.
}

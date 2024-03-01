#![allow(dead_code)]
#![allow(unused)]

macro_rules! skip_zone_zero {
    ($label:tt, $line_blocks:expr) => {
        if $line_blocks[1] == "(0" {
            break $label;
        }
    };
}

pub mod solver_io {
    use itertools::Itertools;
    use orc::common::*;
    use orc::mesh;
    use orc::mesh::Cell;
    use orc::mesh::Face;
    use orc::mesh::Mesh;
    use orc::mesh::Node;
    use regex::Regex;
    use std::fs;
    use std::fs::File;
    use std::io::{self, BufRead};

    pub fn read_mesh(mesh_path: &str) -> mesh::Mesh {
        fn read_mesh_lines(filename: &str) -> io::Result<io::Lines<io::BufReader<File>>> {
            // Example had filename generic type P and ... where P: AsRef<Path> ... in signature but I
            // don't understand it ¯\_(ツ)_/¯
            let file = File::open(filename)?;
            Ok(io::BufReader::new(file).lines())
        }

        // cells: (12 (zone-id first-index last-index type      element-type))
        // faces: (13 (zone-id first-index last-index bc-type   face-type))
        // nodes: (10 (zone-id first-index last-index type      ND))
        fn read_section_header_common(header_line: &str) -> Vec<Uint> {
            // is returning a tuple best?
            let re = Regex::new(r"([0-9a-z]+)").expect("valid regex");
            // let re = Regex::new(r"\(\d+ \(([0-9a-z]+) ([0-9a-z]+) ([0-9a-z]+) ([0-9a-z]+)")
            //     .expect("valid regex");

            let mut items: Vec<Uint> = Vec::new();
            for (_, [s]) in re.captures_iter(header_line).map(|c| c.extract()) {
                items.push(Uint::from_str_radix(s, 16).expect("valid hex"));
            }
            items
        }

        let mut nodes: Vec<Node> = Vec::new();
        let mut faces: Vec<Face> = Vec::new();
        let mut cells: Vec<Cell> = Vec::new();

        let mut node_indices: Vec<Uint> = Vec::new();
        let mut face_indices: Vec<Uint> = Vec::new();
        let mut cell_indices: Vec<Uint> = Vec::new();

        // let mesh = Mesh {
        //     nodes,
        //     faces,
        //     cells,
        // };

        struct CellType {
            id: i8,
            node_count: i16,
            face_count: i16,
        }
        let tet = CellType {
            id: 2,
            node_count: 4,
            face_count: 4,
        };
        let hex = CellType {
            id: 3,
            node_count: 8,
            face_count: 6,
        };

        if let Ok(mesh_file_lines) = read_mesh_lines(mesh_path) {
            let mut lines = mesh_file_lines.flatten();
            let mut section_header_line = lines.next().expect("mesh is at least one line long");
            loop {
                let section_header_blocks: Vec<&str> =
                    section_header_line.split_ascii_whitespace().collect();
                match section_header_blocks[0] {
                    "(0" => (), // comment
                    "(1" => (), // header
                    "(2" => (), // dimensions
                    "(10" => 'read_nodes: {
                        skip_zone_zero!('read_nodes, section_header_blocks);
                        println!("{section_header_line}");
                        let items = read_section_header_common(&section_header_line);
                        let (_, zone_id, start_index, end_index, node_type, dims) = items
                            .iter()
                            .map(|n| *n)
                            .collect_tuple()
                            .expect("correct number of items");
                        println!("Beginning reading nodes from {start_index} to {end_index}.");

                        let mut current_line = lines.next().expect("node section has contents");
                        'node_loop: loop {
                            let line_blocks: Vec<&str> =
                                current_line.split_ascii_whitespace().collect();
                            if line_blocks.len() != 3 {
                                break 'node_loop;
                            }
                            println!("{section_header_line}");
                            let x = line_blocks[0].parse::<Float>();
                            let y = line_blocks[1].parse::<Float>();
                            let z = line_blocks[2].parse::<Float>();
                            if x.is_ok() && y.is_ok() && z.is_ok() {
                                nodes.push(Node {
                                    cell_indices: Vec::new(),
                                    position: Vector {
                                        x: x.unwrap(),
                                        y: y.unwrap(),
                                        z: z.unwrap(),
                                    },
                                    velocity: Vector {
                                        x: 0.,
                                        y: 0.,
                                        z: 0.,
                                    },
                                    pressure: 0.,
                                });
                            } else {
                                break 'node_loop;
                            }
                            match lines.next() {
                                Some(line_contents) => current_line = line_contents,
                                None => break 'node_loop,
                            }
                        }
                    }
                    "(18" => {
                        // skip_zone_zero!(section_header_blocks);
                        println!("Beginning reading shadow faces."); // periodic shadow faces
                    }
                    "(12" => 'read_cells: {
                        // skip_zone_zero!('read_cells, section_header_blocks);
                        // let items = read_section_header_common(&section_header_line);
                        // let (_, zone_id, start_index, end_index, fluid_zone_type, element_type) =
                        //     items
                        //         .iter()
                        //         .map(|n| *n)
                        //         .collect_tuple()
                        //         .expect("correct number of items");
                        // 'cell_loop: loop {
                        //     println!("{current_line}");
                        //     let line_blocks: Vec<&str> =
                        //         current_line.split_ascii_whitespace().collect();
                        //     if line_blocks.len() < 2 {
                        //         break;
                        //     }
                        //     let node_count = line_blocks.len() - 2;
                        //     if face_type != 0
                        //         && face_type != 5
                        //         && face_type != node_count.try_into().unwrap()
                        //     {
                        //         break 'face_loop;
                        //     }
                        //     println!("{section_header_line}");
                        //
                        //     faces.push(Face {
                        //         cell_indices: line_blocks[node_count..]
                        //             .into_iter()
                        //             .map(|cell_id| Uint::from_str_radix(cell_id, 16))
                        //             .flatten()
                        //             .collect(),
                        //         node_indices: line_blocks[..node_count]
                        //             .into_iter()
                        //             .map(|node_id| Uint::from_str_radix(node_id, 16))
                        //             .flatten()
                        //             .collect(),
                        //         centroid: Vector {
                        //             x: 0.,
                        //             y: 0.,
                        //             z: 0.,
                        //         },
                        //         velocity: Vector {
                        //             x: 0.,
                        //             y: 0.,
                        //             z: 0.,
                        //         },
                        //         pressure: 0.,
                        //     });
                        //     match lines.next() {
                        //         Some(line_contents) => current_line = line_contents,
                        //         None => break 'face_loop,
                        //     }
                        //
                        // }
                    }
                    "(13" => 'read_faces: {
                        skip_zone_zero!('read_faces, section_header_blocks);
                        let items = read_section_header_common(&section_header_line);
                        let (_, zone_id, start_index, end_index, boundary_type, face_type) = items
                            .iter()
                            .map(|n| *n)
                            .collect_tuple()
                            .expect("correct number of items");
                        // TODO: Add error checking to not allow unsupported BC types
                        // 2: interior
                        // 3: wall
                        // 4: pressure-inlet, inlet-vent, intake-fan
                        // 5: pressure-outlet, exhaust-fan, outlet-vent
                        // 7: symmetry
                        // 8: periodic-shadow
                        // 9: pressure-far-field
                        // 10: velocity-inlet
                        // 12: periodic
                        // 14: fan, porous-jump, radiator
                        // 20: mass-flow-inlet
                        // 24: interface
                        // 31: parent (hanging node)
                        // 36: outflow
                        // 37: axis
                        // 0: ?
                        // 2: linear face (2 nodes)
                        // 3: triangular face (3 nodes)
                        // 4: quadrilateral face (4 nodes)
                        // 5: polygonal (N nodes)
                        // Is using `usize` bad here?
                        println!("Beginning reading faces."); // cells

                        let mut current_line = lines.next().expect("face section has contents");
                        'face_loop: loop {
                            println!("{current_line}");
                            let line_blocks: Vec<&str> =
                                current_line.split_ascii_whitespace().collect();
                            if line_blocks.len() < 2 {
                                break;
                            }
                            let node_count = line_blocks.len() - 2;
                            if face_type != 0
                                && face_type != 5
                                && face_type != node_count.try_into().unwrap()
                            {
                                break 'face_loop;
                            }
                            println!("{section_header_line}");

                            faces.push(Face {
                                cell_indices: line_blocks[node_count..]
                                    .into_iter()
                                    .map(|cell_id| Uint::from_str_radix(cell_id, 16))
                                    .flatten()
                                    .collect(),
                                node_indices: line_blocks[..node_count]
                                    .into_iter()
                                    .map(|node_id| Uint::from_str_radix(node_id, 16))
                                    .flatten()
                                    .collect(),
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
                            });
                            match lines.next() {
                                Some(line_contents) => current_line = line_contents,
                                None => break 'face_loop,
                            }
                        }
                    }
                    "(58" => (), // cell tree
                    "(59" => (), // face tree
                    "(61" => (), // interface face parents for nonconformal
                    _ => (),
                }
                match lines.next() {
                    Some(line_content) => section_header_line = line_content,
                    None => break,
                }
            }
        }

        

        Mesh {
            nodes,
            faces,
            cells,
        }
    }

    pub fn read_data() {}

    pub fn read_settings() {}

    pub fn write_mesh() {}

    pub fn write_data() {}

    pub fn write_settings() {}
}

pub mod solver {
    use orc::common::*;
    use orc::mesh;
    pub enum PressureVelocityCoupling {
        SIMPLE,
    }

    fn get_velocity_source_term(location: Vector) -> Vector {
        location
    }

    fn initialize_domain() {}

    fn iterate_steady(iteration_count: u32) {
        // 1. Guess the
    }
}

use orc::mesh;
fn main() {
    // Interface: allow user to choose from
    // 1. Read mesh
    // 2. Read data
    // 3. Read settings
    // 4. Write mesh
    // 5. Write data
    // 6. Write settings
    // 7. Run solver
    println!("Hello, world!");
    // let msh = mesh::Mesh{
    //     nodes: Vec<mesh::Node>,
    //     faces: Vec<mesh::Face>,
    //     cells: Vec<mesh::Cell>,
    // };
    solver_io::read_mesh("/home/admin/3x3_cube.msh");
}

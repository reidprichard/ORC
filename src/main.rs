#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

macro_rules! skip_zone_zero {
    ($label:tt, $line_blocks:expr) => {
        if $line_blocks[1] == "(0" {
            break $label;
        }
    };
}

macro_rules! end_section {
    ($label:tt, $parser_state:expr) => {{
        $parser_state = MeshReadState::Parsing;
        break $label;
    }};
}

pub mod solver_io {
    use orc::common::*;
    use orc::mesh;
    use orc::mesh::Cell;
    use orc::mesh::Face;
    use orc::mesh::Mesh;
    use orc::mesh::Node;
    use std::fs;
    use std::fs::File;
    use std::io::{self, BufRead};

    enum MeshReadState {
        Parsing,
        ReadingNodes,
        ReadingFaces,
        ReadingCells,
    }

    pub fn read_mesh(mesh_path: &str) -> mesh::Mesh {
        fn read_mesh_lines(filename: &str) -> io::Result<io::Lines<io::BufReader<File>>> {
            // Example had filename generic type P and ... where P: AsRef<Path> ... in signature but I
            // don't understand it ¯\_(ツ)_/¯
            let file = File::open(filename)?;
            Ok(io::BufReader::new(file).lines())
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

        let mut state = MeshReadState::Parsing;
        let mut start_index: Uint = 0;
        let mut end_index: Uint = 0;

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
            dbg!();
            let mut lines = mesh_file_lines.flatten();
            let mut section_header_line = lines.next().expect("mesh is at least one line long");
            loop {
                let section_header_blocks: Vec<&str> =
                    section_header_line.split_ascii_whitespace().collect();
                match section_header_blocks[0] {
                    "(0" => (), // comment
                    "(1" => (), // header
                    "(2" => (),       // dimensions
                    "(10" => 'read_nodes: {
                        skip_zone_zero!('read_nodes, section_header_blocks);
                        start_index = Uint::from_str_radix(section_header_blocks[2], 16)
                            .expect("third value exists");
                        end_index = Uint::from_str_radix(section_header_blocks[3], 16)
                            .expect("fourth value exists");
                        println!("Beginning reading nodes from {start_index} to {end_index}.");
                        let mut current_line = lines.next().expect("section header has contents");
                        'node_loop: loop {
                            let line_blocks: Vec<&str> =
                                current_line.split_ascii_whitespace().collect();
                            if line_blocks.len() != 3 {
                                end_section!('node_loop, state);
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
                                match lines.next() {
                                    Some(line_contents) => current_line = line_contents,
                                    None => end_section!('node_loop, state),
                                }
                                // current_line = match lines.next() {
                                //     Ok(line) => line,
                                //     Err(e) => e;
                                // }
                            } else {
                                end_section!('node_loop, state);
                            }
                        }
                    }
                    "(18" => {
                        // skip_zone_zero!(section_header_blocks);
                        println!("Beginning reading shadow faces."); // periodic shadow faces
                    }
                    "(12" => 'read_cells: {
                        skip_zone_zero!('read_cells, section_header_blocks);
                        start_index = Uint::from_str_radix(section_header_blocks[2], 16)
                            .expect("third value exists");
                        end_index = Uint::from_str_radix(section_header_blocks[3], 16)
                            .expect("fourth value exists");
                        println!("Beginning reading cells."); // cells
                    }
                    "(13" => 'read_faces: {
                        skip_zone_zero!('read_faces, section_header_blocks);
                        start_index = Uint::from_str_radix(section_header_blocks[2], 16)
                            .expect("third value exists");
                        end_index = Uint::from_str_radix(section_header_blocks[3], 16)
                            .expect("fourth value exists");
                        println!("Beginning reading faces."); // faces
                    }
                    "(58" => (), // cell tree
                    "(59" => (), // face tree
                    "(61" => (), // interface face parents for nonconformal
                    _ => (),
                }
                section_header_line = lines.next().expect("mesh is at least one line long");
                println!("{section_header_line}")
            }
        }
        // for line in read_mesh_lines(mesh_path).unwrap().flatten() {
        //     // 99% sure mesh files are ASCII?
        //     let line_blocks: Vec<&str> = line.split_ascii_whitespace().collect();
        //     match state {
        //         MeshReadState::Parsing => {
        //             match line_blocks[0] {
        //                 "(0" => continue, // comment
        //                 "(1" => continue, // header
        //                 "(2" => (),       // dimensions
        //                 "(10" => {
        //                     skip_zone_zero!(line_blocks);
        //                     state = MeshReadState::ReadingNodes;
        //                     start_index = Uint::from_str_radix(line_blocks[2], 16)
        //                         .expect("third value exists");
        //                     end_index = Uint::from_str_radix(line_blocks[3], 16)
        //                         .expect("fourth value exists");
        //                     println!("Beginning reading nodes from {start_index} to {end_index}.");
        //                     // nodes
        //                 }
        //                 "(18" => {
        //                     skip_zone_zero!(line_blocks);
        //                     println!("Beginning reading shadow faces."); // periodic shadow faces
        //                 }
        //                 "(12" => {
        //                     skip_zone_zero!(line_blocks);
        //                     state = MeshReadState::ReadingCells;
        //                     start_index = Uint::from_str_radix(line_blocks[2], 16)
        //                         .expect("third value exists");
        //                     end_index = Uint::from_str_radix(line_blocks[3], 16)
        //                         .expect("fourth value exists");
        //                     println!("Beginning reading cells."); // cells
        //                 }
        //                 "(13" => {
        //                     skip_zone_zero!(line_blocks);
        //                     state = MeshReadState::ReadingFaces;
        //                     start_index = Uint::from_str_radix(line_blocks[2], 16)
        //                         .expect("third value exists");
        //                     end_index = Uint::from_str_radix(line_blocks[3], 16)
        //                         .expect("fourth value exists");
        //                     println!("Beginning reading faces."); // faces
        //                 }
        //                 "(58" => continue, // cell tree
        //                 "(59" => continue, // face tree
        //                 "(61" => continue, // interface face parents for nonconformal
        //                 _ => continue,
        //             }
        //         }
        //         MeshReadState::ReadingNodes => {
        //             if line_blocks.len() != 3 {
        //                 continue;
        //             }
        //             println!("{line}");
        //             let x = line_blocks[0].parse::<Float>();
        //             let y = line_blocks[1].parse::<Float>();
        //             let z = line_blocks[2].parse::<Float>();
        //             if x.is_ok() && y.is_ok() && z.is_ok() {
        //                 nodes.push(Node {
        //                     cell_indices: Vec::new(),
        //                     position: Vector {
        //                         x: x.unwrap(),
        //                         y: y.unwrap(),
        //                         z: z.unwrap(),
        //                     },
        //                     velocity: Vector {
        //                         x: 0.,
        //                         y: 0.,
        //                         z: 0.,
        //                     },
        //                     pressure: 0.,
        //                 });
        //             } else {
        //                 // todo: read this line on the next pass
        //                 state = MeshReadState::Parsing;
        //             }
        //         }
        //         MeshReadState::ReadingFaces => {}
        //         MeshReadState::ReadingCells => {}
        //     }
        // }
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

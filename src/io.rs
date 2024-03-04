use crate::common::*;
use crate::mesh::*;
use itertools::Itertools;
use log::{debug, info};
use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};

// cells: (12 (zone-id first-index last-index type      element-type))
// faces: (13 (zone-id first-index last-index bc-type   face-type))
// nodes: (10 (zone-id first-index last-index type      ND))

// TODO: Make this only take block 1? Or handle error here in case 1 doesn't exist?
macro_rules! skip_zone_zero {
    ($label:tt, $line_blocks:expr) => {
        if $line_blocks[1] == "(0" {
            break $label;
        }
    };
}

pub fn read_mesh(mesh_path: &str) -> Mesh {
    println!("Beginning reading mesh from {mesh_path}");

    fn read_mesh_lines(filename: &str) -> io::Result<io::Lines<io::BufReader<File>>> {
        // Example had filename generic type P and ... where P: AsRef<Path> ... in signature but I
        // don't understand it ¯\_(ツ)_/¯
        let file = File::open(filename)?;
        Ok(io::BufReader::new(file).lines())
    }

    fn read_section_header_common(header_line: &str) -> Vec<Uint> {
        let re = Regex::new(r"([0-9a-z]+)").expect("valid regex");
        // let re = Regex::new(r"\(\d+ \(([0-9a-z]+) ([0-9a-z]+) ([0-9a-z]+) ([0-9a-z]+)")
        //     .expect("valid regex");

        let mut items: Vec<Uint> = Vec::new();
        for (_, [s]) in re.captures_iter(header_line).map(|c| c.extract()) {
            items.push(Uint::from_str_radix(s, 16).expect("valid hex"));
        }
        items
    }

    let mut nodes: HashMap<Uint, Node> = HashMap::new();
    let mut faces: HashMap<Uint, Face> = HashMap::new();
    let mut cells: HashMap<Uint, Cell> = HashMap::new();
    let mut face_zones: HashMap<Uint, Uint> = HashMap::new();
    let mut cell_zones: HashMap<Uint, Uint> = HashMap::new();

    let mut node_indices: Vec<Uint> = Vec::new();
    let mut face_indices: Vec<Uint> = Vec::new();
    let mut cell_indices: Vec<Uint> = Vec::new();

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
                    info!("section: {section_header_line}");
                    let items = read_section_header_common(&section_header_line);
                    let (_, zone_id, start_index, end_index, node_type, dims) = items
                        .iter()
                        .map(|n| *n)
                        .collect_tuple()
                        .expect("correct number of items");
                    info!("Beginning reading nodes from {start_index} to {end_index}.");

                    let mut current_line = lines.next().expect("node section has contents");
                    let mut node_index = start_index;
                    'node_loop: loop {
                        if current_line == "(" {
                            current_line = lines.next().unwrap();
                            continue 'node_loop;
                        }
                        if current_line.starts_with(")") {
                            break 'node_loop;
                        }
                        debug!("Node {node_index}: {current_line}");
                        let line_blocks: Vec<&str> =
                            current_line.split_ascii_whitespace().collect();
                        if line_blocks.len() == 3 {
                            let x = line_blocks[0].parse::<Float>();
                            let y = line_blocks[1].parse::<Float>();
                            let z = line_blocks[2].parse::<Float>();
                            if x.is_ok() && y.is_ok() && z.is_ok() {
                                nodes.insert(
                                    node_index,
                                    Node {
                                        position: Vector {
                                            x: x.unwrap(),
                                            y: y.unwrap(),
                                            z: z.unwrap(),
                                        },
                                        ..Node::default()
                                    },
                                );
                            } else {
                                break 'node_loop;
                            }
                        }
                        match lines.next() {
                            Some(line_contents) => {
                                current_line = line_contents;
                                node_index += 1;
                            }
                            None => break 'node_loop,
                        }
                    }
                }
                "(18" => {
                    // skip_zone_zero!(section_header_blocks);
                    info!("Beginning reading shadow faces."); // periodic shadow faces
                }
                "(12" => 'read_cells: {
                    skip_zone_zero!('read_cells, section_header_blocks);
                    let items = read_section_header_common(&section_header_line);
                    let (_, zone_id, start_index, end_index, zone_type, element_type) = items
                        .iter()
                        .map(|n| *n)
                        .collect_tuple()
                        .expect("cell section has 6 entries");
                    cell_zones.entry(zone_id).or_insert(zone_type);
                }
                "(13" => 'read_faces: {
                    skip_zone_zero!('read_faces, section_header_blocks);
                    let items = read_section_header_common(&section_header_line);
                    let (_, zone_id, start_index, end_index, boundary_type, face_type) = items
                        .iter()
                        .map(|n| *n)
                        .collect_tuple()
                        .expect("face section has 6 entries");
                    face_zones.entry(zone_id).or_insert(boundary_type);
                    // TODO: Add error checking to not allow unsupported BC types
                    // Is using `usize` bad here?
                    info!("Beginning reading faces."); // cells

                    let mut current_line = lines.next().expect("face section has contents");
                    let mut face_index = start_index;
                    'face_loop: loop {
                        if current_line == "(" {
                            current_line = lines.next().unwrap();
                            continue 'face_loop;
                        }
                        if current_line.starts_with(")") {
                            break 'face_loop;
                        }
                        debug!("Face {face_index}: {current_line}");
                        // if current_line[0] == '(' || current_line[0] == ')' {
                        //
                        // }
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
                        faces.insert(
                            face_index,
                            Face {
                                zone_number: zone_id,
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
                                ..Face::default()
                            },
                        );
                        match lines.next() {
                            Some(line_contents) => {
                                current_line = line_contents;
                                face_index += 1;
                            }
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
    } else {
        panic!("Unable to read mesh.");
    }

    for (face_index, mut face) in &mut faces {
        if face.node_indices.len() < 3 {
            panic!("face has less than 3 nodes");
        }
        let face_nodes: Vec<&Node> = face
            .node_indices
            .iter()
            .map(|n| nodes.get(n).unwrap())
            .collect();
        face.normal = (face_nodes[2].position - face_nodes[1].position)
            .cross(&(face_nodes[1].position - face_nodes[0].position))
            .unit();
        match face_nodes.len() {
            3 => {
                // Triangular face
                face.area = (face_nodes[2].position - face_nodes[1].position)
                    .cross(&(face_nodes[1].position - face_nodes[0].position))
                    .norm()
                    / 2.;
            }
            4 => {
                // Quadrilateral face
                face.area = (face_nodes[3].position - face_nodes[1].position)
                    .cross(&(face_nodes[2].position - face_nodes[0].position))
                    .norm()
                    / 2.;
            }
            node_count => {
                // Polyhedral face - UNTESTED

                // Shoelace formula allows calculation of polygon area in 2D; we need to
                // translate to a 2D coordinate system to allow this
                let axis_1 = (face_nodes[1].position - face_nodes[0].position).unit();
                let axis_2 = axis_1.cross(&face.normal).unit();
                let translate =
                    |x: &Vector| -> (Float, Float) { (x.dot(&axis_1), x.dot(&axis_2)) };
                let mut area: Float = 0.;
                let mut node_index = 0;
                while node_index < face_nodes.len() {
                    let pos_1 = translate(&face_nodes[node_index].position);
                    let pos_2 = translate(&face_nodes[(node_index + 1) % node_count].position);
                    area += pos_1.0 * pos_2.1 - pos_1.1 * pos_2.0;
                    node_index += 1;
                }
                face.area = Float::abs(area) / 2.;
            }
        }
        // I don't really understand the reference semantics here
        for node_index in &face.node_indices {
            face.centroid += nodes[node_index].position;
        }
        face.centroid /= face.node_indices.len();
        info!(
            "Face {}: centroid={}, area={:.2e}",
            face_index, face.centroid, face.area
        );
        for cell_index in &face.cell_indices {
            // Could this check be done with a filter or something?
            if *cell_index == 0 {
                continue;
            }
            let mut cell = cells.entry(*cell_index).or_insert(Cell::default());
            cell.face_indices.push(*face_index);
            // TODO: more rigorous centroid calc
            cell.centroid += face.centroid;
            // TODO: Get cell zones
        }
    }

    for (cell_index, mut cell) in &mut cells {
        cell.centroid /= cell.face_indices.len();
        let cell_faces: Vec<&Face> = cell
            .face_indices
            .iter()
            .map(|n| faces.get(n).unwrap())
            .collect();
        if (cell_faces.len() < 4) {
            panic!("Cell cannot have fewer than 4 faces.");
        }
        for face in &cell_faces {
            cell.volume +=
                Float::abs(face.area * (face.centroid - cell.centroid).dot(&face.normal)) / 3.;
        }
        info!(
            "Cell {}: centroid={}, volume={:.2e}",
            cell_index, cell.centroid, cell.volume
        );
    }

    for (cell_zone_index, cell_zone_type) in &cell_zones {
        info!(
            "Cell zone {}: {}",
            cell_zone_index,
            get_cell_zone_types()[cell_zone_type]
        );
    }

    for (face_zone_index, face_zone_type) in &face_zones {
        info!(
            "Face zone {}: {}",
            face_zone_index,
            get_boundary_condition_types()[face_zone_type]
        );
    }

    println!(
        "Done reading mesh.\nCells: {}\nFaces: {}\nNodes: {}",
        cells.len(),
        faces.len(),
        nodes.len()
    );

    // TODO: Rewrite more concisely
    // Very ugly way to do this
    let node_positions: Vec<Vector> = nodes.iter().map(|(_, n)| n.position).collect();
    let x_min = node_positions
        .iter()
        .fold(Float::INFINITY, |acc, &n| acc.min(n.x));
    let x_max = node_positions
        .iter()
        .fold(Float::NEG_INFINITY, |acc, &n| acc.max(n.x));
    let y_min = node_positions
        .iter()
        .fold(Float::INFINITY, |acc, &n| acc.min(n.y));
    let y_max = node_positions
        .iter()
        .fold(Float::NEG_INFINITY, |acc, &n| acc.max(n.y));
    let z_min = node_positions
        .iter()
        .fold(Float::INFINITY, |acc, &n| acc.min(n.z));
    let z_max = node_positions
        .iter()
        .fold(Float::NEG_INFINITY, |acc, &n| acc.max(n.z));
    println!(
        "Domain extents:\nX:({:.2e}, {:.2e})\nY:({:.1e}, {:.2e})\nZ:({:.2e}, {:.2e})",
        x_min, x_max, y_min, y_max, z_min, z_max
    );

    Mesh {
        nodes,
        faces,
        cells,
        face_zones,
        cell_zones,
    }
}

pub fn read_data() {}

pub fn read_settings() {}

pub fn write_mesh() {}

pub fn write_data() {}

pub fn write_settings() {}

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

    let format_pos_padded = |n: Float| -> String { format!("{n:<10.2e}") };
    let format_pos = |n: Float| -> String { format!("{n:.2e}") };

    fn read_mesh_lines(filename: &str) -> io::Result<io::Lines<io::BufReader<File>>> {
        // Example had filename generic type P and ... where P: AsRef<Path> ... in signature but I
        // don't understand it ¯\_(ツ)_/¯
        let file = File::open(filename)?;
        // let metadata = file.metadata()?;
        // println!("File len: {}", metadata.len());
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
    let mut face_zones: HashMap<Uint, FaceZone> = HashMap::new();
    let mut cell_zones: HashMap<Uint, CellZone> = HashMap::new();

    let mut node_indices: Vec<Uint> = Vec::new();
    let mut face_indices: Vec<Uint> = Vec::new();
    let mut cell_indices: Vec<Uint> = Vec::new();

    let mut dimensions: u8 = 0;

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

    let mut zone_name: String = String::new();

    if let Ok(file_lines) = read_mesh_lines(mesh_path) {
        let mut mesh_file_lines = file_lines.flatten();
        let mut section_header_line = mesh_file_lines
            .next()
            .expect("mesh is at least one line long");
        loop {
            // TODO: Print progress bar to console
            let section_header_blocks: Vec<&str> =
                section_header_line.split_ascii_whitespace().collect();
            match section_header_blocks[0] {
                "(0" => 'read_comment: {
                    zone_name = section_header_line
                        .rsplit_once(" ")
                        .expect("comment has a space")
                        .1
                        .trim_end_matches("\")")
                        .to_string();
                }
                "(1" => (), // header
                "(2" => {
                    // dimensions
                    dimensions = section_header_blocks
                        .get(1)
                        .expect("dimensions section should have two items")
                        .strip_suffix(")")
                        .expect("second item ends with )")
                        .parse()
                        .expect("second item is integer dimension count");
                    if (dimensions != 2 && dimensions != 3) {
                        panic!("Mesh is not 2D or 3D.");
                    }
                }
                "(10" => 'read_nodes: {
                    skip_zone_zero!('read_nodes, section_header_blocks);
                    info!("section: {section_header_line}");
                    let items = read_section_header_common(&section_header_line);
                    let (_, zone_id, start_index, end_index, node_type, dims) = items
                        .iter()
                        .map(|n| *n)
                        .collect_tuple()
                        .expect("nodes header has six items");
                    info!("Beginning reading nodes from {start_index} to {end_index}.");

                    let mut current_line = mesh_file_lines
                        .next()
                        .expect("node section shouldn't be empty");
                    let mut node_index = start_index;
                    'node_loop: loop {
                        if current_line == "(" {
                            current_line = mesh_file_lines.next().unwrap();
                            continue 'node_loop;
                        }
                        if current_line.starts_with(")") {
                            break 'node_loop;
                        }
                        // debug!("Node {node_index}: {current_line}");
                        let line_blocks: Vec<&str> =
                            current_line.split_ascii_whitespace().collect();
                        if line_blocks.len() == dimensions.into() {
                            let x = line_blocks[0].parse::<Float>().expect(&format!(
                                "{} should be a string representation of a float",
                                line_blocks[0]
                            ));
                            let y = line_blocks[1].parse::<Float>().expect(&format!(
                                "{} should be a string representation of a float",
                                line_blocks[1]
                            ));
                            let mut z = 0.;
                            if (dimensions == 3) {
                                z = line_blocks[2].parse::<Float>().expect(&format!(
                                    "{} should be a string representation of a float",
                                    line_blocks[2]
                                ));
                            }
                            nodes.insert(
                                node_index,
                                Node {
                                    position: Vector { x, y, z },
                                    ..Node::default()
                                },
                            );
                            debug!(
                                "Node {}: {} {} {}",
                                node_index,
                                format_pos_padded(x),
                                format_pos_padded(y),
                                format_pos_padded(z)
                            );
                        }
                        match mesh_file_lines.next() {
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
                    cell_zones.entry(zone_id).or_insert(CellZone {
                        zone_type,
                        // name: zone_name.clone(), // Fluent doesn't seem to set comments for zone
                        // name
                    });
                }
                "(13" => 'read_faces: {
                    skip_zone_zero!('read_faces, section_header_blocks);
                    let items = read_section_header_common(&section_header_line);
                    let (_, zone_id, start_index, end_index, boundary_type, face_type) = items
                        .iter()
                        .map(|n| *n)
                        .collect_tuple()
                        .expect("face section has 6 entries");
                    face_zones.entry(zone_id).or_insert(FaceZone {
                        zone_type: BoundaryConditionTypes::try_from(boundary_type)
                            .expect("valid BC type"),
                        name: zone_name.clone(),
                        value: 0.,
                    });
                    info!("Beginning reading faces from zone {zone_id}."); // cells

                    let mut current_line =
                        mesh_file_lines.next().expect("face section has contents");
                    let mut face_index = start_index;
                    'face_loop: loop {
                        if current_line == "(" {
                            current_line = mesh_file_lines.next().unwrap();
                            continue 'face_loop;
                        }
                        if current_line.starts_with(")") {
                            break 'face_loop;
                        }
                        debug!("Face {face_index}: {current_line}");
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
                                zone: zone_id,
                                cell_numbers: line_blocks[node_count..]
                                    .into_iter()
                                    .map(|cell_id| Uint::from_str_radix(cell_id, 16))
                                    .flatten()
                                    .collect(),
                                node_numbers: line_blocks[..node_count]
                                    .into_iter()
                                    .map(|node_id| Uint::from_str_radix(node_id, 16))
                                    .flatten()
                                    .collect(),
                                ..Face::default()
                            },
                        );
                        match mesh_file_lines.next() {
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
            match mesh_file_lines.next() {
                Some(line_content) => section_header_line = line_content,
                None => break,
            }
        }
    } else {
        panic!("Unable to read mesh.");
    }

    for (face_index, mut face) in &mut faces {
        if face.node_numbers.len() < dimensions.into() {
            println!("dimensions: {}", face.node_numbers.len());
            panic!("face has too few nodes");
        }
        let face_nodes: Vec<&Node> = face
            .node_numbers
            .iter()
            .map(|n| nodes.get(n).expect("nodes should have all been read"))
            .collect();
        match dimensions {
            2 => {
                let tangent = face_nodes[1].position - face_nodes[0].position;
                face.normal = if tangent.x == 0. {
                    Vector {
                        x: 1.,
                        y: -tangent.x / tangent.y,
                        z: 0.,
                    }
                } else {
                    Vector {
                        x: -tangent.y / tangent.x,
                        y: 1.,
                        z: 0.,
                    }
                }
                .unit();
            }
            3 => {
                face.normal = (face_nodes[2].position - face_nodes[1].position)
                    .cross(&(face_nodes[1].position - face_nodes[0].position))
                    .unit();
            }
            _ => panic!("dimensions must be 2 or 3"),
        }
        // TGRID format has face normal as defined here pointing toward cell 0
        // If cell 0 does not exist (e.g. face is on boundary), we need to remove
        // that cell and flip the normal vector
        if face.cell_numbers[0] == 0 {
            face.normal = -face.normal;
            face.cell_numbers.remove(0);
        } else if face.cell_numbers[1] == 0 {
            face.cell_numbers.remove(1);
        }
        face.centroid = face
            .node_numbers
            .iter()
            .fold(Vector::zero(), |acc, n| acc + nodes[n].position)
            / (face.node_numbers.len() as Float);
        face.area = match face_nodes.len() {
            0 | 1 => panic!("faces must have 2+ nodes"),
            2 => {
                // 2D
                assert!(dimensions == 2);
                (face_nodes[1].position - face_nodes[0].position).norm()
            }
            node_count => {
                // 3D
                // Both implementations assume face is coplanar
                // Would potentially be faster to treat triangles and quadrilaterals specially but
                // KISS

                // ** Method 1: Shoelace formula - UNTESTED **
                // Shoelace formula allows calculation of polygon area in 2D; we need to
                // translate to a 2D coordinate system to allow this
                // let axis_1 = (face_nodes[1].position - face_nodes[0].position).unit();
                // let axis_2 = axis_1.cross(&face.normal).unit();
                // let translate = |x: &Vector| -> (Float, Float) { (x.dot(&axis_1), x.dot(&axis_2)) };
                // let mut area: Float = 0.;
                // let mut node_index = 0;
                // while node_index < face_nodes.len() {
                //     let pos_1 = translate(&face_nodes[node_index].position);
                //     let pos_2 = translate(&face_nodes[(node_index + 1) % node_count].position);
                //     area += pos_1.0 * pos_2.1 - pos_1.1 * pos_2.0;
                //     node_index += 1;
                // }
                // face.area = Float::abs(area) / 2.;

                // ** Method 2 - decompose into triangles **
                // If polygon is convex (which I think is guaranteed), we can just decompose into
                // triangles
                let calculate_triangle_area =
                    |vertex_1: &Vector, vertex_2: &Vector, vertex_3: &Vector| -> Float {
                        Float::abs(
                            (*vertex_2 - *vertex_1)
                                .cross(&(*vertex_3 - *vertex_1))
                                .norm(),
                        ) / 2.
                    };
                let mut area: Float = face.node_numbers.windows(2).fold(0., |acc, w| {
                    acc + calculate_triangle_area(
                        &face.centroid,
                        &nodes[&w[0]].position,
                        &nodes[&w[1]].position,
                    )
                });
                let first = face.node_numbers[0];
                let last = face.node_numbers[face.node_numbers.len() - 1];
                area + calculate_triangle_area(
                    &face.centroid,
                    &nodes[&first].position,
                    &nodes[&last].position,
                )
            }
        };
        // I don't really understand the reference semantics here
        info!(
            "Face {}: centroid={}, area={:.2e}",
            face_index, face.centroid, face.area
        );
        for cell_index in &face.cell_numbers {
            // Could this check be done with a filter or something?
            if *cell_index == 0 {
                continue;
            }
            let mut cell = cells.entry(*cell_index).or_insert(Cell::default());
            cell.face_numbers.push(*face_index);
            // TODO: more rigorous centroid calc
            cell.centroid += face.centroid;
            // TODO: Get cell zones
        }
    }

    for (cell_index, mut cell) in &mut cells {
        cell.centroid /= cell.face_numbers.len();
        let cell_faces: Vec<&Face> = cell
            .face_numbers
            .iter()
            .map(|n| faces.get(n).unwrap())
            .collect();
        if (cell_faces.len() < (dimensions + 1) as usize) {
            panic!("cell has too few faces");
        }
        // 1/2 * b * h (area of triangle) for 2D
        // 1/3 * b * h (area of pyramid) for 3D
        cell.volume = cell_faces.iter().fold(0. as Float, |acc, f| {
            println!("{}, {}, {}", f.centroid, cell.centroid, f.normal);
            // println!(
            //     "{}: {} * {} = {}",
            //     cell_index,
            //     f.area,
            //     Float::abs((f.centroid - cell.centroid).dot(&f.normal)),
            //     f.area * Float::abs((f.centroid - cell.centroid).dot(&f.normal))
            // );
            acc + f.area * Float::abs((f.centroid - cell.centroid).dot(&f.normal))
                / (dimensions as Float)
        });
        info!(
            "Cell {}: centroid={}, volume={:.2e}",
            cell_index, cell.centroid, cell.volume
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
        "Domain extents:\nX: ({}, {})\nY: ({}, {})\nZ: ({}, {})",
        format_pos_padded(x_min),
        format_pos(x_max),
        format_pos_padded(y_min),
        format_pos(y_max),
        format_pos_padded(z_min),
        format_pos(z_max)
    );

    for (cell_zone_index, cell_zone) in &cell_zones {
        println!(
            "Cell zone {}: {}",
            cell_zone_index,
            get_cell_zone_types()[&cell_zone.zone_type],
        );
    }

    for (face_zone_index, face_zone) in &face_zones {
        println!(
            "Face zone {}: {} ({})",
            face_zone_index, face_zone.zone_type, face_zone.name
        );
    }

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

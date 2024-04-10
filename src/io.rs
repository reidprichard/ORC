#![allow(unused_labels)]

use crate::mesh::*;
use crate::numerical_types::*;
use crate::settings::GradientReconstructionMethods;
use crate::solver::{calculate_pressure_gradient, calculate_velocity_gradient};
use ahash::RandomState;
use itertools::Itertools;
use log::info;
use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::io::{self, BufRead};

// NOTE: In this file, the terms "node" and "vertex" are used interchangeably. A "node" is not a
// cell center; it is a vertex of a cell/face.

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

    fn read_section_header_common(header_line: &str) -> Vec<usize> {
        let re = Regex::new(r"([0-9a-z]+)").expect("valid regex");
        let mut items: Vec<usize> = Vec::new();
        for (_, [s]) in re.captures_iter(header_line).map(|c| c.extract()) {
            items.push(usize::from_str_radix(s, 16).expect("valid hex"));
        }
        items
    }

    macro_rules! new_hashmap {
        () => {
            HashMap::with_hasher(RandomState::with_seeds(3, 1, 4, 1))
        };
    }

    // TODO: BTreeMap?
    let mut vertices_hashmap: HashMap<usize, Vertex, RandomState> = new_hashmap!();
    let mut faces_hashmap: HashMap<usize, Face, RandomState> = new_hashmap!();
    let mut cells_hashmap: HashMap<usize, Cell, RandomState> = new_hashmap!();
    let mut face_zones: HashMap<Uint, FaceZone, RandomState> = new_hashmap!();
    let mut cell_zones: HashMap<Uint, CellZone, RandomState> = new_hashmap!();

    let mut dimensions: u8 = 0;

    let mut zone_name: String = String::new();

    if let Ok(file_lines) = read_mesh_lines(mesh_path) {
        let mut mesh_file_lines = file_lines.map_while(Result::ok);
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
                        .rsplit_once(' ')
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
                        .strip_suffix(')')
                        .expect("second item ends with )")
                        .parse()
                        .expect("second item is integer dimension count");
                    if dimensions != 2 && dimensions != 3 {
                        panic!("Mesh is not 2D or 3D.");
                    }
                }
                "(10" => 'read_nodes: {
                    skip_zone_zero!('read_nodes, section_header_blocks);
                    info!("section: {section_header_line}");
                    let items = read_section_header_common(&section_header_line);
                    let (_, _, start_index, end_index, _, _) = items
                        .iter()
                        .copied()
                        .collect_tuple()
                        .expect("nodes header has six items");
                    info!("Beginning reading nodes from {start_index} to {end_index}.");

                    let mut current_line = mesh_file_lines
                        .next()
                        .expect("node section shouldn't be empty");
                    let mut node_number = start_index;
                    'node_loop: loop {
                        if current_line == "(" {
                            current_line = mesh_file_lines.next().unwrap();
                            continue 'node_loop;
                        }
                        if current_line.starts_with(')') {
                            break 'node_loop;
                        }
                        // debug!("Node {node_index}: {current_line}");
                        let line_blocks: Vec<&str> =
                            current_line.split_ascii_whitespace().collect();
                        if line_blocks.len() == dimensions.into() {
                            let x = line_blocks[0].parse::<Float>().unwrap_or_else(|_| {
                                panic!(
                                    "{} should be a string representation of a float",
                                    line_blocks[0]
                                )
                            });
                            let y = line_blocks[1].parse::<Float>().unwrap_or_else(|_| {
                                panic!(
                                    "{} should be a string representation of a float",
                                    line_blocks[1]
                                )
                            });
                            let mut z = 0.;
                            if dimensions == 3 {
                                z = line_blocks[2].parse::<Float>().unwrap_or_else(|_| {
                                    panic!(
                                        "{} should be a string representation of a float",
                                        line_blocks[2]
                                    )
                                });
                            }
                            vertices_hashmap.insert(
                                node_number - 1,
                                Vertex {
                                    position: Vector { x, y, z },
                                },
                            );
                            // debug!(
                            //     "Node {}: {} {} {}",
                            //     node_number,
                            //     format_pos_padded(x),
                            //     format_pos_padded(y),
                            //     format_pos_padded(z)
                            // );
                        }
                        match mesh_file_lines.next() {
                            Some(line_contents) => {
                                current_line = line_contents;
                                node_number += 1;
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
                    let (_, zone_id, _, _, zone_type, _) = items
                        .iter()
                        .copied()
                        .collect_tuple()
                        .expect("cell section has 6 entries");
                    cell_zones.entry(zone_id as Uint).or_insert(CellZone {
                        zone_type: (zone_type as Uint),
                        // name: zone_name.clone(), // Fluent doesn't seem to set comments for zone
                        // name
                    });
                }
                "(13" => 'read_faces: {
                    skip_zone_zero!('read_faces, section_header_blocks);
                    let items = read_section_header_common(&section_header_line);
                    let (_, zone_id, start_index, _, boundary_type, face_type) = items
                        .iter()
                        .copied()
                        .collect_tuple()
                        .expect("face section has 6 entries");
                    face_zones.entry(zone_id as Uint).or_insert(FaceZone {
                        zone_type: FaceConditionTypes::try_from(boundary_type as Uint)
                            .expect("valid BC type"),
                        name: zone_name.clone(),
                        scalar_value: 0.,
                        vector_value: Vector {
                            x: 0.,
                            y: 0.,
                            z: 0.,
                        },
                    });
                    info!("Beginning reading faces from zone {zone_id}."); // cells

                    let mut current_line =
                        mesh_file_lines.next().expect("face section has contents");
                    let mut face_number = start_index;
                    'face_loop: loop {
                        if current_line == "(" {
                            current_line = mesh_file_lines.next().unwrap();
                            continue 'face_loop;
                        }
                        if current_line.starts_with(')') {
                            break 'face_loop;
                        }
                        // debug!("Face {face_number}: {current_line}");
                        let line_blocks: Vec<&str> =
                            current_line.split_ascii_whitespace().collect();
                        if line_blocks.len() < 2 {
                            break;
                        }
                        let node_count = line_blocks.len() - 2;
                        if face_type != 0 && face_type != 5 && face_type != node_count {
                            break 'face_loop;
                        }
                        faces_hashmap.insert(
                            face_number - 1,
                            Face {
                                zone: zone_id as Uint,
                                cell_indices: line_blocks[node_count..]
                                    .iter()
                                    .map(|cell_num| {
                                        let cell_num = usize::from_str_radix(cell_num, 16).unwrap();
                                        if cell_num > 0 {
                                            cell_num - 1
                                        } else {
                                            usize::MAX
                                        }
                                    })
                                    .collect(),
                                node_indices: line_blocks[..node_count]
                                    .iter()
                                    .map(|node_num_str| {
                                        let node_num =
                                            usize::from_str_radix(node_num_str, 16).unwrap();
                                        if node_num > 0 {
                                            node_num - 1
                                        } else {
                                            usize::MAX
                                        }
                                    })
                                    .collect(),
                                ..Face::default()
                            },
                        );
                        match mesh_file_lines.next() {
                            Some(line_contents) => {
                                current_line = line_contents;
                                face_number += 1;
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
        panic!("Unable to open mesh file for reading.");
    }

    for face_index in 0..faces_hashmap.len() {
        let face = faces_hashmap.get_mut(&face_index).unwrap();
        if face.node_indices.len() < dimensions.into() {
            println!("dimensions: {}", face.node_indices.len());
            panic!("face has too few nodes");
        }
        let face_nodes: Vec<&Vertex> = face
            .node_indices
            .iter()
            .map(|n| vertices_hashmap.get(n).expect("nodes should have all been read"))
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
        if face.cell_indices[0] == usize::MAX {
            face.normal = -face.normal;
            face.cell_indices.remove(0);
        } else if face.cell_indices[1] == usize::MAX {
            face.cell_indices.remove(1);
        }
        face.centroid = face
            .node_indices
            .iter()
            .fold(Vector::zero(), |acc, n| acc + vertices_hashmap[n].position)
            / (face.node_indices.len() as Float);
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
                // let translate = |x: &Vector3| -> (Float, Float) { (x.dot(&axis_1), x.dot(&axis_2)) };
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
                let area: Float = face.node_indices.windows(2).fold(0., |acc, w| {
                    acc + calculate_triangle_area(
                        &face.centroid,
                        &vertices_hashmap[&w[0]].position,
                        &vertices_hashmap[&w[1]].position,
                    )
                });
                let first = face.node_indices[0];
                let last = face.node_indices[node_count - 1];
                area + calculate_triangle_area(
                    &face.centroid,
                    &vertices_hashmap[&first].position,
                    &vertices_hashmap[&last].position,
                )
            }
        };
        // I don't really understand the reference semantics here
        // info!(
        //     "Face {}: centroid={}, area={:.2e}",
        //     face_index, face.centroid, face.area
        // );
        for cell_index in &face.cell_indices {
            // Could this check be done with a filter or something?
            if *cell_index == usize::MAX {
                continue;
            }
            let cell = cells_hashmap.entry(*cell_index).or_default();
            cell.face_indices.push(face_index);
            // TODO: more rigorous centroid calc
            cell.centroid += face.centroid;
            // TODO: Get cell zones
        }
    }

    for cell_index in 0..cells_hashmap.len() {
        let cell = cells_hashmap.get_mut(&cell_index).unwrap();
        cell.centroid /= cell.face_indices.len();
        let cell_faces: Vec<&Face> = cell
            .face_indices
            .iter()
            .map(|n| faces_hashmap.get(n).unwrap())
            .collect();
        if cell_faces.len() < (dimensions + 1) as usize {
            panic!("cell has too few faces");
        }
        // 1/2 * b * h (area of triangle) for 2D
        // 1/3 * b * h (area of pyramid) for 3D
        cell.volume = cell_faces.iter().fold(0. as Float, |acc, f| {
            acc + f.area * Float::abs((f.centroid - cell.centroid).dot(&f.normal))
                / (dimensions as Float)
        });
        // info!(
        //     "Cell {}: centroid={}, volume={:.2e}",
        //     cell_index, cell.centroid, cell.volume
        // );
    }

    println!(
        "Done reading mesh.\nCells: {}\nFaces: {}\nNodes: {}",
        cells_hashmap.len(),
        faces_hashmap.len(),
        vertices_hashmap.len()
    );

    // TODO: Rewrite more concisely
    // Very ugly way to do this
    let node_positions: Vec<Vector> = vertices_hashmap.values().map(|n| n.position).collect();
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

    let mut vertices:Vec<Vertex> = Vec::with_capacity(vertices_hashmap.len());
    let mut faces:Vec<Face> = Vec::with_capacity(faces_hashmap.len());
    let mut cells:Vec<Cell> = Vec::with_capacity(cells_hashmap.len());

    for vertex_index in 0..vertices_hashmap.len() {
        vertices.push(vertices_hashmap[&vertex_index]);
    }
    for face_index in 0..faces_hashmap.len() {
        faces.push(faces_hashmap[&face_index].clone());
    }
    for cell_number in 0..cells_hashmap.len() {
        cells.push(cells_hashmap[&cell_number].clone());
    }

    Mesh {
        vertices,
        faces,
        cells,
        face_zones,
        cell_zones,
    }
}

pub fn read_settings() {}

pub fn read_data(
    data_file_path: &str,
) -> Result<
    (
        DVector<Float>,
        DVector<Float>,
        DVector<Float>,
        DVector<Float>,
    ),
    &'static str,
> {
    let mut u: Vec<Float> = Vec::new();
    let mut v: Vec<Float> = Vec::new();
    let mut w: Vec<Float> = Vec::new();
    let mut p: Vec<Float> = Vec::new();

    let file = File::open(data_file_path);
    match file {
        Ok(f) => {
            println!("Reading solution data from {data_file_path}...");
            let data_file_lines = io::BufReader::new(f).lines();

            for line in data_file_lines {
                line.unwrap()
                    .splitn(3, '\t')
                    .enumerate()
                    .for_each(|(i, chunk)| match i {
                        0 => (),
                        1 => {
                            let uvw_i = Vector::parse(chunk);
                            u.push(uvw_i.x);
                            v.push(uvw_i.y);
                            w.push(uvw_i.z);
                        }
                        2 => {
                            p.push(chunk.parse().unwrap());
                        }
                        _ => panic!(
                            "There should only be three tab-separated columns in the data file."
                        ),
                    });
            }
            Ok((
                DVector::from_column_slice(&u),
                DVector::from_column_slice(&v),
                DVector::from_column_slice(&w),
                DVector::from_column_slice(&p),
            ))
        }
        Err(_) => Err("could not read data file"),
    }
}

pub fn write_data(
    mesh: &Mesh,
    u: &DVector<Float>,
    v: &DVector<Float>,
    w: &DVector<Float>,
    p: &DVector<Float>,
    output_file_name: &str,
) {
    let mut file = File::create(output_file_name).unwrap();
    println!("Writing data to {output_file_name}...");
    for cell_index in 0..mesh.cells.len() {
        let cell = &mesh.cells[cell_index];
        writeln!(
            file,
            "{}\t({:.e}, {:.e}, {:.e})\t{:.e}",
            cell.centroid, u[cell_index], v[cell_index], w[cell_index], p[cell_index],
        )
        .unwrap();
    }
}

pub fn write_data_with_precision(
    mesh: &Mesh,
    u: &DVector<Float>,
    v: &DVector<Float>,
    w: &DVector<Float>,
    p: &DVector<Float>,
    output_file_name: &str,
    decimal_precision: usize,
) {
    let mut file = File::create(output_file_name).unwrap();
    println!("Writing data to {output_file_name}...");
    for cell_index in 0..mesh.cells.len() {
        let cell = &mesh.cells[cell_index];
        writeln!(
            file,
            "{}\t({:.prec$e}, {:.prec$e}, {:.prec$e})\t{:.prec$e}",
            cell.centroid,
            u[cell_index],
            v[cell_index],
            w[cell_index],
            p[cell_index],
            prec = decimal_precision
        )
        .unwrap();
    }
}

#[allow(clippy::too_many_arguments)]
pub fn write_gradients(
    mesh: &Mesh,
    u: &DVector<Float>,
    v: &DVector<Float>,
    w: &DVector<Float>,
    p: &DVector<Float>,
    output_file_name: &str,
    decimal_precision: usize,
    gradient_scheme: GradientReconstructionMethods,
) {
    let mut file = File::create(output_file_name).unwrap();
    println!("Writing data to {output_file_name}...");
    for cell_index in 0..mesh.cells.len() {
        let mut velocity_gradient_str = String::new();
        calculate_velocity_gradient(mesh, u, v, w, cell_index, gradient_scheme)
            .flatten()
            .iter()
            .for_each(|v| {
                velocity_gradient_str +=
                    &format!("{:.prec$e}, ", v, prec = decimal_precision).to_string();
            });
        velocity_gradient_str.strip_suffix(", ").unwrap();

        let mut pressure_gradient_str = String::new();
        calculate_pressure_gradient(mesh, p, cell_index, gradient_scheme)
            .to_vec()
            .iter()
            .for_each(|v| {
                pressure_gradient_str +=
                    &format!("{:.prec$e}, ", v, prec = decimal_precision).to_string();
            });
        pressure_gradient_str.strip_suffix(", ").unwrap();
        writeln!(
            file,
            "{}\t({})\t({})",
            &mesh.cells[cell_index].centroid, velocity_gradient_str, pressure_gradient_str
        )
        .unwrap();
    }
}

pub fn write_settings() {}

pub fn dvector_to_str(a: &DVector<Float>) -> String {
    let mut output = String::from("[");
    for i in 0..a.nrows() {
        let coeff = a[i];
        output += &format!(
            "{: >9.2e}{}",
            coeff,
            if i < a.nrows() - 1 { ", " } else { "" }
        );
    }
    output + "]"
}

pub fn matrix_to_string(a: &CsrMatrix<Float>) -> String {
    let mut s: String = String::from("");
    for i in 0..a.nrows() {
        s += &format!("{i}: ");
        for j in 0..a.ncols() {
            let coeff = a.get_entry(i, j).unwrap().into_value();
            if a.ncols() < 16 {
                if coeff == 0. {
                    s += "          , ";
                } else {
                    s += &format!("{: <9}, ", format!("{coeff:.2e}"));
                }
            } else if coeff != 0. {
                if i == j {
                    s += "*";
                }
                s += &format!("{}={: <9}, ", j, format!("{coeff:.2e}"));
            }
        }
        s += "\n";
    }
    s
}

pub fn print_matrix(a: &CsrMatrix<Float>) {
    println!("{}", matrix_to_string(a));
}

pub fn linear_system_to_string(a: &CsrMatrix<Float>, b: &DVector<Float>) -> String {
    let mut s: String = String::from("");
    for i in 0..a.nrows() {
        s += &format!("{i}: ");
        for j in 0..a.ncols() {
            let coeff = a.get_entry(i, j).unwrap().into_value();
            if a.ncols() < 16 {
                if coeff == 0. {
                    s += "          , ";
                } else {
                    s += &format!("{: <9}, ", format!("{coeff:.2e}"));
                }
            } else if coeff != 0. {
                if i == j {
                    s += "*";
                }
                s += &format!("{}={: <9}, ", j, format!("{coeff:.2e}"));
            }
        }
        s += &format!(" | {:.2e}\n", b[i]);
    }
    s
}

pub fn print_linear_system(a: &CsrMatrix<Float>, b: &DVector<Float>) {
    println!("{}", linear_system_to_string(a, b));
}

pub fn print_vec_scientific(v: &DVector<Float>) {
    // if v.len() == 0 {
    //     return
    // }
    print!("[");
    for i in 0..v.nrows() - 1 {
        print!("{:.2e}, ", v[i]);
    }
    println!("{:.2e}]", v[v.len() - 1]);
}

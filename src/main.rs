#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

pub mod io {
    use orc::common::*;
    use orc::mesh;
    use std::fs;

    pub fn read_mesh(mesh_path: &str, msh: &mut mesh::Mesh) {
        macro_rules! skip_zone_zero {
            ($line_blocks:expr) => {
                if let "(0" = $line_blocks[1] {
                    continue;
                }
            };
        }

        for line in fs::read_to_string(mesh_path)
            .expect("mesh exists and is readable")
            .lines()
        {
            // 99% sure mesh files are ASCII?
            let line_blocks: Vec<&str> = line.split_ascii_whitespace().collect();
            match line_blocks[0] {
                "(0" => continue, // comment
                "(1" => continue, // header
                "(2" => (),       // dimensions
                "(10" => {
                    skip_zone_zero!(line_blocks);
                    println!("Beginning reading nodes."); // nodes
                }
                "(18" => {
                    skip_zone_zero!(line_blocks);
                    println!("Beginning reading shadow faces."); // periodic shadow faces
                }
                "(12" => {
                    skip_zone_zero!(line_blocks);
                    println!("Beginning reading cells."); // cells
                }
                "(13" => {
                    skip_zone_zero!(line_blocks);
                    println!("Beginning reading faces."); // faces
                }
                "(58" => continue, // cell tree
                "(59" => continue, // face tree
                "(61" => continue, // interface face parents for nonconformal
                _ => continue,
            }
            // line.trim_start_matches("(0")
            println!("Line: '{line}'");
        }
        // println!("Mesh contents:\n{mesh_contents}");
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
    let msh = mesh::Mesh{};
    io::read_mesh("/home/admin/3x3_cube.msh", msh);
}

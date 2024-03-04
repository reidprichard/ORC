#![allow(dead_code)]
#![allow(unused)]

fn main() {
    env_logger::init();
    // Interface: allow user to choose from
    // 1. Read mesh
    // 2. Read data
    // 3. Read settings
    // 4. Write mesh
    // 5. Write data
    // 6. Write settings
    // 7. Run solver
    println!("Starting.");
    let mesh = orc::io::read_mesh("./examples/3x3_cube.msh");
    println!("Complete.");
}

# <ins>O</ins>pen <ins>R</ins>ust <ins>C</ins>FD

A simple finite volume CFD solver implemented in Rust with minimal dependencies. Anticipated features:
- Co-located grid
- Unstructured 3D mesh
- Supported cell types:
  - Tetrahedron (4 faces, 4 nodes)
  - Pyramid (5 faces, 5 nodes)
  - Wedge (5 faces, 6 nodes)
  - Hexahedron (6 faces, 8 nodes)
  - Polyhedron (M faces, N nodes)
- SIMPLE pressure-velocity coupling
- TVD momentum schemes (UD/CD currently implemented)
- Green-Gauss gradient reconstruction
- ?? pressure interpolation
- Algebraic multigrid

**Pre-alpha** with basic functionality still in the works.

Roadmap:
- [X] Read TGRID mesh into memory
- [ ] Build solution matrices (75% complete)
- [ ] Initialize flow
- [ ] Iterate steady
- [ ] Read/write solution data
- [ ] Read/write settings
- [ ] CLI
- [ ] Multigrid
- [ ] Iterate transient
- [ ] Standard k-epsilon turbulence model?
- [ ] Add other mesh formats?

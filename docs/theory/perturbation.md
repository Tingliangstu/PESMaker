# Structure Perturbation

PESMaker currently implements a `dpdata.System.perturb`-style structure
perturbation method. The goal is to generate configurations near a known
structure by perturbing both the simulation cell and atomic coordinates.

## Cell perturbation

Let the original cell matrix be:

$$
C =
\begin{pmatrix}
a_x & a_y & a_z \\
b_x & b_y & b_z \\
c_x & c_y & c_z
\end{pmatrix}
$$

where each row is a lattice vector in Cartesian coordinates.

PESMaker samples six random values:

$$
e_i \sim U(-f, f), \quad i = 1, \ldots, 6
$$

where \(f\) is `cell_pert_fraction`. The perturbation matrix is:

$$
P =
\begin{pmatrix}
1 + e_1 & 0.5e_6 & 0.5e_5 \\
0.5e_6 & 1 + e_2 & 0.5e_4 \\
0.5e_5 & 0.5e_4 & 1 + e_3
\end{pmatrix}
$$

The perturbed cell is:

$$
C' = C P
$$

The diagonal terms stretch or compress the cell. The off-diagonal terms apply
shear-like perturbations.

## Coordinate transformation

Before adding independent atomic noise, PESMaker transforms the Cartesian atomic
positions consistently with the cell perturbation:

$$
R' = R P
$$

This keeps atoms deformed with the cell instead of changing only the lattice
vectors.

## Atomic displacement

For `atom_pert_style: normal`, each atom receives a Cartesian displacement:

$$
\Delta r =
\frac{d}{\sqrt{3}}
\begin{pmatrix}
\eta_x \\
\eta_y \\
\eta_z
\end{pmatrix},
\quad
\eta_x,\eta_y,\eta_z \sim \mathcal{N}(0, 1)
$$

where \(d\) is `atom_pert_distance` in Angstrom.

This means the three Cartesian components are sampled independently from a
normal distribution with standard deviation \(d / \sqrt{3}\). For example,
`atom_pert_distance: 0.1` gives each Cartesian component a standard deviation of
approximately \(0.0577\) Angstrom.

## Implemented options

PESMaker currently supports three atomic perturbation styles:

- `normal`: Gaussian Cartesian displacement.
- `uniform`: random direction with radius sampled uniformly inside a sphere.
- `const`: random direction with fixed displacement length.

Random perturbations are opt-in. If `generation.perturb` is omitted, or
`pert_num` is `0`, PESMaker writes only the expanded pristine structures.

A practical explicit perturbation setting is:

```yaml
perturb:
  include_pristine: true
  pert_num: 49
  cell_pert_fraction: 0.03
  atom_pert_distance: 0.1
  atom_pert_style: normal
```

These values generate moderately distorted structures around the starting
geometry and are useful for preparing initial DFT single-point candidates.
The expanded pristine structure is always written as
`pristine_<supercell>.<format>`, for example `pristine_3x3x3.vasp`.
`include_pristine: true` also writes one pristine file for every generated
defect variant when random perturbations are enabled. Defect-variant pristine
files append the variant name, for example
`pristine_3x3x3_single_vacancy_Te_000001.vasp` or
`pristine_3x3x1_line_defect_Te_const_a_000002.vasp`. Without random
perturbations, every generated variant is written once as a pristine structure.

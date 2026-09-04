# Tensor calculus in the reference lane

Run `PYTHONPATH=python python3 examples/tensor_calculus/tensor_calculus_tutorial.py`.
No compiler build or GPU is required. All printed numerical claims are checked.

The tutorial computes Kronecker identities, a Levi-Civita contraction and its
rank-three-storage-free delta rewrite, alpha-normal contraction keys, and the
same physical scalar/vector fields in Cartesian, cylindrical and spherical
charts. Comparisons use Cartesian analytic oracles at each chart's physical
sample points, not unrelated arrays on different grids. Operation counts refer
to the displayed dense formulas, not measured native instructions.

Cylindrical order is `(r, phi, z)`; spherical order is `(r, theta, phi)` with
colatitude theta. Components use the local orthonormal frame. The chart excludes
origins/poles and uses uniform coordinate grids with second-order one-sided
endpoints. Identities with a product rule are convergence tests, not claims of
exact finite-difference product rules. No native curvilinear backend is claimed.

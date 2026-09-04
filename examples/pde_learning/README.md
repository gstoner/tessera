# Learning PDE solutions with Tessera

Run `PYTHONPATH=python python3 examples/pde_learning/pde_learning_tutorial.py`.
No PyTorch, device, compiler build, or downloaded data is required. This is a
small reference-lane reimplementation inspired by the PINN and Deep Kolmogorov
chapters of Jentzen, Kuckuck and von Wurstemberger, *Mathematical Introduction
to Deep Learning* (arXiv:2310.20360v3), not a reproduction of their benchmark.

The first example learns `u_t = 0.005 Δu + u - u³` on `[-1,1]²`, with zero
Dirichlet boundary data, initial data `0.15(1-x²)³(1-y²)³`, and times through
0.1. The ansatz enforces initial/boundary values exactly. Its fixed polynomial
features form a smooth network; `nn.functional.linear` supplies the trainable
readout, `autodiff.grad` supplies parameter gradients, `optim.adam` updates it,
and Philox supplies collocation points. Spatial Laplacians are computed by
`laplacian_exact(jet_trace(...))`. The linearity of differentiation permits
precomputing these exact derivatives for the frozen features. The time
polynomials are differentiated analytically.

This deliberately does not claim trainable hidden-layer PINNs: differentiating
an arbitrary trainable network's jet coefficients with reverse AD needs a
further mixed-mode capability. No finite-difference derivatives enter the loss.
An independent finite-difference/RK4 solve checks the learned solution at three
times, and a finer space/time solve checks the oracle's error. The tutorial
asserts RMS error below 0.002 and oracle disagreement below 0.00015.

The second example learns the heat-semigroup expectation of the quadratic
payoff `|x|²` from Philox Gaussian trajectories. A quadratic feature readout is
trained on noisy Monte Carlo targets and checked on held-out points against
`|x|² + 4 nu t` in two dimensions. Its generator is separately checked using
an exact jet Laplacian. The asserted held-out RMS tolerance is 0.02.

These are bounded correctness examples, not performance or generalization
claims. Every reported loss and error is computed and checked.

# Practice Projects

Here are some project ideas to help reinforce this material.

## Measuring convergence of our Euler solver

We measured convergence with advection by comparing to the exact
solution.  But what about the case when we don't know the exact
solution?  We can use a grid convergence study to assess the
convergence.  Here's how it works.

1. Pick a smooth problem -- a good problem is the acoustic pulse
   described in this paper:

   [A high-order
   finite-volume method for conservation laws on locally refined
   grids](https://msp.org/camcos/2011/6-1/p01.xhtml)

   See section 4.2.  You'll do this in 1-d in our solver with periodic
   BCs.

2. Run the problem at 4 different resolutions, each varying by a
   factor of 2, e.g., 32, 64, 128, and 256 zones.

3. Compute an error between the run with N zones and the run with 2N
   zones as follows:

   * Coarsen the problem with 2N zones down to N zones by averaging
     2 fine zones into a single coarse zone.

   * Compute the :math:`L_2` norm of the difference between the
     coarsened 2N zone run and the N zone run.

   * Do this for all pairs, so for the 4 runs proposed above, you'd
     have 3 errors corresponding to 64-128, 128-256, and 256-512.

4. Plot the errors along with a line representing ideal 2nd order convergence.

optional: try a different integrator, like 4th order Runge-Kutta and
see how that changes the convergence.


## Sedov explosion and spherical symmetry

We solved the equations of hydrodynamics in 1-d Cartesian coordinates.
If we want to model something that is spherically symmetric, we can do
1-d spherical coordinates.

In 1-d spherical coordinates, the equations of hydrodynamics are:

$$
\begin{align*}
\frac{\partial \rho}{\partial t} + \frac{1}{r^2} \frac{\partial (r^2 \rho u)}{\partial r} &= 0 \\
\frac{\partial (\rho u)}{\partial t} + \frac{1}{r^2} \frac{\partial (r^2 \rho u^2)}{\partial r} + \frac{\partial p}{\partial r} &= 0 \\
\frac{\partial (\rho E)}{\partial t} + \frac{1}{r^2} \frac{\partial }{\partial r} \left [ r^2 (\rho E + p) u \right ] &= 0
\end{align*}
$$

The main difference is that the divergence has area and volume terms now.

A good problem to try with this is the Sedov blast wave explosion.  In
this problem, you put a lot of energy into a point at the center (the
origin of our spherical coordinates) and watch a spherical blast wave
move outward.  There is an analytic solution for a gamma-law gas that
can be compared to.

To solve this with the method-of-lines approach, we would need to:

1. add the $r$ terms to the conservative update.

2. implement reflecting boundary conditions at the origin of coordinates.

3. setup the Sedov problem (a lot of sources can give the initial conditions)
   and run and compare to the exact solution.


## HLL Riemann solver

We solved the Riemann problem for the Euler equations exactly, but
many times in practice we use approximate Riemann solvers.  The HLL
solver is a popular solver.  Research this method and implement it in
the Euler code and compare the solutions you get with it to those with
the exact solver.


## Piecewise parabolic reconstruction for advection

In class, we considered piecewise constant and piecewise linear
reconstruction with advection.  The next step is piecewise parabolic
reconstruction.  This is described originally here:

[The Piecewise Parabolic Method (PPM) for Gas-Dynamical Systems](https://crd.lbl.gov/assets/pubs_presos/AMCS/ANAG/A141984.pdf)

We want to try this with our method-of-lines integration scheme, which
considerably simplifies things compared to that original paper.

The basic idea is as follows:

1. In each zone, we construct a parabola, given by CW Eq. 1.4.
   Eqs. 1.5 through 1.10 give the method for computing the 3
   coefficients of the parabola for each cell as well as limiting them
   so as to not introduce any new extrema.
   
2. For the method-of-lines scheme we are using, we simply evaluate the parabola
   on each interface and this gives that's zones edge state.  The CW paper uses
   $\xi$ as the space coordinate, were $\xi_i$ is a zone center and $\xi_{i-1/2}$
   and $\xi_{i+1/2}$ are the left and right edges.  For a parabolic reconstruction
   in zone $i$ of the form $a(\xi)$, our interface states are:
   
   $$a_{i-1/2,R} = a(\xi_{i-1/2})$$
   $$a_{i+1/2,L} = a(\xi_{i+1/2})$$

3. Compare the solution you get with PPM for the Sod problem to the
   one we got with piecewise linear slopes.

## 2-d advection

The linear advection equation in 2-d is:

$$a_t + u a_x + v a_y = 0$$

In conservative form, we'd write this as:

$$\frac{\partial a}{\partial t} + \frac{\partial (u a)}{\partial x} + \frac{\partial (v a)}{\partial y} = 0$$

We can develop a finite volume method by defining an average as:

$$\langle a \rangle_{i,j} = \frac{1}{\Delta x}{\Delta y} \int_{x_{i-1/2}}^{x_{i+1/2}} \int_{y_{j-1/2}}^{y_{j+1/2}} a(x, y) dx dy$$

and our final update would look like (dropping the $\langle \rangle$):

$$\frac{\partial}{\partial t} a_{i,j} = - \frac{1}{\Delta x} (F^{(x)}_{i+1/2,j} - F^{(x)}_{i-1/2,j}) - \frac{1}{\Delta x} (F^{(y)}_{i,j+1/2} - F^{(y)}_{i,j-1/2})$$

where $F^{(x)} = u a$ and $F^{(y)} = v a$.

This can be solved using the same method-of-lines technique we did in
1-d, but now we need to create and manage a 2-d grid, fill ghost cells
on both $x$ and $y$ boundaries, and compute fluxes through both $x$
and $y$ interfaces.  But the flux computations are done simply by
reconstructing in one coordinate direction and solving the Riemann
problem in that direction.

Code up a 2-d advection solver and test it on advecting a Gaussian.


## Non-conservation?

Suppose instead of solving the total energy equation in the Euler
solver, you instead discretized the internal energy evolution
equation:

$$\frac{\partial (\rho e)}{\partial t} + \frac{\partial (\rho e u)}{\partial x} + p \frac{\partial u}{\partial x} = 0$$

You can compute the flux and the $p \partial u/\partial x$ term using the solution from the Riemann problem.

Code this up and run the Sod problem -- how well do you agree with the exact solution?


## Few-body integration and energy



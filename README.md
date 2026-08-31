# Vlasov-Poisson model solver

This simulation of the Vlasov-Poisson system dimensions uses

- [NuFI algorithm](https://doi.org/10.1002/pamm.202300162)
- [deal.ii](https://dealii.org/) FEM package

______________________________________________________________________

# How to use

## Dependencies:

- [deal.ii](https://dealii.org/) >= 9.6.0
- OpenMP
- [perf](https://perfwiki.github.io/main/) + [flameGraph](https://github.com/brendangregg/FlameGraph) (optional)

After cloning the repo to your machine `cd` into it and run:

```
mkdir -p build; cd build/
cmake ..
make
cd ..
```

Then check/change `parameters.lua` and run with `./build/nufi_poisson`.
If you want to run with a [FlameGraph](https://www.brendangregg.com/flamegraphs.html) visualization at the end use `./run`.

______________________________________________________________________

# Further description

dimensions: 1x1v

status: builds, runs, fails on runtime for very high nr of DOFs

solver didnt converge but error was of order e-7 for 9th grid version with 4k dofs, failed on first step after last refinement, probably due to "over refinement" due to the acumulated error at low refinement levels.

notes:

- -O3 compile flag accelerated runtime about 100x
- locator up to O(dim * Max_depth)
- dealii grid is periodic on all directions
- |E| and |E|^2 saved every time from latest solution
- assemble system collects all q_points to be evaluated and hands them to the rhs_function defined in run (NuFISolver::eval_rho)
- eval sequence:
    - eval_rho (called once per it for all q_points)
    - eval_species_density for electrons (and ions) (all points)
        - makes full v_eval (on all V_DIMs)
        - parallel on for(xi<x_size)
            - calls eval_density_at_x for each x
                - calls eval_ftilda_batch in chunks of (xi, v_chunked)
            - density\[xi\] = rho_local (@xi) * v_cell_volume
        - calls eval_ftilda_batch for all X and V (way way too big)
    - density integrated from outputed f 

to-do:

- Look at multigrid solvers (steps 16, 37, 64) (for GPU solver, probably not needed, GPU needed for backwards characteristics)
- add electro-magnetic
- test other f0 s
- test longer simulations with more plotted points




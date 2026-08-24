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

status: builds, runs, fails on runtime

solver didnt converge but error was of order e-7 for 9th grid version with 4k dofs, failed on first step after last refinement, probably due to "over refinement" due to the acumulated error at low refinement levels.

notes:

- -O3 compile flag accelerated runtime about 100x
- Locator not optimized for 1d.
- Works for higher dimensions (to be tested)
- locator up to O(dim * Max_depth)

to-do:

- add ions
- for 1x2v add electro-magnetic
- test other f0 s
- test longer simulations with more plotted points

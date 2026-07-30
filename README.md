# Vlasov-Poisson model solver

This simulation of the Vlasov-Poisson system dimensions uses

- [NuFI algorithm](https://doi.org/10.1002/pamm.202300162)
- [deal.ii](https://dealii.org/) FEM package

______________________________________________________________________

dimensions: 1x1v

notes:

- about 3 times slower than previous locator

status: Working, to be re-reviewed

Refinement working:

- grid versions saved on a vector
- solutions point to a version of the grid

todo:

- add ions

# Vlasov-Poisson model solver

This simulation of the Vlasov-Poisson system dimensions uses

- [NuFI algorithm](https://doi.org/10.1002/pamm.202300162)
- [deal.ii](https://dealii.org/) FEM package

______________________________________________________________________

dimensions: 1x1v

notes:

- Locator not optimized for 1d.
- works for higher dimensions
- locator up to O(dim * Max_depth)
-

status: Working

Refinement working:

- grid versions saved on a vector
- solutions point to a version of the grid

todo:

- add ions

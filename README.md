# Vlasov-Poisson model solver 

This simulation of the Vlasov-Poisson system dimensions uses
- [NuFI algorithm](https://doi.org/10.1002/pamm.202300162)
- [deal.ii](https://dealii.org/) FEM package

---
dimensions: 1x2v

status: working, needs testing, not optimized (at all!)

Refinement working:
- grid versions saved on a vector
- solutions point to a version of the grid

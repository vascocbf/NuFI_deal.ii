# Vlasov-Poisson model solver 

This simulation of the Vlasov-Poisson system in 1x1v dimensions uses
- [NuFI algorithm](https://doi.org/10.1002/pamm.202300162)
- [deal.ii](https://dealii.org/) FEM package

---
status:

Fail
 
Initial E = 0 constant....

last sim:

Timestep 80 / 1000 (simulation time = 8)
cells = 396
 dofs = 1189
Refinement Started
Refinement Finished
Grid written to results/grid_80.gnuplot
Refinement step done in 0.000000[s]
step made in 42.0398 seconds

Saving results...   Time since start = 0

Timestep 81 / 1000 (simulation time = 8.1)
cells = 517
 dofs = 1552

Exception:

--------------------------------------------------------
An error occurred in line <1345> of file </usr/include/deal.II/lac/solver_cg.h> in function
    void dealii::SolverCG<VectorType>::solve(const MatrixType&, VectorType&, const VectorType&, const PreconditionerType&) [with MatrixType = dealii::SparseMatrix<double>; PreconditionerType = dealii::PreconditionIdentity; VectorType = dealii::Vector<double>]
The violated condition was:
    solver_state == SolverControl::success
Additional information:
    Iterative method reported convergence failure in step 5000. The
    residual in the last step was 0.00370817.

    This error message can indicate that you have simply not allowed a
    sufficiently large number of iterations for your iterative solver to
    converge. This often happens when you increase the size of your
    problem. In such cases, the last residual will likely still be very
    small, and you can make the error go away by increasing the allowed
    number of iterations when setting up the SolverControl object that
    determines the maximal number of iterations you allow.

    The other situation where this error may occur is when your matrix is
    not invertible (e.g., your matrix has a null-space), or if you try to
    apply the wrong solver to a matrix (e.g., using CG for a matrix that
    is not symmetric or not positive definite). In these cases, the
    residual in the last iteration is likely going to be large.

Stacktrace:
-----------
#0  ./nufi_poisson: void dealii::SolverCG<dealii::Vector<double> >::solve<dealii::SparseMatrix<double>, dealii::PreconditionIdentity>(dealii::SparseMatrix<double> const&, dealii::Vector<double>&, dealii::Vector<double> const&, dealii::PreconditionIdentity const&)
#1  ./nufi_poisson: PoissonProblem<1>::solve()
#2  ./nufi_poisson: PoissonProblem<1>::solve_step()
#3  ./nufi_poisson: NuFISolver::run()
#4  ./nufi_poisson: main
--------------------------------------------------------

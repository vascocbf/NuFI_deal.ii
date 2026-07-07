#ifndef POISSON_PROBLEM_H
#define POISSON_PROBLEM_H

#include <deal.II/base/function.h>

#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/numerics/vector_tools_evaluate.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/vector_tools_point_gradient.h>
#include <deal.II/numerics/vector_tools_point_value.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nufi/cells.h"
#include "nufi/parameters.h"

using namespace dealii;

// =-=-=-=-= Poisson Solver =-=-=-=-=

template <int dim> class PoissonProblem {
public:
  PoissonProblem(unsigned int degree);

  void initialize();
  void solve_step();
  void coarse_and_refine_grid();
  void run();

  void set_rhs_function(std::function<double(const Point<dim> &)> f);

  const Vector<double> &get_solution() const { return solution; }
  const MappingQ<dim> &get_mapping() const { return mapping; }
  const DoFHandler<dim> &get_dof_handler() const { return dof_handler; }

  std::vector<double> sample_electric_field(double x_min, double x_max,
                                            unsigned int Nx);
  std::vector<double> sample_electric_potential(double x_min, double x_max,
                                                unsigned int Nx);

  std::vector<double>
  eval_vector_grad(const Vector<double> &solution,
                   const std::vector<Point<dim>> &points) const;

  void save_grid_to_file(const std::string &filename) const;

private:
  void create_mesh();
  void setup_system();
  void assemble_system();
  void solve();

  Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution; // phi
  Vector<double> system_rhs;

  std::function<double(const Point<dim> &)> rhs_function;

  MappingQ<dim> mapping;

  CellLocator<dim> cell_locator;
  // std::vector<typename DoFHandler<dim>::active_cell_iterator> active_cells;

  mutable std::vector<double> local_solution_buffer;
  mutable std::unique_ptr<FEPointEvaluation<dim, dim>> evaluator;
};

//====//====//
// Utilities
//====//====//

template <int dim>
void PoissonProblem<dim>::set_rhs_function(
    std::function<double(const Point<dim> &)> f) {
  rhs_function = std::move(f);
}

template <int dim>
PoissonProblem<dim>::PoissonProblem(unsigned int degree)
    : fe(degree), dof_handler(triangulation), mapping(degree) {}

template <int dim>
std::vector<double>
PoissonProblem<dim>::sample_electric_field(double x_min, double x_max,
                                           unsigned int Nx) {
  std::vector<double> E_values(Nx);

  const double dx = (x_max - x_min) / (Nx - 1);

  for (unsigned int i = 0; i < Nx; ++i) {
    const double x = x_min + i * dx;
    const Point<dim> point(x);

    // 1. Find the active cell containing x
    const auto cell_point_pair =
        GridTools::find_active_cell_around_point(mapping, dof_handler, point);

    const auto cell = cell_point_pair.first;
    const Point<dim> &unit_point = cell_point_pair.second;

    // 2. FEPointEvaluation expects an ArrayView of points
    std::vector<Point<dim>> points(1, unit_point);
    ArrayView<const Point<dim>> point_view(points);

    FEPointEvaluation<1, dim> evaluator(mapping, dof_handler.get_fe(),
                                        update_gradients);

    // reinit with ArrayView of points
    evaluator.reinit(cell, point_view);

    Vector<double> local_dofs(dof_handler.get_fe().dofs_per_cell);
    cell->get_dof_values(solution, local_dofs);

    // 3. Evaluate gradient at this point
    evaluator.evaluate(local_dofs, EvaluationFlags::gradients);

    const Tensor<1, dim> grad_phi = evaluator.get_gradient(0);

    // 4. Compute E = -grad(phi)
    E_values[i] = -grad_phi[0];
  }

  return E_values;
}

template <int dim>
std::vector<double>
PoissonProblem<dim>::sample_electric_potential(double x_min, double x_max,
                                               unsigned int Nx) {
  std::vector<double> values(Nx);
  std::vector<Point<dim>> eval_points(Nx);

  double Lx = x_max - x_min;
  double dx = Lx / Nx;

  for (unsigned int i = 0; i < Nx; ++i)
    eval_points[i] = Point<1, double>(x_min + i * dx);

  Utilities::MPI::RemotePointEvaluation<dim, dim> cache;
  cache.reinit(eval_points, triangulation, mapping);

  values = VectorTools::point_values<dim>(cache, dof_handler, solution);

  return values;
}

template <int dim>
std::vector<double> PoissonProblem<dim>::eval_vector_grad(
    const Vector<double> &solution,
    const std::vector<Point<dim>> &points) const {

  std::vector<double> values(points.size());

  for (unsigned int p = 0; p < points.size(); ++p) {

    const auto cell = cell_locator.locate(points[p]);

    cell->get_dof_values(solution, local_solution_buffer.begin(),
                         local_solution_buffer.end());

    evaluator->reinit(cell, ArrayView<const Point<dim>>(&points[p], 1));
    evaluator->evaluate(local_solution_buffer, EvaluationFlags::gradients);

    values[p] = evaluator->get_gradient(0)[0];
  }

  return values;
}

template <int dim>
std::vector<double>
eval_point_grad(const Mapping<dim> &mapping, const DoFHandler<dim> &dof_handler,
                const Vector<double> &solution, const Point<dim> &point) {
  Tensor Ex =
      VectorTools::point_gradient(mapping, dof_handler, solution, point);
  return Ex[0];
}
//
// template <int dim>
// std::vector<double> eval_vector_grad(const Mapping<dim> &mapping,
//                                      const DoFHandler<dim> &dof_handler,
//                                      const Vector<double> &solution,
//                                      const std::vector<Point<dim>> &points) {
//   size_t p_size = points.size();
//
//   std::vector<double> Ex(p_size);
//   for (size_t i = 0; i < p_size; ++i)
//     Ex[i] = eval_point_grad(mapping, dof_handler, solution, points[i]);
//
//   return Ex;
// }

template <int dim>
double eval_point_value(const Mapping<dim> &mapping,
                        const DoFHandler<dim> &dof_handler,
                        const Vector<double> &solution,
                        const Point<dim> &point) {

  return VectorTools::point_value<dim>(mapping, dof_handler, solution, point);
}

template <int dim>
void PoissonProblem<dim>::save_grid_to_file(const std::string &filename) const {
  std::ofstream out(filename);
  GridOut grid_out;
  grid_out.write_svg(triangulation, out);

  std::cout << "Grid written to " << filename << "\n";
}

//======//======//
// dealii Poisson
//======//======//
template <int dim> void PoissonProblem<dim>::create_mesh() {

  GridGenerator::hyper_cube(triangulation, Parameters::X_DOMAIN_LEFT,
                            Parameters::X_DOMAIN_RIGHT);

  std::vector<
      GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
      periodic_faces;

  GridTools::collect_periodic_faces(triangulation, 0, 1, // boundary IDs
                                    0, periodic_faces);

  triangulation.add_periodicity(periodic_faces);

  triangulation.refine_global(Parameters::GLOBAL_REFINEMENT);
}

template <int dim> void PoissonProblem<dim>::setup_system() {

  dof_handler.distribute_dofs(fe);

  constraints.clear();

  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  DoFTools::make_periodicity_constraints(dof_handler, 0, 1, 0, constraints);

  // Gauge fix for periodic Poisson:
  // remove the constant nullspace by pinning one unconstrained DoF.
  // (by Paul Wilhelm)
  types::global_dof_index gauge_dof = numbers::invalid_dof_index;

  for (types::global_dof_index i = 0; i < dof_handler.n_dofs(); ++i) {
    if (!constraints.is_constrained(i)) {
      gauge_dof = i;
      break;
    }
  }

  Assert(gauge_dof != numbers::invalid_dof_index,
         ExcMessage("No unconstrained DoF found for gauge fixing."));

  constraints.add_line(gauge_dof);
  constraints.set_inhomogeneity(gauge_dof, 0.0);

  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  // used for evaluator to avoid running it anytime there is an eval
  cell_locator.rebuild(dof_handler, triangulation);

  local_solution_buffer.resize(fe.n_dofs_per_cell());
  evaluator = std::make_unique<FEPointEvaluation<dim, dim>>(mapping, fe,
                                                            update_gradients);
}

// Paul
template <int dim> void PoissonProblem<dim>::assemble_system() {
  Assert(system_matrix.m() == dof_handler.n_dofs(),
         ExcMessage("Matrix not initialized correctly"));
  system_matrix = 0;
  system_rhs = 0;

  QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);

    cell_matrix = 0;
    cell_rhs = 0;

    for (const auto q : fe_values.quadrature_point_indices()) {
      const double rho = rhs_function(
          fe_values.quadrature_point(q)); // Eval rhs_function at q points

      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                               fe_values.shape_grad(j, q) * fe_values.JxW(q);

      for (const unsigned int i : fe_values.dof_indices())
        cell_rhs(i) += fe_values.shape_value(i, q) * rho * fe_values.JxW(q);
    }

    cell->get_dof_indices(local_dof_indices);

    constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
  }
}

template <int dim> void PoissonProblem<dim>::coarse_and_refine_grid() {
  // Add refinement and coasring algorithm here
  Vector<float> error_per_cell(triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate(
      dof_handler, QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(), solution,
      error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number(
      triangulation, error_per_cell, 0.3,
      0.03); // These numbers are to be reviewd and changed

  triangulation.prepare_coarsening_and_refinement();

  // solution tranafer

  // setup_system

  // solution_transfer interpolate
  // non_zero_constraint.distribute(current solution)
  //
  //======//======//
  // CHECK step-15.

  // rebuild cells with the grid
  cell_locator.rebuild(dof_handler, triangulation);
}

template <int dim> void PoissonProblem<dim>::solve() {

  SolverControl solver_control(Parameters::CONVERGENCE_ITERATIONS,
                               Parameters::CONVERGENCE_LIMIT);
  SolverCG<Vector<double>> solver(solver_control);

  // PreconditionSSOR<SparseMatrix<double>> preconditioner;
  // preconditioner.initialize(system_matrix, 1.2);

  // solver.solve(system_matrix, solution, system_rhs, preconditioner);

  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  constraints.distribute(solution);
}

template <int dim> void PoissonProblem<dim>::initialize() {
  create_mesh();  // build grid
  setup_system(); // distribute DoFs and matrices
}

template <int dim> void PoissonProblem<dim>::solve_step() {
  assemble_system();
  solve();
}

// NuFI doesnt use this, kept only for testing PoissonProblem
template <int dim> void PoissonProblem<dim>::run() {
  create_mesh();
  setup_system();
  assemble_system();
  solve();
}

#endif

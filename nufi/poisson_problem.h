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
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/numerics/vector_tools_evaluate.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nufi/parameters.h"

using namespace dealii;

// =-=-=-=-= Poisson Solver =-=-=-=-=

template <int dim> class PoissonProblem {
public:
  PoissonProblem(unsigned int degree);

  void initialize();
  void solve_step();
  void run();

  void set_rhs_function(std::unique_ptr<Function<dim>> rhs_function);

  const Vector<double> &get_solution() const { return solution; }
  const DoFHandler<dim> &get_dof_handler() const { return dof_handler; }
  const MappingQ<dim> &get_mapping() const { return mapping; }

  std::vector<double> sample_electric_field(double x_min, double x_max,
                                            unsigned int Nx);
  std::vector<double> sample_electric_potential(double x_min, double x_max,
                                                unsigned int Nx);

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

  std::unique_ptr<const Function<dim>> rhs_function;

  MappingQ<dim> mapping;

  // std::unique_ptr<Functions::FEFieldFunction<dim>> fe_field_function;
};

// Utilities

template <int dim>
void PoissonProblem<dim>::set_rhs_function(std::unique_ptr<Function<dim>> rhs) {
  rhs_function = std::move(rhs);
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

// // by GPT to re-re-re-check
// template <int dim> std::vector<double> eval_solution_on_points(
//         const std::vector<Vector<double>> &solutions,
//         const unsigned int n,
//         const std::vector<Point<dim>> &points, // need to be in [x_min,
//         x_max]. I think.... const std::vector<unsigned int> &cell_indices,
//         const DoFHandler<dim> &dof_handler,
//         const MappingQ<dim> &mapping)
// {
//     AssertIndexRange(n, solutions.size());
//     Assert(points.size() == cell_indices.size(),
//            ExcMessage("points and cell_indices must have same size"));
//
//     const Vector<double> &solution = solutions[n];
//
//     std::vector<double> result(points.size());
//
//     // Group points by cell (required for FEPointEvaluation efficiency)
//     std::map<unsigned int, std::vector<unsigned int>> cell_to_point_ids;
//
//     for (unsigned int i = 0; i < points.size(); ++i)
//         cell_to_point_ids[cell_indices[i]].push_back(i);
//
//     FEPointEvaluation<1, dim> evaluator(mapping,
//                                         dof_handler.get_fe(),
//                                         update_values);
//
//     std::vector<Point<dim>> cell_points;
//     Vector<double> local_dofs(dof_handler.get_fe().dofs_per_cell);
//
//     for (const auto &entry : cell_to_point_ids)
//     {
//         const unsigned int cell_id = entry.first;
//         const auto &point_ids = entry.second;
//
//         // these two lines bellow assume some order not sure how or why
//         auto cell = dof_handler.begin_active();
//         std::advance(cell, cell_id);
//
//         // extract points belonging to this cell
//         cell_points.clear();
//         cell_points.reserve(point_ids.size());
//
//         for (unsigned int id : point_ids)
//             cell_points.push_back(points[id]);
//
//         std::vector<types::global_dof_index>
//         indices(dof_handler.get_fe().n_dofs_per_cell());
//         cell->get_dof_indices(indices);
//
//         for (unsigned int i=0;i<indices.size();++i)
//           local_dofs[i] = solution[indices[i]];
//
//         // initialize evaluator on this cell
//         evaluator.reinit(cell, cell_points);
//
//         evaluator.evaluate(local_dofs, EvaluationFlags::values);
//
//         for (unsigned int k = 0; k < point_ids.size(); ++k)
//             result[point_ids[k]] = evaluator.get_value(k);
//     }
//
//     return result;
// }

template <int dim>
double eval_point(const Mapping<dim> &mapping,
                  const DoFHandler<dim> &dof_handler,
                  const Vector<double> &solution, const Point<dim> &point) {
  return VectorTools::point_value<dim>(mapping, dof_handler, solution, point);
}

// dealii Poisson

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

  // fe_field_function =
  // std::make_unique<Functions::FEFieldFunction<dim>>(
  //   dof_handler, solution, mapping);
}
/* (Mine)
template <int dim>
void PoissonProblem<dim>::assemble_system()
{

  system_matrix = 0;
  system_rhs = 0;

  QGauss<dim>  quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values |
                          update_gradients |
                          update_quadrature_points |
                          update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  Assert(rhs_function != nullptr, ExcMessage("RHS function not set"));

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    cell_matrix = 0;
    cell_rhs    = 0;

    for (const auto q : fe_values.quadrature_point_indices())
    {
      const double rho = rhs_function->value(fe_values.quadrature_point(q));

      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) +=
            (fe_values.shape_grad(i, q) * // grad phi_i(x_q)
             fe_values.shape_grad(j, q) * // grad phi_j(x_q)
             fe_values.JxW(q));           // dx

      for (const unsigned int i : fe_values.dof_indices())
        cell_rhs(i) += (fe_values.shape_value(i, q) * // phi_i(x_q)
                        rho *                         // f(x_q)
                        fe_values.JxW(q));            // dx


    }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(cell_matrix,
                                           cell_rhs,
                                           local_dof_indices,
                                           system_matrix,
                                           system_rhs);
    for (const unsigned int i : fe_values.dof_indices())
      for (const unsigned int j : fe_values.dof_indices())
        system_matrix.add(local_dof_indices[i],
                          local_dof_indices[j],
                          cell_matrix(i, j));

    for (const unsigned int i : fe_values.dof_indices())
      system_rhs(local_dof_indices[i]) += cell_rhs(i);

  }
  std::map<types::global_dof_index, double> boundary_values;
  // VectorTools::interpolate_boundary_values(dof_handler,
  //                                          types::boundary_id(0),
  //                                          Functions::ZeroFunction<1>(),
  //                                          boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}
 */

// Paul's, mine's above

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

  Assert(rhs_function != nullptr, ExcMessage("RHS function not set"));

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);

    cell_matrix = 0;
    cell_rhs = 0;

    for (const auto q : fe_values.quadrature_point_indices()) {
      const double rho = rhs_function->value(fe_values.quadrature_point(q));

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

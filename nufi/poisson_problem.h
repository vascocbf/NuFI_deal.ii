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

#include <cstddef>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/vector_tools_point_gradient.h>
#include <deal.II/numerics/vector_tools_point_value.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "nufi/cells.h"
#include "nufi/grids.h"
#include "nufi/parameters.h"
#include "nufi/stopwatch.h"
#include "omp.h"

void save_space_vector(const std::vector<double> &vals,
                       const std::string &filename, size_t it);

using namespace dealii;

// =-=-=-=-= Poisson Solver =-=-=-=-=

template <int X_DIM> class PoissonProblem {
public:
  PoissonProblem(unsigned int degree);

  void initialize();

  // TO-UPDATE for X_DIM template =================
  double solve_step(size_t it, std::vector<GridStructure<X_DIM>> &grid_versions,
                    bool refining = false);
  void coarse_and_refine_grid(size_t it);
  void setup_constraints(AffineConstraints<double> &constraints);
  void run();

  unsigned int get_rhs_size();
  unsigned int get_dof_size();
  void set_rhs_function(
      std::function<std::vector<double>(const std::vector<Point<X_DIM>> &)> f);

  const Vector<double> &get_solution() const { return solution; }
  const MappingQ<X_DIM> &get_mapping() const { return mapping; }
  const DoFHandler<X_DIM> &get_dof_handler() const { return dof_handler; }
  const Triangulation<X_DIM> &get_triangulation() const {
    return triangulation;
  }
  const FE_Q<X_DIM> &get_fe() const { return fe; }
  const AffineConstraints<double> &get_constraints() const {
    return constraints;
  }
  const CellLocator<X_DIM> &get_locator() const { return cell_locator; }
  double get_error_estimate() const { return error_estimate; }

  std::vector<double> sample_electric_field(double x_min, double x_max,
                                            unsigned int Nx);
  std::vector<double> sample_electric_potential(double x_min, double x_max,
                                                unsigned int Nx);

  std::vector<double>
  eval_vector_grad(const Vector<double> &solution,
                   const std::vector<Point<X_DIM>> &points) const;

  void save_grid_to_file(std::string &filename) const;

private:
  void create_mesh();
  void setup_system();
  void assemble_system();
  void solve(size_t it);
  void estimate_error();

  std::function<std::vector<double>(const std::vector<Point<X_DIM>> &)>
      rhs_function;

  MappingQ<X_DIM> mapping;
  FE_Q<X_DIM> fe;
  AffineConstraints<double> constraints;
  Triangulation<X_DIM> triangulation;
  DoFHandler<X_DIM> dof_handler;
  CellLocator<X_DIM> cell_locator;
  Vector<double> solution; // phi
  SparsityPattern sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double> system_rhs;

  const bool PRINT_GAUGE_DOF_POSITION = true;
  double error_estimate = 0.0;
};

//====//====//
// Utilities
//====//====//

template <int X_DIM> unsigned int PoissonProblem<X_DIM>::get_rhs_size() {

  QGauss<X_DIM> quadrature_formula(fe.degree + 1);
  return quadrature_formula.size();
}

template <int X_DIM> unsigned int PoissonProblem<X_DIM>::get_dof_size() {
  return dof_handler.n_dofs();
}

template <int X_DIM>
void PoissonProblem<X_DIM>::set_rhs_function(
    std::function<std::vector<double>(const std::vector<Point<X_DIM>> &)> f) {
  rhs_function = std::move(f);
}

template <int X_DIM>
PoissonProblem<X_DIM>::PoissonProblem(unsigned int degree)
    : mapping(degree), fe(degree), triangulation(), dof_handler(triangulation) {
}

// Uses FEPointEvaluation
//
// TODO: update for dim templating
template <int X_DIM>
std::vector<double>
PoissonProblem<X_DIM>::sample_electric_field(double x_min, double x_max,
                                             unsigned int Nx) {
  std::vector<double> E_values(Nx);

  const double dx = (x_max - x_min) / (Nx - 1);

  for (unsigned int i = 0; i < Nx; ++i) {
    const double x = x_min + i * dx;
    const Point<X_DIM> point(x);

    const auto cell_point_pair =
        GridTools::find_active_cell_around_point(mapping, dof_handler, point);

    const auto cell = cell_point_pair.first;
    const Point<X_DIM> &unit_point = cell_point_pair.second;

    std::vector<Point<X_DIM>> points(1, unit_point);
    ArrayView<const Point<X_DIM>> point_view(points);

    FEPointEvaluation<1, X_DIM> evaluator(mapping, dof_handler.get_fe(),
                                          update_gradients);

    evaluator.reinit(cell, point_view);

    Vector<double> local_dofs(dof_handler.get_fe().dofs_per_cell);
    cell->get_dof_values(solution, local_dofs);

    evaluator.evaluate(local_dofs, EvaluationFlags::gradients);

    const Tensor<1, X_DIM> grad_phi = evaluator.get_gradient(0);

    E_values[i] = -grad_phi[0];
  }

  return E_values;
}

// TODO: update for dim templating.
template <int X_DIM>
std::vector<double>
PoissonProblem<X_DIM>::sample_electric_potential(double x_min, double x_max,
                                                 unsigned int Nx) {
  std::vector<double> values(Nx);
  std::vector<Point<X_DIM>> eval_points(Nx);

  double Lx = x_max - x_min;
  double dx = Lx / Nx;

  for (unsigned int i = 0; i < Nx; ++i)
    eval_points[i] = Point<1, double>(x_min + i * dx);

  Utilities::MPI::RemotePointEvaluation<X_DIM, X_DIM> cache;
  cache.reinit(eval_points, triangulation, mapping);

  values = VectorTools::point_values<X_DIM>(cache, dof_handler, solution);

  return values;
}

template <int X_DIM>
void PoissonProblem<X_DIM>::save_grid_to_file(std::string &filename) const {
  GridOut grid_out;

  if (X_DIM >= 2) {
    filename += ".svg";
    std::ofstream out(filename);
    grid_out.write_svg(triangulation, out);
  } else if (X_DIM == 1) {
    filename += ".vtu";
    std::ofstream out(filename);
    grid_out.write_vtu(triangulation, out);
  }

  std::cout << "Grid written to " << filename << "\n";
}

//======//======//
// dealii Poisson
//======//======//

template <int X_DIM> void PoissonProblem<X_DIM>::create_mesh() {

  Point<X_DIM> lower_left, upper_right;
  for (unsigned int d = 0; d < X_DIM; ++d) {
    lower_left[d] = Parameters::X_DOMAIN_LEFT[d];
    upper_right[d] = Parameters::X_DOMAIN_RIGHT[d];
  }

  GridGenerator::hyper_rectangle(triangulation, lower_left, upper_right,
                                 /*colorize=*/true);

  std::vector<
      GridTools::PeriodicFacePair<typename Triangulation<X_DIM>::cell_iterator>>
      periodic_faces;

  for (unsigned int d = 0; d < X_DIM; ++d)
    GridTools::collect_periodic_faces(triangulation, 2 * d, 2 * d + 1, d,
                                      periodic_faces);

  triangulation.add_periodicity(periodic_faces);

  triangulation.refine_global(Parameters::GLOBAL_REFINEMENT);
}

template <int X_DIM>
void PoissonProblem<X_DIM>::setup_constraints(
    AffineConstraints<double> &constraints) {

  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  for (unsigned int d = 0; d < X_DIM; ++d)
    DoFTools::make_periodicity_constraints(dof_handler, 2 * d, 2 * d + 1, d,
                                           constraints);

  const auto support_points =
      DoFTools::map_dofs_to_support_points(mapping, dof_handler);

  types::global_dof_index gauge_dof = numbers::invalid_dof_index;

  for (const auto &[dof, point] : support_points) {
    if (constraints.is_constrained(dof))
      continue;

    gauge_dof = dof;

    if (PRINT_GAUGE_DOF_POSITION)
      std::cout << " gauge_dof = " << gauge_dof << " gauge_point = " << point[0]
                << std::endl;
    break;
  }

  Assert(gauge_dof != numbers::invalid_dof_index,
         ExcMessage("No gauge DoF found in protected gauge region."));

  constraints.add_line(gauge_dof);
  constraints.set_inhomogeneity(gauge_dof, 0.0);
}

template <int X_DIM> void PoissonProblem<X_DIM>::setup_system() {

  dof_handler.distribute_dofs(fe);

  constraints.clear();

  setup_constraints(constraints);

  constraints.close();

  // DSP
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  // used for evaluator to avoid running it anytime there is an eval
  cell_locator.rebuild(dof_handler, triangulation);
}

template <int X_DIM> void PoissonProblem<X_DIM>::assemble_system() {

  Assert(system_matrix.m() == dof_handler.n_dofs(),
         ExcMessage("Matrix not initialized correctly"));
  system_matrix = 0;
  system_rhs = 0;

  const QGauss<X_DIM> quadrature_formula(fe.degree + 1);
  FEValues<X_DIM> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Eval rhs_function only once for all quadrature points
  const unsigned int n_q_points = quadrature_formula.size();
  std::vector<Point<X_DIM>> all_q_points;
  all_q_points.reserve(triangulation.n_active_cells() * n_q_points);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);
    const auto &q_points = fe_values.get_quadrature_points();
    all_q_points.insert(all_q_points.end(), q_points.begin(), q_points.end());
  }

  Assert(rhs_function,
         ExcMessage("Poisson RHS function has not been initialized."));

  std::cout << "[PoissonProblem::assemble_system] Start of full rho eval..."
            << "\n";
  std::vector<double> all_rho = rhs_function(all_q_points);
  std::cout << "[PoissonProblem::assemble_system] End of full rho eval..."
            << "\n";

  Assert(all_rho.size() == all_q_points.size(),
         ExcMessage("rhs_function returned wrong size"));

  // assemble system
  unsigned int q_offset = 0;
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);

    cell_matrix = 0;
    cell_rhs = 0;

    for (size_t q = 0; q < n_q_points; ++q) {

      for (const unsigned int i : fe_values.dof_indices()) {
        for (const unsigned int j : fe_values.dof_indices()) {
          cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                               fe_values.shape_grad(j, q) * fe_values.JxW(q);
        }

        cell_rhs(i) += fe_values.shape_value(i, q) * all_rho[q_offset + q] *
                       fe_values.JxW(q);
      }
    }
    q_offset += n_q_points;

    cell->get_dof_indices(local_dof_indices);

    constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
  }
}

template <int X_DIM>
void PoissonProblem<X_DIM>::coarse_and_refine_grid(size_t it) {
  std::cout << "Refinement Started..." << "\n";

  Vector<float> error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<X_DIM>::estimate(
      dof_handler, QGauss<X_DIM - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<X_DIM> *>(), solution,
      error_per_cell);

  // GridRefinement::refine_and_coarsen_fixed_number(triangulation,
  // error_per_cell,
  //                                                 0.3, 0.03);
  GridRefinement::refine_and_coarsen_fixed_fraction(
      triangulation, error_per_cell, Parameters::REFINEMENT_TOP_FRACTION,
      Parameters::REFINEMENT_BOTTOM_FRACTION,
      std::numeric_limits<unsigned int>::max(), VectorTools::L2_norm);

  // Avoid coarsing below GLOBAL_REFINEMENT level for CellLocator
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->level() <= static_cast<int>(Parameters::GLOBAL_REFINEMENT))
      cell->clear_coarsen_flag();

  triangulation.execute_coarsening_and_refinement();

  std::cout << "Refinement Finished..." << "\n";

  std::string grid_file_name =
      Parameters::PLOT_DIR + "grid_" + std::to_string(it);
  save_grid_to_file(grid_file_name);
}

template <int X_DIM> void PoissonProblem<X_DIM>::estimate_error() {
  Vector<float> error_per_cell(triangulation.n_active_cells());

  KellyErrorEstimator<X_DIM>::estimate(
      dof_handler, QGauss<X_DIM - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<X_DIM> *>(), solution,
      error_per_cell);

  error_estimate = error_per_cell.l2_norm();
}

template <int X_DIM>
void PoissonProblem<X_DIM>::solve([[maybe_unused]] size_t it) {
  SolverControl solver_control(Parameters::CONVERGENCE_ITERATIONS,
                               Parameters::CONVERGENCE_LIMIT *
                                   system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);

  PreconditionJacobi<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  constraints.distribute(solution);
}

template <int X_DIM> void PoissonProblem<X_DIM>::initialize() {
  create_mesh();
  setup_system();
}

template <int X_DIM>
double PoissonProblem<X_DIM>::solve_step(
    size_t it, std::vector<GridStructure<X_DIM>> &grid_versions,
    bool refining) {
  double refining_time = 0.0;
  if (refining) {
    stopwatch<double> refining_timer;
    coarse_and_refine_grid(it);
    setup_system();
    update_grid_versions(grid_versions, *this);
    refining_time = refining_timer.elapsed();
  }
  assemble_system();
  solve(it);
  estimate_error();

  return refining_time;
}

// NuFI doesnt use this, kept only for testing PoissonProblem
template <int X_DIM> void PoissonProblem<X_DIM>::run() {
  create_mesh();
  setup_system();
  assemble_system();
  solve();
}

#endif

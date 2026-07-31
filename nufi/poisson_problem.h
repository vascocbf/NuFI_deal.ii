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

template <int dim> class PoissonProblem {
public:
  PoissonProblem(unsigned int degree);

  void initialize();
  double solve_step(size_t it, std::vector<GridStructure<1>> &grid_versions,
                    bool refining = false);
  void coarse_and_refine_grid(size_t it);
  void setup_constraints(AffineConstraints<double> &constraints);
  void run();

  unsigned int get_rhs_size();
  unsigned int get_dof_size();
  void set_rhs_function(
      std::function<std::vector<double>(const std::vector<Point<dim>> &)> f);
  // void set_rhs(const Vector<double> &new_rhs) { rhs = new_rhs; }

  const Vector<double> &get_solution() const { return solution; }
  const MappingQ<dim> &get_mapping() const { return mapping; }
  const DoFHandler<dim> &get_dof_handler() const { return dof_handler; }
  const Triangulation<dim> &get_triangulation() const { return triangulation; }
  const FE_Q<dim> &get_fe() const { return fe; }
  const AffineConstraints<double> &get_constraints() const {
    return constraints;
  }
  const CellLocator<dim> &get_locator() const { return cell_locator; }
  double get_error_estimate() const { return error_estimate; }

  std::vector<double> sample_electric_field(double x_min, double x_max,
                                            unsigned int Nx);
  std::vector<double> sample_electric_potential(double x_min, double x_max,
                                                unsigned int Nx);

  std::vector<double>
  eval_vector_grad(const Vector<double> &solution,
                   const std::vector<Point<dim>> &points) const;

  void save_grid_to_file(std::string &filename) const;

private:
  void create_mesh();
  void setup_system();
  void assemble_system();
  void solve(size_t it);
  void estimate_error();

  std::function<std::vector<double>(const std::vector<Point<dim>> &)>
      rhs_function;

  MappingQ<dim> mapping;
  FE_Q<dim> fe;
  AffineConstraints<double> constraints;
  Triangulation<dim> triangulation;
  DoFHandler<dim> dof_handler;
  CellLocator<dim> cell_locator;
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

template <int dim> unsigned int PoissonProblem<dim>::get_rhs_size() {

  QGauss<dim> quadrature_formula(fe.degree + 1);
  return quadrature_formula.size();
}

template <int dim> unsigned int PoissonProblem<dim>::get_dof_size() {
  return dof_handler.n_dofs();
}

template <int dim>
void PoissonProblem<dim>::set_rhs_function(
    std::function<std::vector<double>(const std::vector<Point<dim>> &)> f) {
  rhs_function = std::move(f);
}

template <int dim>
PoissonProblem<dim>::PoissonProblem(unsigned int degree)
    : mapping(degree), fe(degree), triangulation(), dof_handler(triangulation) {
}

// Uses FEPointEvaluation
template <int dim>
std::vector<double>
PoissonProblem<dim>::sample_electric_field(double x_min, double x_max,
                                           unsigned int Nx) {
  std::vector<double> E_values(Nx);

  const double dx = (x_max - x_min) / (Nx - 1);

  for (unsigned int i = 0; i < Nx; ++i) {
    const double x = x_min + i * dx;
    const Point<dim> point(x);

    const auto cell_point_pair =
        GridTools::find_active_cell_around_point(mapping, dof_handler, point);

    const auto cell = cell_point_pair.first;
    const Point<dim> &unit_point = cell_point_pair.second;

    std::vector<Point<dim>> points(1, unit_point);
    ArrayView<const Point<dim>> point_view(points);

    FEPointEvaluation<1, dim> evaluator(mapping, dof_handler.get_fe(),
                                        update_gradients);

    evaluator.reinit(cell, point_view);

    Vector<double> local_dofs(dof_handler.get_fe().dofs_per_cell);
    cell->get_dof_values(solution, local_dofs);

    evaluator.evaluate(local_dofs, EvaluationFlags::gradients);

    const Tensor<1, dim> grad_phi = evaluator.get_gradient(0);

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
std::vector<double>
eval_point_grad(const Mapping<dim> &mapping, const DoFHandler<dim> &dof_handler,
                const Vector<double> &solution, const Point<dim> &point) {
  Tensor Ex =
      VectorTools::point_gradient(mapping, dof_handler, solution, point);
  return Ex[0];
}

template <int dim>
std::vector<double> eval_vector_grad(const Mapping<dim> &mapping,
                                     const DoFHandler<dim> &dof_handler,
                                     const Vector<double> &solution,
                                     const std::vector<Point<dim>> &points) {
  size_t p_size = points.size();

  std::vector<double> Ex(p_size);
  for (size_t i = 0; i < p_size; ++i)
    Ex[i] = eval_point_grad(mapping, dof_handler, solution, points[i]);

  return Ex;
}

template <int dim>
double eval_point_value(const Mapping<dim> &mapping,
                        const DoFHandler<dim> &dof_handler,
                        const Vector<double> &solution,
                        const Point<dim> &point) {

  return VectorTools::point_value<dim>(mapping, dof_handler, solution, point);
}

template <int dim>
void PoissonProblem<dim>::save_grid_to_file(std::string &filename) const {
  GridOut grid_out;

  if (dim >= 2) {
    filename += ".svg";
    std::ofstream out(filename);
    grid_out.write_svg(triangulation, out);
  } else if (dim == 1) {
    filename += ".vtu";
    std::ofstream out(filename);
    grid_out.write_vtu(triangulation, out);
  }

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

template <int dim>
void PoissonProblem<dim>::setup_constraints(
    AffineConstraints<double> &constraints) {

  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  DoFTools::make_periodicity_constraints(dof_handler, 0, 1, 0, constraints);

  const auto support_points =
      DoFTools::map_dofs_to_support_points(mapping, dof_handler);

  types::global_dof_index gauge_dof = numbers::invalid_dof_index;

  // Search only inside the protected region
  for (const auto &[dof, point] : support_points) {
    if (constraints.is_constrained(dof))
      continue;
    const double x = point[0];
    if (x <= Parameters::X_DOMAIN_LEFT + .5) {
      gauge_dof = dof;

      if (PRINT_GAUGE_DOF_POSITION)
        std::cout << " gauge_dof = " << gauge_dof
                  << " gauge_point = " << point[0] << std::endl;
      break;
    }
  }

  Assert(gauge_dof != numbers::invalid_dof_index,
         ExcMessage("No gauge DoF found in protected gauge region."));

  constraints.add_line(gauge_dof);
  constraints.set_inhomogeneity(gauge_dof, 0.0);
}

template <int dim> void PoissonProblem<dim>::setup_system() {

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

template <int dim> void PoissonProblem<dim>::assemble_system() {

  Assert(system_matrix.m() == dof_handler.n_dofs(),
         ExcMessage("Matrix not initialized correctly"));
  system_matrix = 0;
  system_rhs = 0;

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Eval rhs_function only once for all quadrature points
  const unsigned int n_q_points = quadrature_formula.size();
  std::vector<Point<dim>> all_q_points;
  all_q_points.reserve(triangulation.n_active_cells() * n_q_points);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);
    const auto &q_points = fe_values.get_quadrature_points();
    all_q_points.insert(all_q_points.end(), q_points.begin(), q_points.end());
  }

  Assert(rhs_function,
         ExcMessage("Poisson RHS function has not been initialized."));

  std::cout << "Start of full rho eval..." << "\n";
  std::vector<double> all_rho = rhs_function(all_q_points);
  std::cout << "End of full rho eval..." << "\n";

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

template <int dim> void PoissonProblem<dim>::coarse_and_refine_grid(size_t it) {
  std::cout << "Refinement Started..." << "\n";

  Vector<float> error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate(
      dof_handler, QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(), solution,
      error_per_cell);

  // GridRefinement::refine_and_coarsen_fixed_number(triangulation,
  // error_per_cell,
  //                                                 0.3, 0.03);
  GridRefinement::refine_and_coarsen_fixed_fraction(
      triangulation, error_per_cell, Parameters::REFINEMENT_TOP_FRACTION,
      Parameters::REFINEMENT_BOTTOM_FRACTION,
      std::numeric_limits<unsigned int>::max(), VectorTools::L2_norm);

  triangulation.execute_coarsening_and_refinement();

  std::cout << "Refinement Finished..." << "\n";

  std::string grid_file_name =
      Parameters::PLOT_DIR + "grid_" + std::to_string(it);
  save_grid_to_file(grid_file_name);
}

template <int dim> void PoissonProblem<dim>::estimate_error() {
  Vector<float> error_per_cell(triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate(
      dof_handler, QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(), solution,
      error_per_cell);

  error_estimate = error_per_cell.l2_norm();
}

template <int dim> void PoissonProblem<dim>::solve(size_t it) {

  std::cout << "Calling PoissonProblem::solve for time-step " << it << "\n";

  SolverControl solver_control(Parameters::CONVERGENCE_ITERATIONS,
                               Parameters::CONVERGENCE_LIMIT *
                                   system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);

  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  constraints.distribute(solution);

  // std::ofstream out("results/phi_after_solve_" + std::to_string(it) +
  // ".dat"); std::vector<std::pair<double, double>> data;
  //
  // const auto support =
  //     DoFTools::map_dofs_to_support_points(mapping, dof_handler);
  //
  // for (const auto &[dof, p] : support) {
  //   data.emplace_back(p[0], solution[dof]);
  // }
  //
  // std::sort(data.begin(), data.end());
  //
  // for (const auto &[x, value] : data) {
  //   out << x << " " << value << "\n";
  // }
  //
  // std::vector<double> E_x =
  //     sample_electric_field(Parameters::X_DOMAIN_LEFT,
  //                           Parameters::X_DOMAIN_RIGHT, Parameters::PLOT_NX);
  // save_space_vector(E_x, "E_x_after_solve", it);
}

template <int dim> void PoissonProblem<dim>::initialize() {
  create_mesh();  // build grid
  setup_system(); // distribute DoFs and matrices
}

template <int dim>
double PoissonProblem<dim>::solve_step(
    size_t it, std::vector<GridStructure<1>> &grid_versions, bool refining) {
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
template <int dim> void PoissonProblem<dim>::run() {
  create_mesh();
  setup_system();
  assemble_system();
  solve();
}

#endif

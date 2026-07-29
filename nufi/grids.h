#ifndef GRIDS_H
#define GRIDS_H

#include "nufi/cells.h"
#include "nufi/parameters.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <memory>
#include <vector>

using namespace dealii;

template <int dim> class PoissonProblem;

template <int dim> struct GridStructure {
  //==//==//
  // Vars //
  //==//==//
  std::unique_ptr<Triangulation<dim>> triangulation;
  std::unique_ptr<DoFHandler<dim>> dof_handler;
  std::unique_ptr<MappingQ<dim>> mapping;
  std::unique_ptr<FE_Q<dim>> fe;
  std::unique_ptr<AffineConstraints<double>> constraints;
  std::unique_ptr<SparsityPattern> sparsity_pattern;

  std::unique_ptr<CellLocator<dim>> locator;

  unsigned int grid_version = 0;

  // === // === //
  // Evaluator  //
  // === // === //
  std::vector<double>
  eval_vector_grad(const Vector<double> &solution,
                   const std::vector<Point<dim>> &points) const {

    std::vector<double> values(points.size());

    // Uses point_gradient

    // #pragma omp parallel for
    //     for (unsigned int p = 0; p < points.size(); ++p) {
    //
    //       const Tensor<1, dim> grad_phi = VectorTools::point_gradient(
    //           *mapping, *dof_handler, solution, points[p]);
    //
    //       values[p] = grad_phi[0];
    //     }
    //
    //     return values;
    //   }

    // Uses cell locator
#pragma omp parallel
    {
      std::vector<double> local_solution_buffer(fe->n_dofs_per_cell());
      FEPointEvaluation<dim, dim> evaluator(*mapping, *fe, update_gradients);
#pragma omp for
      for (unsigned int p = 0; p < points.size(); ++p) {

        const auto cell_location = locator->locate(points[p]);

        cell_location.info->cell->get_dof_values(solution,
                                                 local_solution_buffer.begin(),
                                                 local_solution_buffer.end());

        evaluator.reinit(
            cell_location.info->cell,
            ArrayView<const Point<dim>>(&cell_location.reference_point, 1));

        evaluator.evaluate(local_solution_buffer, EvaluationFlags::gradients);

        values[p] = evaluator.get_gradient(0)[0];
      }
    }
    return values;
  }
};

template <int dim>
GridStructure<dim> make_grid_snapshot(PoissonProblem<dim> &poisson) {
  bool PRINT_GAUGE_DOF_POSITION = true;
  GridStructure<dim> grid;

  grid.grid_version = 0;

  grid.triangulation = std::make_unique<Triangulation<dim>>();
  grid.triangulation->copy_triangulation(poisson.get_triangulation());

  grid.fe = std::make_unique<FE_Q<dim>>(poisson.get_fe());

  grid.dof_handler = std::make_unique<DoFHandler<dim>>(*grid.triangulation);
  grid.dof_handler->distribute_dofs(*grid.fe);

  grid.mapping = std::make_unique<MappingQ<dim>>(poisson.get_mapping());

  grid.constraints = std::make_unique<AffineConstraints<double>>();

  DoFTools::make_hanging_node_constraints(*grid.dof_handler, *grid.constraints);
  DoFTools::make_periodicity_constraints(*grid.dof_handler, 0, 1, 0,
                                         *grid.constraints);

  const auto support_points =
      DoFTools::map_dofs_to_support_points(*grid.mapping, *grid.dof_handler);

  types::global_dof_index gauge_dof = numbers::invalid_dof_index;

  // Search only inside the protected region
  for (const auto &[dof, point] : support_points) {
    if (grid.constraints->is_constrained(dof))
      continue;
    const double x = point[0];
    if (x <= Parameters::X_DOMAIN_LEFT + .5) {
      gauge_dof = dof;

      if (PRINT_GAUGE_DOF_POSITION)
        std::cout << " gauge_dof = " << gauge_dof
                  << " gauge_point = " << point[0] << "\n";
      break;
    }
  }

  Assert(gauge_dof != numbers::invalid_dof_index,
         ExcMessage("No gauge DoF found in protected gauge region."));

  grid.constraints->add_line(gauge_dof);
  grid.constraints->set_inhomogeneity(gauge_dof, 0.0);
  grid.constraints->close();

  grid.locator = std::make_unique<CellLocator<dim>>();
  grid.locator->rebuild(*grid.dof_handler, *grid.triangulation);

  // START: diagnostics
  AssertThrow(
      poisson.get_constraints().n_constraints() ==
          grid.constraints->n_constraints(),
      ExcMessage(
          "PoissonProblem constraints doesn't match Snapshot constraints"));
  // for (auto c1 = poisson.get_dof_handler().begin_active(),
  //           c2 = grid.dof_handler->begin_active();
  //      c1 != poisson.get_dof_handler().end(); ++c1, ++c2) {
  //   std::vector<types::global_dof_index> d1(c1->get_fe().dofs_per_cell);
  //   std::vector<types::global_dof_index> d2(c2->get_fe().dofs_per_cell);
  //
  //   c1->get_dof_indices(d1);
  //   c2->get_dof_indices(d2);
  //
  //   AssertThrow(d1 == d2, ExcInternalError());
  // }
  // END: diagnostics

  return grid;
}

template <int dim> struct SolutionSnapshot {
  unsigned int grid_version;
  Vector<double> solution;
};

template <int dim>
inline void update_grid_versions(std::vector<GridStructure<dim>> &grid_versions,
                                 PoissonProblem<dim> &poisson) {
  auto grid = make_grid_snapshot(poisson);

  if (!grid_versions.empty())
    grid.grid_version = grid_versions.back().grid_version + 1;

  grid_versions.push_back(std::move(grid));

  std::cout << "@ update_grid_history: "
            << "grid version " << grid.grid_version << " size "
            << poisson.get_solution().size() << "\n";
}

template <int dim>
inline void
update_solution_history(std::vector<SolutionSnapshot<dim>> &solution_history,
                        PoissonProblem<dim> &poisson,
                        unsigned int current_grid_version) {

  SolutionSnapshot<dim> snapshot;

  snapshot.grid_version = current_grid_version;

  // where I might need to change something to pass the correct solution or in
  // the correct form
  Vector<double> solution = poisson.get_solution();
  snapshot.solution = solution;

  std::cout << "@ update_solution_history: "
            << "grid version " << current_grid_version << " size "
            << poisson.get_solution().size() << "\n";

  solution_history.push_back(std::move(snapshot));
}

#endif // !GRIDS_H

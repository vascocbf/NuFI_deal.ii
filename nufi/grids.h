#ifndef GRIDS_H
#define GRIDS_H

#include "nufi/cells.h"

#include <array>
#include <cstddef>
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
#include <omp.h>
#include <unordered_map>
#include <vector>

using namespace dealii;
using std::array;

template <int X_DIM> class PoissonProblem;

template <int X_DIM> struct GridStructure {
  std::unique_ptr<Triangulation<X_DIM>> triangulation;
  std::unique_ptr<DoFHandler<X_DIM>> dof_handler;
  std::unique_ptr<MappingQ<X_DIM>> mapping;
  std::unique_ptr<FE_Q<X_DIM>> fe;

  std::unique_ptr<CellLocator<X_DIM>> locator;

  unsigned int grid_version = 0;

  std::vector<array<double, X_DIM>>
  eval_vector_grad(const Vector<double> &solution,
                   const std::vector<Point<X_DIM>> &points) const {

    const unsigned int n_points = points.size();
    std::vector<array<double, X_DIM>> values(n_points);

    std::vector<CellLocation<X_DIM>> locations(n_points);

#pragma omp parallel for if (!omp_in_parallel())
    for (unsigned int p = 0; p < n_points; ++p)
      locations[p] = locator->locate(points[p]);

    std::unordered_map<unsigned int, std::vector<unsigned int>> cell_to_indices;
    for (unsigned int p = 0; p < n_points; ++p)
      cell_to_indices[locations[p].cell->active_cell_index()].push_back(p);

    std::vector<typename DoFHandler<X_DIM>::active_cell_iterator> cells;
    std::vector<std::vector<unsigned int>> groups;
    cells.reserve(cell_to_indices.size());
    groups.reserve(cell_to_indices.size());

    for (auto &kv : cell_to_indices) {
      groups.push_back(std::move(kv.second));
      cells.push_back(locations[groups.back().front()].cell);
    }

#pragma omp parallel if (!omp_in_parallel())
    {
      std::vector<double> local_solution_buffer(fe->n_dofs_per_cell());
      FEPointEvaluation<1, X_DIM> evaluator(*mapping, *fe, update_gradients);

#pragma omp for
      for (long c = 0; c < static_cast<long>(cells.size()); ++c) {
        const auto &cell = cells[c];
        const auto &idxs = groups[c];

        std::vector<Point<X_DIM>> unit_points(idxs.size());
        for (size_t k = 0; k < idxs.size(); ++k)
          unit_points[k] = locations[idxs[k]].reference_point;

        cell->get_dof_values(solution, local_solution_buffer.begin(),
                             local_solution_buffer.end());

        evaluator.reinit(cell, ArrayView<const Point<X_DIM>>(unit_points));
        evaluator.evaluate(local_solution_buffer, EvaluationFlags::gradients);

        for (size_t k = 0; k < idxs.size(); ++k) {
          for (size_t d = 0; d < X_DIM; ++d)
            values[idxs[k]][d] = evaluator.get_gradient(k)[d];
        }
      }
    }
    return values;
  }
};

template <int X_DIM>
GridStructure<X_DIM> make_grid_snapshot(PoissonProblem<X_DIM> &poisson) {
  GridStructure<X_DIM> grid;

  grid.grid_version = 0;

  grid.triangulation = std::make_unique<Triangulation<X_DIM>>();
  grid.triangulation->copy_triangulation(poisson.get_triangulation());

  grid.fe = std::make_unique<FE_Q<X_DIM>>(poisson.get_fe());

  grid.dof_handler = std::make_unique<DoFHandler<X_DIM>>(*grid.triangulation);
  grid.dof_handler->distribute_dofs(*grid.fe);

  grid.mapping = std::make_unique<MappingQ<X_DIM>>(poisson.get_mapping());

  grid.locator = std::make_unique<CellLocator<X_DIM>>();
  grid.locator->rebuild(*grid.dof_handler, *grid.triangulation);

  return grid;
}

template <int X_DIM> struct SolutionSnapshot {
  unsigned int grid_version;
  Vector<double> solution;
};

template <int X_DIM>
inline void
update_grid_versions(std::vector<GridStructure<X_DIM>> &grid_versions,
                     PoissonProblem<X_DIM> &poisson) {
  auto grid = make_grid_snapshot(poisson);

  if (!grid_versions.empty())
    grid.grid_version = grid_versions.back().grid_version + 1;

  grid_versions.push_back(std::move(grid));

  std::cout << "@ update_grid_history: "
            << "grid version " << grid.grid_version << " size "
            << poisson.get_solution().size() << "\n";
}

template <int X_DIM>
inline void
update_solution_history(std::vector<SolutionSnapshot<X_DIM>> &solution_history,
                        PoissonProblem<X_DIM> &poisson,
                        unsigned int current_grid_version) {

  SolutionSnapshot<X_DIM> snapshot;

  snapshot.grid_version = current_grid_version;

  Vector<double> solution = poisson.get_solution();
  snapshot.solution = solution;

  std::cout << "@ update_solution_history: "
            << "grid version " << current_grid_version << " size "
            << poisson.get_solution().size() << "\n";

  solution_history.push_back(std::move(snapshot));
}

#endif // !GRIDS_H

#ifndef CELLS_H
#define CELLS_H

#include <cmath>
#include <cstddef>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>
#include <vector>

#include "nufi/parameters.h"

using namespace dealii;

template <int X_DIM> struct CellLocation {
  typename DoFHandler<X_DIM>::active_cell_iterator cell;
  Point<X_DIM> reference_point;
};

template <int X_DIM> class CellLocator {
public:
  void rebuild(const DoFHandler<X_DIM> &dof_handler,
               const Triangulation<X_DIM> &triangulation);

  CellLocation<X_DIM> locate(const Point<X_DIM> &p) const;

private:
  const DoFHandler<X_DIM> *dof_handler_ptr = nullptr;
  Point<X_DIM> lower;
  Point<X_DIM> upper;

  // Cached base level = Parameters::GLOBAL_REFINEMENT.
  size_t base_level = 0;
  size_t base_n_per_axis = 1; // 2^base_level

  std::vector<typename Triangulation<X_DIM>::cell_iterator> base_cells;
};

template <int X_DIM>
void CellLocator<X_DIM>::rebuild(const DoFHandler<X_DIM> &dof_handler,
                                 const Triangulation<X_DIM> &triangulation) {
  dof_handler_ptr = &dof_handler;

  AssertThrow(triangulation.n_cells(0) == 1,
              ExcMessage("CellLocator assumes exactly one coarse/root cell."));

  typename Triangulation<X_DIM>::cell_iterator root = triangulation.begin(0);
  lower = root->vertex(0);
  upper = root->vertex(GeometryInfo<X_DIM>::vertices_per_cell - 1);

  base_level = Parameters::GLOBAL_REFINEMENT;
  base_n_per_axis = static_cast<size_t>(1) << base_level; // = 2^base_level

  AssertThrow(X_DIM * base_level <= 24,
              ExcMessage("CellLocator: X_DIM * GLOBAL_REFINEMENT exceeds a "
                         "sane bound; base_cells cache would be too large."));

  const size_t n_base_cells = static_cast<size_t>(1)
                              << (X_DIM * base_level); // = 2^(X_DIM*base_level)
  base_cells.assign(n_base_cells,
                    typename Triangulation<X_DIM>::cell_iterator());

  std::vector<typename Triangulation<X_DIM>::cell_iterator> stack;
  std::vector<unsigned int> index_stack;
  std::vector<unsigned int> depth_stack;
  stack.push_back(root);
  index_stack.push_back(0);
  depth_stack.push_back(0);

  while (!stack.empty()) {
    auto cell = stack.back();
    size_t idx = index_stack.back();
    unsigned int depth = depth_stack.back();
    stack.pop_back();
    index_stack.pop_back();
    depth_stack.pop_back();

    if (depth == base_level) {
      base_cells[idx] = cell;
      continue;
    }

    AssertThrow(cell->has_children(),
                ExcMessage("CellLocator: mesh is not uniformly refined to "
                           "Parameters::GLOBAL_REFINEMENT; base-level cache "
                           "cannot be built. Did you coarsen below the "
                           "global refinement level?"));

    const unsigned int n_children = GeometryInfo<X_DIM>::max_children_per_cell;
    for (unsigned int c = 0; c < n_children; ++c) {
      stack.push_back(cell->child(c));
      index_stack.push_back(idx * n_children + c);
      depth_stack.push_back(depth + 1);
    }
  }
}

template <int X_DIM>
CellLocation<X_DIM> CellLocator<X_DIM>::locate(const Point<X_DIM> &p) const {
  AssertThrow(dof_handler_ptr != nullptr,
              ExcMessage("CellLocator::rebuild() has not been called."));

  Point<X_DIM> p_wrapped;

  // WARNING: assumes all X_DIM s are periodic
  for (unsigned int d = 0; d < X_DIM; ++d) {
    const double L = upper[d] - lower[d];
    double x = p[d] - lower[d];
    x = x - L * std::floor(x / L);
    p_wrapped[d] = lower[d] + x;
  }

  Point<X_DIM> xi;
  for (unsigned int d = 0; d < X_DIM; ++d) {
    xi[d] = (p_wrapped[d] - lower[d]) / (upper[d] - lower[d]);
    xi[d] = std::min(std::max(xi[d], 0.0), 1.0); // of cell at base level
  }

  size_t idx = 0;
  Point<X_DIM> xi_local = xi;
  for (unsigned int l = 0; l < base_level; ++l) {
    const unsigned int child_index =
        GeometryInfo<X_DIM>::child_cell_from_point(xi_local);
    xi_local =
        GeometryInfo<X_DIM>::cell_to_child_coordinates(xi_local, child_index);
    idx = idx * GeometryInfo<X_DIM>::max_children_per_cell + child_index;
  }

  typename Triangulation<X_DIM>::cell_iterator cell = base_cells[idx];
  xi = xi_local;

  while (cell->has_children()) {
    const unsigned int child_index =
        GeometryInfo<X_DIM>::child_cell_from_point(xi);
    xi = GeometryInfo<X_DIM>::cell_to_child_coordinates(xi, child_index);
    cell = cell->child(child_index);
  }

  typename DoFHandler<X_DIM>::active_cell_iterator dof_cell(
      &cell->get_triangulation(), cell->level(), cell->index(),
      dof_handler_ptr);

  CellLocation<X_DIM> location;
  location.cell = dof_cell;
  location.reference_point = xi;
  return location;
}

#endif // !CELLS_H

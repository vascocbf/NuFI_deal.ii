#ifndef CELLS_H
#define CELLS_H

#include <array>
#include <cmath>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>
#include <vector>

#include "nufi/parameters.h"

using namespace dealii;

template <int dim> struct CellLocation {
  typename DoFHandler<dim>::active_cell_iterator cell;
  Point<dim> reference_point;
};

template <int dim> class CellLocator {
public:
  void rebuild(const DoFHandler<dim> &dof_handler,
               const Triangulation<dim> &triangulation);

  CellLocation<dim> locate(const Point<dim> &p) const;

private:
  const DoFHandler<dim> *dof_handler_ptr = nullptr;
  Point<dim> lower;
  Point<dim> upper;

  // Cached base level = Parameters::GLOBAL_REFINEMENT.
  unsigned int base_level = 0;
  unsigned int base_n_per_axis = 1; // 2^base_level
  // Flat lookup, indexed in the same bit-interleaved order that
  // GeometryInfo<dim>::child_cell_from_point produces at each level,
  // so base_cells[idx] can be found with pure bit math, no tree walk.
  std::vector<typename Triangulation<dim>::cell_iterator> base_cells;
};

template <int dim>
void CellLocator<dim>::rebuild(const DoFHandler<dim> &dof_handler,
                               const Triangulation<dim> &triangulation) {
  dof_handler_ptr = &dof_handler;

  AssertThrow(triangulation.n_cells(0) == 1,
              ExcMessage("CellLocator assumes exactly one coarse/root cell."));

  typename Triangulation<dim>::cell_iterator root = triangulation.begin(0);
  lower = root->vertex(0);
  upper = root->vertex(GeometryInfo<dim>::vertices_per_cell - 1);

  base_level = Parameters::GLOBAL_REFINEMENT;
  base_n_per_axis = 1u << base_level;

  // Walk down exactly base_level times ONCE per base cell to build the
  // flat table. This costs O(2^(dim*base_level)) total at rebuild time
  // (proportional to the number of base cells), not per locate() call.
  const unsigned int n_base_cells = 1u << (dim * base_level);
  base_cells.assign(n_base_cells, typename Triangulation<dim>::cell_iterator());

  // Recursive-free BFS/DFS: descend from root, tracking the accumulated
  // child-index bits per level to know where to store each level-G cell.
  std::vector<typename Triangulation<dim>::cell_iterator> stack;
  std::vector<unsigned int> index_stack;
  std::vector<unsigned int> depth_stack;
  stack.push_back(root);
  index_stack.push_back(0);
  depth_stack.push_back(0);

  while (!stack.empty()) {
    auto cell = stack.back();
    unsigned int idx = index_stack.back();
    unsigned int depth = depth_stack.back();
    stack.pop_back();
    index_stack.pop_back();
    depth_stack.pop_back();

    if (depth == base_level) {
      base_cells[idx] = cell;
      continue;
    }

    // At this point cell must have children, since refine_global(base_level)
    // guarantees a fully uniform tree down to base_level.
    AssertThrow(cell->has_children(),
                ExcMessage("CellLocator: mesh is not uniformly refined to "
                           "Parameters::GLOBAL_REFINEMENT; base-level cache "
                           "cannot be built. Did you coarsen below the "
                           "global refinement level?"));

    const unsigned int n_children = GeometryInfo<dim>::max_children_per_cell;
    for (unsigned int c = 0; c < n_children; ++c) {
      stack.push_back(cell->child(c));
      index_stack.push_back(idx * n_children + c);
      depth_stack.push_back(depth + 1);
    }
  }
}

template <int dim>
CellLocation<dim> CellLocator<dim>::locate(const Point<dim> &p) const {
  AssertThrow(dof_handler_ptr != nullptr,
              ExcMessage("CellLocator::rebuild() has not been called."));

  // Step 1: periodic wrap into [lower, upper) per axis.
  Point<dim> p_wrapped;
  for (unsigned int d = 0; d < dim; ++d) {
    const double L = upper[d] - lower[d];
    double x = p[d] - lower[d];
    x = x - L * std::floor(x / L);
    p_wrapped[d] = lower[d] + x;
  }

  // Step 2: reference coordinates in the root cell, clamped against
  // floating-point drift at the domain boundary.
  Point<dim> xi;
  for (unsigned int d = 0; d < dim; ++d) {
    xi[d] = (p_wrapped[d] - lower[d]) / (upper[d] - lower[d]);
    xi[d] = std::min(std::max(xi[d], 0.0), 1.0);
  }

  // Step 3: O(1) jump to the base-level (GLOBAL_REFINEMENT) cell via index
  // math, replacing what used to be `base_level` iterations of the
  // has_children() loop. Must mirror the same bit convention used when
  // building base_cells in rebuild() (child index accumulated as
  // idx = idx*n_children + child_cell_from_point(xi) at each level).
  unsigned int idx = 0;
  Point<dim> xi_local = xi;
  for (unsigned int l = 0; l < base_level; ++l) {
    const unsigned int child_index =
        GeometryInfo<dim>::child_cell_from_point(xi_local);
    xi_local =
        GeometryInfo<dim>::cell_to_child_coordinates(xi_local, child_index);
    idx = idx * GeometryInfo<dim>::max_children_per_cell + child_index;
  }

  typename Triangulation<dim>::cell_iterator cell = base_cells[idx];
  xi = xi_local;

  // Step 4: continue descending only through ADAPTIVE refinement beyond
  // the base level -- this loop now only runs `depth - base_level` times
  // instead of `depth` times.
  while (cell->has_children()) {
    const unsigned int child_index =
        GeometryInfo<dim>::child_cell_from_point(xi);
    xi = GeometryInfo<dim>::cell_to_child_coordinates(xi, child_index);
    cell = cell->child(child_index);
  }

  typename DoFHandler<dim>::active_cell_iterator dof_cell(
      &cell->get_triangulation(), cell->level(), cell->index(),
      dof_handler_ptr);

  CellLocation<dim> location;
  location.cell = dof_cell;
  location.reference_point = xi;
  return location;
}

#endif // !CELLS_H

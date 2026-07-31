#ifndef CELLS_H
#define CELLS_H

#include <cmath>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>

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
  typename Triangulation<dim>::cell_iterator root;
  Point<dim> lower;
  Point<dim> upper;
};

template <int dim>
void CellLocator<dim>::rebuild(const DoFHandler<dim> &dof_handler,
                               const Triangulation<dim> &triangulation) {
  dof_handler_ptr = &dof_handler;

  // Everything below assumes the mesh is a single hyper_cube coarse cell
  // (true for your create_mesh(): GridGenerator::hyper_cube + refine_global).
  AssertThrow(triangulation.n_cells(0) == 1,
              ExcMessage("CellLocator assumes exactly one coarse/root cell."));

  root = triangulation.begin(0);
  lower = root->vertex(0);
  upper = root->vertex(GeometryInfo<dim>::vertices_per_cell - 1);
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

  // Steps 3-5: descend the refinement tree using deal.II's own
  // reference-cell child logic (branch-free, handles dim=1,2,3 uniformly).
  typename Triangulation<dim>::cell_iterator cell = root;
  while (cell->has_children()) {
    const unsigned int child_index =
        GeometryInfo<dim>::child_cell_from_point(xi);
    xi = GeometryInfo<dim>::cell_to_child_coordinates(xi, child_index);
    cell = cell->child(child_index);
  }

  // cell is now active (leaf) -> bind it to the DoFHandler.
  typename DoFHandler<dim>::active_cell_iterator dof_cell(
      &cell->get_triangulation(), cell->level(), cell->index(),
      dof_handler_ptr);

  CellLocation<dim> location;
  location.cell = dof_cell;
  location.reference_point = xi;
  return location;
}

#endif // !CELLS_H

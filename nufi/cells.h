#ifndef CELLS_H
#define CELLS_H

#include <algorithm>
#include <cmath>
#include <limits>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>

using namespace dealii;

// Current Locator needs Grid to be a hypercube !!

template <int dim> struct CellLocation {
  using CellIterator = typename DoFHandler<dim>::cell_iterator;

  CellIterator cell;
  Point<dim> reference_point;
};

template <int dim> class CellLocator {
public:
  void rebuild(const DoFHandler<dim> &dof_handler,
               const Triangulation<dim> &triangulation);

  CellLocation<dim> locate(const Point<dim> &point) const;

private:
  const DoFHandler<dim> *dof_handler_ = nullptr;

  Point<dim> domain_lower_;
  Point<dim> domain_upper_;
};

template <int dim>
void CellLocator<dim>::rebuild(const DoFHandler<dim> &dof_handler,
                               const Triangulation<dim> &triangulation) {
  AssertThrow(&dof_handler.get_triangulation() == &triangulation,
              ExcMessage("CellLocator::rebuild(): DoFHandler and Triangulation "
                         "do not refer to the same mesh."));

  auto root = dof_handler.begin(0);

  AssertThrow(
      root != dof_handler.end(0),
      ExcMessage("CellLocator::rebuild(): triangulation has no level-zero "
                 "cell."));

  auto after_root = root;
  ++after_root;

  AssertThrow(
      after_root == dof_handler.end(0),
      ExcMessage("CellLocator currently requires exactly one coarse cell."));

  dof_handler_ = &dof_handler;

  for (unsigned int d = 0; d < dim; ++d) {
    domain_lower_[d] = std::numeric_limits<double>::max();
    domain_upper_[d] = std::numeric_limits<double>::lowest();
  }

  for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    for (unsigned int d = 0; d < dim; ++d) {
      domain_lower_[d] = std::min(domain_lower_[d], root->vertex(v)[d]);

      domain_upper_[d] = std::max(domain_upper_[d], root->vertex(v)[d]);
    }

  for (unsigned int d = 0; d < dim; ++d) {
    const double length = domain_upper_[d] - domain_lower_[d];

    AssertThrow(length > 0.0,
                ExcMessage("CellLocator::rebuild(): coarse-cell domain has "
                           "non-positive extent."));

    const double scale =
        std::max({1.0, std::abs(domain_lower_[d]), std::abs(domain_upper_[d])});

    const double tolerance =
        100.0 * std::numeric_limits<double>::epsilon() * scale;

    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
      const double coordinate = root->vertex(v)[d];

      const bool lies_on_lower =
          std::abs(coordinate - domain_lower_[d]) <= tolerance;

      const bool lies_on_upper =
          std::abs(coordinate - domain_upper_[d]) <= tolerance;

      AssertThrow(
          lies_on_lower || lies_on_upper,
          ExcMessage(
              "CellLocator requires an axis-aligned hypercube coarse cell."));
    }
  }
}

template <int dim>
CellLocation<dim> CellLocator<dim>::locate(const Point<dim> &point) const {
  AssertThrow(
      dof_handler_ != nullptr,
      ExcMessage("CellLocator::locate(): rebuild() has not been called."));

  Point<dim> reference_point;

  for (unsigned int d = 0; d < dim; ++d) {
    const double length = domain_upper_[d] - domain_lower_[d];

    const double shifted = point[d] - domain_lower_[d];

    double periodic_offset = shifted - length * std::floor(shifted / length);

    if (periodic_offset < 0.0)
      periodic_offset += length;

    if (periodic_offset >= length)
      periodic_offset = 0.0;

    reference_point[d] = periodic_offset / length;
  }

  auto cell = dof_handler_->begin(0);

  while (cell->has_children()) {
    AssertThrow(
        cell->n_children() == GeometryInfo<dim>::max_children_per_cell,
        ExcMessage(
            "CellLocator currently supports isotropic refinement only."));

    const unsigned int child_index =
        GeometryInfo<dim>::child_cell_from_point(reference_point);

    reference_point = GeometryInfo<dim>::cell_to_child_coordinates(
        reference_point, child_index);

    cell = cell->child(child_index);
  }

  AssertThrow(
      cell->is_active(),
      ExcMessage("CellLocator tree traversal did not finish on an active "
                 "cell."));

  AssertThrow(
      GeometryInfo<dim>::is_inside_unit_cell(reference_point, 1e-12),
      ExcMessage(
          "CellLocator produced a reference point outside the unit cell."));

  return CellLocation<dim>{cell, reference_point};
}

#endif // CELLS_H

#ifndef CELLS_H
#define CELLS_H

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>

using namespace dealii;

// Current Locator needs Grid to be a single hypercube coarse cell,
// uniformly refined to `base_level`, with isotropic local refinement on top.
template <int dim> struct CellLocation {
  typename DoFHandler<dim>::active_cell_iterator cell;
  Point<dim> reference_point;
};

template <int dim> class CellLocator {
public:
  void rebuild(const DoFHandler<dim> &dof_handler,
               const Triangulation<dim> &triangulation,
               unsigned int base_level);

  CellLocation<dim> locate(const Point<dim> &point) const;

private:
  const DoFHandler<dim> *dof_handler_ = nullptr;

  Point<dim> domain_lower_;
  Point<dim> domain_upper_;
  std::array<unsigned int, dim> base_n_{};
  std::array<double, dim> base_dx_{};

  // Flat, O(1)-indexable array of the base_level cells.
  std::vector<typename DoFHandler<dim>::cell_iterator> base_cells_;
};

template <int dim>
void CellLocator<dim>::rebuild(const DoFHandler<dim> &dof_handler,
                               const Triangulation<dim> &triangulation,
                               unsigned int base_level) {
  AssertThrow(&dof_handler.get_triangulation() == &triangulation,
              ExcMessage("CellLocator::rebuild(): DoFHandler and Triangulation "
                         "do not refer to the same mesh."));

  dof_handler_ = &dof_handler;

  // --- bounding box of the (single) coarse cell ---
  auto root = dof_handler.begin(0);
  AssertThrow(root != dof_handler.end(0),
              ExcMessage("CellLocator::rebuild(): no level-zero cell."));
  {
    auto after_root = root;
    ++after_root;
    AssertThrow(after_root == dof_handler.end(0),
                ExcMessage("CellLocator currently requires exactly one "
                           "coarse cell."));
  }

  for (unsigned int d = 0; d < dim; ++d) {
    domain_lower_[d] = std::numeric_limits<double>::max();
    domain_upper_[d] = std::numeric_limits<double>::lowest();
  }
  for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    for (unsigned int d = 0; d < dim; ++d) {
      domain_lower_[d] = std::min(domain_lower_[d], root->vertex(v)[d]);
      domain_upper_[d] = std::max(domain_upper_[d], root->vertex(v)[d]);
    }

  // --- build the O(1)-indexable array of base_level cells ---
  unsigned int total = 1;
  for (unsigned int d = 0; d < dim; ++d) {
    const double length = domain_upper_[d] - domain_lower_[d];
    AssertThrow(length > 0.0,
                ExcMessage("CellLocator::rebuild(): non-positive domain "
                           "extent."));
    base_n_[d] =
        1u << base_level; // refine_global(base_level) -> 2^level per axis
    base_dx_[d] = length / base_n_[d];
    total *= base_n_[d];
  }

  base_cells_.assign(total, typename DoFHandler<dim>::cell_iterator());

  for (auto cell = dof_handler.begin(base_level);
       cell != dof_handler.end(base_level); ++cell) {
    const auto bb = cell->bounding_box();
    const auto c_lower = bb.get_boundary_points().first;

    unsigned int idx = 0, stride = 1;
    for (unsigned int d = 0; d < dim; ++d) {
      unsigned int i = static_cast<unsigned int>(
          std::round((c_lower[d] - domain_lower_[d]) / base_dx_[d]));
      i = std::min(i, base_n_[d] - 1);
      idx += i * stride;
      stride *= base_n_[d];
    }
    base_cells_[idx] = cell;
  }

  for (const auto &c : base_cells_)
    AssertThrow(c.state() == IteratorState::valid,
                ExcMessage("CellLocator::rebuild(): failed to fill the base "
                           "grid — mesh isn't uniformly refined to "
                           "'base_level' as expected."));
}

template <int dim>
CellLocation<dim> CellLocator<dim>::locate(const Point<dim> &point) const {
  AssertThrow(dof_handler_ != nullptr,
              ExcMessage("CellLocator::locate(): rebuild() has not been "
                         "called."));

  std::array<unsigned int, dim> base_index;
  Point<dim> reference_point; // local coords, updated at each descent step

  // 1) periodic wrap + O(1) base-cell index per axis
  for (unsigned int d = 0; d < dim; ++d) {
    const double length = domain_upper_[d] - domain_lower_[d];
    double shifted = point[d] - domain_lower_[d];
    shifted -= length * std::floor(shifted / length);
    if (shifted >= length)
      shifted = 0.0;

    const double xi = shifted / base_dx_[d];
    const unsigned int i =
        std::min(base_n_[d] - 1, static_cast<unsigned int>(std::floor(xi)));
    base_index[d] = i;
    reference_point[d] = xi - i; // fraction within the base cell
  }

  // 2) O(1) lookup of the base-level cell
  unsigned int idx = 0, stride = 1;
  for (unsigned int d = 0; d < dim; ++d) {
    idx += base_index[d] * stride;
    stride *= base_n_[d];
  }
  auto cell = base_cells_[idx];

  // 3) O(R_level_max) descent — only costs anything for cells that are
  //    actually locally refined beyond base_level
  while (cell->has_children()) {
    AssertThrow(cell->n_children() == GeometryInfo<dim>::max_children_per_cell,
                ExcMessage("CellLocator currently supports isotropic "
                           "refinement only."));
    const unsigned int child_index =
        GeometryInfo<dim>::child_cell_from_point(reference_point);
    reference_point = GeometryInfo<dim>::cell_to_child_coordinates(
        reference_point, child_index);
    cell = cell->child(child_index);
  }

  AssertThrow(cell->is_active(),
              ExcMessage("CellLocator tree traversal did not finish on an "
                         "active cell."));

  return CellLocation<dim>{typename DoFHandler<dim>::active_cell_iterator(cell),
                           reference_point};
}

#endif // CELLS_H

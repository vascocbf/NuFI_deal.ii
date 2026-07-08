#ifndef CELLS_H
#define CELLS_H

#include <algorithm>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>
#include <vector>

using namespace dealii;

template <int dim> struct CellInfo {
  // what needs to be given to evaluator
  typename DoFHandler<dim>::active_cell_iterator cell;
  // usefull for locator
  Point<dim> lower;
  Point<dim> upper;
  double h;
};

template <int dim> class CellLocator {
public:
  using CellIterator = typename DoFHandler<dim>::active_cell_iterator;

  void rebuild(const DoFHandler<dim> &dof_handler,
               const Triangulation<dim> &triangulation);

  CellIterator locate(const Point<dim> &p) const;

private:
  std::vector<CellInfo<dim>> cells;
};

template <int dim>
void CellLocator<dim>::rebuild(const DoFHandler<dim> &dof_handler,
                               const Triangulation<dim> &triangulation) {

  cells.clear();
  cells.reserve(triangulation.n_active_cells());

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    CellInfo<dim> info;

    info.cell = cell;
    info.lower = cell->vertex(0);
    info.upper = cell->vertex(GeometryInfo<dim>::vertices_per_cell - 1);

    info.h = info.upper[0] - info.lower[0];

    cells.push_back(info);
  }

  std::sort(cells.begin(), cells.end(),
            [](const CellInfo<dim> &a, const CellInfo<dim> &b) {
              return a.lower[0] < b.lower[0];
            });
}

template <int dim>
typename DoFHandler<dim>::active_cell_iterator
CellLocator<dim>::locate(const Point<dim> &p) const {
  static_assert(dim == 1,
                "Current CellLocator implementation only supports 1D.");

  AssertThrow(!cells.empty(),
              ExcMessage("CellLocator::rebuild() has not been called."));

  const double x = p[0];

  auto it = std::upper_bound(cells.begin(), cells.end(), x,
                             [](double value, const CellInfo<dim> &cell) {
                               return value < cell.lower[0];
                             }); // returns cell to the right of cell with x

  if (it == cells.begin())
    it = cells.begin();
  else
    --it;

  // Safety check: make sure the point is really inside this cell
  AssertThrow(x >= it->lower[0] - 1e-12 && x <= it->upper[0] + 1e-12,
              ExcMessage("CellLocator failed to find containing cell."));

  return it->cell;
}

#endif // !CELLS_H

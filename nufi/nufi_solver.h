#ifndef NUFI_SOLVER_H
#define NUFI_SOLVER_H

#include <boost/qvm/mat_access.hpp>
#include <cmath>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/vector_tools.h>
#include <vector>

#include "nufi/fields.h" // dont remove
#include "nufi/grids.h"
#include "nufi/parameters.h"

using namespace dealii;

class NuFISolver {
public:
  NuFISolver();

  std::vector<double>
  eval_rho(unsigned int n, std::vector<double> &x,
           const std::vector<GridStructure<1>> &grid_struct,
           const std::vector<SolutionSnapshot<1>> &phi_history,
           const unsigned int Nv = Parameters::NV) const;

  std::vector<double>
  eval_ftilda_batch(unsigned int n, std::vector<double> X,
                    std::vector<double> U,
                    const std::vector<GridStructure<1>> &grid_structures,
                    const std::vector<SolutionSnapshot<1>> &phi_history) const;

  std::vector<double>
  eval_f(unsigned int n, std::vector<double> x, double u,
         const std::vector<GridStructure<1>> &grid_struct,
         const std::vector<SolutionSnapshot<1>> &phi_history) const;

  std::vector<double>
  eval_rho_points(unsigned int n, const std::vector<Point<1>> &points,
                  const std::vector<GridStructure<1>> &grid_struct,
                  const std::vector<SolutionSnapshot<1>> &phi_history,
                  const unsigned int Nv) const;

private:
  unsigned int Nt = std::floor(Parameters::TMAX / Parameters::DT);

  double Lx = Parameters::LX;

  double x_min = Parameters::X_DOMAIN_LEFT;
  double x_max = Parameters::X_DOMAIN_RIGHT;

  unsigned int order;
};
#endif

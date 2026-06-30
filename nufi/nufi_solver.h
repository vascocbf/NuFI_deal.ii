#ifndef NUFI_SOLVER_H
#define NUFI_SOLVER_H

#include <boost/qvm/mat_access.hpp>
#include <cmath>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/vector_tools.h>
#include <vector>

#include "nufi/fields.h" //dont remove
#include "nufi/parameters.h"
#include "nufi/poisson_problem.h"

using namespace dealii;

class NuFISolver {
public:
  NuFISolver();

  void run();
  std::vector<double> eval_rho(unsigned int n, std::vector<double> &x,
                               const PoissonProblem<1> &poisson,
                               const std::vector<Vector<double>> &phi_history,
                               const unsigned int Nv = Parameters::NV) const;
  std::vector<double>
  eval_ftilda(unsigned int, std::vector<double> &x, double u,
              const PoissonProblem<1> &poisson,
              const std::vector<Vector<double>> &phi_history) const;
  std::vector<double>
  eval_f(unsigned int n, std::vector<double> &x, double u,
         const PoissonProblem<1> &poisson,
         const std::vector<Vector<double>> &phi_history) const;

private:
  unsigned int Nt = std::floor(Parameters::TMAX / Parameters::DT);
  unsigned int Nx = Parameters::CALC_NX;

  double Lx = Parameters::LX;

  double x_min = Parameters::X_DOMAIN_LEFT;
  double x_max = Parameters::X_DOMAIN_RIGHT;

  std::vector<double> rho;

  unsigned int order;

  PoissonProblem<1> poisson;
};
#endif

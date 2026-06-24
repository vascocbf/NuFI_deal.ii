#ifndef NUFI_SOLVER_H
#define NUFI_SOLVER_H

#include <boost/qvm/mat_access.hpp>
#include <vector>
#include <cmath>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/fe_field_function.h>

#include "nufi/parameters.h"
#include "nufi/poisson_problem.h"
#include "nufi/fields.h" //dont remove

using namespace dealii;

class NuFISolver
{
public:
  NuFISolver();

  void run();
  double eval_rho(unsigned int n, double x, const PoissonProblem<1> &poisson, unsigned int Nv = Parameters::NV) const;
  double eval_ftilda(unsigned int n, double x, double u, const PoissonProblem<1> &poisson) const;
  double eval_f(unsigned int n, double x, double u, const PoissonProblem<1> &poisson) const;

private:


  unsigned int Nt = std::floor(Parameters::TMAX/Parameters::DT);
  unsigned int Nx = Parameters::CALC_NX;

  double Lx = Parameters::LX;

  double x_min = Parameters::X_DOMAIN_LEFT;
  double x_max = Parameters::X_DOMAIN_RIGHT;

  std::vector<double> rho;

  unsigned int order;

  PoissonProblem<1> poisson;

};

template<unsigned int dim>
class ChargeDensity_NuFI : public Function<dim>
{
  public:
    ChargeDensity_NuFI(const double *rho_values, unsigned int Nx)
      : Function<dim>(), rho(rho_values), Nx(Nx) {}

    virtual double value(const Point<dim> &p,
                         [[maybe_unused]] const unsigned int component = 0) const override
    {
      const double x = p[0];

      // Map x -> grid index
      const double L = Parameters::LX;
      const double dx = L / (Nx-1);

      int i = static_cast<int>(std::floor((x - Parameters::X_DOMAIN_LEFT) / dx));

      // periodic wrap
      i = (i % Nx + Nx) % Nx;

      return rho[i];
    }

  private:
    const double *rho;
    const unsigned int Nx;
};

#endif

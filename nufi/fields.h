#ifndef FIELDS_H
#define FIELDS_H

#include "nufi/parameters.h"
#include "poisson_problem.h"
#include <cmath>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

using namespace dealii;

inline std::vector<int> Indices_of_points(const std::vector<double> &points, double x_min, double x_max, double dx, int grid_type=0)
{ 
    // grid type:
    // 0 => uniform
    // 1 => non uniform (TODO)
    
    if (dx <= 0.0) {
        throw std::invalid_argument("dx must be positive");
    }
    if (x_max <= x_min) {
        throw std::invalid_argument("x_max must be > x_min");
    }

    std::vector<int> indices;
    indices.reserve(points.size());

    switch (grid_type) {
        case 0:
            {
            const double L = x_max - x_min;
            const int N = std::floor(L/dx);


            for (double x : points) //GPT loop, to check
            {
                x-= x_min;
                x = x - L * std::floor(x/L);

                int i = static_cast<int>(std::floor(x / dx));

                // safety: handle rare edge case due to floating precision
                if (i == N) i = 0;

                indices.push_back(i);
            }
            }
        case 1:
            {
            throw std::invalid_argument("Case for non uniform grid is not completed");
            }
        default:
            throw std::invalid_argument("Invalid grid_type argument");

    }
    return indices;
}

inline double f0(const double x, const double v,
                 const double eps = Parameters::EPS,
                 const double k = Parameters::WAVE_NR) {
  const double prefactor =
      Parameters::F0_FACTOR * (1.0 + eps * std::cos(k * x));
  const double gaussian = v * v * std::exp(-0.5 * v * v);

  return prefactor * gaussian;
}

inline double compute_rho(const double x,
                          const unsigned int Nv = Parameters::NV) {
  const double dv =
      (Parameters::V_DOMAIN_RIGHT - Parameters::V_DOMAIN_LEFT) / Nv;

  double integral = 0.0;

  for (unsigned int i = 0; i < Nv; ++i) {
    const double v = Parameters::V_DOMAIN_LEFT + (i + 0.5) * dv;
    integral += f0(x, v) * dv;
  }

  return 1.0 - integral;
}

double eval(double x, const PoissonProblem<1> &poisson) noexcept {
    return poisson.evaluate_potential(Point<1>(x));
}

inline double integral_space_vector(const PoissonProblem<1> &poisson,
                                    double dx = Parameters::PLOT_DX,
                                    size_t Nx = Parameters::PLOT_NX) {
  double integral = 0.0;
  double xmin = Parameters::X_DOMAIN_LEFT;
#pragma omp parallel for reduction(+ : integral)
  for (size_t i = 0; i < Nx; ++i) {
    double x = xmin + i * dx;
    integral += eval(x, poisson);
  }
  return integral * dx;
};

inline double integral_space_vector_squared(const PoissonProblem<1> &poisson,
                                            double dx = Parameters::PLOT_DX,
                                            size_t Nx = Parameters::PLOT_NX) {
  double integral = 0.0;
  double xmin = Parameters::X_DOMAIN_LEFT;
#pragma omp parallel for reduction(+ : integral)
  for (size_t i = 0; i < Nx; ++i) {
    double x = xmin + i * dx;
    double val = eval(x, poisson);
    integral += val * val;
  }
  return integral * dx;
};

class Gradient {
public:
  Gradient(double xmin, double xmax, unsigned int Nx)
      : xmin_(xmin), xmax_(xmax), Nx_(Nx) {
    if (xmax_ <= xmin_) {
      throw std::invalid_argument("xmax must be greater than xmin");
    }
  }

  std::vector<double> compute(const std::vector<double> &values) const {
    size_t n = values.size();
    if (n < 2) {
      throw std::invalid_argument("Need at least 2 points");
    }

    std::vector<double> grad(n);

    double dx = (xmax_ - xmin_) / (n - 1);
    // periodic boundaries
    grad[0] = -(values[1] - values[n - 1]) / (2.0 * dx);
    grad[n - 1] = -(values[0] - values[n - 2]) / (2.0 * dx);

    for (size_t i = 1; i < n - 1; ++i) {
      grad[i] = -(values[i + 1] - values[i - 1]) / (2.0 * dx);
    }

    return grad;
  }

private:
  double xmin_;
  double xmax_;
  [[maybe_unused]] unsigned int Nx_;
};

template <int dim>
class ChargeDensity : public Function<dim> // only uses f0
{
public:
  ChargeDensity(double eps, double k, unsigned int Nv)
      : Function<dim>(1), eps(eps), k(k), Nv(Nv) {}

  virtual double
  value(const Point<dim> &p,
        [[maybe_unused]] const unsigned int component = 0) const override {
    return compute_rho(p[0], Nv);
  }

private:
  const double eps;
  const double k;
  const unsigned int Nv;
};

#endif

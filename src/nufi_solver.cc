#include "nufi/nufi_solver.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/fe_field_function.h>

#include <iostream>
#include <memory>
#include <ostream>
#include <cstddef>
#include <vector>

#include "nufi/parameters.h"
#include "nufi/save_results.h"
#include "nufi/poisson_problem.h"
#include "nufi/fields.h"
#include "nufi/stopwatch.h"

using namespace dealii;

double NuFISolver::eval_ftilda(unsigned int n,
                                      double x,
                                      double u,
                                      const PoissonProblem<1> &poisson) const
{
  if ( n == 0 ) return f0(x,u);
  
  double Ex;

  // We omit the initial half-step.

  while ( --n )
  {
      x  = x - Parameters::DT *u;
      Ex = -eval(x, poisson);
      u  = u + Parameters::DT *Ex;
  }

  // The final half-step.
  x -= Parameters::DT*u;
  Ex = -eval(x, poisson);
  u += 0.5*Parameters::DT*Ex;

  return f0(x,u);
}

double NuFISolver::eval_f(unsigned int n,
                                      double x,
                                      double u,
                                      const PoissonProblem<1> &poisson) const
{
  if ( n == 0 ) return f0(x,u);
  
  double Ex;

  // Initial half-step.
  Ex = -eval(x, poisson);
  u += 0.5*Parameters::DT * Ex;

  while ( --n )
  {
      x  = x - Parameters::DT *u;
      Ex = -eval(x, poisson);
      u  = u + Parameters::DT *Ex;
  }

  // The final half-step.
  x -= Parameters::DT*u;
  Ex = -eval(x, poisson);
  u += 0.5*Parameters::DT*Ex;

  return f0(x,u);
}

double NuFISolver::eval_rho(const unsigned int n,
                          const double x,
                          const PoissonProblem<1> &poisson,
                          const unsigned int Nv) const
{
  const double dv = (Parameters::V_DOMAIN_RIGHT - Parameters::V_DOMAIN_LEFT) / Nv;
  const double v_min = Parameters::V_DOMAIN_LEFT;

  double integral = 0.0;

#pragma omp parallel for reduction (+ : integral)
  for (unsigned int i = 0; i < Nv; ++i)
    integral += eval_ftilda(n, x, v_min + i * dv, poisson);
  
  return 1.0 - integral*dv;
}

void NuFISolver::run()
{
  std::cout << "Building E_sline\n\n";

  using std::abs;
  using std::max;

  std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,sizeof(double)*Nx)), std::free };

  std::vector<double> int_E_squared;
  int_E_squared.reserve(Nt);

  if ( rho == nullptr ) throw std::bad_alloc {};

  Gradient grad(x_min, x_max, Nx);

  double total_time = 0;

  for (unsigned int it = 0; it < Nt; ++it)
  {
      stopwatch<double> timer;
      
      double time_elapsed_before = timer.elapsed();

      std::cout << "Timestep " << it << " / " << Nt << " (simulation time = "<< it*Parameters::DT << ")"<< std::endl;

      // compute rho

      double dx = Parameters::CALC_DX;
      
      #pragma omp parallel for
    	for(size_t i = 0; i<Nx; i++)
    	{
        double x = Parameters::X_DOMAIN_LEFT + i*dx;
        double ith_rho = eval_rho(it, x, poisson, Parameters::NV); 

        AssertThrow(std::isfinite(ith_rho), ExcMessage("NaN detected in rho"));
    		rho.get()[i] =  ith_rho;
    	}

      poisson.set_rhs_function(std::make_unique<ChargeDensity_NuFI<1>>(rho.get(), Nx));
      poisson.solve_step();

      std::vector<double> sampled_potential = poisson.sample_electric_potential(x_min, x_max, Nx); // Solution of FE

      // These have been tested to be equivalent
      // ////////////////////////////////////////////////
      // std::vector<double> E_vals = grad.compute(sampled_potential); // vector grad of FE solution
      // save_space_vector(E_vals, "electric", it);
      // std::vector<double> E_vals_deal = poisson.sample_electric_field(x_min, x_max, Nx); // FE grad of soution
      // save_space_vector(E_vals_deal, "electricdeal", it);
      // ////////////////////////////////////////////////                                             


      // interpolate and save current field

      std::vector<double> E_x(Nx,0.0) ;
      #pragma omp parallel for
      for(size_t ix=0; ix<Nx; ++ix)
      {
        E_x[ix] = -eval(Parameters::X_DOMAIN_LEFT+ix*dx, poisson);
      }

      double timer_elapsed = timer.elapsed();
      total_time += timer_elapsed;
      
      std::cout << "step made in "<< timer_elapsed-time_elapsed_before <<" seconds\n\n";
      if (it % Parameters::PLOT_FREQUENCY == 0)
      {
        std::cout << "Saving results...   ";
        save_f(*this, it, poisson, Parameters::PLOT_NX, Parameters::NV, "results/ftilda_" + std::to_string(it) + ".dat");
        save_rho(*this, it, poisson, Parameters::PLOT_NX, "results/rho_" + std::to_string(it) + ".dat");
        // save_Efield(it, coeffs.get(), 128, "results/field_" + std::to_string(it) + ".dat");
        save_space_vector(E_x, "field", it);

        double int_val = 0.5 * integral_space_vector_squared(poisson);
        int_E_squared.push_back(int_val);  
        save_space_vector(int_E_squared, "electricint", it);
        std::cout << "Time since start = "<< total_time<<"\n\n";
      }
  }

  std::cout << "NuFI simulation finished in "<< total_time <<" seconds.\n";
}

NuFISolver::NuFISolver()
  : order(Parameters::FE_DEGREE),
    poisson(order)
{
  std::cout << "Initializing dealii Poisson Solver\n";
  poisson.initialize();
}

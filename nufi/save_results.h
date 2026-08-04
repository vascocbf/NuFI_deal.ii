#ifndef SAVE_RESULTS_H
#define SAVE_RESULTS_H

#include "nufi/grids.h"
#include <deal.II/lac/vector.h>
#include <string>
#include <vector>

class NuFISolver;

struct RhoESnapshot {
  unsigned int it = 0;
  double time = 0.0;
  std::vector<double> x_eval;
  std::vector<double> rho;
  std::vector<double> E;
  double int_E_sqr = 0;
};

RhoESnapshot
compute_rho_E_snapshot(const NuFISolver &solver, unsigned int n,
                       std::vector<GridStructure<1>> &grid_struct,
                       std::vector<SolutionSnapshot<1>> &phi_history,
                       unsigned int Nx_out, unsigned int Nv = Parameters::NV);

void flush_rho_E_history(std::vector<RhoESnapshot> &history,
                         const std::string &dir);

struct DiagnosticsSnapshot {
  unsigned int Nx = 0;
  unsigned int Nv = 0;
  std::vector<double> x_eval;
  std::vector<double> v_eval;
  std::vector<double> f;
  std::vector<double> rho;
  std::vector<double> E;
};

DiagnosticsSnapshot
compute_diagnostics(const NuFISolver &solver, unsigned int n,
                    std::vector<GridStructure<1>> &grid_struct,
                    std::vector<SolutionSnapshot<1>> &phi_history,
                    unsigned int Nx_out, unsigned int Nv_out);

void save_f(const DiagnosticsSnapshot &snap, const std::string &filepath);
void save_rho(const DiagnosticsSnapshot &snap, const std::string &filepath);
void save_Efield(const DiagnosticsSnapshot &snap, const std::string &filepath);
double compute_int_E_squared(const std::vector<double> &E, double Lx);

void save_time_series(const std::vector<double> &t,
                      const std::vector<double> &values,
                      const std::string &filepath);

#endif

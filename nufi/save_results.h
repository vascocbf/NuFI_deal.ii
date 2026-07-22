#ifndef SAVE_RESULTS_H
#define SAVE_RESULTS_H

#include "nufi/grids.h"
#include <deal.II/lac/vector.h>
#include <string>
#include <vector>

class NuFISolver;

void save_f(const NuFISolver &solver, unsigned int n, const double v2_0,
            std::vector<GridStructure<1>> &grid_struct,
            std::vector<SolutionSnapshot<1>> &phi_history, unsigned int Nx_out,
            unsigned int Nv_out, const std::string &filename);

void save_rho(const NuFISolver &solver, unsigned int n,
              std::vector<GridStructure<1>> &grid_struct,
              std::vector<SolutionSnapshot<1>> &phi_history,
              unsigned int Nx_out, const std::string &filename);

void save_Efield(unsigned int it, std::vector<GridStructure<1>> &grid_versions,
                 std::vector<SolutionSnapshot<1>> &phi_history,
                 unsigned int Nx_out = Parameters::PLOT_NX);

void save_space_vector(const std::vector<double> &vals,
                       const std::string &filename, size_t it);

#endif

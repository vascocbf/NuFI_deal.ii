#include <filesystem>
#include <iostream>
#include <nufi/nufi_solver.h>
#include <omp.h>

void clear_results_directory(const std::string &dir) {
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir))
    return;

  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    if (std::filesystem::is_regular_file(entry)) {
      std::filesystem::remove(entry.path());
      std::cout << "Deleted: " << entry.path() << '\n';
    }
  }
}

int main() {
  omp_set_max_active_levels(1);
  std::cout << "Threads: " << omp_get_max_threads() << "\n";
  try {
    clear_results_directory("results");

    NuFISolver solver;
    solver.run();

  } catch (const std::exception &exc) {
    std::cerr << "\nException:\n" << exc.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "\nUnknown exception!\n";
    return 1;
  }

  return 0;
}

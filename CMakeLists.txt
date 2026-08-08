cmake_minimum_required(VERSION 3.13.4)
project(nufi_poisson)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

find_package(deal.II 9.6.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR
    "\n*** Could not locate deal.II. ***\n"
    "Pass -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
  )
endif()

deal_ii_initialize_cached_variables()

find_package(OpenMP REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LUA REQUIRED lua5.4)

# -------------------------

add_library(nufi_lib
  src/nufi_solver.cc
  src/save_results.cc
  #  src/blas.cc
)

target_include_directories(nufi_lib PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${LUA_INCLUDE_DIRS}
)

deal_ii_setup_target(nufi_lib)

target_link_libraries(nufi_lib
  OpenMP::OpenMP_CXX
  ${LUA_LIBRARIES}
) 

target_compile_options(nufi_lib PRIVATE -O3 -g -fno-omit-frame-pointer)

# -------------------------

add_executable(nufi_poisson
  src/main.cc
)

target_link_libraries(nufi_poisson
  nufi_lib
  OpenMP::OpenMP_CXX
)

deal_ii_setup_target(nufi_poisson)

target_compile_options(nufi_poisson PRIVATE -O3 -g -fno-omit-frame-pointer)

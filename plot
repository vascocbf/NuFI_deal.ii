#!/usr/bin/env gnuplot
#
# plot -- generate final-state plots (rho, E, |E|^2 time series, f) from
# nufi_poisson simulation output.
#
# Usage:
#   ./plot
#
# Requires gnuplot >= 5.0 (uses column(), sprintf(), system()).
# On Debian/Ubuntu:  sudo apt install gnuplot

results_dir = "./results" # .dat files
save_dir    = "./results" # save dir for the .png plots

X_DIM = 1
V_DIM = 1
dt = 0.0626 # 1/16

# 0 -> full plots
# 1 -> slice plots
slice_plots = 0

# For slice plots' titles
slice_coords = "x2 = 0, v2 = 0"

# For file name
slice_suffix = "_slice"

# Which component of E to plot
E_component = 0

plot_E_squared_slice = 0

log_E_squared_timeseries = 1 

# x-axis range for rho/E/f plots. Leave both as "" to autoscale.
x_min = "0"
x_max = "12.566"

img_width  = 1000
img_height = 700
font_size  = 12

# color scheme for the f(x,v) plot. One of:
#   "rainbow"    -> classic rainbow (rgbformulae 33,13,10)
#   "pm3d"       -> traditional pm3d blue-green-yellow-red (rgbformulae 7,5,15)
#   "grayscale"  -> muted grayscale (rgbformulae 21,22,23)
#   "diverging"  -> blue-white-red, good if f has a natural zero (rgbformulae 30,31,32)
#   "viridis"    -> matplotlib-style viridis (needs gnuplot >= 5.2, else falls back)
palette_choice = "viridis"

set terminal pngcairo size img_width,img_height font sprintf("Helvetica,%d", font_size)
system(sprintf("mkdir -p %s", save_dir))

if (!slice_plots && X_DIM != 1) {
  print "plot: WARNING - full plots are only intended for X_DIM == 1. Set slice_plots = 1 for higher-dimensional runs."
}
if (!slice_plots && V_DIM != 1 && X_DIM!=1) {
  print "plot: WARNING - full f(x,v) plotting assumes V_DIM == 1 and X_DIM==1."
}

# number of x-columns actually present in the rho/E files
x_cols = (slice_plots ? 1 : X_DIM)
# number of v-columns actually present in the f file
v_cols = (slice_plots ? 1 : V_DIM)

rho_col = x_cols + 1
E_col   = x_cols + 1 + E_component
Esq_col = x_cols + X_DIM + 1        # only used if plot_E_squared_slice
f_col   = x_cols + v_cols + 1

rho_suffix = (slice_plots ? slice_suffix : "")
E_suffix   = (slice_plots ? slice_suffix : "")
f_suffix   = (slice_plots ? slice_suffix : "")

rho_glob = sprintf("%s/rho%s_*.dat", results_dir, rho_suffix)

xr_lo = (x_min eq "" ? "*" : x_min)
xr_hi = (x_max eq "" ? "*" : x_max)

# --- find the last saved timestep (rho/E/f are saved together in main.cc) --
last_it = system(sprintf("ls %s 2>/dev/null | sed -E 's/.*_([0-9]+)\\.dat/\\1/' | sort -n | tail -1", rho_glob))

if (last_it eq "") {
  print sprintf("plot: no files matching %s found -- check results_dir / slice_plots / slice_suffix", rho_glob)
  exit
}

rho_file  = sprintf("%s/rho%s_%s.dat", results_dir, rho_suffix, last_it)
E_file    = sprintf("%s/E%s_%s.dat",   results_dir, E_suffix,   last_it)
f_file    = sprintf("%s/f%s_%s.dat",   results_dir, f_suffix,   last_it)
Esqr_file = sprintf("%s/int_E_sqr.dat", results_dir)

print sprintf("plot: using it = %s", last_it)
print sprintf("plot:   rho -> %s", rho_file)
print sprintf("plot:   E   -> %s", E_file)
print sprintf("plot:   f   -> %s", f_file)

slice_tag = (slice_plots ? sprintf(" [%s]", slice_coords) : "")

set output sprintf("%s/final_rho.png", save_dir)
set title sprintf("Final Charge Density {/Symbol r}(x), t = %.2f%s", (last_it+0)*dt, slice_tag)
set xlabel "x"
set ylabel "{/Symbol r}(x)"
set grid
set xrange [xr_lo:xr_hi]
unset key
plot rho_file using 1:(column(rho_col)) with lines lw 2 lc rgb "#1f77b4"

set output sprintf("%s/final_E.png", save_dir)
set title sprintf("Final Electric Field E_%d(x), it = %.2f%s", E_component, (last_it+0)*dt, slice_tag)
set xlabel "x"
set ylabel sprintf("E_%d", E_component)
set grid

if (slice_plots && plot_E_squared_slice) {
  set y2tics
  set y2label "|E|^2"
  set key top right
  plot E_file using 1:(column(E_col))   axes x1y1 with lines lw 2 lc rgb "#d62728" title sprintf("E_%d", E_component), \
       E_file using 1:(column(Esq_col)) axes x1y2 with lines lw 2 lc rgb "#2ca02c" title "|E|^2"
  unset y2tics
  unset y2label
  unset key
} else {
  unset key
  plot E_file using 1:(column(E_col)) with lines lw 2 lc rgb "#d62728"
}

# --- computation time plots ------------------------------------------------
sim_time_file = sprintf("%s/simulation_time.dat", results_dir)

if (system(sprintf("test -s %s && echo 1 || echo 0", sim_time_file)) eq "0") {
  print sprintf("plot: WARNING - %s not found or empty, skipping timing plots.", sim_time_file)
} else {
  last_iter = system(sprintf("tail -n 1 %s | awk '{print $1}'", sim_time_file))
  set xrange [0:(last_iter eq "" ? "*" : last_iter+0)]

  set output sprintf("%s/step_time.png", save_dir)
  set title "Per-iteration Computation Time"
  set xlabel "iteration"
  set ylabel "time [s]"
  set grid
  set key top left
  plot sim_time_file using 1:2 with lines lw 2 lc rgb "#1f77b4" title "step time", \
       sim_time_file using 1:4 with lines lw 2 lc rgb "#ff7f0e" title "compute time", \
       #sim_time_file using 1:6 with lines lw 2 lc rgb "#2ca02c" title "plot\\_time"
  unset key

  # Plot 2: cumulative time (total_time, cumulative compute_time, cumulative plot_time)
  set output sprintf("%s/cumulative_time.png", save_dir)
  set title "Cumulative Computation Time"
  set xlabel "iteration"
  set ylabel "cumulative time [s]"
  set grid
  set key top left
  plot sim_time_file using 1:3 with lines lw 2 lc rgb "#1f77b4" title "total time", \
       sim_time_file using 1:4 smooth cumulative with lines lw 2 lc rgb "#ff7f0e" title "compute time (cumulative)", \
       #sim_time_file using 1:6 smooth cumulative with lines lw 2 lc rgb "#2ca02c" title "plot\\_time (cum)"
  unset key
  unset xrange
}

set output sprintf("%s/E_squared_timeseries.png", save_dir)
set title "1/2 {/Symbol \362} E^2 dx vs. time"
set xlabel "time [s]"
set ylabel "1/2 {/Symbol \362} E^2 dx"
set grid
last_t = system(sprintf("tail -n 1 %s | awk '{print $1}'", Esqr_file))
set xrange [0:(last_t eq "" ? "*" : last_t+0)]
unset key
if (log_E_squared_timeseries) {
  set logscale y
}
plot Esqr_file using 1:2 with lines lw 2 lc rgb "#9467bd"
unset logscale y
unset xrange

set output sprintf("%s/final_f.png", save_dir)
set title sprintf("Phase Space Density f(x,v), it = %.2f%s", (last_it+0)*dt, slice_tag)
set xlabel "x"
set ylabel "v"
unset grid
set pm3d map
if (palette_choice eq "rainbow") {
  set palette rgbformulae 33,13,10
} else { if (palette_choice eq "pm3d") {
  set palette rgbformulae 7,5,15
} else { if (palette_choice eq "grayscale") {
  set palette rgbformulae 21,22,23
} else { if (palette_choice eq "diverging") {
  set palette rgbformulae 30,31,32
} else { if (palette_choice eq "viridis") {
  set palette defined (0 '#440154', 0.25 '#3b528b', 0.5 '#21908d', 0.75 '#5dc963', 1 '#fde725')
} else {
  print sprintf("plot: unknown palette_choice '%s', falling back to rainbow", palette_choice)
  set palette rgbformulae 33,13,10
}}}}}
set cblabel "f(x,v)"
set xrange [xr_lo:xr_hi]
splot f_file using 1:2:(column(f_col)) with pm3d

unset output

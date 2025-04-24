begin
    using Pkg
    Pkg.activate(pwd()[end-14:end] == "nonlocal_arrays" ? "." : "../")
end

using Distributed
using ProgressMeter
using CairoMakie, GLMakie, LinearAlgebra, StaticArrays, FFTW
# Set up distributed environment
N_procs = 6
if nprocs() == 1
    addprocs(N_procs)
end

# Ensure all necessary packages are available on all workers
@everywhere begin
    using NonlinearSolve, SparseDiffTools, SparseConnectivityTracer, ADTypes, ForwardDiff
    using LinearSolve, DifferentialEquations, SparseArrays, AtomicArrays, NonlocalArrays
end

# Define paths for saving data and figures
const PATH_DATA = "../Data/"
const PATH_FIGS = "../Figs/"

# ----------------------------------------------------------------
# Function to compute simulation for one combination of parameters.
# ----------------------------------------------------------------
@everywhere function compute_simulation(params::Dict{String,Any})
    # Expected keys: "a", "B_z", "deltas", "POLARIZATION", "amplitude", "angle_k"
    a           = params["a"]
    B_z         = params["Bz"]
    delta_val   = params["deltas"]
    POLARIZATION = params["POLARIZATION"]
    amplitude   = params["amplitude"]
    angle_k     = params["anglek"]
    Nx          = params["Nx"]
    Ny          = params["Ny"]

    coll, _, _, OmR = build_fourlevel_system(; a=a, Nx=Nx, Ny=Ny, 
                                               delta_val=delta_val,
                                               POL=POLARIZATION,
                                               amplitude=amplitude,
                                               angle_k=angle_k,
                                               field_func=AtomicArrays.field.gauss)

    # Initialize product state.
    state0 = AtomicArrays.fourlevel_meanfield.ProductState(length(coll.atoms))
    # Compute the steady state.
    steady_state = AtomicArrays.fourlevel_meanfield.steady_state_nonlinear(coll, OmR, B_z, state0)
    return steady_state
end

# ----------------------------------------------------------------
# Generalized sweep over multiple parameters.
# ----------------------------------------------------------------
function run_sweep(sweep_params::AbstractDict{String, X}) where X
    # Each key in sweep_params maps to a collection of values to sweep.
    keys_list = collect(keys(sweep_params))
    values_list = [sweep_params[k] for k in keys_list]
    # Form the Cartesian product of the parameter values.
    prod_iter = Base.Iterators.product(values_list...)
    # Total number of combinations is the product of the lengths.
    total = prod(length, values_list)
    
    # Prepare arrays to collect the parameter dictionaries and their futures.
    params_vec = Vector{Dict{String,Any}}()
    futures = Vector{Future}()
    
    # Dispatch all simulation tasks concurrently.
    for combo in prod_iter
        combo_dict = Dict{String,Any}()
        for (k, v) in zip(keys_list, combo)
            combo_dict[k] = v
        end
        push!(params_vec, combo_dict)
        push!(futures, @spawnat :any compute_simulation(combo_dict))
    end

    # Now fetch all results, updating the progress bar.
    results = Dict{Dict{String,Any}, Any}()
    p = Progress(total, desc = "Simulating steady states")
    for (combo_dict, fut) in zip(params_vec, futures)
        results[combo_dict] = fetch(fut)
        next!(p)
    end
    return results
end

# ----------------------------------------------------------------
# Example sweep parameters.
# Users can choose which parameters to vary and their ranges.
# (By convention, keys here do not include underscores.)
sweep_params = Dict(
    "a"            => 0.2,#range(0.1, 1.0, length=200),
    "Bz"           => 0.2,
    "deltas"       => range(-1.8, 1.8, length=200),
    "POLARIZATION" => ("R", "L"),
    "amplitude"    => 0.02,
    "anglek"       => [[0.0, 0.0]],
    "Nx"           => 6,
    "Ny"           => 6,
)

# Run the sweep over all parameter combinations.
all_results = run_sweep(sweep_params)

# Save the sweep results using the provided save_sweep interface.
savepath = save_sweep(PATH_DATA, all_results;
                      description = "steady-states-sweep",
                      sweep_params = sweep_params)
@info "Sweep results saved to $savepath"

parse_sweep_filename(savepath)

load_sweep(savepath)
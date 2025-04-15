# Activate environment
begin
    using Pkg
    Pkg.activate(pwd()[end-14:end] == "nonlocal_arrays" ? "." : "../")
end

# Set up distributed environment
using Distributed
N_procs = 6
if nprocs() == 1
    addprocs(N_procs)
end

# Ensure all necessary packages are available on all workers
@everywhere begin
    using NonlinearSolve, SparseDiffTools, SparseConnectivityTracer, ADTypes, ForwardDiff
    using LinearSolve, DifferentialEquations, SparseArrays, AtomicArrays, NonlocalArrays
end
using CairoMakie, GLMakie, LinearAlgebra
using Serialization, Printf

"""
    save_result(path, result, N, B_z, pol, amplitude, a)

Serialize a result to a file with a descriptive name.
"""
function save_result(path::String, result,
                     N::Int, B_z::Real,
                     pol::String, amplitude::Float64, a::Float64)
    filename = joinpath(path,
        @sprintf("result_N%d_Bz%.4f_%s_E%.4f_a%.3f.bson", N, B_z, pol, amplitude, a))
    serialize(filename, result)
    return filename
end

"""
    load_all_results(path)

Deserialize all .bson result files in the directory.
"""
function load_all_results(path::String)
    files = filter(f -> endswith(f, ".bson"), readdir(path; join=true))
    return [deserialize(file) for file in sort(files)]
end

"""
    parse_result_filename(filename::String)

Extract parameters (N, B_z, polarization, amplitude, a) from filename string.
"""
function parse_result_filename(filename::String)
    pattern = r"result_N(\d+)_Bz([-\d.]+)_([RL])_E([\d.]+)_a([\d.]+)\.bson"
    m = match(pattern, basename(filename))
    isnothing(m) && error("Invalid filename format: $filename")

    N        = parse(Int, m.captures[1])
    B_z      = parse(Float64, m.captures[2])
    pol      = m.captures[3]
    E0       = parse(Float64, m.captures[4])
    a_val    = parse(Float64, m.captures[5])

    return (; N, B_z, POLARIZATION = pol, amplitude = E0, a = a_val)
end


"""
    save_sweep(path::String, data;
               description::String = "results", kwargs...)

Save any `data` object with a filename auto-generated from key-value pairs in `kwargs`,
plus a short description. Supports Float, Tuple, Vector, and String keyword arguments.

Examples:
save_sweep(PATH_DATA, data; description="states", Bz=(0.1, 0.2), N=100, POL="R", a=0.2)

Creates a file like:
    states_Bz_0.1_to_0.2_N_100_POL_R_a_0.2.bson
"""
function save_sweep(path::String, data;
                    description::String = "results", kwargs...)

    mkpath(path)
    parts = [description]
    for (key, val) in pairs(kwargs)
        strval = if isa(val, AbstractVector) && !isempty(val)
            @sprintf("%s_%.3f_to_%.3f", key, minimum(val), maximum(val))
        elseif isa(val, Tuple) && length(val) == 2
            @sprintf("%s_%.3f_to_%.3f", key, val[1], val[2])
        elseif isa(val, AbstractFloat)
            @sprintf("%s_%.3f", key, val)
        elseif isa(val, Int)
            string(key, "_", val)
        elseif isa(val, AbstractString)
            string(key, "_", val)
        else
            error("Unsupported kwarg type for key = $key, val = $val")
        end
        push!(parts, strval)
    end

    filename = join(parts, "_") * ".bson"
    filepath = joinpath(path, filename)
    serialize(filepath, data)
    return filepath
end


"""
    load_sweep(filepath::String)

Load data from a sweep result file (saved via `save_sweep`).
"""
load_sweep(filepath::String) = deserialize(filepath)


# Function to extract positions from the FourLevelAtomCollection and plot them
function plot_atoms_with_field(coll::AtomicArrays.FourLevelAtomCollection, field::AtomicArrays.field.EMField)
    CairoMakie.activate!()
    # Extract atom positions
    xs = [atom.position[1] for atom in coll.atoms]
    ys = [atom.position[2] for atom in coll.atoms]
    zs = [atom.position[3] for atom in coll.atoms]

    # Create a new figure and add a 3D axis.
    fig = Figure(size = (800, 600))
    ax = Axis3(fig[1, 1];
        title = "3D Array of Atoms with Field Vectors",
        xlabel = "x", ylabel = "y", zlabel = "z",
        perspectiveness = 0.8
    )

    # Plot atom positions
    scatter!(ax, xs, ys, zs; markersize = 10, color = :blue)

    # Field origin
    field_origin = Point3f(field.position_0...)

    # Arrow scaling
    k_scale = 1.0
    pol_scale = 1.0

    # k-vector arrow
    k_arrow = Vec3f(normalize(field.k_vector)...) * k_scale
    arrows!(ax, [field_origin], [k_arrow]; linewidth = 0.02, arrowsize = 0.04, color = :red)
    text!(ax, "k-vector"; position = field_origin + k_arrow, align = (:center, :bottom), color = :red)

    # polarization arrow
    pol_arrow = Vec3f(real.(field.polarisation)...) * pol_scale
    pol_arrow_im = Vec3f(imag(field.polarisation)...) * pol_scale
    arrows!(ax, [field_origin], [pol_arrow]; linewidth = 0.02, arrowsize = 0.04, color = :green)
    arrows!(ax, [field_origin], [pol_arrow_im]; linewidth = 0.02, arrowsize = 0.04, color = :green)
    text!(ax, "polarization"; position = field_origin + pol_arrow, align = (:center, :top), color = :green)

    return fig
end

"""
    parse_sweep_filename(filename::String) -> Dict{String, Any}

Parses filenames like:
"steady_states_Bz_-0.200_to_0.200_num_vals_36_N_100_POL_R_E0_0.020_a_0.200_geometry_rect.bson"
into a dictionary of parameter names and values.
"""
function parse_sweep_filename(filename::String)::Dict{String, Any}
    # Remove directory components and file extension.
    base = splitext(basename(filename))[1]
    # Split base name on underscores.
    tokens = split(base, '_')
    
    # By convention, the first token is the file prefix (with no underscores) and is discarded.
    if length(tokens) > 1
        tokens = tokens[2:end]
    end

    params = Dict{String, Any}()
    i = 1
    while i <= length(tokens)
        # Ensure there is at least one token available for a value.
        if i == length(tokens)
            break
        end
        key = tokens[i]
        val_token = tokens[i+1]
        # Check if a range is indicated: key, value, "to", value.
        if i + 2 <= length(tokens) && tokens[i+2] == "to" && i + 3 <= length(tokens)
            try
                v1 = parse(Float64, val_token)
                v2 = parse(Float64, tokens[i+3])
                params[key] = (v1, v2)
            catch err
                # If parsing fails, fall back to using the raw token.
                params[key] = val_token
            end
            i += 4
        else
            # For a single value, attempt conversion.
            iv = tryparse(Int, val_token)
            if iv !== nothing
                params[key] = iv
            else
                fv = tryparse(Float64, val_token)
                if fv !== nothing
                    params[key] = fv
                else
                    params[key] = val_token
                end
            end
            i += 2
        end
    end

    return params
end



# Define paths (used for saving data, figures, etc.)
const PATH_DATA = "../Data"
const PATH_FIGS = "../Figs"

# Build the atomic collection and fields on main process
begin
    a, Nx, Ny = 0.2, 10, 10
    positions = AtomicArrays.geometry.rectangle(a, a; Nx=Nx, Ny=Ny,
                                position_0=[(-Nx/2+0.5)*a,(-Ny/2+0.5)*a,0.0])
    N = length(positions)

    pols = AtomicArrays.polarizations_spherical(N)
    gam = [AtomicArrays.gammas(0.25)[m] for m in 1:3, j in 1:N]
    deltas = [(i == 1) ? 0.0 : 0.2 for i in 1:N]

    coll = AtomicArrays.FourLevelAtomCollection(positions; deltas, polarizations=pols, gammas=gam)

    POLARIZATION = "R"
    amplitude, k_mod = 0.02, 2π
    angle_k = [0*π/6, 0.0]  # x-z plane
    polarisation = POLARIZATION == "R" ? [1.0, -1.0im, 0.0] : [1.0, 1.0im, 0.0]
    waist_radius = 0.3 * a * sqrt(Nx * Ny)
    field = AtomicArrays.field.EMField(amplitude, k_mod, angle_k, polarisation;
                                       position_0=[0.0, 0.0, 0.0], waist_radius)
    field_func = AtomicArrays.field.gauss

    OmR = AtomicArrays.field.rabi(field, field_func, coll)
    println("System built with N = $N atoms.")
end

plot_atoms_with_field(coll, field)

# Cache for Jacobian sparsity patterns
@everywhere const jacobian_sparsity_cache = Dict{Tuple{Int, Int}, SparseMatrixCSC{Float64, Int}}()

# Steady-state computation
@everywhere function steady_state_problem(A::AtomicArrays.FourLevelAtomCollection, Om_R::Array{ComplexF64,2}, B_z::Real, state0::AtomicArrays.fourlevel_meanfield.ProductState; abstol=1e-8, reltol=1e-8, maxiters=100)
    N = state0.N
    Omega = AtomicArrays.interaction.OmegaTensor_4level(A)
    Gamma = AtomicArrays.interaction.GammaTensor_4level(A)
    w = [A.atoms[n].delta + B_z * m for m = -1:1, n = 1:N]
    p = (w, Om_R, Omega, Gamma)

    function f_steady!(du, u, p)
        AtomicArrays.fourlevel_meanfield.f(du, u, p, 0.0)
    end

    function jacobian!(J, u, p)
        SparseDiffTools.forwarddiff_color_jacobian!(J, (du, u) -> f_steady!(du, u, p), u)
    end

    u0 = copy(state0.data)
    cache_key = (length(u0), length(p))
    sparsity = get!(jacobian_sparsity_cache, cache_key) do
        detector = TracerSparsityDetector()
        jacobian_sparsity((du, u) -> f_steady!(du, u, p), u0, u0, detector)
    end

    nlfun = NonlinearFunction(f_steady!, jac=jacobian!, jac_prototype=sparsity)
    prob = NonlinearProblem(nlfun, u0, p)
    linsolve = LinearSolve.KrylovJL_GMRES()
    sol = solve(prob, NewtonRaphson(linsolve=linsolve); abstol, reltol, maxiters)

    state0.data .= sol.u
    return state0
end

# Parallel worker job
@everywhere function compute_for_Bz(B_z_val, A, Om_R, initial_state)
    steady_state_problem(A, Om_R, B_z_val, deepcopy(initial_state))
end

# Set up B_z sweep and run in parallel
B_z_vals = range(-0.2, 0.2, length=6*N_procs)
state0 = AtomicArrays.fourlevel_meanfield.ProductState(N)

results = pmap(Bz -> compute_for_Bz(Bz, coll, OmR, state0), B_z_vals)

# Results now contains steady states for each B_z
results
savepath = save_sweep(PATH_DATA, results;
                      description = "steady-states",
                      Bz = B_z_vals, numvals = length(B_z_vals),
                      N = N, POL = POLARIZATION,
                      E0 = field.amplitude, a = a,
                      geometry = "rect", theta = field.angle_k[1])


# Analysis
results_loaded = load_sweep(savepath)
test_dict = parse_sweep_filename(savepath)

sigmas_m = map(i -> AtomicArrays.fourlevel_meanfield.sigma_matrices(results, i)[1], eachindex(results))
sigmas_mm = map(i -> AtomicArrays.fourlevel_meanfield.sigma_matrices(results, i)[2], eachindex(results))

zlim = 1*a*Nx
trans_result = [AtomicArrays.field.transmission_reg(field, field_func, coll, sigmas_m[i]; samples=400, zlim=zlim)[1] for i = 1:6*N_procs]


# Scattered and total field
begin
    # Options: "xy", "xz", "yz"
    plane = "xz"   # change this to "xz" or "yz" as desired
    index_Bz = 1

    # === Define grid parameters ===
    n_points = 100
    factor_scale = 3.5
    grid_min, grid_max = -Nx*a/2*factor_scale, Nx*a/2*factor_scale
    coord1_range = range(grid_min, grid_max; length=n_points)
    coord2_range = range(grid_min, grid_max; length=n_points)


    # Depending on the plane, choose the two varying coordinates and fix the third:
    if plane == "xy"
        fixed_value = Nx*a/2 + 0.2                           # fixed z
        fixed_index = 3
        label1, label2, fixed_label = "x", "y", "z"
    elseif plane == "xz"
        fixed_value = (mod(Nx, 2) == 0) ? 0.0 : a/2          # fixed y
        fixed_index = 2
        label1, label2, fixed_label = "x", "z", "y"
    elseif plane == "yz"
        fixed_value = Nx*a/2 + 0.2                           # fixed x
        fixed_index = 1
        label1, label2, fixed_label = "y", "z", "x"
    else
        error("Unknown plane. Choose 'xy', 'xz', or 'yz'.")
    end

    nx, ny = length(coord1_range), length(coord2_range)

    I_total = zeros(nx, ny)
    I_scatt = zeros(nx, ny)

    # === Update atom positions according to the selected plane ===
    # For "xy": use (x,y), for "xz": use (x,z), for "yz": use (y,z)
    if plane == "xy"
        atom_coord1 = [atom.position[1] for atom in coll.atoms]
        atom_coord2 = [atom.position[2] for atom in coll.atoms]
    elseif plane == "xz"
        atom_coord1 = [atom.position[1] for atom in coll.atoms]
        atom_coord2 = [atom.position[3] for atom in coll.atoms]
    elseif plane == "yz"
        atom_coord1 = [atom.position[2] for atom in coll.atoms]
        atom_coord2 = [atom.position[3] for atom in coll.atoms]
    end

    atom_x = [atom.position[1] for atom in coll.atoms]
    atom_y = [atom.position[2] for atom in coll.atoms]
    atom_z = [atom.position[3] for atom in coll.atoms]

    # === Compute the scattered field on the grid ===
    # In each case, r is built by inserting the two grid coordinates into the proper indices.
    for (i, a) in enumerate(coord1_range)
        for (j, b) in enumerate(coord2_range)
            r = zeros(3)
            # Set the two varying coordinates.
            if plane == "xy"
                r[1] = a; r[2] = b; r[3] = fixed_value
            elseif plane == "xz"
                r[1] = a; r[2] = fixed_value; r[3] = b
            elseif plane == "yz"
                r[1] = fixed_value; r[2] = a; r[3] = b
            end
            # Compute the scattered field at r using the provided function.
            # It is assumed that sigmas_m is defined appropriately.
            E = AtomicArrays.field.scattered_field(r, coll, sigmas_m[index_Bz])
            E_t = field_func(r, field) + E
            I_t = (norm(E_t)^2 / abs(field.amplitude)^2)
            I_s = (norm(E)^2 / abs(field.amplitude)^2)
            
            # Save the real and absolute values for each Cartesian component.
            I_total[i, j] = I_t
            I_scatt[i, j] = I_s
        end
    end
end

let
    CairoMakie.activate!()
    fig = Figure(size = (1500, 700))

    nlevels = 50

    # --- Row 1: x-component (Eₓ) ---
    ax1 = Axis(fig[1, 1], title = "I_t, "*POLARIZATION*", E₀ = "*string(round(field.amplitude, digits=3)) * ", B_z = "*string(round(B_z_vals[index_Bz], digits=3)), xlabel = label1, ylabel = label2,
               titlesize = 24, xlabelsize = 20, ylabelsize = 20,
               xticklabelsize = 18, yticklabelsize = 18)
    ax2 = Axis(fig[1, 2], title = "I_sc, "*POLARIZATION*", E₀ = "*string(round(field.amplitude, digits=3)) * ", B_z = "*string(round(B_z_vals[index_Bz], digits=3)), xlabel = label1, ylabel = label2,
               titlesize = 24, xlabelsize = 20, ylabelsize = 20,
               xticklabelsize = 18, yticklabelsize = 18)
    hm1 = contourf!(ax1, coord1_range, coord2_range, I_total,
                    colormap = :plasma, 
                    levels = range(0, 1.0*maximum(I_total), nlevels), 
                    # colorscale=log10,
                    )
    scatter!(ax1, atom_coord1, atom_coord2, markersize = 8, color = :white)
    hm2 = contourf!(ax2, coord1_range, coord2_range, I_scatt,
                    colormap = :plasma, 
                    levels = range(0, 1.0*maximum(I_scatt), nlevels), 
                    # colorscale=log10,
                    )
    scatter!(ax2, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[1, 3], hm1, label="Iₜ/|E₀|²",
             labelsize = 20, ticklabelsize = 18,
             width = 15, height = Relative(1))
    Colorbar(fig[1, 4], hm2, label="Iₛ/|E₀|²",
             labelsize = 20, ticklabelsize = 18,
             width = 15, height = Relative(1))

    # save(PATH_FIGS*"4level_I_t_a"*string(a)*"_"*POLARIZATION*"_theta"*string(round(field.angle_k[1], digits=2))*"_N"*string(N)*".pdf", fig, px_per_unit=4)
    fig
end


let 
    f = Figure(size=(1000, 300))
    ax1 = Axis(f[1, 1], title="Polrization = "*POLARIZATION, xlabel="t", ylabel="Transmission")
    # Plot sublevels
    lines!(ax1, B_z_vals, trans_result, linewidth=2)
    f
end
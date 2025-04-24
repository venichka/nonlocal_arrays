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

# # Cache for Jacobian sparsity patterns
# @everywhere const jacobian_sparsity_cache = Dict{Tuple{Int, Int}, SparseMatrixCSC{Float64, Int}}()
# # ----------------------------------------------------------------
# # Steady-state simulation function.
# # ----------------------------------------------------------------
# @everywhere function steady_state_problem(A::AtomicArrays.FourLevelAtomCollection, 
#     Om_R::Array{ComplexF64,2}, B_z::Real, state0::AtomicArrays.fourlevel_meanfield.ProductState; 
#     abstol=1e-8, reltol=1e-8, maxiters=100)
    
#     N = state0.N
#     Omega = AtomicArrays.interaction.OmegaTensor_4level(A)
#     Gamma = AtomicArrays.interaction.GammaTensor_4level(A)
#     w = [A.atoms[n].delta + B_z * m for m = -1:1, n = 1:N]
#     p = (w, Om_R, Omega, Gamma)

#     function f_steady!(du, u, p)
#         AtomicArrays.fourlevel_meanfield.f(du, u, p, 0.0)
#     end

#     function jacobian!(J, u, p)
#         SparseDiffTools.forwarddiff_color_jacobian!(J, (du, u) -> f_steady!(du, u, p), u)
#     end

#     u0 = copy(state0.data)
#     cache_key = (length(u0), length(p))
#     sparsity = get!(jacobian_sparsity_cache, cache_key) do
#         detector = TracerSparsityDetector()
#         jacobian_sparsity((du, u) -> f_steady!(du, u, p), u0, u0, detector)
#     end

#     nlfun = NonlinearFunction(f_steady!, jac=jacobian!, jac_prototype=sparsity)
#     prob = NonlinearProblem(nlfun, u0, p)
#     linsolve = LinearSolve.KrylovJL_GMRES()
#     sol = solve(prob, NewtonRaphson(linsolve=linsolve); abstol, reltol, maxiters)

#     state0.data .= sol.u
#     return state0
# end

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
    # steady_state = steady_state_problem(coll, OmR, B_z, state0)
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

@show all_results


# Define fixed parameter values
fixed_params = Dict(
    "a" => 0.2,
    "Bz" => 0.2,
    "POLARIZATION" => "R",
    "amplitude" => 0.02,
    "anglek" => [0.0, 0.0],
    "Nx" => 6,
    "Ny" => 6,
)
fixed_params_0 = Dict(
    "a" => 0.2,
    # "deltas" => 0.0,
    "Bz" => 0.2,
    "amplitude" => 0.02,
    "anglek" => [0.0, 0.0],
    "Nx" => 6,
    "Ny" => 6,
)

# Define transmission function
transmission_func = (result, params) -> begin
    coll, field, field_func, _ = build_fourlevel_system(merge(params, Dict("field_func" => AtomicArrays.field.gauss)))
    sigmas_m = AtomicArrays.fourlevel_meanfield.sigma_matrices([result], 1)[1]
    zlim = 2.0
    AtomicArrays.field.transmission_reg(field, field_func, coll, sigmas_m; samples=400, zlim=zlim)[1]
end

transmission_func_new = (result, params) -> begin
    coll, field, field_func, _ = build_fourlevel_system(merge(params, Dict("field_func" => AtomicArrays.field.gauss)))
    sigmas_m = AtomicArrays.fourlevel_meanfield.sigma_matrices([result], 1)[1]
    zlim = 2.0
    coefs = TR_coefficients(field, coll, sigmas_m;
                    beam=:gauss,
                    surface=:hemisphere,
                    samples=400,
                    zlim=zlim,
                    size=[5.0,5.0])
    # coefs = transmission_reg(field,
    #                       field_func,
    #                       coll, sigmas_m,
    #                       samples=400, zlim=zlim)
    coefs[2] #+ coefs[2]
end

GLMakie.activate!()
CairoMakie.activate!()
fig = plot_sweep_quantity(all_results, transmission_func, "deltas";
    fixed_params=fixed_params, ylabel="Transmission")

plot_sweep_multicurve(all_results,
                                # transmission_func,
                                transmission_func_new,
                                "deltas",
                                "POLARIZATION";
                                fixed_params=fixed_params_0)







function transmission_reg(E::AtomicArrays.field.Field, 
                          inc_wave_function::Function,
                          S::Union{SpinCollection, FourLevelAtomCollection}, sigmam::AbstractArray;
                          samples::Int=50, zlim::Real=1000.0, angle::Vector=[π, π])

    # Extract front/back position L depending on wave vector
    L = 0.0
    zlim = (E.angle_k[1] >= π/2) ? -zlim : zlim

    # Generate hemisphere points via Fibonacci method
    θ, φ = AtomicArrays.field.fibonacci_angles(samples)
    r = [zlim * [sin(θ[j]) * cos(φ[j]),
                 sin(θ[j]) * sin(φ[j]),
                 cos(θ[j]) + L / zlim] for j in 1:samples]

    ΔΩ = 2π/samples                     # equal‑area weight
    area_factor = zlim^2                # R² term in ∫ I dΩ

    # Precompute incoming and total fields
    E_in = inc_wave_function(r, E)  # vector{vector}
    total_fields = field.total_field(inc_wave_function, r, E, S, sigmam)

    # Compute normalization
    # E_in2 = (π * E.waist_radius^2 * abs(E.amplitude)^2) / 4#sum(intensity.(E_in))
    E_in2 = sum(field.intensity.(E_in)) * (ΔΩ*area_factor)

    # Compute transmission
    E_out2 = 0.0
    for j in 1:samples
        E_in_j = E_in[j]
        tf_j = total_fields[j]
        for k in 1:samples
            tf_k = total_fields[k]
            E_in_k = E_in[k]
            E_out2 += abs(conj(E_in_j') * (tf_j' * tf_k) * conj(E_in_k))*(ΔΩ*area_factor)^2
        end
    end
    # E_out2 = 0.0
    # for j in 1:samples
    #     E_in_j = E.polarisation
    #     tf_j = total_fields[j]
    #     E_out2 += abs(conj(E_in_j') * (tf_j' * tf_j) * conj(E_in_j))*(ΔΩ*area_factor)
    # end

    E_in4 = abs2(E_in2)
    return E_out2 / E_in4, 1 - E_out2 / E_in4
        # alternative way
    # for j in 1:samples
    #     r_j = zlim * [sin(θ[j])*cos(φ[j]),
    #                   sin(θ[j])*sin(φ[j]),
    #                   cos(θ[j]) + L/zlim]
    #     r[j] = r_j
    #     E_in = inc_wave_function(r_j, E)
    #     E_out2 += abs(E_in'*
    #     total_field(inc_wave_function,r_j, E, S, sigmam)*total_field(inc_wave_function,r_j, E, S, sigmam)'*E_in)
    #     E_in2 += abs(E_in'*E_in)^2
    # end
end

function scattered_field_0(r::AbstractVector,
                           A::AtomicArrays.FourLevelAtomCollection,
                           sigmas_m::AbstractMatrix; k::Real=2*π)
    M, N = size(A.gammas)
    C = 3/4*A.gammas
    return sum(C[m, n] * 
               sigmas_m[m, n] *
               AtomicArrays.field.GreenTensor(r .- A.atoms[n].position, k) *
               A.polarizations[m, :, n] for m in 1:M, n in 1:N)
end


function TR_coefficients(E::AtomicArrays.field.EMField,
                         collection, σ;
                         beam::Symbol      = :plane,
                         surface::Symbol   = :hemisphere,
                         polarization      = nothing,
                         samples::Int      = 50,
                         zlim::Real        = 1_000.0,
                         size::Vector      = [5.0,5.0])
                                
    # ---------- 1.  Set helpers ----------
    inc_wave = beam === :plane ? AtomicArrays.field.plane : AtomicArrays.field.gauss
    pol      = polarization === nothing ? E.polarisation :
                polarization ./ norm(polarization)      # ensure unit‑norm
    k̂        = E.k_vector / norm(E.k_vector)           # propagation direction
                                
    # analytic incident intensity  (c = ε₀ = 1)
    intensity(v) = abs2( dot(pol, v) )/2
    I_inc = abs2(E.amplitude)/2
    P_inc = (π*E.waist_radius^2/2) * I_inc
                                
    # ---------- 2.  Make integration grids ----------
    if surface === :hemisphere
        θ, φ = AtomicArrays.field.fibonacci_angles(samples)
        # forward (+k̂) & backward (−k̂) hemispheres
        r_fwd =  Vector{Vector{Float64}}(undef, samples)
        r_bwd =  Vector{Vector{Float64}}(undef, samples)
        for j in eachindex(θ)
            rot_vec = [sin(θ[j])*cos(φ[j]),
                       sin(θ[j])*sin(φ[j]),
                       cos(θ[j])]
            # build orthonormal basis with k̂ as new +z
            # fast Gram–Schmidt
            zax = k̂
            xax = abs(zax[3]) < 0.9 ? normalize(cross(zax,[0,0,1])) :
                                      normalize(cross(zax,[0,1,0]))
            yax = cross(zax,xax)
            R   = hcat(xax,yax,zax)          # 3×3 rotation matrix
            dir = R*rot_vec
            r_fwd[j] =  zlim* dir           # along +k̂
            r_bwd[j] = -zlim* dir           # along −k̂
        end
        ΔΩ = 2π/samples                     # equal‑area weight
        area_factor = zlim^2                # R² term in ∫ I dΩ
    else  # :plane
        # square grid centred on optical axis
        x = range(-0.5*size[1], stop=0.5*size[1], length=samples)
        y = range(-0.5*size[2], stop=0.5*size[2], length=samples)
        r_fwd = [ [xx,yy, zlim] for yy in y, xx in x ] |> vec
        r_bwd = [ [xx,yy,-zlim] for yy in y, xx in x ] |> vec
        dA   = (size[1]/(samples-1))*(size[2]/(samples-1))
    end
                                
    # ---------- 3.  Pre‑compute fields ----------
    # (broadcasting works because inc_wave / total_field already have dot‑methods)
    E_in_fwd  = inc_wave(r_fwd, E)
    E_sc_fwd = AtomicArrays.field.scattered_field(r_fwd, collection, σ)#/sqrt(3)
    # E_sc_fwd = scattered_field_0.(r_fwd, Ref(collection), Ref(σ))#/sqrt(3)
    E_tot_fwd = E_in_fwd .+ E_sc_fwd
    E_in_bwd  = inc_wave(r_bwd, E)
    E_sc_bwd = AtomicArrays.field.scattered_field(r_bwd, collection, σ)#/sqrt(3)
    # E_sc_bwd = scattered_field_0.(r_bwd, Ref(collection), Ref(σ))#/sqrt(3)
    E_tot_bwd = E_in_bwd .+ E_sc_bwd
                                
    # ---------- 4.  Integrate power ----------
    if surface === :hemisphere
        # projected |E|² sum  (∝ intensity); pol·E selects co‑polarised power
        P_fwd = ΔΩ*area_factor * sum( intensity.(E_tot_fwd) )
        P_bwd = ΔΩ*area_factor * sum( intensity.(E_sc_bwd) )#.- intensity.(E_in_bwd))
        P_inc = beam === :plane ? I_inc*2*pi*zlim^2 : P_inc # analytic = I_inc*π zlim²
        # P_inc = ΔΩ*area_factor * sum( intensity.(E_in_fwd) )
    else  # plane
        P_fwd = dA * sum( intensity.(E_tot_fwd) )
        P_bwd = dA * sum( intensity.(E_sc_bwd) )#- intensity.(E_in_bwd))
        P_inc = I_inc * size[1] * size[2]*2              # analytic = I_inc*size[1]*size[2]
        # P_inc = dA * sum( intensity.(E_in_fwd) )
    end
                                
    # ---------- 5.  Coefficients ----------
    T = P_fwd / P_inc
    R = P_bwd / P_inc
    return T, R
end












let
GLMakie.activate!()
# Get the points
R = 1.0
L = 0.0
nθ, nφ = 20, 40
# rf, rb = hemisphere(R, nθ, nφ)

# Generate hemisphere points via Fibonacci method
samples = 400
θ, φ = AtomicArrays.field.fibonacci_angles(samples)
rf = [R * [sin(θ[j]) * cos(φ[j]),
             sin(θ[j]) * sin(φ[j]),
             cos(θ[j]) + L / R] for j in 1:samples]
rb = [-R * [sin(θ[j]) * cos(φ[j]),
             sin(θ[j]) * sin(φ[j]),
             cos(θ[j]) - L / R] for j in 1:samples]

# Extract x, y, z for both sets
xf, yf, zf = map(v -> getindex.(rf, v), 1:3)
xb, yb, zb = map(v -> getindex.(rb, v), 1:3)

# Plotting
fig = Figure()
ax = Axis3(fig[1, 1], title = "Hemisphere points",
           xlabel = "x", ylabel = "y", zlabel = "z")

scatter!(ax, xf, yf, zf, markersize = 4, color = :blue, label = "Forward")
scatter!(ax, xb, yb, zb, markersize = 4, color = :red, label = "Backward")
axislegend(ax)
fig
end
pars[idxs][312]
# Scattered and total field
begin
    # Options: "xy", "xz", "yz"
    plane = "xz"   # change this to "xz" or "yz" as desired



    coll, E_in, field_func, _ = build_fourlevel_system(merge(fixed_params_0, Dict("field_func" => AtomicArrays.field.gauss)))

    results = []
    pars = []
    for (params, result) in all_results
        append!(pars, [params])
        append!(results, [result])
    end
    idxs = sortperm(pars, by=p -> p["deltas"])

    sigmas_m = AtomicArrays.fourlevel_meanfield.sigma_matrices([results[idxs][312]], 1)[1]

    Nx = fixed_params_0["Nx"]
    Ny = fixed_params_0["Ny"]
    a = fixed_params_0["a"]

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

    # === Preallocate arrays to store field projections ===
    Re_field_x = zeros(nx, ny)
    Abs_field_x = zeros(nx, ny)
    Re_t_field_x = zeros(nx, ny)
    Abs_t_field_x = zeros(nx, ny)
    Re_field_y = zeros(nx, ny)
    Abs_field_y = zeros(nx, ny)
    Re_t_field_y = zeros(nx, ny)
    Abs_t_field_y = zeros(nx, ny)
    Re_field_z = zeros(nx, ny)
    Abs_field_z = zeros(nx, ny)
    Re_t_field_z = zeros(nx, ny)
    Abs_t_field_z = zeros(nx, ny)

    I_total = zeros(nx, ny)

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
            E = AtomicArrays.field.scattered_field(r, coll, sigmas_m)/sqrt(3)
            E_t = field_func(r, E_in) + E
            I_t = (norm(E_t)^2 / abs(E_in.amplitude)^2)
            
            # Save the real and absolute values for each Cartesian component.
            Re_field_x[i, j] = real(E[1])
            Abs_field_x[i, j] = abs(E[1])
            Re_field_y[i, j] = real(E[2])
            Abs_field_y[i, j] = abs(E[2])
            Re_field_z[i, j] = real(E[3])
            Abs_field_z[i, j] = abs(E[3])
            Re_t_field_x[i, j] = real(E_t[1])
            Abs_t_field_x[i, j] = abs(E_t[1])
            Re_t_field_y[i, j] = real(E_t[2])
            Abs_t_field_y[i, j] = abs(E_t[2])
            Re_t_field_z[i, j] = real(E_t[3])
            Abs_t_field_z[i, j] = abs(E_t[3])
            I_total[i, j] = I_t
        end
    end
end

# Plots
# fields
let
    CairoMakie.activate!()
    fig = Figure(size = (900, 1100))

    Re_x = Re_t_field_x
    Abs_x = Abs_t_field_x
    Re_y = Re_t_field_y
    Abs_y = Abs_t_field_y
    Re_z = Re_t_field_z
    Abs_z = Abs_t_field_z

    nlevels = 10

    # --- Row 1: x-component (Eₓ) ---
    ax1 = Axis(fig[1, 1], title = "Re(Eₓ)", xlabel = label1, ylabel = label2)
    hm1 = contourf!(ax1, coord1_range, coord2_range, Re_x, colormap = :plasma, levels=nlevels)
    scatter!(ax1, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[1, 2], hm1, width = 15, height = Relative(1))

    ax2 = Axis(fig[1, 3], title = "Abs(Eₓ)", xlabel = label1, ylabel = label2)
    hm2 = contourf!(ax2, coord1_range, coord2_range, Abs_x, colormap = :plasma, levels=nlevels)
    scatter!(ax2, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[1, 4], hm2, width = 15, height = Relative(1))

    # --- Row 2: y-component (Eᵧ) ---
    ax3 = Axis(fig[2, 1], title = "Re(Eᵧ)", xlabel = label1, ylabel = label2)
    hm3 = contourf!(ax3, coord1_range, coord2_range, Re_y, colormap = :plasma, levels=nlevels)
    scatter!(ax3, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[2, 2], hm3, width = 15, height = Relative(1))

    ax4 = Axis(fig[2, 3], title = "Abs(Eᵧ)", xlabel = label1, ylabel = label2)
    hm4 = contourf!(ax4, coord1_range, coord2_range, Abs_y, colormap = :plasma, levels=nlevels)
    scatter!(ax4, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[2, 4], hm4, width = 15, height = Relative(1))

    # --- Row 3: z-component (E_z) ---
    ax5 = Axis(fig[3, 1], title = "Re(E_z)", xlabel = label1, ylabel = label2)
    hm5 = contourf!(ax5, coord1_range, coord2_range, Re_z, colormap = :plasma, levels=nlevels)
    scatter!(ax5, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[3, 2], hm5, width = 15, height = Relative(1))

    ax6 = Axis(fig[3, 3], title = "Abs(E_z)", xlabel = label1, ylabel = label2)
    hm6 = contourf!(ax6, coord1_range, coord2_range, Abs_z, colormap = :plasma, levels=nlevels)
    scatter!(ax6, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[3, 4], hm6, width = 15, height = Relative(1))

    # save(PATH_FIGS*"4level_E_t_a"*string(a)*"_"*POLARIZATION*"_theta"*string(round(field.angle_k[1], digits=2))*"_N"*string(N)*".pdf", fig, px_per_unit=4)
    fig
end

let
    CairoMakie.activate!()
    fig = Figure(size = (800, 700))

    nlevels = 50

    # --- Row 1: x-component (Eₓ) ---
    ax1 = Axis(fig[1, 1], title = "Intensity, E₀ = "*string(round(E_in.amplitude, digits=3)), xlabel = label1, ylabel = label2,
               titlesize = 24, xlabelsize = 20, ylabelsize = 20,
               xticklabelsize = 18, yticklabelsize = 18)
    hm1 = contourf!(ax1, coord1_range, coord2_range, I_total,
                    colormap = :plasma, 
                    levels = range(0, 1.0*maximum(I_total), nlevels), 
                    # colorscale=log10,
                    )
    scatter!(ax1, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[1, 2], hm1, label="I/|E₀|²",
             labelsize = 20, ticklabelsize = 18,
             width = 15, height = Relative(1))

    # save(PATH_FIGS*"4level_I_t_a"*string(a)*"_"*POLARIZATION*"_theta"*string(round(field.angle_k[1], digits=2))*"_N"*string(N)*".pdf", fig, px_per_unit=4)
    fig
end


module NonlocalArrays

using LinearAlgebra
using QuantumOptics
using AtomicArrays
using CairoMakie
using DataFrames, CSV, Printf, Serialization

export save_result, save_sweep, load_sweep, load_all_results, parse_sweep_filename

export plot_atoms_with_field, build_fourlevel_system, plot_sweep_quantity, plot_sweep_multicurve, plot_sweep_heatmap

"""
    NonlocalArrays.comm(A, B)
Computes the commututator of two operators: A and B
"""
function comm(A::Operator, B::Operator)
    return A * B - B * A
end
function comm(A::Matrix, B::Matrix)
    return A * B - B * A
end

"""
    NonlocalArrays.anticomm(A, B)
Computes the anticommututator of two operators: A and B
"""
function anticomm(A::Operator, B::Operator)
    return A * B + B * A
end
function anticomm(A::Matrix, B::Matrix)
    return A * B + B * A
end

function gamma_detuned(gamma::Real, delt::Real; omega::Real=2*pi)
    return gamma*(1.0 + 3.0*delt/omega + 3.0*delt^2/omega^3 + delt^3/omega^3)
end

"""
    NonlocalArrays.correlation_3op_1t([tspan, ]rho0, H, J, A, B, C; <keyword arguments>)
Calculate one time correlation values ``⟨A(0)B(\\tau)C(0)⟩``.
The calculation is done by multiplying the initial density operator
with ``C \\rho A`` performing a time evolution according to a master equation
and then calculating the expectation value ``\\mathrm{Tr} \\{B ρ\\}``
Without the `tspan` argument the points in time are chosen automatically from
the ode solver and the final time is determined by the steady state termination
criterion specified in [`steadystate.master`](@ref).
# Arguments
* `tspan`: Points of time at which the correlation should be calculated.
* `rho0`: Initial density operator.
* `H`: Operator specifying the Hamiltonian.
* `J`: Vector of jump operators.
* `A`: Operator at time `t=0`.
* `B`: Operator at time `t=\\tau`.
* `C`: Operator at time `t=0`.
* `rates=ones(N)`: Vector or matrix specifying the coefficients (decay rates)
        for the jump operators.
* `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function correlation_3op_1t(tspan, rho0::Operator, H::AbstractOperator, J,
                     A, B, C;
                     rates=nothing,
                     Jdagger=dagger.(J),
                     kwargs...)
    function fout(t, rho)
        expect(B, rho)
    end
    t,u = timeevolution.master(tspan, C*rho0*A, H, J; rates=rates, Jdagger=Jdagger,
                        fout=fout, kwargs...)
    return u
end

"""
    NonlocalArrays.correlation_3op_1t(rho0, H, J, A, B, C; <keyword arguments>)
Calculate steady-state correlation values ``⟨A(0)B(0)C(0)⟩``.
The calculation is done by multiplying the initial density operator
with ``C \\rho A`` and then calculating the expectation value ``\\mathrm{Tr} \\{B C \\rho A\\}``
# Arguments
* `rho0`: Initial density operator.
* `A`: Operator at time `t=0`.
* `B`: Operator at time `t=\\tau`.
* `C`: Operator at time `t=0`.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function correlation_3op_1t(rho0::Operator, A, B, C; kwargs...)
    return expect(B, C*rho0*A)
end


"""
    NonlocalArrays.coherence_function_g2([tspan, ]rho0, H, J, A_op; <keyword arguments>)
Calculate one time correlation values

``g^{(2)}(\\tau) =
        \\frac{\\langle A^\\dagger(0)A^\\dagger(\\tau)A(\\tau)A(0)\\rangle}
        {\\langle A^\\dagger(\\tau)A(\\tau)\\rangle
         \\langle A^\\dagger(0)A(0)\\rangle}``.

# Arguments
* `tspan`: Points of time at which the correlation should be calculated.
* `H`: Operator specifying the Hamiltonian.
* `J`: Vector of jump operators.
* `A_op`: Operator at time `t=0`.
* `rates=ones(N)`: Vector or matrix specifying the coefficients (decay rates)
        for the jump operators.
* `rho0=nothing`: Initial density operator, if nothing `rho0 = rho_ss`.
* `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function coherence_function_g2(tspan, H::AbstractOperator, J, A_op;
                     rates=nothing,
                     rho0=nothing,
                     Jdagger=dagger.(J),
                     kwargs...)
    function fout(t, rho)
        expect(dagger(A_op) * A_op, rho)
    end
    if isnothing(rho0)
        rho0 = QuantumOptics.steadystate.eigenvector(H, J; rates=rates)
        n_ss = [expect(dagger(A_op) * A_op, rho0)]
    else
        _, n_ss = timeevolution.master(tspan, rho0, H, J; rates=rates, Jdagger=Jdagger,
                        fout=fout, kwargs...)
    end
    t,u = timeevolution.master(tspan, A_op*rho0*dagger(A_op), H, J; rates=rates, Jdagger=Jdagger,
                        fout=fout, kwargs...)
    return u ./ (n_ss[1] * n_ss)
end


"""
    NonlocalArrays.jump_op_source_mode(Γ, J)
Calculate the source-mode jump operators

``\\hat{J}_l = \\sqrt{\\lambda_l} \\mathbf{b}_l^T \\hat{\\mathbf{\\Sigma}}``.

# Arguments
* `Γ`: matrix of decay rates
* `J`: Vector of common jump operators.

# Return
* `J_s`: vector of the source-mode jump operators
"""
function jump_op_source_mode(Γ, J; tol=1e-10)
    N = length(J)
    λ_g, β_g = eigen(Γ)
    λ_g[abs.(λ_g) .< tol] .= 0.0
    J_s = [sqrt.(λ_g[l]) * β_g[:,l]' * J for l = 1:N]
    return J_s
end

"""
    NonlocalArrays.collective_ops_all(J::Vector)
Computes collective oprators given the individual jump operators of atoms, J.
# Arguments
* `J`: vector of jump operators

# Return
* [`J_z`, `J_z_sum`, `J_sum`, `J2`]
"""
function collective_ops_all(J::Vector)
    J_z = 0.5 * [comm(dagger(j), j) for j in J]
    J_z_sum = sum(J_z)
    J_sum = sum(J)
    J2 = 0.5 * (anticomm(dagger(J_sum), J_sum)) + J_z_sum^2
    return J_z, J_z_sum, J_sum, J2
end

"""
    NonlocalArrays.dicke_state(N::Int, j::Number, m::Number, J::Vector{Operator})

Dicke state in uncoupled basis.
# Arguments
* `j` = N / 2
* `m` ∈ [-j, j]
* `J` is vector of collapse operators

# Return
* `|j,m>`
"""
function dicke_state(N::Int, j::Number, m::Number, J::Vector)
    norm_coef = sqrt(factorial(Int(j+m)) / (factorial(N)*factorial(Int(j-m))))
    e_state = AtomicArrays.quantum.blochstate(0.0, 0.0, N)
    return norm_coef * (sum(J))^Int(j - m) * e_state
end

"""
    NonlocalArrays.spherical_basis_jm_4(J::Vector; γ::Real=1.0)
Calculates the basis vectors of the total momentum operator ``|j, m\rangle``
for a system of 4 two-level atoms.
    TODO: write a general function for arb N using Clebsch-Gordan coefficients

# Arguments
* `J::Vector` -- vector of individual jump operators of atoms

# Return
* `|j,m>` -- vector of vectors |2,m> (m = -2,...,2), |1,m> (m = -1,0,1), |0,0>, note that |1,m> are threefold degenerate, |0,0> is twofold degenerate
"""
function spherical_basis_jm_4(J::Vector; γ::Real = 1.0)
    N = length(J)
    Γ = γ * ones(N, N)
    _, J_s = diagonaljumps(Γ, J)
    G = Ket(basis(J[1]), [(i < 2^N) ? 0 : 1 for i in 1:2^N])
    ψ_1m1_2, ψ_1m1_1, ψ_1m1_3, ψ_2m1 = [dagger(j) * G for j in J_s]
    ψ_00_1 = 1/sqrt(2) * dagger(sum(J) - 2*J[end]) * ψ_1m1_1
    ψ_00_2 = 1/sqrt(2) * dagger(sum(J) - 2*J[end]) * ψ_1m1_2
    Jsd = dagger(sum(J))
    return [[G, ψ_2m1, Jsd * ψ_2m1/sqrt(6), Jsd^2 * ψ_2m1/6, Jsd^3 * ψ_2m1/12],
           [[ψ_1m1_1, ψ_1m1_2, ψ_1m1_3],
            [Jsd*ψ_1m1_1, Jsd*ψ_1m1_2, Jsd*ψ_1m1_3] ./ sqrt(2), 
            [Jsd^2*ψ_1m1_1, Jsd^2*ψ_1m1_2, Jsd^2*ψ_1m1_3] ./ 2], 
           [[ψ_00_1, ψ_00_2]]]
end

# TODO: change these functions accordingly for spherical_basis_jm_N function
function find_state_index(states::Vector{Vector}, j::Int, m::Int;
                          degeneracy::Union{Int, Nothing}=nothing)
    # Determine min and max j from the states input
    max_j = length(states) - 1  # max_j is determined by the length of states
    min_j = 0  # min_j is assumed to be 0 based on the structure
    
    # Create j_map dynamically based on max_j
    j_map = Dict(max_j => 1)
    for i in 1:max_j
        j_map[max_j - i] = i + 1
    end
    
    # Ensure j is within the valid range
    if j < min_j || j > max_j
        error("Invalid j value. Valid range is $min_j to $max_j.")
    end
    
    # Ensure m is within the valid range of -j to +j
    if m < -j || m > j
        error("Invalid m value. For j = $j, valid range is -$j to +$j.")
    end

    # Map j to the correct index in `states`
    j_idx = j_map[j]

    # Now map m to the correct sublist in the selected j vector
    m_idx = j + 1 + m  # for example, m = -j will be 1, m = -j+1 will be 2, and so on

    # Retrieve the specific |j,m> state(s)
    state_group = states[j_idx][m_idx]

    # Determine degeneracy based on the structure
    if isa(state_group, Vector)
        num_degenerate = length(state_group)
        if isnothing(degeneracy)
            return j_idx, m_idx  # Return the whole set of degenerate states
        elseif degeneracy > num_degenerate || degeneracy < 1
            error("Invalid degeneracy index. Available indices: 1 to $num_degenerate.")
        else
            return j_idx, m_idx, degeneracy
        end
    else
        # No degeneracy, return the single state
        return j_idx, m_idx
    end
end

function find_flattened_state_index(states::Vector{Vector}, j::Int, m::Int;
                                    degeneracy::Union{Int, Nothing}=nothing)
    # Determine min and max j from the states input
    max_j = length(states) - 1  # max_j is determined by the length of states
    min_j = 0  # min_j is assumed to be 0 based on the structure
    
    # Create j_map dynamically based on max_j
    j_map = Dict(max_j => 1)
    for i in 1:max_j
        j_map[max_j - i] = i + 1
    end
    
    # Ensure j is within the valid range
    if j < min_j || j > max_j
        error("Invalid j value. Valid range is $min_j to $max_j.")
    end
    
    # Ensure m is within the valid range of -j to +j
    if m < -j || m > j
        error("Invalid m value. For j = $j, valid range is -$j to +$j.")
    end
    
    # Map j to the correct index in `states`
    j_idx = j_map[j]

    # Now map m to the correct sublist in the selected j vector
    m_idx = j + 1 + m  # for example, m = -j will be 1, m = -j+1 will be 2, and so on

    # Retrieve the specific |j,m> state(s)
    state_group = states[j_idx][m_idx]

    # Flatten the states vector
    flattened_states = vcat(vcat(states...)...)

    # Determine degeneracy based on the structure and find the correct index
    if isa(state_group, Vector)
        num_degenerate = length(state_group)
        if isnothing(degeneracy)
            return findall(x -> x in state_group, flattened_states)  # Return all indices for the degenerate states
        elseif degeneracy > num_degenerate || degeneracy < 1
            error("Invalid degeneracy index. Available indices: 1 to $num_degenerate.")
        else
            specific_state = state_group[degeneracy]
            return findfirst(x -> x == specific_state, flattened_states)  # Return the index of the specific state
        end
    else
        # No degeneracy, return the index of the single state
        return findfirst(x -> x == state_group, flattened_states)
    end
end
# Function when degeneracy is provided (index_jm = (j, m, d))
function find_flattened_state_index(states::Vector{Vector}, index_jm::Tuple{Int, Int, Int})
    j, m, d = index_jm
    return find_flattened_state_index(states, j, m; degeneracy=d)
end
# Function when degeneracy is not provided (index_jm = (j, m))
function find_flattened_state_index(states::Vector{Vector}, index_jm::Tuple{Int, Int})
    j, m = index_jm
    return find_flattened_state_index(states, j, m)
end

# Function that takes a tuple (j, m, d) and returns "|j,m>_d"
function state_string(index_jm::Tuple{Int, Int, Int})
    j, m, d = index_jm
    return "|$j,$m>$d"
end
# Function that takes a tuple (j, m) and returns "|j,m>"
function state_string(index_jm::Tuple{Int, Int})
    j, m = index_jm
    return "|$j,$m>"
end


"""
    NonlocalArrays.D_angle(θ::Number, ϕ::Number, S::SpinCollection)
Calculate the source-mode jump operators

``D(\\theta, \\phi) = \\frac{3}{8 \\pi} \\left( 1 - (\\mu \\cdot \\hat{r}(\\theta, \\phi)^2) \\right)``.

# Arguments
* `θ`: angle between z axis and radius-vector
* `ϕ`: angle in xy plane starting from x axis
* `S`: spin collection

# Return
* `D_θϕ`: angular distribution
"""
function D_angle(θ::Number, ϕ::Number, S::SpinCollection)
    μ = S.polarizations[1]
    r_n = [sin.(θ)*cos.(ϕ), sin.(θ)*sin.(ϕ), cos(θ)]
    return 3.0 / (8.0*pi) * (1 - (μ' * r_n)^2)
end


"""
    NonlocalArrays.jump_op_direct_detection(r::Vector, dΩ::Number, S::SpinCollection, J)
Calculate the direct-detection jump operators

``\\hat{S}(\\theta, \\phi) = \\sqrt{\\gamma D(\\theta, \\phi) d\\Omega} \\sum_{j=1}^N e^{-i k_0 \\hat{r}(\\theta, \\phi) \\cdot \\mathbf{r}_j} \\hat{\\sigma}_j``

``D(\\theta, \\phi) = \\frac{3}{8\\pi} \\left( 1 - \\left[ \\hat{\\mu} \\cdot \\hat{r}(\\theta, \\phi) \\right]^2\\right)``

# Arguments
* `r`: radius-vector (depends on ``\\theta, \\phi``)
* `dΩ`: element of solid angle in direction ``r(\\theta, \\phi)``
* `S`: spin collection
* `k_0`: wave number, ``k_0 = \\omega_0 / c``, where ``\\omega_0`` is a transition frequency of a atom
* `J`: Vector of common jump operators.

TODO: take into account different atomic frequencies

# Return
* `S_op`: vector of the source-mode jump operators
"""
function jump_op_direct_detection(r::Vector, dΩ::Number, S::SpinCollection, k_0::Number, J)
    N = length(S.gammas)
    μ = S.polarizations[1]
    γ = S.gammas[1]
    r_n = r ./ norm(r)
    D_θϕ = 3.0 / (8.0*pi) * (1 - (μ' * r_n)[1]^2)
    S_op = sqrt.(γ * D_θϕ * dΩ) * sum([exp(-im*k_0*(r_n' * S.spins[j].position)[1])*J[j]
                                       for j = 1:N])
    return S_op
end


function jump_op_direct_detection(θ::Real, ϕ::Real, dΩ::Number, S::SpinCollection, k_0::Number, J)
    N = length(S.gammas)
    μ = S.polarizations[1]
    γ = S.gammas[1]
    r_n = [sin.(θ)*cos.(ϕ), sin.(θ)*sin.(ϕ), cos(θ)]
    D_θϕ = 3.0 / (8.0*pi) * (1 - (μ' * r_n)[1]^2)
    S_op = sqrt.(γ * D_θϕ * dΩ) * sum([exp(-im*k_0*(r_n' * S.spins[j].position)[1])*J[j]
                                       for j = 1:N])
    return S_op
end


function compute_w_tau(jump_t)
    n = length(jump_t)
    w_tau = Vector{Float64}(undef, n-1)
    @inbounds for j in 1:(n-1)
        w_tau[j] = jump_t[j+1] - jump_t[j]
    end
    return filter!(x -> x >= 0, w_tau)
end

function compute_w_tau_n(w_tau_n, idx_no_stat, jump_t, jump_i, i)
    jumps = jump_t[jump_i .== i]
    jumps_dist = diff(jumps)
    jumps_dist = jumps_dist[jumps_dist .>= 0]
    if isempty(jumps_dist)
        push!(idx_no_stat, i)
        print(i, " ")
    end
    push!(w_tau_n, jumps_dist)
end

function g2_0_jump_opers(rho_ss::Operator, J_s)
    N = length(J_s)
    num = real(sum([NonlocalArrays.correlation_3op_1t(rho_ss, 
    dagger(J_s[i]), dagger(J_s[j])*J_s[j], J_s[i]) for i = 1:N, j = 1:N]))
    denom = real(sum([QuantumOptics.expect(dagger(J_s[i])*J_s[i], rho_ss) for i = 1:N]).^2)
    return num / denom
end

function g2_tau_jump_opers(rho_ss::Operator, J_s, H, tspan)
    N = length(J_s)
    num = real(sum([NonlocalArrays.correlation_3op_1t(tspan, rho_ss, H, J_s, 
    dagger(J_s[i]), dagger(J_s[j])*J_s[j], J_s[i]) for i = 1:N, j = 1:N]))
    denom = real(sum([QuantumOptics.expect(dagger(J_s[i])*J_s[i], rho_ss) for i = 1:N]).^2)
    return num ./ denom
end

# Function to retrieve parameters from the CSV file based on specified fields
function get_parameters_csv(csv_file, state, N, geometry, detuning_symmetry, direction)
    # Read the CSV file into a DataFrame
    df = CSV.read(csv_file, DataFrame)

    # Filter the DataFrame based on the specified fields
    filtered_df = filter(row -> row.State_proj_max == state && row.N == N && row.geometry == geometry &&
                                row.detuning_symmetry == detuning_symmetry && row.Direction == direction, df)

    # Check if any rows match the criteria
    if nrow(filtered_df) == 0
        println("No matching parameters found.")
        return nothing
    end

    # Extract the desired parameters
    a = filtered_df.a[1]
    E₀ = filtered_df.E₀[1]
    Δ_params = zeros(Float64, N)
    for i in 1:N
        Δ_params[i] = filtered_df[!, Symbol("Δ_$i")][1]
    end

    return Dict("a" => a, "E_0" => E₀, "Δ_vec" => Δ_params)
end

# Saving serialization

"""
    NonlocalArrays.save_result(path, result, N, B_z, pol, amplitude, a)

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
    NonlocalArrays.load_all_results(path)

Deserialize all .bson result files in the directory.
"""
function load_all_results(path::String)
    files = filter(f -> endswith(f, ".bson"), readdir(path; join=true))
    return [deserialize(file) for file in sort(files)]
end

"""
    NonlocalArrays.parse_result_filename(filename::String)

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
    NonlocalArrays.save_sweep(path::String, data;
               description::String = "results", kwargs...)

Save any `data` object with a filename auto-generated from key-value pairs in `kwargs`,
plus a short description. Supports Float, Tuple, Vector, and String keyword arguments.

Examples:
save_sweep(PATH_DATA, data; description="states", Bz=(0.1, 0.2), N=100, POL="R", a=0.2)

Creates a file like:
    states_Bz_0.1_to_0.2_N_100_POL_R_a_0.2.bson

Or parameters can be passed as a single dict:
sweep_params = Dict(
    "a"            => (0.2, 0.25),
    "Bz"          => collect(range(-0.2, 0.2, length=3)),
    "deltas"       => (0.2, 0.3),
    "POLARIZATION" => ("R", "L"),
    "amplitude"    => (0.02, 0.03),
    "anglek"      => [[π/6, 0.0], [π/4, 0.1]]
)
savepath = save_sweep_0(PATH_DATA, data;
                      description = "steady-states-sweep",
                      sweep_params = sweep_params)
"""
function save_sweep(path::String, data;
                    description::String = "results", kwargs...)
    mkpath(path)
    parts = [description]

    # 1) Process sweep_params
    if haskey(kwargs, :sweep_params)
        for (key, val) in pairs(kwargs[:sweep_params])
            k = string(key)

            # --- Tuple of two numeric vectors (e.g. ([π/6,0],[π/4,0.1])) ---
            if isa(val, Tuple) && length(val) == 2 &&
               isa(val[1], AbstractVector{<:Number}) &&
               isa(val[2], AbstractVector{<:Number})
                minv, maxv = val[1], val[2]
                s1 = "[" * join([ @sprintf("%.3f", x) for x in minv ], ",") * "]"
                s2 = "[" * join([ @sprintf("%.3f", x) for x in maxv ], ",") * "]"
                push!(parts, "$(k)_$(s1)_to_$(s2)")
                push!(parts, "n$(k)_2")

            # --- Vector of vectors of numbers (e.g. [[π/6,0],[π/4,0.1],...]) ---
            elseif isa(val, AbstractVector) &&
                   all(x -> isa(x, AbstractVector{<:Number}), val)
                minv, maxv = val[1], val[end]
                s1 = "[" * join([ @sprintf("%.3f", x) for x in minv ], ",") * "]"
                s2 = "[" * join([ @sprintf("%.3f", x) for x in maxv ], ",") * "]"
                push!(parts, "$(k)_$(s1)_to_$(s2)")
                push!(parts, "n$(k)_$(length(val))")

            # — tuple of strings or chars —
            elseif isa(val, Tuple) && all(x->isa(x,AbstractString)||isa(x,Char), val)
                strs = string.(val)
                sweep = join(strs, "_to_")
                push!(parts, "$(k)_$(sweep)", "n$(k)_$(length(val))")

            # — vector of strings or chars —
            elseif isa(val, AbstractVector) && all(x->isa(x,AbstractString)||isa(x,Char), val)
                strs = string.(val)
                if length(val) > 1
                    sweep = join(strs, "_to_")
                    push!(parts, "$(k)_$(sweep)", "n$(k)_$(length(val))")
                else
                    push!(parts, "$(k)_$(strs[1])")
                end

            # - integer scalar -
            elseif isa(val, Int)
                push!(parts, "$(k)_$(val)")

            # — float scalar —
            elseif isa(val, AbstractFloat)
                push!(parts, @sprintf("%s_%.3f", k, val))

            # — string scalar —
            elseif isa(val, AbstractString) || isa(val, Char)
                push!(parts, "$(k)_$(val)")

            # — numeric vector of Numbers —
            elseif isa(val, AbstractVector) && all(x->isa(x,Number), val)
                if length(val) > 2
                    push!(parts, @sprintf("%s_%.3f_to_%.3f", k, minimum(val), maximum(val)))
                    push!(parts, "n$(k)_$(length(val))")
                elseif length(val) == 2
                    s = "[" * join([ @sprintf("%.3f", x) for x in val ], ",") * "]"
                    push!(parts, "$(k)_$s")
                else
                    x = val[1]
                    if isa(x, Int)
                        push!(parts, "$(k)_$(x)")
                    else
                        push!(parts, @sprintf("%s_%.3f", k, x))
                    end
                end

            # — range of numbers —
            elseif isa(val, AbstractRange)
                arr = collect(val)
                push!(parts, @sprintf("%s_%.3f_to_%.3f", k, minimum(arr), maximum(arr)))
                push!(parts, "n$(k)_$(length(arr))")

            # — tuple of two numbers —
            elseif isa(val, Tuple) && length(val)==2 && all(x->isa(x,Number), val)
                push!(parts, @sprintf("%s_%.3f_to_%.3f", k, val[1], val[2]))
                push!(parts, "n$(k)_2")

            else
                error("Unsupported sweep_params[$k] = $val of type $(typeof(val))")
            end
        end
    end

    # 2) Process other kwargs
    for (key, val) in pairs(kwargs)
        if key === :sweep_params; continue; end
        k = string(key)
        strval = if isa(val, Int)
            "$(k)_$(val)"
        elseif isa(val, AbstractFloat)
            @sprintf("%s_%.3f", k, val)
        elseif isa(val, AbstractString)
            "$(k)_$(val)"
        elseif isa(val, AbstractVector) && all(x->isa(x,Number), val)
            if length(val) > 2
                @sprintf("%s_%.3f_to_%.3f", k, minimum(val), maximum(val))
            elseif length(val)==2
                s = "[" * join([ @sprintf("%.3f", x) for x in val ], ",") * "]"
                "$(k)_$s"
            else
                x = val[1]
                isa(x,Int) ? "$(k)_$(x)" : @sprintf("%s_%.3f", k, x)
            end
        elseif isa(val, Tuple) && length(val)==2 && all(x->isa(x,Number), val)
            @sprintf("%s_%.3f_to_%.3f", k, val[1], val[2])
        else
            error("Unsupported kwarg $k => $val of type $(typeof(val))")
        end
        push!(parts, strval)
    end

    # 3) Save
    filename = join(parts, "_") * ".bson"
    filepath = joinpath(path, filename)
    Serialization.serialize(filepath, data)
    return filepath
end

"""
    load_sweep(filepath::AbstractString) -> (data, params)

Reads the .bson file at `filepath`, deserializes the contents back into
`data`, and parses `filepath`’s filename to recover its sweep parameters.
"""
function load_sweep(filepath::AbstractString)
    data   = Serialization.deserialize(filepath)
    params = parse_sweep_filename(filepath)
    return data, params
end

"""
    NonlocalArrays.parse_sweep_filename(filename::String) -> Dict{String, Any}

Parses filenames like:
"steady_states_Bz_-0.200_to_0.200_num_vals_36_N_100_POL_R_E0_0.020_a_0.200_geometry_rect.bson"
into a dictionary of parameter names and values.
"""
function parse_sweep_filename(filename::String)::Dict{String, Any}
    # Remove directory components and file extension.
    base = splitext(basename(filename))[1]
    # Split the base name on underscores.
    tokens = split(base, '_')
    
    # By convention, discard the first token (the file prefix).
    if length(tokens) > 1
        tokens = tokens[2:end]
    end

    params = Dict{String, Any}()
    i = 1
    while i <= length(tokens)
        # If there is no value token, break.
        if i == length(tokens)
            break
        end

        key = tokens[i]
        # Check if a range is indicated by the literal token "to"
        if i + 2 <= length(tokens) && tokens[i+2] == "to" && i + 3 <= length(tokens)
            part1 = tokens[i+1]
            part2 = tokens[i+3]
            # If both parts are enclosed in brackets, parse them as vectors.
            if startswith(strip(part1), "[") && endswith(strip(part1), "]") &&
               startswith(strip(part2), "[") && endswith(strip(part2), "]")
                params[key] = (parse_vector(part1), parse_vector(part2))
            else
                # Otherwise, try to parse both parts as numbers.
                n1 = tryparse(Float64, part1)
                n2 = tryparse(Float64, part2)
                if n1 !== nothing && n2 !== nothing
                    params[key] = (n1, n2)
                else
                    # Otherwise, treat them as strings.
                    params[key] = (part1, part2)
                end
            end
            i += 4
        else
            # Single value: attempt to parse as Int first, then Float; if that fails, leave as String.
            val_token = tokens[i+1]
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

"""
    NonlocalArrays.build_fourlevel_system(; a=0.2, Nx=4, Ny=4,
                                          delta_val=0.2,
                                          POL="R",
                                          amplitude=0.02,
                                          angle_k=[π/6, 0.0])

Builds a four-level atomic system consisting of:

- A rectangular atomic array
- Polarizations and decay rates
- An incident electric field
- Rabi frequencies calculated from the field

# Keyword Arguments

- `a`: Lattice spacing (default 0.2)
- `Nx`, `Ny`: Number of atoms along x and y axes
- `delta_val`: Detuning value for all atoms except the first
- `gamma`: Spontaneous decay rate of all three transitions
- `POL`: "R" or "L" circular polarization
- `amplitude`: Electric field amplitude
- `field_func`: Shape of the incident beam
- `angle_k`: Incident field direction `[θ, φ]`

# Returns

- `coll`: `FourLevelAtomCollection`
- `field`: `EMField` structure
- `field_func`: function used to evaluate the field at a point
- `OmR`: Rabi frequencies at each atom position
"""
function build_fourlevel_system(; a=0.2, Nx=4, Ny=4,
                                delta_val=0.0,
                                gamma=0.25,
                                POL="R",
                                amplitude=0.02,
                                field_func=AtomicArrays.field.gauss,
                                angle_k=[0.0, 0.0])
    # TODO: adjust for more generosity
    # Compute positions
    positions = AtomicArrays.geometry.rectangle(a, a; Nx=Nx, Ny=Ny,
                    position_0 = [(-Nx/2 + 0.5)*a, (-Ny/2 + 0.5)*a, 0.0])
    N = length(positions)

    # Build polarization and decay profile
    pols = AtomicArrays.polarizations_spherical(N)
    gam  = [AtomicArrays.gammas(gamma)[m] for m in 1:3, j in 1:N]
    deltas = [(i == 1) ? delta_val : delta_val for i in 1:N]

    # Build atomic collection
    coll = AtomicArrays.FourLevelAtomCollection(positions;
                    deltas=deltas, polarizations=pols, gammas=gam)

    # Create field
    k_mod = 2π
    polarisation = POL == "R" ? [1.0, -1.0im, 0.0] : [1.0, 1.0im, 0.0]
    waist_radius = 0.3 * a * sqrt(Nx * Ny)
    field = AtomicArrays.field.EMField(amplitude, k_mod, angle_k, polarisation;
                    position_0=[0.0, 0.0, 0.0], waist_radius)

    # Compute Rabi frequency at atom positions
    OmR = AtomicArrays.field.rabi(field, field_func, coll)

    return coll, field, field_func, OmR
end
"""
    build_fourlevel_system(params::Dict)

Overload that takes a dictionary of parameters and forwards them as keyword arguments.
Supports both POL and POLARIZATION, and angle_k or anglek.
"""
function build_fourlevel_system(params::Dict{String,Any})
    # Resolve aliases
    POL = get(params, "POL", get(params, "POLARIZATION", "R"))
    angle_k = get(params, "angle_k", get(params, "anglek", [0.0, 0.0]))

    # Forward values using the original function
    return build_fourlevel_system(; 
        a = get(params, "a", 0.2),
        Nx = get(params, "Nx", 4),
        Ny = get(params, "Ny", 4),
        delta_val = get(params, "delta_val", get(params, "deltas", 0.0)),
        gamma = get(params, "gamma", 0.25),
        POL = POL,
        amplitude = get(params, "amplitude", 0.02),
        field_func = get(params, "field_func", AtomicArrays.field.gauss),
        angle_k = angle_k,
    )
end

# ------- Helper functions -------

# Helper function: parse a vector string like "[0.524,0.000]" into a vector of Float64.
function parse_vector(s::AbstractString)
    # Remove any leading/trailing whitespace and square brackets.
    s = strip(s, ['[', ']'])
    # Split on comma.
    parts = split(s, ',')
    # Parse each part as Float64, rounding via @sprintf can be done later if needed.
    return [parse(Float64, strip(p)) for p in parts]
end

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
    plot_sweep_quantity(results, quantity_func, xparam; fixed_params=Dict(), label_func=identity, sort_x=true)

Plot a single quantity versus one swept parameter.
"""
function plot_sweep_quantity(results::Dict{Dict{String,Any}, Any},
                             quantity_func::Function,
                             xparam::String;
                             fixed_params::Dict{String,Any}=Dict(),
                             label_func::Function=identity,
                             sort_x::Bool=true,
                             ylabel="Quantity")
    xs, ys = Float64[], Float64[]
    for (params, result) in results
        match = all(get(params, k, nothing) == v for (k, v) in fixed_params)
        if match && haskey(params, xparam)
            push!(xs, params[xparam])
            push!(ys, quantity_func(result, params))
        end
    end
    if sort_x
        ord = sortperm(xs)
        xs, ys = xs[ord], ys[ord]
    end
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = xparam, ylabel = ylabel, title = "Sweep: $xparam")
    lines!(ax, xs, ys, label = label_func(fixed_params))
    # axislegend(ax)
    return fig
end

"""
    plot_sweep_multicurve(results, quantity_func, xparam, curve_param; fixed_params=Dict())

Plot multiple curves with varying `curve_param`, each curve showing quantity vs `xparam`.
"""
function plot_sweep_multicurve(results::Dict{Dict{String,Any}, Any},
                                quantity_func::Function,
                                xparam::String,
                                curve_param::String;
                                fixed_params::Dict{String,Any}=Dict())
    grouped = Dict{Any, Vector{Tuple{Float64, Float64}}}()
    for (params, result) in results
        match = all(get(params, k, nothing) == v for (k, v) in fixed_params)
        if match && haskey(params, xparam) && haskey(params, curve_param)
            xval = params[xparam]
            curveval = params[curve_param]
            yval = quantity_func(result, params)
            push!(get!(grouped, curveval, []), (xval, yval))
        end
    end
    unzip(pairs::Vector{<:Tuple}) = map(x -> getindex.(pairs, x), 1:2)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = xparam, ylabel = "Quantity", title = "$xparam vs $curve_param")
    for (curveval, data) in grouped
        xs, ys = unzip(data)
        ord = sortperm(xs)
        lines!(ax, xs[ord], ys[ord], label = "$curve_param = $curveval")
    end
    axislegend(ax)
    return fig
end

"""
    plot_sweep_heatmap(results, quantity_func, xparam, yparam; fixed_params=Dict())

Generate a heatmap of quantity vs two parameters.
"""
function plot_sweep_heatmap(results::Dict{Dict{String,Any}, Any},
                             quantity_func::Function,
                             xparam::String,
                             yparam::String;
                             fixed_params::Dict{String,Any}=Dict())
    vals = Dict{Tuple{Float64, Float64}, Float64}()
    xvals, yvals = Float64[], Float64[]
    for (params, result) in results
        match = all(get(params, k, nothing) == v for (k, v) in fixed_params)
        if match && haskey(params, xparam) && haskey(params, yparam)
            x = Float64(params[xparam])
            y = Float64(params[yparam])
            push!(xvals, x)
            push!(yvals, y)
            vals[(x, y)] = quantity_func(result, params)
        end
    end
    xgrid = sort(unique(xvals))
    ygrid = sort(unique(yvals))
    Z = [get(vals, (x, y), NaN) for y in ygrid, x in xgrid]

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = xparam, ylabel = yparam, title = "Heatmap: $xparam vs $yparam")
    heatmap!(ax, xgrid, ygrid, Z)
    Colorbar(fig[1, 2], ax)
    return fig
end


end # module NonlocalArrays

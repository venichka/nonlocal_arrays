module NonlocalArrays

using LinearAlgebra
using QuantumOptics
using AtomicArrays
using DataFrames, CSV

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


end # module NonlocalArrays

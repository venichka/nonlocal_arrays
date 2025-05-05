module Correlations

using LinearAlgebra, QuantumOptics, AtomicArrays
using ..Algebra
export correlation_3op_1t, coherence_function_g2,
       jump_op_source_mode, collective_ops_all,
       dicke_state, spherical_basis_jm_4,
       find_state_index, find_flattened_state_index,
       state_string, g2_0_jump_opers, g2_tau_jump_opers

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

end
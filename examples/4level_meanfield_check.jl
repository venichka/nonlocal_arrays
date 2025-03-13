begin
    if pwd()[end-14:end] == "nonlocal_arrays"
        PATH_ENV = "."
    else
        PATH_ENV = "../"
    end
    using Pkg
    Pkg.activate(PATH_ENV)
end

begin
    using Revise
    using LinearAlgebra
    using QuantumOptics
    using DifferentialEquations
    using CairoMakie
    using AtomicArrays

    using NonlocalArrays
end

function average_values(ops, rho_t)
    T = length(rho_t)  # number of time points

    # Distinguish whether `ops` is a single operator or an array.
    if isa(ops, Operator)
        # Single operator => result is a 1D vector of length T
        av = Vector{Float64}(undef, T)
        for t in 1:T
            av[t] = real(trace(ops * rho_t[t]))
        end
        return av
    else
        s = size(ops)  # shape of the ops array
        outshape = (T, s...)
        av = Array{ComplexF64}(undef, outshape)
        for idx in CartesianIndices(s)
            op_ij = ops[idx]  # this is one operator
            for t in 1:T
                av[t, Tuple(idx)...] = tr(op_ij * rho_t[t])
            end
        end

        return av
    end
end

function population_ops(A::AtomicArrays.FourLevelAtomCollection)
    P_m1 = SparseOpType[embed(AtomicArrays.fourlevel_quantum.basis(A), j,
                     AtomicArrays.fourlevel_quantum.sigmas_ee_[1]) 
                    for j=1:length(A.atoms)]
    P_0 = SparseOpType[embed(AtomicArrays.fourlevel_quantum.basis(A), j,
                     AtomicArrays.fourlevel_quantum.sigmas_ee_[2]) 
                    for j=1:length(A.atoms)]
    P_p1 = SparseOpType[embed(AtomicArrays.fourlevel_quantum.basis(A), j,
                     AtomicArrays.fourlevel_quantum.sigmas_ee_[3]) 
                    for j=1:length(A.atoms)]
    return P_m1, P_0, P_p1
end

# u = [σ₋₁¹, σ₀¹, σ₊₁¹, σ₋₁²,..., σ₋₁¹⁺,...,]
function meanfield_spherical!(du, u, p, t)
    N = Int(length(u) / 15)
    w, OmR, Omega, Gamma = p
    sm = reshape(view(u, 1:3*N), (3, N))
    sp = reshape(view(u, 3*N+1:2*(3*N)), (3, N))
    smm = reshape(view(u, 2*(3*N)+1:15*N), (3, 3, N))
    dsm = reshape(view(du, 1:3*N), (3, N))
    dsp = reshape(view(du, 3*N+1:2*(3*N)), (3, N))
    dsmm = reshape(view(du, 2*(3*N)+1:15*N), (3, 3, N))
    # sigma equations
    @inbounds for n=1:N
        for m=1:3
            dsm[m,n] = (-1im*w[m,n]*sm[m,n])
            dsp[m,n] = (+1im*w[m,n]*sp[m,n])
            for m1 = 1:3
                if m1 == m
                    s_bar = smm[m1,m,n] - 1 + sum([smm[i,i,n] for i=1:3])
                    s_bar_p = smm[m,m1,n] - 1 + sum([smm[i,i,n] for i=1:3])
                else
                    s_bar = smm[m1,m,n]
                    s_bar_p = smm[m,m1,n]
                end
                dsm[m,n] += -1im*conj(OmR[m1,n])*s_bar
                dsp[m,n] += +1im*OmR[m1,n]*s_bar_p
                for m2 = 1:3
                    if m1 == m
                        dsm[m,n] += - 0.5*Gamma[n,n,m,m2]*sm[m2,n]
                        dsp[m,n] += - 0.5*Gamma[n,n,m,m2]*sp[m2,n]
                    else
                        continue
                        # this case gives 0 because of orthogonality of states
                        # dsm[m,n] += - 0.5*Gamma[n,n,m1,m2]*sm[m,n]*smm[m1,m2,n]
                        # dsp[m,n] += - 0.5*Gamma[n,n,m1,m2]*sp[m,n]*smm[m2,m1,n]  # TODO: check if gamma is symmetric: m1-m2 == m2-m1
                    end
                    for n2 = 1:N
                        if n2 == n
                            continue
                        end
                        dsm[m,n] += (1im*Omega[n,n2,m1,m2]+
                                     0.5*Gamma[n,n2,m1,m2])*s_bar*sm[m2,n2] 
                        dsp[m,n] += (- 1im*Omega[n,n2,m1,m2]+
                                     0.5*Gamma[n,n2,m1,m2])*s_bar_p*sp[m2,n2] 
                    end
                end
            end
        end
    end
    # population equations
    @inbounds for n = 1:N
        for m = 1:3
            for m1 = 1:3
                dsmm[m1,m,n] = (1im*(w[m1,n] - w[m,n])*smm[m1,m,n] -
                                1im*OmR[m1,n]*sm[m,n] +
                                1im*conj(OmR[m,n])*sp[m1,n])
                for m2 = 1:3
                    dsmm[m1,m,n] += (- 0.5*Gamma[n,n,m2,m1]*smm[m2,m,n] -
                                     0.5*Gamma[n,n,m,m2]*smm[m1,m2,n])
                    for n1 = 1:N
                        if n1 == n
                            continue
                        end
                        dsmm[m1,m,n] += ((1im*Omega[n1,n,m2,m1] +
                                          0.5*Gamma[n1,n,m2,m1])*
                                          sp[m2,n1]*sm[m,n] +
                                         (-1im*Omega[n,n1,m,m2] +
                                          0.5*Gamma[n,n1,m,m2] )*
                                          sp[m1,n]*sm[m2,n1])
                    end
                end
            end
        end
    end
end

# BUILDING THE SYSTEM

# Build the collection
positions = [
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    # [20.4, 0.0, 0.0],
    # [30.6, 0.0, 0.0]
]
N = length(positions)

pols = AtomicArrays.polarizations_spherical(N)
gam = [AtomicArrays.gammas(0.25)[m] for m=1:3, j=1:N]
# deltas = [0.0 for i = 1:N]
# deltas = [0.0, 0.0, 0.0, 0.0]
deltas = [0.0, 0.01]

coll = AtomicArrays.FourLevelAtomCollection(positions;
    deltas = deltas,
    polarizations = pols,
    gammas = gam
)

# Define a plane wave field in +y direction:
amplitude = 0.4
k_mod = 2π
angle_k = [0.0, π/2]  # => +y direction
polarisation = [1.0, 1.0im, 0.0]
pos_0 = [0.0, 0.0, 0.0]

field = AtomicArrays.field.EMField(amplitude, k_mod, angle_k, polarisation; position_0=pos_0)
external_drive = AtomicArrays.field.rabi(field, AtomicArrays.field.plane, coll)

B_f = 0.2
w = [deltas[n]+B_f*m for m = -1:1, n = 1:N]

# Build the Hamiltonian and jump operators
H = AtomicArrays.fourlevel_quantum.Hamiltonian(coll; magnetic_field=B_f,
                external_drive=external_drive,
                dipole_dipole=true)

Γ_fl, J_ops = AtomicArrays.fourlevel_quantum.JumpOperators(coll; flatten=true)
Γ = AtomicArrays.interaction.GammaTensor_4level(coll)
Ω = AtomicArrays.interaction.OmegaTensor_4level(coll)

# Master equation time evolution
begin
    b = AtomicArrays.fourlevel_quantum.basis(coll)
    # initial state => all ground
    ψ0 = basisstate(b, [AtomicArrays.fourlevel_quantum.idx_g for i = 1:N])
    ρ0 = dm(ψ0)
    tspan = [0.0:0.1:200.0;]
    t, rho_t = timeevolution.master_h(tspan, ψ0, H, J_ops; rates=Γ_fl)
end

begin
    av_J = average_values(J_ops, rho_t)
    P_m1, P_0, P_p1 = population_ops(coll)

    # We'll define arrays to store population vs time: shape = (length(t), N)
    pop_e_minus = zeros(length(t), N)
    pop_e_0     = zeros(length(t), N)
    pop_e_plus  = zeros(length(t), N)

    for n in 1:N
        for (k, ρ) in enumerate(rho_t)
            pop_e_minus[k, n] = real(tr(P_m1[n] * ρ))
            pop_e_0[k, n]     = real(tr(P_0[n] * ρ))
            pop_e_plus[k, n]  = real(tr(P_p1[n] * ρ))
        end
    end
end

# Meanfield time dynamics
begin
    u0 = [0.0im for i = 1:15*N]
    tspan = (0.0, 200.0)
    # tspan = [0.0:0.1:200.0;]
    p = [w, external_drive, Ω, Γ]
    prob = ODEProblem(meanfield_spherical!, u0, tspan, p)
    sol = solve(prob)
end

function plot_populations(pop_e_minus, pop_e_0, pop_e_plus, t)
    fig = Figure(size=(1000, 300))
    ax1 = Axis(fig[1, 1], title="m = -1", xlabel="t", ylabel="Population")
    ax2 = Axis(fig[1, 2], title="m = 0", xlabel="t", ylabel="Population")
    ax3 = Axis(fig[1, 3], title="m = +1", xlabel="t", ylabel="Population")
    # Define colors for different atoms
    colors = Makie.wong_colors(N)
    # Plot sublevel m = -1
    for n in 1:N
        lines!(ax1, t, pop_e_minus[:, n], label="Atom $n", color=colors[n], linewidth=2)
    end
    # Plot sublevel m = 0
    for n in 1:N
        lines!(ax2, t, pop_e_0[:, n], label="Atom $n", color=colors[n], linewidth=2)
    end
    # Plot sublevel m = +1
    for n in 1:N
        lines!(ax3, t, pop_e_plus[:, n], label="Atom $n", color=colors[n], linewidth=2)
    end
    # Add legend
    Legend(fig[1, 4], ax1, "Atoms", framevisible=false)
    fig
end

begin
    p_e_minus_mf = [real(reshape(view(sol(i),2*(3*N)+1:15*N), (3, 3, N))[1,1,j])
                    for i in t, j = 1:N]
    p_e_0_mf = [real(reshape(view(sol(i),2*(3*N)+1:15*N), (3, 3, N))[2,2,j])
                    for i in t, j = 1:N]
    p_e_plus_mf = [real(reshape(view(sol(i),2*(3*N)+1:15*N), (3, 3, N))[3,3,j])
                    for i in t, j = 1:N]
end

let 
    f = Figure(size=(1000, 300))
    ax1 = Axis(f[1, 1], title="m = -1", xlabel="t", ylabel="Population")
    ax2 = Axis(f[1, 2], title="m = 0", xlabel="t", ylabel="Population")
    ax3 = Axis(f[1, 3], title="m = +1", xlabel="t", ylabel="Population")
    # Define colors for different atoms
    colors = Makie.wong_colors(N)
    colors_mf = cgrad(:viridis, N, categorical=false)
    # Plot sublevels
    for n in 1:N
        lines!(ax1, t, pop_e_minus[:,n], label="Atom $n, q", color=colors[n], linewidth=2)
        lines!(ax1, t, p_e_minus_mf[:, n], label="Atom $n, mf", color=colors_mf[n], linewidth=2)
        lines!(ax2, t, pop_e_0[:,n], label="Atom $n", color=colors[n], linewidth=2)
        lines!(ax2, t, p_e_0_mf[:, n], label="Atom $n, mf", color=colors_mf[n], linewidth=2)
        lines!(ax3, t, pop_e_plus[:,n], label="Atom $n", color=colors[n], linewidth=2)
        lines!(ax3, t, p_e_plus_mf[:, n], label="Atom $n, mf", color=colors_mf[n], linewidth=2)
    end
    # Add legend
    Legend(f[1, 4], ax1, "Atoms", framevisible=false)
    f
end


plot_populations(pop_e_minus, pop_e_0, pop_e_plus, t)
plot_populations(p_e_minus_mf, p_e_0_mf, p_e_plus_mf, t)

let 
    f = Figure(size=(1000, 300))
    ax1 = Axis(f[1, 1], title="m = -1", xlabel="t", ylabel="Difference")
    ax2 = Axis(f[1, 2], title="m = 0", xlabel="t", ylabel="Difference")
    ax3 = Axis(f[1, 3], title="m = +1", xlabel="t", ylabel="Difference")
    # Define colors for different atoms
    colors = Makie.wong_colors(N)
    # Plot sublevels
    for n in 1:N
        lines!(ax1, t, abs.(pop_e_minus[:,n] .- p_e_minus_mf[:, n]), label="Atom $n", color=colors[n], linewidth=2)
        lines!(ax2, t, abs.(pop_e_0[:,n] .- p_e_0_mf[:, n]), label="Atom $n", color=colors[n], linewidth=2)
        lines!(ax3, t, abs.(pop_e_plus[:,n] .- p_e_plus_mf[:, n]), label="Atom $n", color=colors[n], linewidth=2)
    end
    # Add legend
    Legend(f[1, 4], ax1, "Atoms", framevisible=false)
    f
end
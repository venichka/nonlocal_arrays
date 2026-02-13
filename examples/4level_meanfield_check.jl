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

function rotate_xy_to_xz(positions::Vector{Vector{Float64}})
    rotation_matrix = [1.0  0.0  0.0;
                       0.0  0.0 -1.0;
                       0.0  1.0  0.0]

    return [rotation_matrix * pos for pos in positions]
end

# Rodrigues' rotation formula
function rotate_positions(positions::Vector{Vector{Float64}}, axis::Vector{Float64}, angle_rad::Float64)
    # Normalize rotation axis
    axis = axis / norm(axis)

    # Skew-symmetric matrix K
    K = [
        0.0        -axis[3]   axis[2];
        axis[3]     0.0      -axis[1];
       -axis[2]     axis[1]   0.0
    ]

    # Rodrigues rotation matrix
    rotation_matrix = I + sin(angle_rad)*K + (1 - cos(angle_rad))*(K*K)

    # Apply rotation
    return [rotation_matrix * pos for pos in positions]
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
    # P_m1 = SparseOpType[embed(AtomicArrays.fourlevel_quantum.basis(A), j,
    #                  AtomicArrays.fourlevel_quantum.sigmas_ee_[1]) 
    #                 for j=1:length(A.atoms)]
    # P_0 = SparseOpType[embed(AtomicArrays.fourlevel_quantum.basis(A), j,
    #                  AtomicArrays.fourlevel_quantum.sigmas_ee_[2]) 
    #                 for j=1:length(A.atoms)]
    # P_p1 = SparseOpType[embed(AtomicArrays.fourlevel_quantum.basis(A), j,
    #                  AtomicArrays.fourlevel_quantum.sigmas_ee_[3]) 
    #                 for j=1:length(A.atoms)]
    Ps = [SparseOpType[embed(AtomicArrays.fourlevel_quantum.basis(A), j,
                     AtomicArrays.fourlevel_quantum.sigmas_eg_[k] * 
                     AtomicArrays.fourlevel_quantum.sigmas_ge_[l]) 
                    for j=1:length(A.atoms)] for k=1:3, l=1:3]
    # return P_m1, P_0, P_p1, P_cross
    return Ps
end

# BUILDING THE SYSTEM

# Build the collection
begin
    # positions = [
    #     [0.0, 0.0, 0.0],
    #     [0.5, 0.0, 0.0],
    #     # [20.4, 0.0, 0.0],
    #     # [30.6, 0.0, 0.0]
    # ]
    a = 0.6
    positions = AtomicArrays.geometry.rectangle(a, a; Nx=2, Ny=2)
    # positions = rotate_xy_to_xz(positions)
    N = length(positions)

    pols = AtomicArrays.polarizations_spherical(N)
    gam = [AtomicArrays.gammas(0.25)[m] for m=1:3, j=1:N]
    deltas = [0.0 for i = 1:N]
    # deltas = [0.0, 0.0, 0.0, 0.0]
    # deltas = [0.0, 0.01]

    coll = AtomicArrays.FourLevelAtomCollection(positions;
        deltas = deltas,
        polarizations = pols,
        gammas = gam
    )

    # Define a plane wave field in +y direction:
    amplitude = 0.01
    k_mod = 2π
    # angle_k = [0.0, π/2]  # => +y direction
    angle_k = [0.0, 0.0]  # => +z direction
    polarisation = [1.0, 1.0im, 0.0]
    pos_0 = [0.0, 0.0, 0.0]

    field = AtomicArrays.field.EMField(amplitude, k_mod, angle_k, polarisation; position_0=pos_0)
    external_drive = AtomicArrays.field.rabi(field, AtomicArrays.field.plane, coll)

    B_f = 0.1
    w = [deltas[n]+B_f*m for m = -1:1, n = 1:N]
    Γ = AtomicArrays.interaction.GammaTensor_4level(coll)
    Ω = AtomicArrays.interaction.OmegaTensor_4level(coll)
end


# Build the Hamiltonian and jump operators
H = AtomicArrays.fourlevel_quantum.Hamiltonian(coll; magnetic_field=B_f,
                external_drive=external_drive,
                dipole_dipole=true)

Γ_fl, J_ops = AtomicArrays.fourlevel_quantum.JumpOperators(coll; flatten=true)

test = (B_f == 0.0) ? 1 : 2

# Master equation time evolution
begin
    b = AtomicArrays.fourlevel_quantum.basis(coll)
    # initial state => all ground
    ψ0 = basisstate(b, [(i == 1) ? AtomicArrays.fourlevel_quantum.idx_e_plus : 
    AtomicArrays.fourlevel_quantum.idx_g for i = 1:N])
    ρ0 = dm(ψ0)
    tspan = [0.0:0.1:200.0;]
    t, rho_t = timeevolution.master_h(tspan, ψ0, H, J_ops; rates=Γ_fl)
end

begin
    av_J = average_values(J_ops, rho_t)
    Ps = population_ops(coll)

    pops_q_t = zeros((3, 3, length(t), N))
    for i=1:3, j=1:3
        for n in 1:N
            for (k, ρ) in enumerate(rho_t)
                pops_q_t[i,j,k,n] = real(tr(Ps[i,j][n] * ρ))
            end
        end
    end
end

# Meanfield time dynamics
begin
    u0 = [0.0im for i = 1:12*N]
    for n = 1:1
        reshape(view(u0, (3*N)+1:12*N), (3, 3, N))[3,3,n] = 1.0
    end
    tspan = (0.0, 200.0)
    p = (w, external_drive, Ω, Γ)
    prob = ODEProblem(AtomicArrays.fourlevel_meanfield.f_sym, u0, tspan, p)
    sol = solve(prob)
end

function plot_populations(pop_e_minus, pop_e_0, pop_e_plus, t)
    fig = Figure(size=(1000, 300))
    ax1 = Axis(fig[1, 1], title="m = -1", xlabel="t", ylabel="Population")
    ax2 = Axis(fig[1, 2], title="m = 0", xlabel="t", ylabel="Population")
    ax3 = Axis(fig[1, 3], title="m = +1", xlabel="t", ylabel="Population")
    # Define colors for different atoms
    # colors = Makie.wong_colors(N)
    colors = cgrad(:viridis, N, categorical=true)
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

t = [0.0:0.1:200.0;]

begin
    p_e_minus_mf = [real(reshape(view(sol(i),(3*N)+1:12*N), (3, 3, N))[1,1,j])
                    for i in t, j = 1:N]
    p_e_0_mf = [real(reshape(view(sol(i),(3*N)+1:12*N), (3, 3, N))[2,2,j])
                    for i in t, j = 1:N]
    p_e_plus_mf = [real(reshape(view(sol(i),(3*N)+1:12*N), (3, 3, N))[3,3,j])
                    for i in t, j = 1:N]
    s_e_minus_mf = [reshape(view(sol(i),1:3*N), (3, N))[1,j]
                    for i in t, j = 1:N]
    s_e_0_mf = [reshape(view(sol(i),1:3*N), (3, N))[2,j]
                    for i in t, j = 1:N]
    s_e_plus_mf = [reshape(view(sol(i),1:3*N), (3, N))[3,j]
                    for i in t, j = 1:N]
end

let 
    f = Figure(size=(1000, 300))
    ax1 = Axis(f[1, 1], title="m = -1", xlabel="t", ylabel="Population")
    ax2 = Axis(f[1, 2], title="m = 0", xlabel="t", ylabel="Population")
    ax3 = Axis(f[1, 3], title="m = +1", xlabel="t", ylabel="Population")
    # Define colors for different atoms
    colors = Makie.wong_colors(N)
    colors_mf = cgrad(:viridis, N, categorical=true)
    # Plot sublevels
    for n in 1:N
        lines!(ax1, t, pops_q_t[1,1,:,n], label="Atom $n, q", color=colors[n], linewidth=2)
        lines!(ax1, t, p_e_minus_mf[:, n], label="Atom $n, mf", color=colors_mf[n], linewidth=2)
        lines!(ax2, t, pops_q_t[2,2,:,n], label="Atom $n", color=colors[n], linewidth=2)
        lines!(ax2, t, p_e_0_mf[:, n], label="Atom $n, mf", color=colors_mf[n], linewidth=2)
        lines!(ax3, t, pops_q_t[3,3,:,n], label="Atom $n", color=colors[n], linewidth=2)
        lines!(ax3, t, p_e_plus_mf[:, n], label="Atom $n, mf", color=colors_mf[n], linewidth=2)
    end
    # Add legend
    Legend(f[1, 4], ax1, "Atoms", framevisible=false)
    f
end


plot_populations(pops_q_t[1,2,:,:], pops_q_t[1,3,:,:], pops_q_t[2,3,:,:], t)
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
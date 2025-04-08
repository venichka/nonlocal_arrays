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
    using QuantumOptics, QuantumCumulants
    using ModelingToolkit
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
    N = Int(length(u) / 12)
    w, OmR, Omega, Gamma = p
    sm = reshape(view(u, 1:3*N), (3, N))
    smm = reshape(view(u, (3*N)+1:12*N), (3, 3, N))
    dsm = reshape(view(du, 1:3*N), (3, N))
    dsmm = reshape(view(du, (3*N)+1:12*N), (3, 3, N))
    # sigma equations
    @inbounds for n=1:N
        for m=1:3
            dsm[m,n] = ((-1im*w[m,n] - 0.5*Gamma[n,n,m,m])*sm[m,n])
            for m1 = 1:3
                if m1 == m
                    s_bar = smm[m1,m,n] - 1 + sum([smm[i,i,n] for i=1:3])
                else
                    s_bar = smm[m1,m,n]
                end
                dsm[m,n] += -1im*conj(OmR[m1,n])*s_bar
                for m2 = 1:3
                    for n2 = 1:N
                        if n2 == n
                            continue
                        end
                        dsm[m,n] += (1im*Omega[n,n2,m1,m2]+
                                     0.5*Gamma[n,n2,m1,m2])*s_bar*sm[m2,n2] 
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
                                1im*conj(OmR[m,n])*sm[m1,n]')
                for m2 = 1:3
                    dsmm[m1,m,n] += - (0.5*Gamma[n,n,m2,m1]*smm[m2,m,n] +
                                     0.5*Gamma[n,n,m,m2]*smm[m1,m2,n])
                    for n1 = 1:N
                        if n1 == n
                            continue
                        end
                        dsmm[m1,m,n] += ((1im*Omega[n1,n,m2,m1] -
                                          0.5*Gamma[n1,n,m2,m1])*
                                          sm[m2,n1]'*sm[m,n] +
                                         (-1im*Omega[n,n1,m,m2] -
                                          0.5*Gamma[n,n1,m,m2] )*
                                          sm[m1,n]'*sm[m2,n1])
                    end
                end
            end
        end
    end
end

# u = [σ₋₁¹, σ₀¹, σ₊₁¹, σ₋₁²,..., σ₋₁¹⁺,...,]
function meanfield_spherical_real!(du, u, p, t)
    N = Int(length(u) / 24)
    w, OmR, Omega, Gamma = p
    OmR_r = real(OmR); OmR_i = imag(OmR)
    Omega_r = real(Omega); Omega_i = imag(Omega)
    Gamma_r = real(Gamma); Gamma_i = imag(Gamma)
    xm = reshape(view(u, 1:3*N), (3, N))
    ym = reshape(view(u, 3*N+1:6*N), (3, N))
    xmm = reshape(view(u, (6*N)+1:15*N), (3, 3, N))
    ymm = reshape(view(u, (15*N)+1:24*N), (3, 3, N))
    dxm = reshape(view(du, 1:3*N), (3, N))
    dym = reshape(view(du, 3*N+1:6*N), (3, N))
    dxmm = reshape(view(du, (6*N)+1:15*N), (3, 3, N))
    dymm = reshape(view(du, (15*N)+1:24*N), (3, 3, N))
    # sigma equations
    @inbounds for n=1:N
        for m=1:3
            dxm[m,n] = w[m,n]*ym[m,n]
            dym[m,n] = -w[m,n]*xm[m,n]
            for m1 = 1:3
                dxm[m,n] -= 0.5*(Gamma_r[n,n,m,m1]*xm[m1,n] -
                                 Gamma_i[n,n,m,m1]*ym[m1,n])
                dym[m,n] -= 0.5*(Gamma_r[n,n,m,m1]*ym[m1,n] +
                                 Gamma_i[n,n,m,m1]*xm[m1,n])
                if m1 == m
                    x_bar = xmm[m1,m,n] - 1 + sum([xmm[i,i,n] for i=1:3])
                else
                    x_bar = xmm[m1,m,n]
                end
                dxm[m,n] -= OmR_i[m1,n]*x_bar - OmR_r[m1,n]*ymm[m1,m,n]
                dym[m,n] -= OmR_r[m1,n]*x_bar + OmR_i[m1,n]*ymm[m1,m,n]
                for m2 = 1:3
                    for n2 = 1:N
                        if n2 == n
                            continue
                        end
                        dxm[m,n]+=((0.5*Gamma_r[n,n2,m1,m2]-
                                    Omega_i[n,n2,m1,m2])*
                                    (x_bar*xm[m2,n2] - ymm[m1,m,n]*ym[m2,n2]) -
                                   (0.5*Gamma_i[n,n2,m1,m2]+
                                    Omega_r[n,n2,m1,m2])*
                                    (x_bar*ym[m2,n2] + ymm[m1,m,n]*xm[m2,n2]))
                        dym[m,n]+=((0.5*Gamma_i[n,n2,m1,m2]+
                                    Omega_r[n,n2,m1,m2])*
                                    (x_bar*xm[m2,n2] - ymm[m1,m,n]*ym[m2,n2]) +
                                   (0.5*Gamma_r[n,n2,m1,m2]-
                                    Omega_i[n,n2,m1,m2])*
                                    (x_bar*ym[m2,n2] + ymm[m1,m,n]*xm[m2,n2]))
                    end
                end
            end
        end
    end
    # population equations
    @inbounds for n = 1:N
        for m = 1:3
            for m1 = 1:3
                dxmm[m1,m,n] = (-(w[m1,n]-w[m,n])*ymm[m1,m,n]
                                +OmR_r[m1,n]*ym[m,n]+OmR_i[m1,n]*xm[m,n]
                                +OmR_r[m,n]*ym[m1,n]+OmR_i[m,n]*xm[m1,n])
                dymm[m1,m,n] = ((w[m1,n]-w[m,n])*xmm[m1,m,n]
                                -OmR_r[m1,n]*xm[m,n]+OmR_i[m1,n]*ym[m,n]
                                +OmR_r[m,n]*xm[m1,n]-OmR_i[m,n]*ym[m1,n])
                for m2 = 1:3
                    dxmm[m1,m,n] -= 0.5*(Gamma_r[n,n,m2,m1]*xmm[m2,m,n] -
                                         Gamma_i[n,n,m2,m1]*ymm[m2,m,n] +
                                         Gamma_r[n,n,m,m2]*xmm[m1,m2,n] -
                                         Gamma_i[n,n,m,m2]*ymm[m1,m2,n])
                    dymm[m1,m,n] -= 0.5*(Gamma_r[n,n,m2,m1]*ymm[m2,m,n] +
                                         Gamma_i[n,n,m2,m1]*xmm[m2,m,n] +
                                         Gamma_r[n,n,m,m2]*ymm[m1,m2,n] +
                                         Gamma_i[n,n,m,m2]*xmm[m1,m2,n])
                    for n1 = 1:N
                        if n1 == n
                            continue
                        end
                        dxmm[m1,m,n] -=((0.5*Gamma_r[n1,n,m2,m1]+
                                        Omega_i[n1,n,m2,m1])*
                                        (xm[m2,n1]*xm[m,n]+ym[m2,n1]*ym[m,n])-
                                        (0.5*Gamma_i[n1,n,m2,m1]-
                                        Omega_r[n1,n,m2,m1])*
                                        (xm[m2,n1]*ym[m,n]-ym[m2,n1]*xm[m,n])+
                                        (0.5*Gamma_r[n,n1,m,m2]-
                                        Omega_i[n,n1,m,m2])*
                                        (xm[m1,n]*xm[m2,n1]+ym[m1,n]*ym[m2,n1])-
                                        (0.5*Gamma_i[n,n1,m,m2]+
                                        Omega_r[n,n1,m,m2])*
                                        (xm[m1,n]*ym[m2,n1]-ym[m1,n]*xm[m2,n1]))
                        dymm[m1,m,n] -=((0.5*Gamma_i[n1,n,m2,m1]-
                                        Omega_r[n1,n,m2,m1])*
                                        (xm[m2,n1]*xm[m,n]+ym[m2,n1]*ym[m,n])+
                                        (0.5*Gamma_r[n1,n,m2,m1]+
                                        Omega_i[n1,n,m2,m1])*
                                        (xm[m2,n1]*ym[m,n]-ym[m2,n1]*xm[m,n])+
                                        (0.5*Gamma_i[n,n1,m,m2]+
                                        Omega_r[n,n1,m,m2])*
                                        (xm[m1,n]*xm[m2,n1]+ym[m1,n]*ym[m2,n1])+
                                        (0.5*Gamma_r[n,n1,m,m2]-
                                        Omega_i[n,n1,m,m2])*
                                        (xm[m1,n]*ym[m2,n1]-ym[m1,n]*xm[m2,n1]))
                    end
                end
            end
        end
    end
    @inbounds for m = 1:3
        for m1 = m:3
            dxmm[m, m1, :] .= dxmm[m1, m, :]
            dymm[m, m1, :] .= -dymm[m1, m, :]
            if m1 == m
                dymm[m, m1, :] .= 0.0
            end
        end
    end
end

function scattered_field(r::Vector, A::AtomicArrays.FourLevelAtomCollection,
    sigmas_m::Matrix, k_field::Number=2π)
    M, N = size(A.gammas)
    C = 3.0/4.0 * A.gammas
    return sum(C[m,n] * sigmas_m[m,n] * 
               GreenTensor(r-A.atoms[n].position, k_field) *
               A.polarizations[m,:,n] for m = 1:M, n = 1:N)
end

# Build the collection
begin
    a = 0.2; Nx = 10; Ny = 10;
    positions = AtomicArrays.geometry.rectangle(a, a; Nx=Nx, Ny=Ny,
                                position_0=[(-Nx/2+0.5)*a,(-Ny/2+0.5)*a,0.0])
    positions = rotate_xy_to_xz(positions)
    N = length(positions)

    pols = AtomicArrays.polarizations_spherical(N)
    gam = [AtomicArrays.gammas(0.25)[m] for m=1:3, j=1:N]
    deltas = [(i == 1) ? 0.1 : 0.0 for i = 1:N]

    coll = AtomicArrays.FourLevelAtomCollection(positions;
        deltas = deltas,
        polarizations = pols,
        gammas = gam
    )

    # Define a plane wave field in +y direction:
    amplitude = 0.1
    k_mod = 2π
    angle_k = [0.0, π/2]  # => +y direction
    polarisation = [1.0, 1.0im, 0.0]
    pos_0 = [0.0, 0.0, 0.0]

    field = AtomicArrays.field.EMField(amplitude, k_mod, angle_k, polarisation; position_0=pos_0)
    external_drive = AtomicArrays.field.rabi(field, AtomicArrays.field.plane, coll)

    B_f = 0.2
    w = [deltas[n]+B_f*m for m = -1:1, n = 1:N]
    Γ = AtomicArrays.interaction.GammaTensor_4level(coll)
    Ω = AtomicArrays.interaction.OmegaTensor_4level(coll)
end

# Q cumulants eqs
begin
    # Parameters
    # @cnumbers N M

    # Hilbertspace
    h = NLevelSpace(:qubit, (:g, Symbol("\\sigma_-"), :π, Symbol("\\sigma_+")))
    
    w_m1(i) = IndexedVariable(Symbol("\\omega^{(-1)}"), i)
    w_0(i) = IndexedVariable(Symbol("\\omega^{(0)}"), i)
    w_p1(i) = IndexedVariable(Symbol("\\omega^{(+1)}"), i)
    Γ_m1_0(i,j) = IndexedVariable(Symbol("\\Gamma^{(-1,0)}"), i, j)
    Γ_m1_p1(i,j) = IndexedVariable(Symbol("\\Gamma^{(-1,+1)}"), i, j)
    Γ_0_p1(i,j) = IndexedVariable(Symbol("\\Gamma^{(0,+1)}"), i, j)
    Γ_m1_m1(i,j) = IndexedVariable(Symbol("\\Gamma^{(-1,-1)}"), i, j)
    Γ_p1_p1(i,j) = IndexedVariable(Symbol("\\Gamma^{(+1,+1)}"), i, j)
    Γ_0_0(i,j) = IndexedVariable(Symbol("\\Gamma^{(0,0)}"), i, j)
    Ω_m1_0(i,j) = IndexedVariable(Symbol("\\Omega^{(-1,0)}"), i, j)
    Ω_m1_p1(i,j) = IndexedVariable(Symbol("\\Omega^{(-1,+1)}"), i, j)
    Ω_0_p1(i,j) = IndexedVariable(Symbol("\\Omega^{(0,+1)}"), i, j)
    Ω_m1_m1(i,j) = IndexedVariable(Symbol("\\Omega^{(-1,-1)}"), i, j)
    Ω_p1_p1(i,j) = IndexedVariable(Symbol("\\Omega^{(+1,+1)}"), i, j)
    Ω_0_0(i,j) = IndexedVariable(Symbol("\\Omega^{(0,0)}"), i, j)
    g_m1(i) = IndexedVariable(Symbol("g^{(-1)}"), i)
    g_0(i) = IndexedVariable(Symbol("g^{(0)}"), i)
    g_p1(i) = IndexedVariable(Symbol("g^{(+1)}"), i)
    gc_m1(i) = IndexedVariable(Symbol("g^{(-1)*}"), i)
    gc_0(i) = IndexedVariable(Symbol("g^{(0)*}"), i)
    gc_p1(i) = IndexedVariable(Symbol("g^{(+1)*}"), i)
    γ(α,β) = IndexedVariable(Symbol("\\gamma"), α, β)

    i = Index(h, :i, N, h)
    j = Index(h, :j, N, h)
    k = Index(h, :k, N, h)
    l = Index(h, :l, N, h)
    α = Index(h, :α, 3*N, h)
    β = Index(h, :β, 3*N, h)
    
    
    # Operators
    # Hilbert space
    h_spin(i) = NLevelSpace("qubit_$i", (:g, Symbol("\\sigma_-"), :π, Symbol("\\sigma_+")))
    h_tot = tensor([h_spin(i) for i=1:N]...)
    # Operators
    s_m1(i) = Transition(h_tot,Symbol("J_{$i}"),:g,Symbol("\\sigma_-"),i)
    s_0(i) = Transition(h_tot,Symbol("J_{$i}"),:g,:π,i)
    s_p1(i) = Transition(h_tot,Symbol("J_{$i}"),:g,Symbol("\\sigma_+"),i)

    # Hamiltonian

    H_a = sum(w_m1(i)*s_m1(i)'*s_m1(i) + 
            w_0(i)*s_0(i)'*s_0(i) + 
            w_p1(i)*s_p1(i)'*s_p1(i) for i=1:N)
    H_f = -sum(g_m1(i)*s_m1(i) + gc_m1(i)*s_m1(i)' + 
             g_0(i)*s_0(i) + gc_0(i)*s_0(i)' +
             g_p1(i)*s_p1(i) + gc_p1(i)*s_p1(i)' for i=1:N)
    H_dd = sum((Ω_m1_m1(i,j))*s_m1(i)'*s_m1(j) + 
             (Ω_0_0(i,j))*s_0(i)'*s_0(j) +
             (Ω_p1_p1(i,j))*s_p1(i)'*s_p1(j) + 
             (Ω_m1_0(i,j))*s_m1(i)'*s_0(j) + 
             (Ω_m1_0(i,j))*s_0(i)'*s_m1(j) +
             (Ω_m1_p1(i,j))*s_m1(i)'*s_p1(j) + 
             (Ω_m1_p1(i,j))*s_p1(i)'*s_m1(j) +
             (Ω_0_p1(i,j))*s_0(i)'*s_p1(j) + 
             (Ω_0_p1(i,j))*s_p1(i)'*s_0(j)
             for i=1:N for j=1:N)

    H = H_a + H_f + H_dd
    
    # Jumps
    J = vcat([[s_m1(i), s_0(i), s_p1(i)] for i = 1:N]...)
    
    # Rates
    rates = [γ(i,j) for i=1:3N, j= 1:3N]#[Γ0(c1,c2) for c1=1:N, c2=1:N]
end

begin
    # list of operators
    n_order = 1
    s(n) = [s_m1(n), s_0(n), s_p1(n)]
    if n_order > 1
        ops = vcat([s_m1(i) for i = 1:N],
                [s_m1(i)' * s_m1(i) for i = 1:N])
    else
        ops = vcat(
            [s(n)[m] for n = 1:N for m = 1:3],
            [s(n)[m]' * s(n)[m1] for n = 1:N for m = 1:3 for m1 = m:3]
        )
    end
    
    eqs = QuantumCumulants.meanfield(ops, H, J; rates=rates, order=n_order, simplify=true)

    eqs_comp = complete(eqs) #automatically complete the system

# Build an ODESystem out of the MeanfieldEquations
    @named sys = ODESystem(eqs_comp)
    print("Model is built")
end


# Quantum dynamics
begin
    # Build the Hamiltonian and jump operators
    H = AtomicArrays.fourlevel_quantum.Hamiltonian(coll; magnetic_field=B_f,
                    external_drive=external_drive,
                    dipole_dipole=true)

    Γ_fl, J_ops = AtomicArrays.fourlevel_quantum.JumpOperators(coll; flatten=true)

    b = AtomicArrays.fourlevel_quantum.basis(coll)
    # initial state => all ground
    ψ0 = basisstate(b, [(i == 1) ? AtomicArrays.fourlevel_quantum.idx_g : 
    AtomicArrays.fourlevel_quantum.idx_g for i = 1:N])
    ρ0 = dm(ψ0)
    tspan = [0.0:0.1:400.0;]
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


begin
    # list of symbolic indexed parameters
    w_m1_i = [w_m1(i) for i=1:N]
    w_0_i = [w_0(i) for i=1:N]
    w_p1_i = [w_p1(i) for i=1:N]
    Ω_m1_0_ij = [Ω_m1_0(i,j) for i=1:N for j=1:N]
    Ω_m1_p1_ij = [Ω_m1_p1(i,j) for i=1:N for j=1:N]
    Ω_0_p1_ij = [Ω_0_p1(i,j) for i=1:N for j=1:N]
    Ω_m1_m1_ij = [Ω_m1_m1(i,j) for i=1:N for j=1:N]
    Ω_p1_p1_ij = [Ω_p1_p1(i,j) for i=1:N for j=1:N]
    Ω_0_0_ij = [Ω_0_0(i,j) for i=1:N for j=1:N]
    g_m1_i = [g_m1(i) for i=1:N]
    g_0_i = [g_0(i) for i=1:N]
    g_p1_i = [g_p1(i) for i=1:N]
    gc_m1_i = [gc_m1(i) for i=1:N]
    gc_0_i = [gc_0(i) for i=1:N]
    gc_p1_i = [gc_p1(i) for i=1:N]
    γ_ij = [γ(i, j) for i=1:3N for j=1:3N]

    w_m1_ = w[1,:]
    w_0_ = w[2,:]
    w_p1_ = w[3,:]
    Ω_m1_0_ = Ω[:, :, 1, 2]
    Ω_m1_p1_ = Ω[:, :, 1, 3]
    Ω_0_p1_ = Ω[:, :, 2, 3]
    Ω_m1_m1_ = Ω[:, :, 1, 1]
    Ω_0_0_ = Ω[:, :, 2, 2]
    Ω_p1_p1_ = Ω[:, :, 1, 1]
    g_m1_ = external_drive[1,:]
    g_0_ = external_drive[2,:]
    g_p1_ = external_drive[3,:]
    γ_ = Γ_fl
end

begin
    # initial state
    u0 = zeros(ComplexF64, length(eqs_comp))
    reshape(view(u0, 3N+1:3N+3*2*N),(3,2,N))[3,2,:] .= 0.0 + 0.0im

    # list of parameters
    ps = [w_m1_i; w_0_i; w_p1_i; 
        Ω_m1_0_ij; Ω_m1_p1_ij; Ω_0_p1_ij; 
        Ω_m1_m1_ij; Ω_p1_p1_ij; Ω_0_0_ij;
        g_m1_i; g_0_i; g_p1_i; gc_m1_i; gc_0_i; gc_p1_i;
        γ_ij
    ]
    pn = collect(Iterators.flatten([
        w_m1_, w_0_, w_p1_, 
        Ω_m1_0_, Ω_m1_p1_, Ω_0_p1_, Ω_m1_m1_, Ω_p1_p1_, Ω_0_0_,
        g_m1_, g_0_, g_p1_, conj.(g_m1_), conj.(g_0_), conj.(g_p1_),
        γ_
    ]))
    p0 = ps .=> pn
    tend = 400.
    tlist = range(0, tend, 200)
    
    prob = ODEProblem(sys,u0,(0.0,tend),p0)
    sol = solve(prob)
    # sol = solve(prob, Kvaerno5(autodiff=false), alg_hints=[:stiff], reltol=1e-8, abstol=1e-8, maxiters=1e7, saveat=tlist)
end

begin
    p_e_minus_qc = [real(reshape(view(sol(i),(3*N)+1:9*N), (3, 2, N))[1,1,j])
                    for i in t, j = 1:N]
    p_e_0_qc = [real(reshape(view(sol(i),(3*N)+1:9*N), (3, 2, N))[1,2,j])
                    for i in t, j = 1:N]
    p_e_plus_qc = [real(reshape(view(sol(i),(3*N)+1:9*N), (3, 2, N))[3,2,j])
                    for i in t, j = 1:N]
    s_e_minus_qc = [reshape(view(sol(i),1:3*N), (3, N))[1,j]
                    for i in t, j = 1:N]
    s_e_0_qc = [reshape(view(sol(i),1:3*N), (3, N))[2,j]
                    for i in t, j = 1:N]
    s_e_plus_qc = [reshape(view(sol(i),1:3*N), (3, N))[3,j]
                    for i in t, j = 1:N]
end

# Meanfield time dynamics
begin
    u0 = [0.0im for i = 1:12*N]
    for n = 1:1
        reshape(view(u0, (3*N)+1:12*N), (3, 3, N))[3,3,n] = 0.0
    end
    tspan = (0.0, 400.0)
    # tspan = [0.0:0.1:200.0;]
    p = (w, external_drive, Ω, Γ)
    prob = ODEProblem(meanfield_spherical!, u0, tspan, p)
    sol = solve(prob)
end

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

# Meanfield time dynamics: real
begin
    u0 = [0.0 for i = 1:24*N]
    for n = 1:1
        reshape(view(u0, (6*N)+1:15*N), (3, 3, N))[3,3,n] = 0.0
    end
    tspan = (0.0, 400.0)
    # tspan = [0.0:0.1:200.0;]
    p = (w, external_drive, Ω, Γ)
    prob = ODEProblem(meanfield_spherical_real!, u0, tspan, p)
    sol = solve(prob)
end

begin
    p_e_minus_mf_r = [reshape(view(sol(i),(6*N)+1:15*N), (3, 3, N))[1,1,j]
                    for i in t, j = 1:N]
    p_e_0_mf_r = [reshape(view(sol(i),(6*N)+1:15*N), (3, 3, N))[2,2,j]
                    for i in t, j = 1:N]
    p_e_plus_mf_r = [reshape(view(sol(i),(6*N)+1:15*N), (3, 3, N))[3,3,j]
                    for i in t, j = 1:N]
    s_e_minus_mf_r = [reshape(view(sol(i),1:3*N).+
                              1im*view(sol(i),3*N+1:6*N), (3, N))[1,j]
                      for i in t, j = 1:N]
    s_e_0_mf_r = [reshape(view(sol(i),1:3*N).+
                              1im*view(sol(i),3*N+1:6*N), (3, N))[2,j]
                      for i in t, j = 1:N]
    s_e_plus_mf_r = [reshape(view(sol(i),1:3*N).+
                              1im*view(sol(i),3*N+1:6*N), (3, N))[3,j]
                      for i in t, j = 1:N]
end

let 
    f = Figure(size=(1000, 300))
    ax1 = Axis(f[1, 1], title="m = -1", xlabel="t", ylabel="Population")
    ax2 = Axis(f[1, 2], title="m = 0", xlabel="t", ylabel="Population")
    ax3 = Axis(f[1, 3], title="m = +1", xlabel="t", ylabel="Population")
    # Define colors for different atoms
    colors = cgrad(:reds, N, categorical=true)
    colors_qc = cgrad(:greens, N, categorical=true)
    colors_mf = cgrad(:blues, N, categorical=true)
    colors_mf_r = cgrad(:army, N, categorical=true)
    # Plot sublevels
    for n in 1:N
        lines!(ax1, t, pop_e_minus[:,n], label="Atom $n, q", color=colors[n], linewidth=2)
        lines!(ax1, t, p_e_minus_qc[:, n], label="Atom $n, qc", color=colors_qc[n], linewidth=2)
        lines!(ax1, t, p_e_minus_mf[:, n], label="Atom $n, mf", color=colors_mf[n], linewidth=2)
        lines!(ax1, t, p_e_minus_mf_r[:, n], label="Atom $n, mf_r", color=colors_mf_r[n], linewidth=2)
        lines!(ax2, t, pop_e_0[:,n], label="Atom $n", color=colors[n], linewidth=2)
        lines!(ax2, t, p_e_0_qc[:, n], label="Atom $n, qc", color=colors_qc[n], linewidth=2)
        lines!(ax2, t, p_e_0_mf[:, n], label="Atom $n, mf", color=colors_mf[n], linewidth=2)
        lines!(ax2, t, p_e_0_mf_r[:, n], label="Atom $n, mf_r", color=colors_mf_r[n], linewidth=2)
        lines!(ax3, t, pop_e_plus[:,n], label="Atom $n", color=colors[n], linewidth=2)
        lines!(ax3, t, p_e_plus_qc[:, n], label="Atom $n, qc", color=colors_qc[n], linewidth=2)
        lines!(ax3, t, p_e_plus_mf[:, n], label="Atom $n, mf", color=colors_mf[n], linewidth=2)
        lines!(ax3, t, p_e_plus_mf_r[:, n], label="Atom $n, mf_r", color=colors_mf_r[n], linewidth=2)
    end
    # Add legend
    Legend(f[1, 4], ax1, "Atoms", framevisible=false)
    f
end


let 
    f = Figure(size=(1000, 300))
    ax1 = Axis(f[1, 1], title="m = -1", xlabel="t", ylabel="Dipole moment")
    ax2 = Axis(f[1, 2], title="m = 0", xlabel="t", ylabel="Dipole moment")
    ax3 = Axis(f[1, 3], title="m = +1", xlabel="t", ylabel="Dipole moment")

    av_J_m = reshape(av_J, (length(t), 3, N))

    # Define colors for different atoms
    colors = cgrad(:reds, N, categorical=true)
    colors_qc = cgrad(:greens, N, categorical=true)
    colors_mf = cgrad(:blues, N, categorical=true)
    colors_mf_r = cgrad(:army, N, categorical=true)
    # Plot sublevels
    for n in 1:N
        lines!(ax1, t, imag(av_J_m[:, 1, n]), label="Atom $n, q", color=colors[n], linewidth=2)
        lines!(ax1, t, imag(s_e_minus_qc[:, n]), label="Atom $n, qc", color=colors_qc[n], linewidth=2)
        lines!(ax1, t, imag(s_e_minus_mf[:, n]), label="Atom $n, mf", color=colors_mf[n], linewidth=2)
        lines!(ax1, t, imag(s_e_minus_mf_r[:, n]), label="Atom $n, mf_r", color=colors_mf_r[n], linewidth=2)
        lines!(ax2, t, imag(av_J_m[:, 2, n]), label="Atom $n", color=colors[n], linewidth=2)
        lines!(ax2, t, imag(s_e_0_qc[:, n]), label="Atom $n, qc", color=colors_qc[n], linewidth=2)
        lines!(ax2, t, imag(s_e_0_mf[:, n]), label="Atom $n, mf", color=colors_mf[n], linewidth=2)
        lines!(ax2, t, imag(s_e_0_mf_r[:, n]), label="Atom $n, mf_r", color=colors_mf_r[n], linewidth=2)
        lines!(ax3, t, imag(av_J_m[:, 3, n]), label="Atom $n", color=colors[n], linewidth=2)
        lines!(ax3, t, imag(s_e_plus_qc[:, n]), label="Atom $n, qc", color=colors_qc[n], linewidth=2)
        lines!(ax3, t, imag(s_e_plus_mf[:, n]), label="Atom $n, mf", color=colors_mf[n], linewidth=2)
        lines!(ax3, t, imag(s_e_plus_mf_r[:, n]), label="Atom $n, mf_r", color=colors_mf_r[n], linewidth=2)
    end
    # Add legend
    Legend(f[1, 4], ax1, "Atoms", framevisible=false)
    f
end

# === Choose which plane to visualize ===
begin
    # Options: "xy", "xz", "yz"
    plane = "xy"   # change this to "xz" or "yz" as desired

    # sigmas_m = reshape(av_J[end,:],(3,N))
    sigmas_m = reshape(view(sol(tspan[end]),1:3*N).+
                       1im*view(sol(tspan[end]),3*N+1:6*N),(3,N))

    # === Define grid parameters ===
    grid_min, grid_max, grid_step = -2, 2, 0.02

    # Depending on the plane, choose the two varying coordinates and fix the third:
    if plane == "xy"
        coord1_range = grid_min:grid_step:grid_max  # x
        coord2_range = grid_min:grid_step:grid_max  # y
        fixed_value = Nx*a/2 +0.2                           # fixed z
        fixed_index = 3
        label1, label2, fixed_label = "x", "y", "z"
    elseif plane == "xz"
        coord1_range = grid_min:grid_step:grid_max  # x
        coord2_range = grid_min:grid_step:grid_max  # z
        fixed_value = 0.1                           # fixed y
        fixed_index = 2
        label1, label2, fixed_label = "x", "z", "y"
    elseif plane == "yz"
        coord1_range = grid_min:grid_step:grid_max  # y
        coord2_range = grid_min:grid_step:grid_max  # z
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
    Re_field_y = zeros(nx, ny)
    Abs_field_y = zeros(nx, ny)
    Re_field_z = zeros(nx, ny)
    Abs_field_z = zeros(nx, ny)

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
            E = scattered_field(r, coll, sigmas_m)
            # Save the real and absolute values for each Cartesian component.
            Re_field_x[i, j] = real(E[1])
            Abs_field_x[i, j] = abs(E[1])
            Re_field_y[i, j] = real(E[2])
            Abs_field_y[i, j] = abs(E[2])
            Re_field_z[i, j] = real(E[3])
            Abs_field_z[i, j] = abs(E[3])
        end
    end
end

begin
    fig = Figure(size = (900, 1100))

    nlevels = 10

    # --- Row 1: x-component (Eₓ) ---
    ax1 = Axis(fig[1, 1], title = "Re(Eₓ)", xlabel = label1, ylabel = label2)
    hm1 = contourf!(ax1, coord1_range, coord2_range, Re_field_x, colormap = :plasma, levels=nlevels)
    scatter!(ax1, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[1, 2], hm1, width = 15, height = Relative(1))

    ax2 = Axis(fig[1, 3], title = "Abs(Eₓ)", xlabel = label1, ylabel = label2)
    hm2 = contourf!(ax2, coord1_range, coord2_range, Abs_field_x, colormap = :plasma, levels=nlevels)
    scatter!(ax2, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[1, 4], hm2, width = 15, height = Relative(1))

    # --- Row 2: y-component (Eᵧ) ---
    ax3 = Axis(fig[2, 1], title = "Re(Eᵧ)", xlabel = label1, ylabel = label2)
    hm3 = contourf!(ax3, coord1_range, coord2_range, Re_field_y, colormap = :plasma, levels=nlevels)
    scatter!(ax3, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[2, 2], hm3, width = 15, height = Relative(1))

    ax4 = Axis(fig[2, 3], title = "Abs(Eᵧ)", xlabel = label1, ylabel = label2)
    hm4 = contourf!(ax4, coord1_range, coord2_range, Abs_field_y, colormap = :plasma, levels=nlevels)
    scatter!(ax4, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[2, 4], hm4, width = 15, height = Relative(1))

    # --- Row 3: z-component (E_z) ---
    ax5 = Axis(fig[3, 1], title = "Re(E_z)", xlabel = label1, ylabel = label2)
    hm5 = contourf!(ax5, coord1_range, coord2_range, Re_field_z, colormap = :plasma, levels=nlevels)
    scatter!(ax5, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[3, 2], hm5, width = 15, height = Relative(1))

    ax6 = Axis(fig[3, 3], title = "Abs(E_z)", xlabel = label1, ylabel = label2)
    hm6 = contourf!(ax6, coord1_range, coord2_range, Abs_field_z, colormap = :plasma, levels=nlevels)
    scatter!(ax6, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[3, 4], hm6, width = 15, height = Relative(1))

    fig
end

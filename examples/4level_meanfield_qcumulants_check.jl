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

# Build the collection
begin
    a = 0.6
    positions = AtomicArrays.geometry.rectangle(a, a; Nx=2, Ny=2)
    positions = rotate_xy_to_xz(positions)
    N = length(positions)

    pols = AtomicArrays.polarizations_spherical(N)
    gam = [AtomicArrays.gammas(0.25)[m] for m=1:3, j=1:N]
    deltas = [0.0 for i = 1:N]

    coll = AtomicArrays.FourLevelAtomCollection(positions;
        deltas = deltas,
        polarizations = pols,
        gammas = gam
    )

    # Define a plane wave field in +y direction:
    amplitude = 0.0
    k_mod = 2π
    angle_k = [0.0, π/2]  # => +y direction
    polarisation = [1.0, 0.0im, 0.0]
    pos_0 = [0.0, 0.0, 0.0]

    field = AtomicArrays.field.EMField(amplitude, k_mod, angle_k, polarisation; position_0=pos_0)
    external_drive = AtomicArrays.field.rabi(field, AtomicArrays.field.plane, coll)

    B_f = 0.0
    w = [deltas[n]+B_f*m for m = -1:1, n = 1:N]
    Γ = AtomicArrays.interaction.GammaTensor_4level(coll)
    Ω = AtomicArrays.interaction.OmegaTensor_4level(coll)
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
    ψ0 = basisstate(b, [(i == 1) ? AtomicArrays.fourlevel_quantum.idx_e_plus : 
    AtomicArrays.fourlevel_quantum.idx_g for i = 1:N])
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

begin
    # Parameters
    N_qubits = 4 #number of qubits
    @cnumbers N M

    # Hilbertspace
    # h = FockSpace(:qubit)
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
    γ(i) = IndexedVariable(Symbol("\\gamma"), i)

    i = Index(h, :i, N, h)
    j = Index(h, :j, N, h)
    k = Index(h, :k, N, h)
    l = Index(h, :l, N, h)
    α = Index(h, :α, 3*N, h)
    β = Index(h, :β, 3*N, h)
    
    
    # Operators
    # a(m) = IndexedOperator(Destroy(h, :a), m)
    s_m1(i) = IndexedOperator(Transition(h,:J,:g,Symbol("\\sigma_-")), i)
    s_0(i) = IndexedOperator(Transition(h,:J,:g,:π), i)
    s_p1(i) = IndexedOperator(Transition(h,:J,:g,Symbol("\\sigma_+")), i)
end

begin
    # Hamiltonian

    H_a = ∑(w_m1(i)*s_m1(i)'*s_m1(i) + 
            w_0(i)*s_0(i)'*s_0(i) + 
            w_p1(i)*s_p1(i)'*s_p1(i), i)
    H_f = -∑(g_m1(i)*s_m1(i) + gc_m1(i)*s_m1(i)' + 
             g_0(i)*s_0(i) + gc_0(i)*s_0(i)' +
             g_p1(i)*s_p1(i) + gc_p1(i)*s_p1(i)', i)
    H_dd = ∑((Ω_m1_m1(i,j) - 0.5im*Γ_m1_m1(i,j))*s_m1(i)'*s_m1(j) + 
             (Ω_0_0(i,j) - 0.5im*Γ_0_0(i,j))*s_0(i)'*s_0(j) +
             (Ω_p1_p1(i,j) - 0.5im*Γ_p1_p1(i,j))*s_p1(i)'*s_p1(j) + 
             (Ω_m1_0(i,j) - 0.5im*Γ_m1_0(i,j))*s_m1(i)'*s_0(j) + 
             (Ω_m1_0(i,j) - 0.5im*Γ_m1_0(i,j))*s_0(i)'*s_m1(j) +
             (Ω_m1_p1(i,j) - 0.5im*Γ_m1_p1(i,j))*s_m1(i)'*s_p1(j) + 
             (Ω_m1_p1(i,j) - 0.5im*Γ_m1_p1(i,j))*s_p1(i)'*s_m1(j) +
             (Ω_0_p1(i,j) - 0.5im*Γ_0_p1(i,j))*s_0(i)'*s_p1(j) + 
             (Ω_0_p1(i,j) - 0.5im*Γ_0_p1(i,j))*s_p1(i)'*s_0(j), j, i)
    H_nh = ∑(-1im*γ(i)*(s_m1(i)'*s_m1(i) + s_0(i)'*s_0(i) + s_p1(i)'*s_p1(i)), i)
    H = H_a + H_f + H_dd# + H_nh
    
    # Jumps
    J = [s_m1(i), s_0(i), s_p1(i)]
    
    # Rates
    rates = [γ(i), γ(i), γ(i)]
end

begin
    # list of operators
    n_order = 1
    ops = (n_order > 1) ? [s_m1(k), s_m1(k)'*s_m1(l)] : [s_m1(k),s_0(k),s_p1(k),
        s_m1(k)'*s_m1(k), s_0(k)'*s_0(k), s_p1(k)'*s_p1(k),
        s_m1(k)'*s_0(k), s_m1(k)'*s_p1(k),s_0(k)'*s_p1(k)
    ]
    
    eqs = meanfield(ops, H, J; rates=rates, order=n_order, simplify=true)
    # eqs = meanfield(ops, H; order=n_order, simplify=true)
end

begin
    eqs_comp = complete(eqs) #automatically complete the system
end

# Build an ODESystem out of the MeanfieldEquations
begin
    me_comp = evaluate(eqs_comp; limits=(N=>N_qubits))
    @named sys = ODESystem(me_comp)
    print("Model is built")
end

begin
    # list of symbolic indexed parameters
    w_m1_i = [w_m1(i) for i=1:N_qubits]
    w_0_i = [w_0(i) for i=1:N_qubits]
    w_p1_i = [w_p1(i) for i=1:N_qubits]
    Γ_m1_0_ij = [Γ_m1_0(i,j) for i=1:N_qubits for j=1:N_qubits]
    Γ_m1_p1_ij = [Γ_m1_p1(i,j) for i=1:N_qubits for j=1:N_qubits]
    Γ_0_p1_ij = [Γ_0_p1(i,j) for i=1:N_qubits for j=1:N_qubits]
    Γ_m1_m1_ij = [Γ_m1_m1(i,j) for i=1:N_qubits for j=1:N_qubits]
    Γ_p1_p1_ij = [Γ_p1_p1(i,j) for i=1:N_qubits for j=1:N_qubits]
    Γ_0_0_ij = [Γ_0_0(i,j) for i=1:N_qubits for j=1:N_qubits]
    Ω_m1_0_ij = [Ω_m1_0(i,j) for i=1:N_qubits for j=1:N_qubits]
    Ω_m1_p1_ij = [Ω_m1_p1(i,j) for i=1:N_qubits for j=1:N_qubits]
    Ω_0_p1_ij = [Ω_0_p1(i,j) for i=1:N_qubits for j=1:N_qubits]
    Ω_m1_m1_ij = [Ω_m1_m1(i,j) for i=1:N_qubits for j=1:N_qubits]
    Ω_p1_p1_ij = [Ω_p1_p1(i,j) for i=1:N_qubits for j=1:N_qubits]
    Ω_0_0_ij = [Ω_0_0(i,j) for i=1:N_qubits for j=1:N_qubits]
    g_m1_i = [g_m1(i) for i=1:N_qubits]
    g_0_i = [g_0(i) for i=1:N_qubits]
    g_p1_i = [g_p1(i) for i=1:N_qubits]
    gc_m1_i = [gc_m1(i) for i=1:N_qubits]
    gc_0_i = [gc_0(i) for i=1:N_qubits]
    gc_p1_i = [gc_p1(i) for i=1:N_qubits]
    γ_i = [γ(i) for i=1:N_qubits]
end

begin
    # initial state
    u0 = zeros(ComplexF64, length(me_comp))
    u0[4*N_qubits+1:5*N_qubits] .= 1.0 + 0.0im

    # list of parameters
    ps = [w_m1_i; w_0_i; w_p1_i; 
        Γ_m1_0_ij; Γ_m1_p1_ij; Γ_0_p1_ij; 
        Γ_m1_m1_ij; Γ_p1_p1_ij; Γ_0_0_ij;
        Ω_m1_0_ij; Ω_m1_p1_ij; Ω_0_p1_ij; 
        Ω_m1_m1_ij; Ω_p1_p1_ij; Ω_0_0_ij;
        g_m1_i; g_0_i; g_p1_i; gc_m1_i; gc_0_i; gc_p1_i;
        γ_i
    ]
    pn = collect(Iterators.flatten([
        w_m1_, w_0_, w_p1_, 
        Γ_m1_0_, Γ_m1_p1_, Γ_0_p1_, Γ_m1_m1_, Γ_p1_p1_, Γ_0_0_,
        Ω_m1_0_, Ω_m1_p1_, Ω_0_p1_, Ω_m1_m1_, Ω_p1_p1_, Ω_0_0_,
        g_m1_, g_0_, g_p1_, conj.(g_m1_), conj.(g_0_), conj.(g_p1_),
        γ_
    ]))
    p0 = ps .=> pn
    tend = 200.
    tlist = range(0, tend, 200)
    
    prob = ODEProblem(sys,u0,(0.0,tend),p0)
    sol = solve(prob)
    # sol = solve(prob, Kvaerno5(autodiff=false), alg_hints=[:stiff], reltol=1e-8, abstol=1e-8, maxiters=1e7, saveat=tlist)
end
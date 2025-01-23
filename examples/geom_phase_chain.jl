# geom phase in atomic chain

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
    using CairoMakie
    using ProgressMeter
    # using GLMakie
    using AtomicArrays

    using NonlocalArrays
end

# aliases
begin
    const sigma_matrices = AtomicArrays.meanfield.sigma_matrices
    const mapexpect = AtomicArrays.meanfield.mapexpect
    const mapexpect_mpc = AtomicArrays.mpc.mapexpect
    const sigma_matrices_mpc = AtomicArrays.mpc.sigma_matrices
end

# constants
begin
    const EMField = field.EMField
    em_inc_function = AtomicArrays.field.plane  # or field.gauss
    NMAX = 200

    N_1 = 20
    N_2 = 20
    N = N_1 + N_2
    const PATH_FIGS, PATH_DATA = ["../Figs/", "../Data/"]
    METHOD = "meanfield"  # quantum, meanfield, mpc
end

k_vec(θ, φ, K=1.0) = K.*[sin(θ), cos(θ)*sin(φ), cos(θ)*cos(φ)]

# system's parameters
begin
    a = 0.3  # lattice constant
    a_1 = a  # lattice constant
    a_2 = a + 0.0  # lattice constant
    d = 0.4  # distance between arrays
    γ = 0.1  # decay rate of an individual atom
    # e_dipole = [0., 0, 1]  # dipole moment of atoms
    e_dipole = normalize.([AtomicArrays.field.vec_rotate([1.,0,0], pi/2, phi) 
                          for phi = range(-pi/2, pi/2, N)])  # dipole moment of atoms


    pos_1 = geometry.chain_dir(a_1, N_1; dir="z", 
                               pos_0=[-d/2, 0, - (N_1 - 1) * a_1 / 2])
    pos_2 = geometry.chain_dir(a_2, N_2; dir="z",
                               pos_0=[d/2, 0, - (N_2 - 1) * a_2 / 2])
    pos = vcat(pos_1, pos_2)
    Delt = 0.0  # detunings
    S = SpinCollection(pos, e_dipole; gammas=γ, deltas=Delt)

    # incident field
    E_ampl = 0.05 + 0im
    E_kvec = 2π
    E_w_0 = 0.5  # width of the gaussian profile
    E_pos0 = [0.0, 0.0, 0.0]
    E_polar = [1.0, 1im, 0.0im] #rand(ComplexF64, 3)
    E_angle = [1.0 * pi / 2, 0.0]  # {θ, φ}

    E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
                    position_0=E_pos0, waist_radius=E_w_0)
    # Field-spin interaction
    E_vec = [em_inc_function(S.spins[k].position, E_inc) for k = 1:N]
    Om_R = field.rabi(E_vec, S.polarizations)
end

# check incident field distribution
let
    x = range(-3.0, 3.0, NMAX)
    y = 0.0
    z = range(-5.0, 5.0, NMAX)
    x_xy = x
    y_xy = x
    z_xy = 0.0
    e_field = Matrix{ComplexF64}(undef, length(x), length(z))
    e_field_xy = Matrix{ComplexF64}(undef, length(x_xy), length(y_xy))
    for i in eachindex(x)
        for j in eachindex(z)
            e_field[i, j] = em_inc_function([x[i], y, z[j]], E_inc)[3]
            e_field_xy[i, j] = em_inc_function([x_xy[i], y_xy[j], z_xy],
                                               E_inc)[3]
        end
    end
    f = Figure(size=(700, 600), fontsize=20)
    ax = [Axis(f[i, j], aspect=DataAspect(); 
        xlabel="x",
        ylabel=i==1 ? "z" : "y",
    ) for i in 1:2, j in 1:2]
    hm1 = heatmap!(ax[1,1], x, z, real(e_field))
    scatter!(ax[1,1], [(p[1], p[3]) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    hm2 = heatmap!(ax[1,2], x, z, angle.(e_field),
                   colormap=:cyclic_tritanopic_cwrk_40_100_c20_n256)
    scatter!(ax[1,2], [(p[1], p[3]) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    hm3 = heatmap!(ax[2,1], x_xy, y_xy, real(e_field_xy))
    scatter!(ax[2,1], [(p[1], p[2]) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    hm4 = heatmap!(ax[2,2], x_xy, y_xy, angle.(e_field_xy),
                   colormap=:cyclic_tritanopic_cwrk_40_100_c20_n256)
    scatter!(ax[2,2], [(p[1], p[2]) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    Colorbar(f[1,3], hm1; label=L"E_0")
    Colorbar(f[1,4], hm2; label=L"\phi(E_0)")
    Colorbar(f[2,3], hm3; label=L"E_0")
    Colorbar(f[2,4], hm4; label=L"\phi(E_0)")
    f
end

# system Hamiltonian
if METHOD == "quantum"

    Γ, J = AtomicArrays.quantum.JumpOperators(S)
    Jdagger = [dagger(j) for j = J]
    spsm = [Jdagger[i] * J[j] for i = 1:N, j = 1:N]
    Jx, Jy, Jz = [J .+ Jdagger, 
                  1.0im*(J .- Jdagger),
                  Jdagger .* J .- J .* Jdagger]
    # Ω = AtomicArrays.interaction.OmegaMatrix(S)
    H = AtomicArrays.quantum.Hamiltonian(S) - sum(Om_R[j] * J[j] +
                                                            conj(Om_R[j]) * Jdagger[j]
                                                            for j = 1:N)

    H.data
    # Non-Hermitian Hamiltonian
    H_nh = H - 0.5im * sum(Γ[j,k] * Jdagger[j] * J[k] for j = 1:N, k = 1:N)
    # Liouvillian
    L = liouvillian(H, J; rates=Γ)
    w, v = eigenstates(dense(H))

    # Jump operators description
    J_s = NonlocalArrays.jump_op_source_mode(Γ, J)
end

# Dynamics
begin
    # Initial state (Bloch state)
    phi = 0.
    theta = pi/1.

    # Time evolution
    T = [0:1.0:1000;]
    if METHOD == "quantum"
        # Quantum: master equation
        sx_t = zeros(Float64, N, length(T))
        sy_t = zeros(Float64, N, length(T))
        sz_t = zeros(Float64, N, length(T))
        sm_t = zeros(ComplexF64, N, length(T))

        embed(op::Operator,i) = QuantumOptics.embed(AtomicArrays.quantum.basis(S), i, op)

        function fout(t, rho)
            j = findfirst(isequal(t), T)
            for i = 1:N
                sx_t[i, j] = real(expect(Jx[i], rho))
                sy_t[i, j] = real(expect(Jy[i], rho))
                sz_t[i, j] = real(expect(Jz[i], rho))
                sm_t[i, j] = expect(J[i], rho)
            end
            if t == T[end]
                return rho
            end
            return nothing
        end

        Ψ₀ = AtomicArrays.quantum.blochstate(phi,theta,N)
        ρ₀ = Ψ₀⊗dagger(Ψ₀)
        _, ρ_t = QuantumOptics.timeevolution.master_h(T, ρ₀, H, J; fout=fout, rates=Γ)
        ρ_ss_t = ρ_t[end]
        # TODO: make exception if ρ_ss cannot be computed
        ρ_ss = QuantumOptics.steadystate.iterative(H, J; rates=Γ)  # steady-state
        sm_mat = [expect(J[i], ρ_ss) for i = 1:N]
    elseif METHOD == "meanfield"
        state0 = AtomicArrays.meanfield.blochstate(phi, theta, N)
        tout, state_mf_t = AtomicArrays.meanfield.timeevolution_field(T, S, Om_R, state0)
        # Expectation values
        sx_t = [mapexpect(AtomicArrays.meanfield.sx, state_mf_t, i) for i=1:N]
        sy_t = [mapexpect(AtomicArrays.meanfield.sy, state_mf_t, i) for i=1:N]
        sz_t = [mapexpect(AtomicArrays.meanfield.sz, state_mf_t, i) for i=1:N]
        sm_t = 0.5*(sx_t - 1im.*sy_t)
        # make arrays
        sx_t = reduce(vcat, transpose.(sx_t))
        sy_t = reduce(vcat, transpose.(sy_t))
        sz_t = reduce(vcat, transpose.(sz_t))
        sm_t = reduce(vcat, transpose.(sm_t))
        t_ind = length(T)
        _, _, _, sm_mat, _ = sigma_matrices(state_mf_t, t_ind)
    end
end

# plot dynamics of Jx, Jy, Jz
let 
    f = Figure(size=(700, 700), fontsize=20)
    labels_y = [L"\sigma_x", L"\sigma_y", L"\sigma_z"]
    axs = [Axis(f[j, 1]; 
           xlabel=j == length(labels_y) ? L"\tau" : "", 
           ylabel=labels_y[j])
        for j in eachindex(labels_y)]
    for i in 1:N
        lines!(axs[1], T, sx_t[i, :], label="$i")
        lines!(axs[2], T, sy_t[i, :], label="$i")
        lines!(axs[3], T, sz_t[i, :], label="$i")
    end
    axislegend.(axs)
    f
end

# scattered field distribution
function intensity_xy(sm_mat; nmax=50, n_a=1, z_a=0.0)
    # X-Y view
    x = range(-n_a*a+pos[1][1], pos[end][1]+n_a*a, nmax)
    y = range(-n_a*a+pos[1][1], pos[end][1]+n_a*a, nmax)
    z = z_a*a
    I = zeros(length(x), length(y))
    E_tot = [Vector{ComplexF64}() for _ in 1:nmax, _ in 1:nmax]
    E_sc = [Vector{ComplexF64}() for _ in 1:nmax, _ in 1:nmax]
    lk = ReentrantLock()
    progress = Progress(length(x))
    Threads.@threads for i in eachindex(x)
        for j in eachindex(y)
            E_tot[i,j] = AtomicArrays.field.total_field(em_inc_function,
                                                        [x[i],y[j],z],
                                                        E_inc,
                                                        S, sm_mat)
            I[i,j] = norm(E_tot[i, j])^2 / abs(E_ampl)^2
            E_tot[i,j] = E_tot[i,j] / E_ampl
            E_sc[i,j] = AtomicArrays.field.scattered_field([x[i],y[j],z],
                                                           S, sm_mat)/E_ampl
        end
        next!(progress)
    end
    finish!(progress)
    return x, y, I, E_tot, E_sc
end

function intensity_xz(sm_mat; nmax=50, n_a=1, y_a=0.0)
    # X-Z view
    x = range(-n_a*a+pos[1][1], pos[end][1]+n_a*a, nmax)
    y = y_a*a
    z = range(-n_a*a+pos[1][3], pos[end][3]+n_a*a, nmax)
    I = zeros(length(x), length(z))
    E_tot = [Vector{ComplexF64}() for _ in 1:nmax, _ in 1:nmax]
    E_sc = [Vector{ComplexF64}() for _ in 1:nmax, _ in 1:nmax]
    lk = ReentrantLock()
    progress = Progress(length(x))
    Threads.@threads for i in eachindex(x)
        for j in eachindex(z)
            E_tot[i,j] = AtomicArrays.field.total_field(em_inc_function,
                                                        [x[i],y,z[j]],
                                                        E_inc,
                                                        S, sm_mat)
            I[i,j] = norm(E_tot[i, j])^2 / abs(E_ampl)^2
            E_tot[i,j] = E_tot[i,j] / E_ampl
            E_sc[i,j] = AtomicArrays.field.scattered_field([x[i],y,z[j]],
                                                           S, sm_mat)/E_ampl
        end
        next!(progress)
    end
    finish!(progress)
    return x, z, I, E_tot, E_sc
end

begin
    nmax = 200
    n_a = 1
    z_a = 0.0*(N_1 / 2 + 1.0)
    y_a = 0.5
    x_xy, y_xy, I_xy, E_tot_xy, E_sc_xy = intensity_xy(sm_mat; nmax=nmax,
                                                       n_a=n_a, z_a=z_a)
    x_xz, z_xz, I_xz, E_tot_xz, E_sc_xz = intensity_xz(sm_mat; nmax=nmax,
                                                       n_a=n_a, y_a=y_a)
end

# phase
let
    # data
    Z_x_xy = [angle(E_tot_xy[i, j][1]) for i = 1:nmax, j = 1:nmax]
    Z_y_xy = [angle(E_tot_xy[i, j][2]) for i = 1:nmax, j = 1:nmax]
    Z_z_xy = [angle(E_tot_xy[i, j][3]) for i = 1:nmax, j = 1:nmax]
    Z_x_xz = [angle(E_tot_xz[i, j][1]) for i = 1:nmax, j = 1:nmax]
    Z_y_xz = [angle(E_tot_xz[i, j][2]) for i = 1:nmax, j = 1:nmax]
    Z_z_xz = [angle(E_tot_xz[i, j][3]) for i = 1:nmax, j = 1:nmax]

    # figure
    f = Figure(size=(1000, 1200), fontsize=20)
    axs = [Axis(f[i, j]; aspect=1,#DataAspect(),
           xlabel= i == 4 ? "x/a" : "", 
           ylabel=j == 1 ? "y/a" : "z/a")
        for i in 1:4, j in 1:2]
    cmap = :cyclic_tritanopic_cwrk_40_100_c20_n256	
    
    p11 = heatmap!(axs[1, 1], x_xy/a, y_xy/a, I_xy; colorscale=identity)
    scatter!(axs[1, 1], [(p[1]/a, p[2]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    lines!(axs[1, 1], x_xy / a, y_a * ones(nmax);
           color=:white, linewidth=1.5, linestyle=:dash)
    p12 = heatmap!(axs[1, 2], x_xz/a, z_xz/a, I_xz; colorscale=identity)
    scatter!(axs[1, 2], [(p[1]/a, p[3]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    lines!(axs[1, 2], x_xz / a, z_a * ones(nmax);
           color=:white, linewidth=1.5, linestyle=:dash)
    Colorbar(f[1,3], p11)
    Colorbar(f[1,4], p12; label=L"I")
    p21 = heatmap!(axs[2, 1], x_xy/a, y_xy/a, Z_x_xy; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[2, 1], [(p[1]/a, p[2]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    p22 = heatmap!(axs[2, 2], x_xz/a, z_xz/a, Z_x_xz; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[2, 2], [(p[1]/a, p[3]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    Colorbar(f[2,3], p21)
    Colorbar(f[2,4], p22; label=L"\phi(E_x)")
    p31 = heatmap!(axs[3, 1], x_xy/a, y_xy/a, Z_y_xy; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[3, 1], [(p[1]/a, p[2]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    p32 = heatmap!(axs[3, 2], x_xz/a, z_xz/a, Z_y_xz; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[3, 2], [(p[1]/a, p[3]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    Colorbar(f[3,3], p31)
    Colorbar(f[3,4], p32; label=L"\phi(E_y)")
    p41 = heatmap!(axs[4, 1], x_xy/a, y_xy/a, Z_z_xy; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[4, 1], [(p[1]/a, p[2]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    p42 = heatmap!(axs[4, 2], x_xz/a, z_xz/a, Z_z_xz; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[4, 2], [(p[1]/a, p[3]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    Colorbar(f[4,3], p41)
    Colorbar(f[4,4], p42; label=L"\phi(E_z)")
    f
end

# real total electric field
let
    # data
    Z_x_xy = [real(E_tot_xy[i, j][1]) for i = 1:nmax, j = 1:nmax]
    Z_y_xy = [real(E_tot_xy[i, j][2]) for i = 1:nmax, j = 1:nmax]
    Z_z_xy = [real(E_tot_xy[i, j][3]) for i = 1:nmax, j = 1:nmax]
    Z_x_xz = [real(E_tot_xz[i, j][1]) for i = 1:nmax, j = 1:nmax]
    Z_y_xz = [real(E_tot_xz[i, j][2]) for i = 1:nmax, j = 1:nmax]
    Z_z_xz = [real(E_tot_xz[i, j][3]) for i = 1:nmax, j = 1:nmax]

    # figure
    f = Figure(size=(1000, 1200), fontsize=20)
    axs = [Axis(f[i, j]; aspect=1,#DataAspect(),
           xlabel= i == 4 ? "x/a" : "", 
           ylabel=j == 1 ? "y/a" : "z/a")
        for i in 1:4, j in 1:2]
    cmap = :viridis
    p11 = heatmap!(axs[1, 1], x_xy/a, y_xy/a, I_xy; colorscale=identity)
    scatter!(axs[1, 1], [(p[1]/a, p[2]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    lines!(axs[1, 1], x_xy / a, y_a * ones(nmax);
           color=:white, linewidth=1.5, linestyle=:dash)
    p12 = heatmap!(axs[1, 2], x_xz/a, z_xz/a, I_xz; colorscale=identity)
    scatter!(axs[1, 2], [(p[1]/a, p[3]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    lines!(axs[1, 2], x_xz / a, z_a * ones(nmax);
           color=:white, linewidth=1.5, linestyle=:dash)
    Colorbar(f[1,3], p11)
    Colorbar(f[1,4], p12; label=L"I")
    p21 = heatmap!(axs[2, 1], x_xy/a, y_xy/a, Z_x_xy; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[2, 1], [(p[1]/a, p[2]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    p22 = heatmap!(axs[2, 2], x_xz/a, z_xz/a, Z_x_xz; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[2, 2], [(p[1]/a, p[3]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    Colorbar(f[2,3], p21)
    Colorbar(f[2,4], p22; label=L"\mathrm{Re}(E_x)")
    p31 = heatmap!(axs[3, 1], x_xy/a, y_xy/a, Z_y_xy; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[3, 1], [(p[1]/a, p[2]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    p32 = heatmap!(axs[3, 2], x_xz/a, z_xz/a, Z_y_xz; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[3, 2], [(p[1]/a, p[3]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    Colorbar(f[3,3], p31)
    Colorbar(f[3,4], p32; label=L"\mathrm{Re}(E_y)")
    p41 = heatmap!(axs[4, 1], x_xy/a, y_xy/a, Z_z_xy; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[4, 1], [(p[1]/a, p[2]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    p42 = heatmap!(axs[4, 2], x_xz/a, z_xz/a, Z_z_xz; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[4, 2], [(p[1]/a, p[3]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    Colorbar(f[4,3], p41)
    Colorbar(f[4,4], p42; label=L"\mathrm{Re}(E_z)")
    f
end

# real scattered electric field
let
    # data
    Z_x_xy = [real(E_sc_xy[i, j][1]) for i = 1:nmax, j = 1:nmax]
    Z_y_xy = [real(E_sc_xy[i, j][2]) for i = 1:nmax, j = 1:nmax]
    Z_z_xy = [real(E_sc_xy[i, j][3]) for i = 1:nmax, j = 1:nmax]
    Z_x_xz = [real(E_sc_xz[i, j][1]) for i = 1:nmax, j = 1:nmax]
    Z_y_xz = [real(E_sc_xz[i, j][2]) for i = 1:nmax, j = 1:nmax]
    Z_z_xz = [real(E_sc_xz[i, j][3]) for i = 1:nmax, j = 1:nmax]

    # figure
    f = Figure(size=(1000, 1200), fontsize=20)
    axs = [Axis(f[i, j]; aspect=1,#DataAspect(),
           xlabel= i == 4 ? "x/a" : "", 
           ylabel=j == 1 ? "y/a" : "z/a")
        for i in 1:4, j in 1:2]
    cmap = :viridis
    p11 = heatmap!(axs[1, 1], x_xy/a, y_xy/a, I_xy; colorscale=identity)
    scatter!(axs[1, 1], [(p[1]/a, p[2]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    lines!(axs[1, 1], x_xy / a, y_a * ones(nmax);
           color=:white, linewidth=1.5, linestyle=:dash)
    p12 = heatmap!(axs[1, 2], x_xz/a, z_xz/a, I_xz; colorscale=identity)
    scatter!(axs[1, 2], [(p[1]/a, p[3]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    lines!(axs[1, 2], x_xz / a, z_a * ones(nmax);
           color=:white, linewidth=1.5, linestyle=:dash)
    Colorbar(f[1,3], p11)
    Colorbar(f[1,4], p12; label=L"I")
    p21 = heatmap!(axs[2, 1], x_xy/a, y_xy/a, Z_x_xy; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[2, 1], [(p[1]/a, p[2]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    p22 = heatmap!(axs[2, 2], x_xz/a, z_xz/a, Z_x_xz; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[2, 2], [(p[1]/a, p[3]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    Colorbar(f[2,3], p21)
    Colorbar(f[2,4], p22; label=L"\mathrm{Re}(E_x)")
    p31 = heatmap!(axs[3, 1], x_xy/a, y_xy/a, Z_y_xy; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[3, 1], [(p[1]/a, p[2]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    p32 = heatmap!(axs[3, 2], x_xz/a, z_xz/a, Z_y_xz; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[3, 2], [(p[1]/a, p[3]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    Colorbar(f[3,3], p31)
    Colorbar(f[3,4], p32; label=L"\mathrm{Re}(E_y)")
    p41 = heatmap!(axs[4, 1], x_xy/a, y_xy/a, Z_z_xy; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[4, 1], [(p[1]/a, p[2]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    p42 = heatmap!(axs[4, 2], x_xz/a, z_xz/a, Z_z_xz; colorscale=identity, 
                   colormap=cmap)
    scatter!(axs[4, 2], [(p[1]/a, p[3]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    Colorbar(f[4,3], p41)
    Colorbar(f[4,4], p42; label=L"\mathrm{Re}(E_z)")
    f
end

# animate the fields plot
let
    nmax = 200
    n_a = 20
    z_a = 5.5
    y_a = 3.0
    function compute_intensity(sm_mat, nmax, n_a, z_a, y_a)
        x_xy, y_xy, I_xy, _, _ = intensity_xy(sm_mat; nmax=nmax, n_a=n_a, z_a=z_a)
        x_xz, z_xz, I_xz, _, _ = intensity_xz(sm_mat; nmax=nmax, n_a=n_a, y_a=y_a)
        return x_xy, y_xy, I_xy, x_xz, z_xz, I_xz
    end

    sm_mat = sm_t[:, 1]
    x_xy, y_xy, I_xy, x_xz, z_xz, I_xz = compute_intensity(sm_mat, nmax,
                                                        n_a, z_a, y_a)

    # Create the figure and axes
    f = Figure(size=(1000, 500), fontsize=20)
    axs = [Axis(f[1, j]; aspect=DataAspect(),
        xlabel="x/a", 
        ylabel=j == 1 ? "y/a" : "z/a")
        for j in 1:2]
    p1 = heatmap!(axs[1], x_xy/a, y_xy/a, I_xy; 
                colorscale=identity, colorrange=(0.95,1.05))
    p2 = heatmap!(axs[2], x_xz/a, z_xz/a, I_xz;
                colorscale=identity, colorrange=(0.95,1.05))
    scatter1 = scatter!(axs[1], [(p[1]/a, p[2]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    scatter2 = scatter!(axs[2], [(p[1]/a, p[3]/a) for p in pos], color=:white, strokecolor=:black, strokewidth=1)
    lines1 = lines!(axs[1], x_xy / a, y_a * ones(nmax); color=:white, linewidth=1.5, linestyle=:dash)
    lines2 = lines!(axs[2], x_xz / a, z_a * ones(nmax); color=:white, linewidth=1.5, linestyle=:dash)

    Colorbar(f[1,3], p1)
    Colorbar(f[1,4], p2; label=L"I")

    # Animation loop
    framerate = 30
    timestamps = collect(eachindex(T))

    record(f, PATH_FIGS*"field_intensity_animation.mp4", timestamps;
        framerate=framerate) do t_index
        sm_mat = sm_t[:, t_index]
        x_xy, y_xy, I_xy, x_xz, z_xz, I_xz = compute_intensity(sm_mat, nmax,
                                                            n_a, z_a, y_a)
        # Update heatmaps
        p1[3] = I_xy
        p2[3] = I_xz
        # Update scatter plots and lines
        scatter1[1] = [(p[1] / a, p[2] / a) for p in pos]
        scatter2[1] = [(p[1] / a, p[3] / a) for p in pos]
        lines1[1] = x_xy / a
        lines1[2] = y_a * ones(nmax)
        lines2[1] = x_xz / a
        lines2[2] = z_a * ones(nmax)
    end
end
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
    using Base.Threads
    using LinearAlgebra
    using QuantumOptics, QuantumCumulants
    using ModelingToolkit
    using DifferentialEquations
    using CairoMakie, GLMakie
    using AtomicArrays

    using NonlocalArrays
end

PATH_DATA, PATH_FIGS = "../Data", "../Figs/"

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
    a = 0.2; Nx = 20; Ny = 20;
    positions = AtomicArrays.geometry.rectangle(a, a; Nx=Nx, Ny=Ny,
                                position_0=[(-Nx/2+0.5)*a,(-Ny/2+0.5)*a,0.0])
    # positions = rotate_xy_to_xz(positions)
    N = length(positions)

    pols = AtomicArrays.polarizations_spherical(N)
    gam = [AtomicArrays.gammas(0.25)[m] for m=1:3, j=1:N]
    deltas = [(i == 1) ? 0.0 : 0.2 for i = 1:N]

    coll = AtomicArrays.FourLevelAtomCollection(positions;
        deltas = deltas,
        polarizations = pols,
        gammas = gam
    )

    POLARIZATION = "R"
    # Define a plane wave field in +y direction:
    amplitude = 0.02
    k_mod = 2π
    # angle_k = [0.0, π/2]  # => +y direction
    angle_k = [1*π/6.0, 0.0]  # => if ϕ = 0, θ rotates in x-z plane
    if POLARIZATION == "R"
        polarisation = [1.0, -1.0im, 0.0]  # -i is R, +i is L
    elseif POLARIZATION == "L"
        polarisation = [1.0, +1.0im, 0.0]  # -i is R, +i is L
    end
    pos_0 = [0.0, 0.0, 0.0]
    waist_radius = 0.3*a*sqrt(Nx*Ny)

    field = AtomicArrays.field.EMField(amplitude, k_mod, angle_k, polarisation; position_0=pos_0, waist_radius=waist_radius)
    field_function = AtomicArrays.field.gauss
    OmR = AtomicArrays.field.rabi(field, field_function, coll)

    B_z = 0.2
    w = [deltas[n]+B_z*m for m = -1:1, n = 1:N]
    Γ = AtomicArrays.interaction.GammaTensor_4level(coll)
    Ω = AtomicArrays.interaction.OmegaTensor_4level(coll)
    print("System's built")
end

# Time dynamics
begin
    u0 = AtomicArrays.fourlevel_meanfield.ProductState(N)
    for n = 1:1
        reshape(view(u0.data, (3*N)+1:12*N), (3, 3, N))[3,3,n] = 0.0
    end
    tspan = [0.0:0.1:400.0;]
    tout, state_mf_t = AtomicArrays.fourlevel_meanfield.timeevolution(tspan,
                                            coll, OmR, B_z, u0);
    print("Time dynamics' computed")
end

# Average values
begin
    T = length(tout)
    sigmas_m, _ = AtomicArrays.fourlevel_meanfield.sigma_matrices(state_mf_t, T)

    # Preallocate the expectation arrays.
    # p_e_* are real expectation values, s_e_* might be complex.
    p_e_minus_mf = zeros(Float64, T, N)
    p_e_0_mf     = zeros(Float64, T, N)
    p_e_plus_mf  = zeros(Float64, T, N)

    s_e_minus_mf = zeros(ComplexF64, T, N)
    s_e_0_mf     = zeros(ComplexF64, T, N)
    s_e_plus_mf  = zeros(ComplexF64, T, N)

    # Loop over each time step and atom.
    # @inbounds for t in 1:T
    Threads.@threads for t in 1:T
        # Compute the operators only once per time step.
        state = state_mf_t[t]
        s_val   = AtomicArrays.fourlevel_meanfield.sm(state)   # shape (3, N)
        smm_val = AtomicArrays.fourlevel_meanfield.smm(state)  # shape (3, 3, N)
        for n in 1:N
            p_e_minus_mf[t, n] = real(smm_val[1, 1, n])
            p_e_0_mf[t, n]     = real(smm_val[2, 2, n])
            p_e_plus_mf[t, n]  = real(smm_val[3, 3, n])
            s_e_minus_mf[t, n] = s_val[1, n]
            s_e_0_mf[t, n]     = s_val[2, n]
            s_e_plus_mf[t, n]  = s_val[3, n]
        end
    end
end

# Scattered and total field
begin
    # Options: "xy", "xz", "yz"
    plane = "xz"   # change this to "xz" or "yz" as desired

    # === Define grid parameters ===
    factor_scale = 3.5
    grid_min, grid_max, grid_step = -Nx*a/2*factor_scale, Nx*a/2*factor_scale, 0.02*factor_scale

    # Depending on the plane, choose the two varying coordinates and fix the third:
    if plane == "xy"
        coord1_range = grid_min:grid_step:grid_max  # x
        coord2_range = grid_min:grid_step:grid_max  # y
        fixed_value = Nx*a/2 + 0.2                           # fixed z
        fixed_index = 3
        label1, label2, fixed_label = "x", "y", "z"
    elseif plane == "xz"
        coord1_range = grid_min:grid_step:grid_max  # x
        coord2_range = grid_min:grid_step:grid_max  # z
        fixed_value = (mod(Nx, 2) == 0) ? 0.0 : a/2          # fixed y
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
            E = AtomicArrays.field.scattered_field(r, coll, sigmas_m)
            E_t = field_function(r, field) + E
            I_t = (norm(E_t)^2 / abs(field.amplitude)^2)
            
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

    save(PATH_FIGS*"4level_E_t_a"*string(a)*"_"*POLARIZATION*"_theta"*string(round(field.angle_k[1], digits=2))*"_N"*string(N)*".pdf", fig, px_per_unit=4)
    fig
end

let
    CairoMakie.activate!()
    fig = Figure(size = (800, 700))

    nlevels = 50

    # --- Row 1: x-component (Eₓ) ---
    ax1 = Axis(fig[1, 1], title = "Intensity, "*POLARIZATION*", E₀ = "*string(round(field.amplitude, digits=3)), xlabel = label1, ylabel = label2,
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

    save(PATH_FIGS*"4level_I_t_a"*string(a)*"_"*POLARIZATION*"_theta"*string(round(field.angle_k[1], digits=2))*"_N"*string(N)*".pdf", fig, px_per_unit=4)
    fig
end


let 
    GLMakie.activate!()
    # Create a meshgrid for plotting
    X, Z = [x for x in coord1_range, z in coord2_range], [z for x in coord1_range, z in coord2_range]
    Y = fill(fixed_value, size(X))  # Fixed y-coordinate plane

    fig = Figure(size=(1000,800))

    zoom_fac = 1.3

    ax = Axis3(fig[1,1],
    xlabel="x", ylabel="y", zlabel="z",
    titlesize = 24, xlabelsize = 20, ylabelsize = 20,
    xticklabelsize = 18, yticklabelsize = 18,
    azimuth = -pi/3,              # optimal horizontal angle
    elevation = pi/10,             # optimal tilt downwards
    perspectiveness = 0.8,
    aspect = :data,
    )

    surface!(ax, X, Y, Z;   # explicit horizontal y-plane at fixed y
        color = I_total, colormap = :plasma, 
        transparency = true, 
        shading = NoShading,
        alpha = 0.8)

    scatter!(ax, atom_x, atom_y, atom_z; 
        color=:red, markersize=12, strokewidth=1, strokecolor=:black,
        
        )
    ax.viewmode = :fitzoom

    Colorbar(fig[1,2], limits=(minimum(I_total), maximum(I_total)),
        colormap=:plasma, label="I / |E₀|²",
        labelsize = 20, ticklabelsize = 18,
        width = 15, height = Relative(1))
    
    hidedecorations!.(ax; grid = false)
    hidespines!(ax)

    # GLMakie ONLY: Set the camera's up vector to align with the y-axis
    upvector = Vec3f(0, 1, 0)
    # Define the camera's position and look-at point
    eyeposition = zoom_fac * Vec3f(-1, 1, -1)  # Position the camera
    lookat = Vec3f(0, 0, 0)       # Look towards the origin
    # Apply the camera settings
    cam = cameracontrols(ax.scene)
    cam3d!(ax.scene; clipping_mode = :static)
    update_cam!(ax.scene, eyeposition, lookat, upvector)

    save(PATH_FIGS*"4level_I_t_a"*string(a)*"_"*POLARIZATION*"_theta"*string(round(field.angle_k[1], digits=2))*"_N"*string(N)*"_3D.png", fig, px_per_unit=4)

    fig
end


let 
    f = Figure(size=(1000, 300))
    ax1 = Axis(f[1, 1], title="m = -1", xlabel="t", ylabel="Population")
    ax2 = Axis(f[1, 2], title="m = 0", xlabel="t", ylabel="Population")
    ax3 = Axis(f[1, 3], title="m = +1", xlabel="t", ylabel="Population")
    # Define colors for different atoms
    colors = cgrad(:reds, N, categorical=true)
    # Plot sublevels
    for n in 1:N
        lines!(ax1, tout, p_e_minus_mf[:, n], label="Atom $n, mf", color=colors[n], linewidth=2)
        lines!(ax2, tout, p_e_0_mf[:, n], label="Atom $n, mf", color=colors[n], linewidth=2)
        lines!(ax3, tout, p_e_plus_mf[:, n], label="Atom $n, mf", color=colors[n], linewidth=2)
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

    # Define colors for different atoms
    colors = cgrad(:reds, N, categorical=true)
    # Plot sublevels
    for n in 1:N
        lines!(ax1, tout, imag(s_e_minus_mf[:, n]), label="Atom $n, mf", color=colors[n], linewidth=2)
        lines!(ax2, tout, imag(s_e_0_mf[:, n]), label="Atom $n, mf", color=colors[n], linewidth=2)
        lines!(ax3, tout, imag(s_e_plus_mf[:, n]), label="Atom $n, mf", color=colors[n], linewidth=2)
    end
    # Add legend
    Legend(f[1, 4], ax1, "Atoms", framevisible=false)
    f
end

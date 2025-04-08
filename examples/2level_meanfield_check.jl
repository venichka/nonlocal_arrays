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

begin
    a = 0.6
    positions = AtomicArrays.geometry.rectangle(a, a; Nx=10, Ny=10)
    positions = rotate_xy_to_xz(positions)
    N = length(positions)

    pols = [1.0, 1.0im, 0]
    gam = 0.25/3
    deltas = [0.05 for i = 1:N]

    coll = AtomicArrays.SpinCollection(positions, pols, gam; deltas=deltas)

    # Define a plane wave field in +y direction:
    amplitude = 0.001
    k_mod = 2π
    angle_k = [0.0, π/2]  # => +y direction
    polarisation = [1.0, 1.0im, 0.0]
    pos_0 = [0.0, 0.0, 0.0]

    field = AtomicArrays.field.EMField(amplitude, k_mod, angle_k, polarisation; position_0=pos_0)
    external_drive = AtomicArrays.field.rabi(field, AtomicArrays.field.plane, coll)
end

begin
    u0 = AtomicArrays.meanfield.blochstate(0.0, pi, N)
    tspan = [0.0:0.1:200.0;]
    t, u = AtomicArrays.meanfield.timeevolution_field(tspan, coll, external_drive,
                                                    u0)
    sz_mf = [AtomicArrays.meanfield.sz(u[i]) for i in eachindex(t)]
end


let 
    f = Figure(size=(500, 300))
    ax1 = Axis(f[1, 1], title="2-level", xlabel="t", ylabel="Populations")
    # Define colors for different atoms
    colors_mf = cgrad(:viridis, N, categorical=true)
    # Plot sublevels
    for n in 1:N
        lines!(ax1, t, [(sz_mf[i][n] .+ 1) ./ 2 for i in eachindex(t)], label="Atom $n", color=colors_mf[n], linewidth=2)
    end
    # Add legend
    Legend(f[1, 2], ax1, "Atoms", framevisible=false)
    f
end
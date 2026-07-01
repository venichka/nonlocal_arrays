"""
Fig 3(a) option 3: Numerical field intensity maps from finite arrays.
Three regime points: C<0.5, C=0.9, C>0.99.
For each: Gaussian σ⁺ beam at the σ⁺ resonance frequency.
Shows |E_total|² in the x-z plane (y=0 slice).
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using NonlocalArrays
using AtomicArrays
using AtomicArrays: interaction, field
using StaticArrays, LinearAlgebra, DelimitedFiles

outdir = joinpath(@__DIR__, "..", "..", "Data")
mkpath(outdir)

γ₀ = 0.25; γ_m = γ₀/3; k₀ = 2π
Nside = 20

# Three regime points (a/λ, Δ_B/Γ₀)
regimes = [
    (a=0.45, DB=0.15, label="C_low"),
    (a=0.65, DB=0.75, label="C_090"),
    (a=0.80, DB=1.50, label="C_099"),
]

# Field evaluation grid
Nx_grid = 120; Nz_grid = 180
x_range = (-8.0, 8.0)
z_range = (-20.0, 15.0)
x_pts = range(x_range[1], x_range[2], length=Nx_grid)
z_pts = range(z_range[1], z_range[2], length=Nz_grid)

for reg in regimes
    a = reg.a
    ΔB_Gamma0 = reg.DB
    ΔB = ΔB_Gamma0 * γ_m  # Zeeman in γ_m units
    gt = 3 / (4π * a^2)
    x_val = 2 * ΔB_Gamma0 / gt
    C_val = 2 * x_val^2 / (2 * x_val^2 + 1)
    println("\n===== $(reg.label): a/λ=$(a), Δ_B/Γ₀=$(ΔB_Gamma0), C=$(round(C_val, digits=3)) =====")

    # Build array with delta_val=0 (detuning added manually to H_eff)
    # Use Gaussian beam for incident field
    waist = 3.0 * a * Nside / 4  # beam waist ~ 3/4 of array width
    coll, E, ff, OmR = build_fourlevel_system(;
        a=a, Nx=Nside, Ny=Nside, delta_val=0.0, gamma=γ₀,
        POL="R", amplitude=0.01,  # POL="R" = paper σ⁻ (show transmission at σ⁺ resonance)
        field_func=AtomicArrays.field.gauss,
        angle_k=[0.0, 0.0])

    N = Nside^2

    # Build H_eff
    println("  Building H_eff...")
    Ω_mat = interaction.OmegaMatrix_4level(coll)
    Γ_mat = interaction.GammaMatrix_4level(coll)
    H_eff = Ω_mat .- 0.5im .* Γ_mat

    # Add Zeeman
    for n in 1:N, m in 1:3
        H_eff[3*(n-1)+m, 3*(n-1)+m] += [-1, 0, 1][m] * ΔB
    end

    # Add detuning to place drive at σ⁺ resonance (δ ≈ -ΔB in γ_m units)
    # The σ⁺ resonance for the m=+1 level: eigenvalue real part = 0
    # when δ + ΔB + J̃ = 0 → δ = -(ΔB + J̃)
    # Approximate: δ ≈ -ΔB (neglecting J̃ for the field map)
    δ_res = -ΔB
    for n in 1:N, m in 1:3
        H_eff[3*(n-1)+m, 3*(n-1)+m] += δ_res
    end

    # Drive from Rabi frequencies (Gaussian beam profile)
    drive = zeros(ComplexF64, 3*N)
    for n in 1:N, m in 1:3
        drive[3*(n-1)+m] = OmR[m, n]
    end

    # Solve steady state
    println("  Solving linear system...")
    s = H_eff \ drive

    # Extract coherences
    σ = zeros(ComplexF64, 3, N)
    for n in 1:N, m in 1:3
        σ[m, n] = s[3*(n-1)+m]
    end

    d_sph = AtomicArrays.polarizations_spherical()

    # Compute total field on the grid
    println("  Computing field on $(Nx_grid)×$(Nz_grid) grid...")
    intensity = zeros(Nz_grid, Nx_grid)

    for (ix, xv) in enumerate(x_pts)
        for (iz, zv) in enumerate(z_pts)
            r_pt = [xv, 0.0, zv]

            # Scattered field from all atoms
            E_sc = zeros(ComplexF64, 3)
            for n in 1:N
                pos = coll.atoms[n].position
                dr = r_pt - pos
                dr_norm = norm(dr)
                if dr_norm < 0.05  # skip near-field singularity
                    continue
                end
                rhat = dr / dr_norm
                kr = k₀ * dr_norm

                # Dyadic Green tensor
                g_scalar = exp(im * kr) / (4π * dr_norm)
                A_coeff = (1 + im/kr - 1/kr^2)
                B_coeff = (-1 - 3im/kr + 3/kr^2)
                I3 = [1.0 0 0; 0 1.0 0; 0 0 1.0]
                G_dyad = g_scalar * (A_coeff * I3 + B_coeff * (rhat * rhat'))

                # Sum over m-channels: p_m = d_m * σ_m^(n)
                for m in 1:3
                    if abs(σ[m, n]) < 1e-15
                        continue
                    end
                    μ = d_sph[m, :]
                    E_sc .+= k₀^2 * G_dyad * (μ * σ[m, n])
                end
            end

            # Incident field (Gaussian beam)
            E_inc = AtomicArrays.field.gauss(r_pt, E)

            # Total
            E_tot = E_inc + E_sc
            intensity[iz, ix] = real(E_tot' * E_tot)
        end
    end

    # Save intensity
    fname = joinpath(outdir, "fig3a_field_$(reg.label).csv")
    open(fname, "w") do io
        write(io, "# Nx=$Nx_grid Nz=$Nz_grid x_range=$(x_range) z_range=$(z_range)\n")
        write(io, "# a=$(a) DB=$(ΔB_Gamma0) C=$(round(C_val,digits=3)) Nside=$Nside delta_res=$(δ_res) waist=$(waist)\n")
        writedlm(io, intensity)
    end

    # Save atom positions
    fname_pos = joinpath(outdir, "fig3a_atoms_$(reg.label).csv")
    open(fname_pos, "w") do io
        write(io, "# x_pos  y_pos\n")
        for n in 1:N
            pos = coll.atoms[n].position
            writedlm(io, [pos[1] pos[2]])
        end
    end
    println("  Saved: $(reg.label)")
end

println("\nDone.")

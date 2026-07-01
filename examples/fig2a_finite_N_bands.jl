"""
Fig 2(a) comparison: finite-N Bloch bands from H_eff vs Python infinite-array.
Computes bands via bloch_3x3_H for N = 10, 20, 30 and saves CSV for overlay.

Parameters: a/λ = 0.7, Δ_B/Γ₀ = 0.5
Path: M → Γ → X → M  (matching Python convention)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using NonlocalArrays
using AtomicArrays
using AtomicArrays: interaction
using StaticArrays
using LinearAlgebra
using DelimitedFiles

# =====================================================================
#  Parameters
# =====================================================================
a_over_lambda = 0.55
a = a_over_lambda           # λ₀ = 1
γ₀ = 0.25                   # total single-atom decay rate
γ_m = γ₀ / 3.0             # per-channel decay = natural unit of H_eff
DeltaB_over_gamma = 0.5

# Zeeman shift in the SAME units as H_eff (γ_m = γ₀/3)
ΔB = DeltaB_over_gamma * γ_m

outdir = joinpath(@__DIR__, "..", "..", "Data")
mkpath(outdir)

# =====================================================================
#  Loop over array sizes  (N = total atoms: 9, 100, 400)
# =====================================================================
for Nside in [10, 20, 30]
    println("\n========== N = $(Nside)×$(Nside) ==========")
    Nx, Ny = Nside, Nside
    N = Nx * Ny

    # Build atomic collection
    coll, _, _, _ = build_fourlevel_system(;
        a=a, Nx=Nx, Ny=Ny, delta_val=0.0, gamma=γ₀,
        POL="R", amplitude=0.01, angle_k=[0.0, 0.0])

    # Get Ω and Γ matrices (3N × 3N)
    println("  Building Ω matrix...")
    Ω_mat = interaction.OmegaMatrix_4level(coll)
    println("  Building Γ matrix...")
    Γ_mat = interaction.GammaMatrix_4level(coll)

    # Form H_eff = Ω - i·Γ/2 + Zeeman diagonal
    H_eff = Ω_mat .- 0.5im .* Γ_mat

    # Add Zeeman shifts: for each atom n, m-channels are at indices 3(n-1)+1,2,3
    # corresponding to m = -1, 0, +1.
    # ΔB is in units of γ₀ = Γ₀. The H_eff matrix has natural units of
    # γ_m = γ₀/3 (from the interaction functions). To make the Zeeman term
    # consistent, we add it in the same raw units (γ₀) — no rescaling needed,
    # since we'll handle units at the output stage.
    zeeman_shifts = [-ΔB, 0.0, +ΔB]
    for n in 1:N
        for m in 1:3
            idx = 3*(n-1) + m
            H_eff[idx, idx] += zeeman_shifts[m]
        end
    end

    # Extract 2D positions as Vector{SVector{2,Float64}}
    r = [SVector{2,Float64}(coll.atoms[n].position[1], coll.atoms[n].position[2])
         for n in 1:N]

    # Compute bands along M → Γ → X → M (matching Python path)
    println("  Computing bands...")
    Nk = 200
    knodes_MGXM = [(π/a, π/a), (0.0, 0.0), (π/a, 0.0), (π/a, π/a)]

    compute_W = (kx, ky) -> NonlocalArrays.GeomField.bloch_3x3_H(H_eff, r, kx, ky)
    ω_bands, Γ_bands, s = NonlocalArrays.GeomField.path_bands(knodes_MGXM;
        Nk=Nk, compute_W=compute_W,
        n_bands=3, threads=true, keep_k=true, return_gamma=true)

    # Normalize to Γ₀ units by dividing by γ_m = γ₀/3
    # (the natural unit of the AtomicArrays H_eff matrix).
    # Zeeman was also added in γ_m units, so everything is consistent.
    ω_n = ω_bands ./ γ_m
    Γ_n = Γ_bands ./ γ_m

    # Print Γ-point values (index Nk = end of first segment M→Γ)
    ig = Nk
    println("  Γ-point shifts/Γ₀: $(round.(ω_n[:, ig], digits=4))")
    println("  Γ-point widths/Γ₀: $(round.(Γ_n[:, ig], digits=4))")

    # Save (label by total atom count N = Nside²)
    Ntotal = Nside * Nside
    fname = joinpath(outdir, "fig2a_julia_N$(Ntotal)_bands.csv")
    open(fname, "w") do io
        write(io, "# s omega1 omega2 omega3 Gamma1 Gamma2 Gamma3\n")
        write(io, "# Ntotal=$Ntotal ($(Nside)x$(Nside)) a/lambda=$a_over_lambda DeltaB/gamma=$DeltaB_over_gamma norm=gamma_m\n")
        writedlm(io, hcat(s, ω_n', Γ_n'))
    end
    println("  Saved: $fname")
end

println("\nAll done.")

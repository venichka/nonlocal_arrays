"""
Figure 2(a): Band structure along M → Γ → X → M for a square array.
Julia version using NonlocalArrays / AtomicArrays.
Saves data for comparison with the Python regularized-GF and Rayleigh methods.

Parameters (matching the paper plan):
  a/λ = 0.7,  Δ_B/γ = 0.5
  λ₀ = 1, k₀ = 2π, all frequencies in units of Γ₀

Convention mapping (Julia ↔ Python):
  Julia: γ₀ = total single-atom decay (= Γ₀ in the paper)
         gammas(γ₀) = [γ₀/3, γ₀/3, γ₀/3]  per m-channel
         Omega(..., γ_m, γ_m') = 0.75 √(γ_m γ_m') G(...)
         band eigenvalues are in units of γ₀
  Python: all Δ, Γ, eigenvalues in units of Γ₀
  → They should match directly when using the same a/λ and Δ_B/Γ₀.
"""

# Activate project
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using NonlocalArrays
using AtomicArrays
using StaticArrays
using LinearAlgebra
using DelimitedFiles

# =====================================================================
#  Parameters
# =====================================================================
a_over_lambda = 0.7
a = a_over_lambda           # λ₀ = 1
γ₀ = 0.25                   # total single-atom decay rate (arbitrary units)

DeltaB_over_gamma = 0.5     # Zeeman splitting in units of γ₀ = Γ₀

# Zeeman detunings: Δ = [Δ_{m=-1}, Δ_{m=0}, Δ_{m=+1}]
# In Julia ordering: m = 1,2,3 ↔ m = -1, 0, +1
ΔB = DeltaB_over_gamma * γ₀
Δ = [-ΔB, 0.0, +ΔB]

# Dipole orientations (spherical basis, unit vectors)
d_sph = AtomicArrays.polarizations_spherical()
# d_sph[1,:] = m=-1, d_sph[2,:] = m=0, d_sph[3,:] = m=+1
μ = [SVector{3}(d_sph[m, :]) for m in 1:3]

# Decay rates per channel
γ_vec = AtomicArrays.gammas(γ₀)   # [γ₀/3, γ₀/3, γ₀/3]

# =====================================================================
#  Compute bands: infinite-array real-space sum
# =====================================================================
println("Computing bands via real-space lattice sum (Julia) ...")
Nk = 200
Nmax = 60   # real-space truncation

# With Zeeman
ω_bands, Γ_bands, s = bands_GXMG(a, μ, γ_vec, Δ;
    ωc=0.0, Nmax=Nmax, Nk=Nk, keep_k=true, return_gamma=true)

# Without Zeeman (reference)
Δ0 = [0.0, 0.0, 0.0]
ω_bands0, Γ_bands0, s0 = bands_GXMG(a, μ, γ_vec, Δ0;
    ωc=0.0, Nmax=Nmax, Nk=Nk, keep_k=true, return_gamma=true)

# =====================================================================
#  Normalize to Γ₀ units and print Γ-point values
# =====================================================================
# bands_GXMG returns eigenvalues in the same units as γ₀
# To express in units of Γ₀ = γ₀, divide by γ₀
ω_norm = ω_bands ./ γ₀     # shifts / Γ₀
Γ_norm = Γ_bands ./ γ₀     # widths / Γ₀

ω_norm0 = ω_bands0 ./ γ₀
Γ_norm0 = Γ_bands0 ./ γ₀

# Γ-point is at the boundary between M→Γ and Γ→X segments
# In the Julia path (M→Γ→X→M), Γ is at index Nk+1 ≈ Nk
# Actually path_bands uses Nk points per segment (excluding endpoint),
# so Γ is at index Nk (end of first segment)
idx_Gamma = Nk
println("\n=== Values at Γ point (index $idx_Gamma) ===")
println("  a/λ = $a_over_lambda,  Δ_B/Γ₀ = $DeltaB_over_gamma")
γ̃_analytic = 3 / (4π * a_over_lambda^2)
println("  γ̃/Γ₀ (analytic) = $(round(γ̃_analytic, digits=4))")
println("\n  Julia real-space sum:")
println("    Shifts Δ/Γ₀: $(round.(ω_norm[:, idx_Gamma], digits=5))")
println("    Widths Γ/Γ₀: $(round.(Γ_norm[:, idx_Gamma], digits=5))")

# =====================================================================
#  Save data for comparison
# =====================================================================
outdir = joinpath(@__DIR__, "..", "..", "Data")
mkpath(outdir)

# Save as CSV: columns = s, ω₁, ω₂, ω₃, Γ₁, Γ₂, Γ₃
npts = length(s)
data = hcat(s, ω_norm', Γ_norm')
header = "# s  omega1  omega2  omega3  Gamma1  Gamma2  Gamma3\n"
header *= "# a/lambda=$a_over_lambda  DeltaB/gamma=$DeltaB_over_gamma  gamma0=$γ₀  Nmax=$Nmax\n"

fname = joinpath(outdir, "fig2a_julia_bands.csv")
open(fname, "w") do io
    write(io, header)
    writedlm(io, data)
end

# Also save B=0 reference
data0 = hcat(s0, ω_norm0', Γ_norm0')
fname0 = joinpath(outdir, "fig2a_julia_bands_B0.csv")
open(fname0, "w") do io
    write(io, "# s  omega1  omega2  omega3  Gamma1  Gamma2  Gamma3\n")
    write(io, "# a/lambda=$a_over_lambda  B=0  gamma0=$γ₀  Nmax=$Nmax\n")
    writedlm(io, data0)
end

println("\nSaved: $fname")
println("Saved: $fname0")

# =====================================================================
#  Quick console plot (optional, requires UnicodePlots)
# =====================================================================
try
    using UnicodePlots
    println("\n--- Shifts (Δ/Γ₀) ---")
    plt = lineplot(s, ω_norm[1,:]; name="band 1", xlabel="s", ylabel="Δ/Γ₀")
    lineplot!(plt, s, ω_norm[2,:]; name="band 2")
    lineplot!(plt, s, ω_norm[3,:]; name="band 3")
    display(plt)

    println("\n--- Widths (Γ/Γ₀) ---")
    plt2 = lineplot(s, Γ_norm[1,:]; name="band 1", xlabel="s", ylabel="Γ/Γ₀")
    lineplot!(plt2, s, Γ_norm[2,:]; name="band 2")
    lineplot!(plt2, s, Γ_norm[3,:]; name="band 3")
    display(plt2)
catch
    println("(Install UnicodePlots for terminal plots)")
end

println("\nDone. Compare with Python output in Data/fig2a_python_bands.npz")

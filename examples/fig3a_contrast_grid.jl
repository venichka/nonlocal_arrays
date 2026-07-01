"""
Fig 3(a): Compute helicity contrast C at a grid of (a/λ, Δ_B/Γ₀).

Convention clarification:
  AtomicArrays POL="R" = [1,-i,0] = lab e_minus = drives m=-1
  AtomicArrays POL="L" = [1,+i,0] = lab e_plus  = drives m=+1

  Paper's σ⁺ drives m=+1, paper's σ⁻ drives m=-1.
  So: paper σ⁺ = code POL="L", paper σ⁻ = code POL="R".

  Paper R₊ = reflectance for σ⁺ inc (m=+1 resonance) = code R from POL="L"
  Paper R₋ = reflectance for σ⁻ inc (m=-1 resonance) = code R from POL="R"

  Lab circular basis: e_plus = [1,i,0]/√2, e_minus = [1,-i,0]/√2
  Julia dot(a,b) = Σ conj(a)·b — correct inner product without extra conj.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using NonlocalArrays
using AtomicArrays
using AtomicArrays: interaction
using StaticArrays, LinearAlgebra, DelimitedFiles

outdir = joinpath(@__DIR__, "..", "..", "Data")
mkpath(outdir)

function compute_R_copol(H_work, drive, d_sph, e_inc, C_pref, E0_amp, N)
    """Compute co-polarized specular reflectance for a given drive/channel."""
    s = H_work \ drive
    p = zeros(ComplexF64, 3)
    for n in 1:N, m in 1:3, α in 1:3
        p[α] += d_sph[m, α] * s[3*(n-1)+m]
    end
    p ./= N
    # Reflection amplitude in the incident channel
    r = C_pref * dot(e_inc, p) / E0_amp
    return abs2(r)  # = |r|² (reflectance, since |a_inc|=1 for normalized e_inc)
end

function compute_contrast_grid(a_over_lambda, Nside, DeltaB_values;
                               n_scan=60, scan_half_width=3.0)
    a = a_over_lambda
    γ₀ = 0.25; γ_m = γ₀ / 3.0; k₀ = 2π
    N = Nside^2
    t0 = time()

    # Build collection once
    coll, E_R, _, OmR_R = build_fourlevel_system(;
        a=a, Nx=Nside, Ny=Nside, delta_val=0.0, gamma=γ₀,
        POL="R", amplitude=0.01,
        field_func=AtomicArrays.field.plane, angle_k=[0.0, 0.0])

    _, E_L, _, OmR_L = build_fourlevel_system(;
        a=a, Nx=Nside, Ny=Nside, delta_val=0.0, gamma=γ₀,
        POL="L", amplitude=0.01,
        field_func=AtomicArrays.field.plane, angle_k=[0.0, 0.0])

    Ω_mat = interaction.OmegaMatrix_4level(coll)
    Γ_mat = interaction.GammaMatrix_4level(coll)
    H_base = Ω_mat .- 0.5im .* Γ_mat

    # Drive vectors
    drive_R = zeros(ComplexF64, 3*N)  # POL="R" → drives m=-1 → paper's σ⁻
    drive_L = zeros(ComplexF64, 3*N)  # POL="L" → drives m=+1 → paper's σ⁺
    for n in 1:N, m in 1:3
        drive_R[3*(n-1)+m] = OmR_R[m, n]
        drive_L[3*(n-1)+m] = OmR_L[m, n]
    end

    d_sph = AtomicArrays.polarizations_spherical()
    e_plus  = ComplexF64[1, im, 0] / √2   # lab circular basis
    e_minus = ComplexF64[1, -im, 0] / √2
    C_pref = im * 3π * γ_m / (2 * k₀^2 * a^2)
    E0_amp = E_R.amplitude

    # POL="R" is in e_minus channel, POL="L" is in e_plus channel
    # Co-pol reflection of POL="R" → project onto e_minus
    # Co-pol reflection of POL="L" → project onto e_plus

    println("    Setup: $(round(time()-t0, digits=1))s")

    H_work = copy(H_base)
    results = zeros(length(DeltaB_values), 4)

    for (iB, ΔB_Γ₀) in enumerate(DeltaB_values)
        ΔB = ΔB_Γ₀ * γ_m

        H_Zeeman = copy(H_base)
        for n in 1:N, m in 1:3
            H_Zeeman[3*(n-1)+m, 3*(n-1)+m] += [-1, 0, 1][m] * ΔB
        end

        # Scan δ to find peak of paper's R₊ (= code POL="L" co-pol in e_plus)
        δ_scan = range(-scan_half_width * γ_m, scan_half_width * γ_m, length=n_scan)

        R_plus_best = 0.0    # paper R₊
        R_minus_at_peak = 0.0  # paper R₋ at the same δ

        for δ_code in δ_scan
            copyto!(H_work, H_Zeeman)
            for n in 1:N, m in 1:3
                H_work[3*(n-1)+m, 3*(n-1)+m] += δ_code
            end

            # Paper R₊ = co-pol R from POL="L" (drives m=+1), project onto e_plus
            R_plus_trial = compute_R_copol(H_work, drive_L, d_sph, e_plus,
                                           C_pref, E0_amp, N)

            if R_plus_trial > R_plus_best
                R_plus_best = R_plus_trial
                # Paper R₋ = co-pol R from POL="R" (drives m=-1), project onto e_minus
                R_minus_at_peak = compute_R_copol(H_work, drive_R, d_sph, e_minus,
                                                   C_pref, E0_amp, N)
            end
        end

        C_val = (R_plus_best + R_minus_at_peak) > 1e-10 ?
            (R_plus_best - R_minus_at_peak) / (R_plus_best + R_minus_at_peak) : 0.0

        results[iB, :] = [a_over_lambda, ΔB_Γ₀, C_val, R_plus_best]
    end

    return results
end

# =====================================================================
# Dense grid for color map
a_values = range(0.35, 0.90, length=20)
DeltaB_values = range(0.05, 2.8, length=25)

for Nside in [10]
    N = Nside^2
    println("\n====== $(Nside)×$(Nside) array ======")
    all_results = zeros(0, 4)
    for a_val in a_values
        println("  a/λ = $a_val ...")
        res = compute_contrast_grid(a_val, Nside, DeltaB_values;
                                     n_scan=60, scan_half_width=3.0)
        all_results = vcat(all_results, res)
        gt = 3 / (4π * a_val^2)
        i_mid = div(length(DeltaB_values), 2)
        println("    ΔB/Γ₀=$(round(DeltaB_values[i_mid],digits=2)): " *
                "C=$(round(res[i_mid,3],digits=3)), R⁺=$(round(res[i_mid,4],digits=4))")
    end
    fname = joinpath(outdir, "fig3a_julia_N$(N).csv")
    open(fname, "w") do io
        write(io, "# a_over_lambda  DeltaB_over_Gamma0  C  R_plus_peak\n")
        write(io, "# Nside=$Nside dot_fix=true\n")
        writedlm(io, all_results)
    end
    println("  Saved: $fname")
end
println("\nDone.")

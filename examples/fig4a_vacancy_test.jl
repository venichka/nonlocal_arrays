"""
Fig 4(a) test: Compare two vacancy approaches on small arrays.
Approach A: Rebuild collection with surviving atoms only
Approach B: Mask full H_eff (zero out vacant rows/cols, solve reduced system)

Both should give identical R⁺, R⁻, C for the same vacancy realization.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using NonlocalArrays
using AtomicArrays
using AtomicArrays: interaction
using StaticArrays, LinearAlgebra, DelimitedFiles, Random

# =====================================================================
#  Common setup
# =====================================================================
a_over_lambda = 0.7
a = a_over_lambda
γ₀ = 0.25
γ_m = γ₀ / 3.0
k₀ = 2π

# Δ_B such that x=2 at f=1: Δ_B/Γ₀ = γ̃/Γ₀ = 0.487
gamma_tilde_Gamma0 = 3.0 / (4π * a_over_lambda^2)
DeltaB_Gamma0 = gamma_tilde_Gamma0  # x = 2ΔB/γ̃ = 2 at f=1
ΔB = DeltaB_Gamma0 * γ_m

d_sph = AtomicArrays.polarizations_spherical()
e_plus  = ComplexF64[1, im, 0] / √2
e_minus = ComplexF64[1, -im, 0] / √2

function compute_R_from_s(s, N_atoms, d_sph, e_channel, C_pref, E0_amp)
    p = zeros(ComplexF64, 3)
    for n in 1:N_atoms, m in 1:3, α in 1:3
        p[α] += d_sph[m, α] * s[3*(n-1)+m]
    end
    p ./= N_atoms
    r = C_pref * dot(e_channel, p) / E0_amp
    return abs2(r)
end

# =====================================================================
#  Approach A: Rebuild collection for surviving atoms
# =====================================================================
function approach_A(positions_full, mask, ΔB, γ₀, a, n_scan, scan_hw)
    γ_m = γ₀ / 3.0
    k₀ = 2π
    d_sph = AtomicArrays.polarizations_spherical()

    # Surviving positions
    pos_surv = positions_full[mask]
    N_surv = length(pos_surv)
    if N_surv < 2
        return 0.0, 0.0, 0.0
    end

    # Build collection manually
    pols = AtomicArrays.polarizations_spherical(N_surv)
    gam = [AtomicArrays.gammas(γ₀)[m] for m in 1:3, j in 1:N_surv]
    deltas = zeros(N_surv)
    coll = AtomicArrays.FourLevelAtomCollection(pos_surv;
        deltas=deltas, polarizations=pols, gammas=gam)

    Ω_mat = interaction.OmegaMatrix_4level(coll)
    Γ_mat = interaction.GammaMatrix_4level(coll)
    H_base = Ω_mat .- 0.5im .* Γ_mat

    for n in 1:N_surv, m in 1:3
        H_base[3*(n-1)+m, 3*(n-1)+m] += [-1, 0, 1][m] * ΔB
    end

    # Drives
    E0_amp = 0.01
    # σ⁺ (paper) = POL="L" = [1,i,0], drives m=+1
    E_L = normalize(ComplexF64[1, im, 0])
    # σ⁻ (paper) = POL="R" = [1,-i,0], drives m=-1
    E_R = normalize(ComplexF64[1, -im, 0])

    drive_L = zeros(ComplexF64, 3*N_surv)
    drive_R = zeros(ComplexF64, 3*N_surv)
    for n in 1:N_surv, m in 1:3
        # Ω_m = d_m^* · E / ℏ (in code units)
        # Match AtomicArrays.field.rabi convention: conj(μ' * E) = Σ μ_k * conj(E_k)
        drive_L[3*(n-1)+m] = sum(d_sph[m,:] .* conj(E_L)) * E0_amp
        drive_R[3*(n-1)+m] = sum(d_sph[m,:] .* conj(E_R)) * E0_amp
    end

    C_pref = im * 3π * γ_m / (2 * k₀^2 * a^2)
    e_plus  = ComplexF64[1, im, 0] / √2
    e_minus = ComplexF64[1, -im, 0] / √2

    # Scan detuning
    H_work = copy(H_base)
    δ_scan = range(-scan_hw * γ_m, scan_hw * γ_m, length=n_scan)

    Rp_best = 0.0; Rm_at_peak = 0.0

    for δ_code in δ_scan
        copyto!(H_work, H_base)
        for n in 1:N_surv, m in 1:3
            H_work[3*(n-1)+m, 3*(n-1)+m] += δ_code
        end

        F = lu(H_work)

        s_L = F \ drive_L
        Rp_trial = compute_R_from_s(s_L, N_surv, d_sph, e_plus, C_pref, E0_amp)

        if Rp_trial > Rp_best
            Rp_best = Rp_trial
            s_R = F \ drive_R
            Rm_at_peak = compute_R_from_s(s_R, N_surv, d_sph, e_minus, C_pref, E0_amp)
        end
    end

    C_val = (Rp_best + Rm_at_peak) > 1e-10 ?
        (Rp_best - Rm_at_peak) / (Rp_best + Rm_at_peak) : 0.0

    return Rp_best, Rm_at_peak, C_val
end

# =====================================================================
#  Approach B: Mask full H_eff
# =====================================================================
function approach_B(H_full, drive_L_full, drive_R_full, mask, ΔB, γ₀, a,
                    N_full, n_scan, scan_hw)
    γ_m = γ₀ / 3.0
    k₀ = 2π
    d_sph = AtomicArrays.polarizations_spherical()

    # Build index map: which 3×3 blocks survive
    surviving = findall(mask)
    N_surv = length(surviving)
    if N_surv < 2
        return 0.0, 0.0, 0.0
    end

    # Extract sub-matrix and sub-drives
    idx = Int[]
    for n in surviving
        push!(idx, 3*(n-1)+1, 3*(n-1)+2, 3*(n-1)+3)
    end

    H_sub = H_full[idx, idx]
    dL_sub = drive_L_full[idx]
    dR_sub = drive_R_full[idx]

    # Add Zeeman
    for i in 1:N_surv, m in 1:3
        H_sub[3*(i-1)+m, 3*(i-1)+m] += [-1, 0, 1][m] * ΔB
    end

    C_pref = im * 3π * γ_m / (2 * k₀^2 * a^2)
    E0_amp = 0.01
    e_plus  = ComplexF64[1, im, 0] / √2
    e_minus = ComplexF64[1, -im, 0] / √2

    H_work = copy(H_sub)
    δ_scan = range(-scan_hw * γ_m, scan_hw * γ_m, length=n_scan)

    Rp_best = 0.0; Rm_at_peak = 0.0

    for δ_code in δ_scan
        copyto!(H_work, H_sub)
        for i in 1:N_surv, m in 1:3
            H_work[3*(i-1)+m, 3*(i-1)+m] += δ_code
        end

        F = lu(H_work)

        s_L = F \ dL_sub
        Rp_trial = compute_R_from_s(s_L, N_surv, d_sph, e_plus, C_pref, E0_amp)

        if Rp_trial > Rp_best
            Rp_best = Rp_trial
            s_R = F \ dR_sub
            Rm_at_peak = compute_R_from_s(s_R, N_surv, d_sph, e_minus, C_pref, E0_amp)
        end
    end

    C_val = (Rp_best + Rm_at_peak) > 1e-10 ?
        (Rp_best - Rm_at_peak) / (Rp_best + Rm_at_peak) : 0.0

    return Rp_best, Rm_at_peak, C_val
end

# =====================================================================
#  Test on 5×5 and 10×10
# =====================================================================
for Nside in [5, 10]
    println("\n$('='^ 60)")
    println("Testing $(Nside)×$(Nside) array")
    println("$('='^ 60)")

    N_full = Nside^2

    # Build full collection and H_eff (without Zeeman — added per approach)
    coll_full, E_dummy, _, _ = build_fourlevel_system(;
        a=a, Nx=Nside, Ny=Nside, delta_val=0.0, gamma=γ₀,
        POL="R", amplitude=0.01,
        field_func=AtomicArrays.field.plane, angle_k=[0.0, 0.0])

    positions_full = [coll_full.atoms[n].position for n in 1:N_full]

    Ω_full = interaction.OmegaMatrix_4level(coll_full)
    Γ_full = interaction.GammaMatrix_4level(coll_full)
    H_full = Ω_full .- 0.5im .* Γ_full  # NO Zeeman yet

    # Full drives (without Zeeman, for approach B)
    E0_amp = 0.01
    E_L = normalize(ComplexF64[1, im, 0])
    E_R = normalize(ComplexF64[1, -im, 0])
    drive_L_full = zeros(ComplexF64, 3*N_full)
    drive_R_full = zeros(ComplexF64, 3*N_full)
    for n in 1:N_full, m in 1:3
        # Match AtomicArrays.field.rabi: Σ μ_k * conj(E_k)
        drive_L_full[3*(n-1)+m] = sum(d_sph[m,:] .* conj(E_L)) * E0_amp
        drive_R_full[3*(n-1)+m] = sum(d_sph[m,:] .* conj(E_R)) * E0_amp
    end

    n_scan = 40
    scan_hw = 4.0

    # Test at several filling fractions with same random seed
    for f in [0.6, 0.8, 0.9, 1.0]
        for trial in 1:3
            Random.seed!(1000 * Nside + round(Int, 100*f) + trial)
            mask = rand(N_full) .< f

            if f == 1.0
                mask .= true
            end

            N_surv = sum(mask)

            t_A = @elapsed Rp_A, Rm_A, C_A = approach_A(
                positions_full, mask, ΔB, γ₀, a, n_scan, scan_hw)

            t_B = @elapsed Rp_B, Rm_B, C_B = approach_B(
                H_full, drive_L_full, drive_R_full, mask, ΔB, γ₀, a,
                N_full, n_scan, scan_hw)

            match_R = isapprox(Rp_A, Rp_B, atol=1e-6) && isapprox(Rm_A, Rm_B, atol=1e-6)
            match_C = isapprox(C_A, C_B, atol=1e-6)

            status = (match_R && match_C) ? "✓ MATCH" : "✗ MISMATCH"

            println("  f=$f trial=$trial N_surv=$N_surv: $status " *
                    "(A: R⁺=$(round(Rp_A,digits=4)) C=$(round(C_A,digits=4)) $(round(t_A,digits=2))s) " *
                    "(B: R⁺=$(round(Rp_B,digits=4)) C=$(round(C_B,digits=4)) $(round(t_B,digits=2))s)")

            if !match_R || !match_C
                println("    ΔR⁺=$(Rp_A-Rp_B), ΔR⁻=$(Rm_A-Rm_B), ΔC=$(C_A-C_B)")
            end
        end
    end
end

println("\nDone.")

"""
Fig 4(a) production: C and R⁺_peak vs filling f, disorder-averaged.
Approach B (mask full H_eff) for speed.
Parameters: a/λ=0.7, Δ_B chosen so x=2 at f=1, array 20×20.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using NonlocalArrays
using AtomicArrays
using AtomicArrays: interaction
using StaticArrays, LinearAlgebra, DelimitedFiles, Random, Statistics

outdir = joinpath(@__DIR__, "..", "..", "Data")
mkpath(outdir)

# =====================================================================
#  Parameters
# =====================================================================
a = 0.75
γ₀ = 0.25
γ_m = γ₀ / 3.0
k₀ = 2π
Nside = 30
N_full = Nside^2

gamma_tilde_Gamma0 = 3.0 / (4π * a^2)
DeltaB_Gamma0 = gamma_tilde_Gamma0  # x=2 at f=1
ΔB = DeltaB_Gamma0 * γ_m

n_realizations = 50
f_values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
n_scan = 50
scan_hw = 4.0

println("a/λ = $a, γ̃/Γ₀ = $(round(gamma_tilde_Gamma0, digits=4)), " *
        "ΔB/Γ₀ = $(round(DeltaB_Gamma0, digits=4))")
println("Array: $(Nside)×$(Nside), $n_realizations realizations per f")

# =====================================================================
#  Build full H_eff and drives ONCE
# =====================================================================
println("Building full $(Nside)×$(Nside) interaction matrices...")

coll_full, _, _, _ = build_fourlevel_system(;
    a=a, Nx=Nside, Ny=Nside, delta_val=0.0, gamma=γ₀,
    POL="R", amplitude=0.01,
    field_func=AtomicArrays.field.plane, angle_k=[0.0, 0.0])

Ω_full = interaction.OmegaMatrix_4level(coll_full)
Γ_full = interaction.GammaMatrix_4level(coll_full)
H_full_noZeeman = Ω_full .- 0.5im .* Γ_full

d_sph = AtomicArrays.polarizations_spherical()
e_plus  = ComplexF64[1, im, 0] / √2
e_minus = ComplexF64[1, -im, 0] / √2
C_pref = im * 3π * γ_m / (2 * k₀^2 * a^2)
E0_amp = 0.01

E_L = normalize(ComplexF64[1, im, 0])   # paper σ⁺
E_R = normalize(ComplexF64[1, -im, 0])  # paper σ⁻

drive_L_full = zeros(ComplexF64, 3*N_full)
drive_R_full = zeros(ComplexF64, 3*N_full)
for n in 1:N_full, m in 1:3
    drive_L_full[3*(n-1)+m] = sum(d_sph[m,:] .* conj(E_L)) * E0_amp
    drive_R_full[3*(n-1)+m] = sum(d_sph[m,:] .* conj(E_R)) * E0_amp
end

println("Setup done.\n")

# =====================================================================
#  Vacancy sweep
# =====================================================================
function compute_one_realization(H_full_noZeeman, drive_L_full, drive_R_full,
                                 mask, ΔB, d_sph, e_plus, e_minus,
                                 C_pref, E0_amp, N_full, n_scan, scan_hw, γ_m)
    surviving = findall(mask)
    N_surv = length(surviving)
    if N_surv < 2
        return 0.0, 0.0, 0.0
    end

    idx = Int[]
    for n in surviving
        push!(idx, 3*(n-1)+1, 3*(n-1)+2, 3*(n-1)+3)
    end

    H_sub = H_full_noZeeman[idx, idx]
    dL = drive_L_full[idx]
    dR = drive_R_full[idx]

    # Add Zeeman
    for i in 1:N_surv, m in 1:3
        H_sub[3*(i-1)+m, 3*(i-1)+m] += [-1, 0, 1][m] * ΔB
    end

    H_work = copy(H_sub)
    δ_scan = range(-scan_hw * γ_m, scan_hw * γ_m, length=n_scan)

    Rp_best = 0.0
    Rm_at_peak = 0.0

    for δ_code in δ_scan
        copyto!(H_work, H_sub)
        for i in 1:N_surv, m in 1:3
            H_work[3*(i-1)+m, 3*(i-1)+m] += δ_code
        end

        F = lu(H_work)

        # R⁺ (paper): POL="L" → e_plus
        s_L = F \ dL
        p_L = zeros(ComplexF64, 3)
        for i in 1:N_surv, m in 1:3, α in 1:3
            p_L[α] += d_sph[m, α] * s_L[3*(i-1)+m]
        end
        p_L ./= N_surv
        r_plus = C_pref * dot(e_plus, p_L) / E0_amp
        Rp_trial = abs2(r_plus)

        if Rp_trial > Rp_best
            Rp_best = Rp_trial
            # R⁻ (paper): POL="R" → e_minus
            s_R = F \ dR
            p_R = zeros(ComplexF64, 3)
            for i in 1:N_surv, m in 1:3, α in 1:3
                p_R[α] += d_sph[m, α] * s_R[3*(i-1)+m]
            end
            p_R ./= N_surv
            r_minus = C_pref * dot(e_minus, p_R) / E0_amp
            Rm_at_peak = abs2(r_minus)
        end
    end

    C_val = (Rp_best + Rm_at_peak) > 1e-10 ?
        (Rp_best - Rm_at_peak) / (Rp_best + Rm_at_peak) : 0.0

    return Rp_best, Rm_at_peak, C_val
end

# Results: f, C_mean, C_std, Rp_mean, Rp_std
results = zeros(length(f_values), 5)

for (i_f, f) in enumerate(f_values)
    Rp_all = zeros(n_realizations)
    C_all  = zeros(n_realizations)

    for r in 1:n_realizations
        Random.seed!(1000*i_f + r)  # reproducible
        mask = f < 1.0 ? (rand(N_full) .< f) : trues(N_full)

        Rp, Rm, C = compute_one_realization(
            H_full_noZeeman, drive_L_full, drive_R_full,
            mask, ΔB, d_sph, e_plus, e_minus,
            C_pref, E0_amp, N_full, n_scan, scan_hw, γ_m)

        Rp_all[r] = Rp
        C_all[r] = C
    end

    results[i_f, :] = [f, mean(C_all), std(C_all),
                         mean(Rp_all), std(Rp_all)]

    println("f=$(round(f,digits=2)):  C=$(round(results[i_f,2],digits=4))±$(round(results[i_f,3],digits=4))  " *
            "R⁺=$(round(results[i_f,4],digits=4))±$(round(results[i_f,5],digits=4))")
end

# Save
fname = joinpath(outdir, "fig4a_vacancy_N$(N_full).csv")
open(fname, "w") do io
    write(io, "# f  C_mean  C_std  Rp_mean  Rp_std\n")
    write(io, "# Nside=$Nside a=$a DeltaB_Gamma0=$(round(DeltaB_Gamma0,digits=4)) n_real=$n_realizations\n")
    writedlm(io, results)
end
println("\nSaved: $fname")

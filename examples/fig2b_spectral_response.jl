"""
Fig 2(b): Spectral response R±(δ), T±(δ) for finite arrays at normal incidence.
Fixed dot() conjugation: use dot(e, p) not dot(conj(e), p).

Convention (matching paper):
  Paper R₊ = code POL="L" co-pol in e_plus  (drives m=+1)
  Paper R₋ = code POL="R" co-pol in e_minus (drives m=-1)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using NonlocalArrays
using AtomicArrays
using AtomicArrays: interaction
using StaticArrays, LinearAlgebra, DelimitedFiles

outdir = joinpath(@__DIR__, "..", "..", "Data")
mkpath(outdir)

function compute_spectral(a_over_lambda, DeltaB_Gamma0, Nside;
                          n_det=200, delta_range=(-3.0, 3.0))
    a = a_over_lambda
    γ₀ = 0.25; γ_m = γ₀ / 3.0; k₀ = 2π
    ΔB = DeltaB_Gamma0 * γ_m
    N = Nside^2

    # Build once
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

    for n in 1:N, m in 1:3
        H_base[3*(n-1)+m, 3*(n-1)+m] += [-1, 0, 1][m] * ΔB
    end

    # Drives: POL="L" → paper σ⁺, POL="R" → paper σ⁻
    drive_L = zeros(ComplexF64, 3*N)  # paper σ⁺
    drive_R = zeros(ComplexF64, 3*N)  # paper σ⁻
    for n in 1:N, m in 1:3
        drive_L[3*(n-1)+m] = OmR_L[m, n]
        drive_R[3*(n-1)+m] = OmR_R[m, n]
    end

    d_sph = AtomicArrays.polarizations_spherical()
    e_plus  = ComplexF64[1, im, 0] / √2
    e_minus = ComplexF64[1, -im, 0] / √2
    C_pref = im * 3π * γ_m / (2 * k₀^2 * a^2)
    E0_amp = E_R.amplitude

    deltas_Gamma0 = range(delta_range[1], delta_range[2], length=n_det)

    # Output: paper convention (R₊ = σ⁺, R₋ = σ⁻)
    Rp = zeros(n_det); Rm = zeros(n_det)
    Tp = zeros(n_det); Tm = zeros(n_det)
    H_work = copy(H_base)

    for (i, δ_Γ₀) in enumerate(deltas_Gamma0)
        δ_code = δ_Γ₀ * γ_m
        copyto!(H_work, H_base)
        for n in 1:N, m in 1:3
            H_work[3*(n-1)+m, 3*(n-1)+m] += δ_code
        end

        F = lu(H_work)

        # Paper R₊: POL="L" co-pol in e_plus
        s_L = F \ drive_L
        p_L = zeros(ComplexF64, 3)
        for n in 1:N, m in 1:3, α in 1:3
            p_L[α] += d_sph[m, α] * s_L[3*(n-1)+m]
        end
        p_L ./= N
        r_plus = C_pref * dot(e_plus, p_L) / E0_amp
        Rp[i] = abs2(r_plus)
        Tp[i] = abs2(1.0 + r_plus)  # t = 1 + r for co-pol channel

        # Paper R₋: POL="R" co-pol in e_minus
        s_R = F \ drive_R
        p_R = zeros(ComplexF64, 3)
        for n in 1:N, m in 1:3, α in 1:3
            p_R[α] += d_sph[m, α] * s_R[3*(n-1)+m]
        end
        p_R ./= N
        r_minus = C_pref * dot(e_minus, p_R) / E0_amp
        Rm[i] = abs2(r_minus)
        Tm[i] = abs2(1.0 + r_minus)
    end

    return collect(deltas_Gamma0), Rp, Rm, Tp, Tm
end

for a_lam in [0.55, 0.75]
    for Nside in [10, 20, 30]
        N = Nside^2
        gt = 3 / (4π * a_lam^2)
        println("\n===== a/λ=$a_lam, $(Nside)×$(Nside), γ̃/Γ₀=$(round(gt,digits=4)) =====")
        deltas, Rp, Rm, Tp, Tm = compute_spectral(a_lam, 0.5, Nside)
        i_res = argmax(Rp)
        println("  R⁺ peak = $(round(Rp[i_res],digits=4)) at δ/Γ₀ = $(round(deltas[i_res],digits=3))")
        println("  R⁻ there = $(round(Rm[i_res],digits=4))")
        println("  R⁺+T⁺ = $(round(Rp[i_res]+Tp[i_res],digits=4))")

        a_str = replace(string(a_lam), "."=>"")
        fname = joinpath(outdir, "fig2b_julia_a$(a_str)_N$(N).csv")
        open(fname, "w") do io
            write(io, "# delta_over_Gamma0 R_plus R_minus T_plus T_minus\n")
            write(io, "# a/lambda=$a_lam Nside=$Nside DeltaB/Gamma0=0.5 dot_fix=true paper_convention=true\n")
            writedlm(io, hcat(deltas, Rp, Rm, Tp, Tm))
        end
        println("  Saved: $fname")
    end
end
println("\nDone.")

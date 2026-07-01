"""
Fig 4(b): Experimental prediction — B = 1 mT (x ≈ 2.2).
30×30 array, a/λ=0.75, f=0.95, γ_nr/Γ₀=0.001.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using NonlocalArrays
using AtomicArrays
using AtomicArrays: interaction
using StaticArrays, LinearAlgebra, DelimitedFiles, Random, Statistics

outdir = joinpath(@__DIR__, "..", "..", "Data")
mkpath(outdir)

a = 0.75; γ₀ = 0.25; γ_m = γ₀/3; k₀ = 2π
Nside = 30; N_full = Nside^2

DeltaB_Gamma0 = 0.459   # B = 1 mT
ΔB = DeltaB_Gamma0 * γ_m
gamma_nr_Gamma0 = 0.001
γ_nr = gamma_nr_Gamma0 * γ_m
f = 0.95
n_realizations = 30
n_det = 200
delta_range = (-2.0, 2.0)  # narrower range since peaks are closer

println("B=1mT: a/λ=$a, Δ_B/Γ₀=$DeltaB_Gamma0, f=$f, $(Nside)×$(Nside), $n_realizations realizations")

println("Building interaction matrices...")
coll, _, _, _ = build_fourlevel_system(;
    a=a, Nx=Nside, Ny=Nside, delta_val=0.0, gamma=γ₀,
    POL="R", amplitude=0.01,
    field_func=AtomicArrays.field.plane, angle_k=[0.0, 0.0])

Ω_full = interaction.OmegaMatrix_4level(coll)
Γ_full = interaction.GammaMatrix_4level(coll)
H_full_base = Ω_full .- 0.5im .* Γ_full

d_sph = AtomicArrays.polarizations_spherical()
e_plus  = ComplexF64[1, im, 0] / √2
e_minus = ComplexF64[1, -im, 0] / √2
C_pref = im * 3π * γ_m / (2 * k₀^2 * a^2)
E0_amp = 0.01

E_L = normalize(ComplexF64[1, im, 0])
E_R = normalize(ComplexF64[1, -im, 0])

drive_L_full = zeros(ComplexF64, 3*N_full)
drive_R_full = zeros(ComplexF64, 3*N_full)
for n in 1:N_full, m in 1:3
    drive_L_full[3*(n-1)+m] = sum(d_sph[m,:] .* conj(E_L)) * E0_amp
    drive_R_full[3*(n-1)+m] = sum(d_sph[m,:] .* conj(E_R)) * E0_amp
end
println("Setup done.")

deltas_Gamma0 = range(delta_range[1], delta_range[2], length=n_det)

Rp_all = zeros(n_det, n_realizations)
Rm_all = zeros(n_det, n_realizations)
Tp_all = zeros(n_det, n_realizations)
Tm_all = zeros(n_det, n_realizations)

for r in 1:n_realizations
    println("  Realization $r/$n_realizations ...")
    Random.seed!(42 + r)
    mask = rand(N_full) .< f
    surviving = findall(mask)
    N_surv = length(surviving)

    idx = Int[]
    for n in surviving
        push!(idx, 3*(n-1)+1, 3*(n-1)+2, 3*(n-1)+3)
    end

    H_sub = H_full_base[idx, idx]
    dL = drive_L_full[idx]
    dR = drive_R_full[idx]

    for i in 1:N_surv, m in 1:3
        H_sub[3*(i-1)+m, 3*(i-1)+m] += [-1, 0, 1][m] * ΔB - im * γ_nr / 2
    end

    H_work = copy(H_sub)

    for (j, δ_Γ₀) in enumerate(deltas_Gamma0)
        δ_code = δ_Γ₀ * γ_m
        copyto!(H_work, H_sub)
        for i in 1:N_surv, m in 1:3
            H_work[3*(i-1)+m, 3*(i-1)+m] += δ_code
        end

        F = lu(H_work)

        s_L = F \ dL
        p_L = zeros(ComplexF64, 3)
        for i in 1:N_surv, m in 1:3, α in 1:3
            p_L[α] += d_sph[m, α] * s_L[3*(i-1)+m]
        end
        p_L ./= N_surv
        r_plus = C_pref * dot(e_plus, p_L) / E0_amp
        Rp_all[j, r] = abs2(r_plus)
        Tp_all[j, r] = abs2(1.0 + r_plus)

        s_R = F \ dR
        p_R = zeros(ComplexF64, 3)
        for i in 1:N_surv, m in 1:3, α in 1:3
            p_R[α] += d_sph[m, α] * s_R[3*(i-1)+m]
        end
        p_R ./= N_surv
        r_minus = C_pref * dot(e_minus, p_R) / E0_amp
        Rm_all[j, r] = abs2(r_minus)
        Tm_all[j, r] = abs2(1.0 + r_minus)
    end
end

Rp_mean = vec(mean(Rp_all, dims=2))
Rm_mean = vec(mean(Rm_all, dims=2))
Tp_mean = vec(mean(Tp_all, dims=2))
Tm_mean = vec(mean(Tm_all, dims=2))

i_peak = argmax(Rp_mean)
C_at_peak = (Rp_mean[i_peak]-Rm_mean[i_peak])/(Rp_mean[i_peak]+Rm_mean[i_peak])
println("\nR⁺ peak = $(round(Rp_mean[i_peak], digits=4)) at δ/Γ₀ = $(round(deltas_Gamma0[i_peak], digits=3))")
println("R⁻ at peak = $(round(Rm_mean[i_peak], digits=4))")
println("C at peak = $(round(C_at_peak, digits=4))")

fname = joinpath(outdir, "fig4b_experimental_1mT.csv")
open(fname, "w") do io
    write(io, "# delta_Gamma0 Rp_mean Rm_mean Tp_mean Tm_mean\n")
    write(io, "# a=$a DeltaB_Gamma0=$DeltaB_Gamma0 gamma_nr=$gamma_nr_Gamma0 f=$f B=1mT Nside=$Nside n_real=$n_realizations\n")
    writedlm(io, hcat(collect(deltas_Gamma0), Rp_mean, Rm_mean, Tp_mean, Tm_mean))
end
println("Saved: $fname")

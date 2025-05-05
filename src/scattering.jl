module Scattering

using LinearAlgebra, AtomicArrays
export stokes, transmission_reflection_new,   # rename without _new
       chiral_mirror_metrics

# ────────────────── 1. Stokes helper ───────────────────────────────────
"""
    stokes(Ex, Ey) -> (I, Q, U, V, dop)

Return the four Stokes parameters and degree of polarisation (DOP)
for a monochromatic field with transverse components `Ex, Ey`.
"""
@inline function stokes(Ex::Complex, Ey::Complex)
    I = abs2(Ex) + abs2(Ey)
    Q = abs2(Ex) - abs2(Ey)
    U = 2 * real(Ex * conj(Ey))
    V = -2 * imag(Ex * conj(Ey))
    dop = I ≈ 0 ? 0.0 : sqrt(Q^2 + U^2 + V^2) / I
    return (I=I, Q=Q, U=U, V=V, dop=dop)
end



# ────────────────── Stokes (I,V only) ────────────────────────────────
@inline stokes_IV(Ex, Ey) = (I = abs2(Ex)+abs2(Ey),
                              V = -2im*Ex*conj(Ey) |> imag)

# ────────────────── main routine ─────────────────────────────────────
"""
    transmission_reflection(E, collection, σ;
                            beam = :plane | :gauss,
                            surface = :hemisphere | :plane,
                            pol_basis = :circular | :linear,
                            samples = 50,
                            zlim = 100λ,
                            size = (5.0,5.0),
                            return_helicity = true,
                            return_positions = false)

Returns

* `(T, R)` (total) if `return_helicity = false`
* `(Tσ⁺, Tσ⁻, Rσ⁺, Rσ⁻, T, R)` if `return_helicity = true`
* plus sampled point arrays when `return_positions = true`.
"""
function transmission_reflection_new(E::AtomicArrays.field.EMField,
                                     coll, σ;
                                     beam::Symbol      = :plane,
                                     surface::Symbol   = :hemisphere,
                                     pol_basis::Symbol = :circular,
                                     samples::Int      = 50,
                                     zlim::Real        = 10.0,
                                     size::Tuple       = (5.0,5.0),
                                     return_helicity::Bool = false,
                                     return_positions::Bool = false,
                                     return_powers::Bool = false)

    # ------------------ 1. incident‑field builder ----------------------
    inc_wave = beam === :plane ? AtomicArrays.field.plane :
                                 AtomicArrays.field.gauss

    # helicity or linear analysers
    e_plus, e_minus = if pol_basis === :circular
        [1,  im, 0]/√2,  [1,-im,0]/√2
    elseif pol_basis === :linear
        [1,0,0],         [0,1,0]
    else
        error("pol_basis must be :circular or :linear")
    end

    # ------------------ 2. sample points -------------------------------
    if surface === :hemisphere
        θ, φ = AtomicArrays.field.fibonacci_angles(samples)   # forward hemi
        ΔΩ   = 2π/samples
        rfwd = [ zlim*[sin(t)*cos(p), sin(t)*sin(p), cos(t)] for (t,p) in zip(θ,φ) ]
        rbwd = [ zlim*[sin(t)*cos(p), sin(t)*sin(p),-cos(t)] for (t,p) in zip(θ,φ) ]
        weight = zlim^2*ΔΩ
    elseif surface === :plane
        x = range(-0.5*size[1], stop=0.5*size[1], length=samples)
        y = range(-0.5*size[2], stop=0.5*size[2], length=samples)
        dA = (size[1]/(samples-1))*(size[2]/(samples-1))
        rfwd = [ [xx,yy, zlim] for yy in y, xx in x ] |> vec
        rbwd = [ [xx,yy,-zlim] for yy in y, xx in x ] |> vec
        weight = dA
    else
        error("surface must be :hemisphere or :plane")
    end

    # ------------------ 3. evaluate fields -----------------------------
    Et_fwd = AtomicArrays.field.total_field(inc_wave, rfwd, E, coll, σ)
    Ei_fwd = inc_wave.(rfwd, Ref(E))
    Esc_bwd = AtomicArrays.field.scattered_field(rbwd, coll, σ)
    
    # --------------- power sums --------------------------------------
    Pp_fwd = sum(abs2.(dot.(Et_fwd, Ref(e_plus ))))
    Pm_fwd = sum(abs2.(dot.(Et_fwd, Ref(e_minus))))
    P_inc  = sum(abs2.(dot.(Ei_fwd, Ref(E.polarisation ))))           # same analyser
    Pp_bwd = sum(abs2.(dot.(Esc_bwd, Ref(e_plus ))))
    Pm_bwd = sum(abs2.(dot.(Esc_bwd, Ref(e_minus))))

    Pp_fwd *= 0.5*weight;  Pm_fwd *= 0.5*weight
    Pp_bwd *= 0.5*weight;  Pm_bwd *= 0.5*weight
    P_inc  *= 0.5*weight

    Tσp, Tσm = Pp_fwd/P_inc,  Pm_fwd/P_inc
    Rσp, Rσm = Pp_bwd/P_inc,  Pm_bwd/P_inc
    Ttot, Rtot = Tσp+Tσm, Rσp+Rσm

    # --------------- named‑tuple output ------------------------------
    if return_powers && return_helicity
        core = (Pp_fwd  = Pp_fwd,
                Pm_fwd = Pm_fwd,
                Pp_bwd  = Pp_bwd,
                Pm_bwd = Pm_bwd,
                P_inc = P_inc,
                P_T = Ttot*P_inc,
                P_R = Rtot*P_inc) 
    elseif !return_powers && return_helicity
        core = (T_sigma_plus  = Tσp,
                T_sigma_minus = Tσm,
                R_sigma_plus  = Rσp,
                R_sigma_minus = Rσm,
                T = Ttot,
                R = Rtot)
    elseif !return_powers && !return_helicity
        core = (T = Ttot, R = Rtot)
    elseif return_powers && !return_helicity
        core = (P_T = Ttot*P_inc, P_R = Rtot*P_inc, P_inc = P_inc)
    end

    if return_positions
        core = merge(core, (rfwd = rfwd, rbwd = rbwd))
    end
    return (; core...)          # NamedTuple
end

"""
    chiral_mirror_metrics(Tσp, Tσm, Rσp, Rσm;
                          Rσp_phase = nothing, Rσm_phase = nothing,
                          dop_back  = nothing,
                          thresh_CD = 0.9,
                          thresh_T  = 0.05,
                          thresh_DOP = 0.99,
                          thresh_phase = π/4)

Compute circular‑dichroism and contrast ratios and test optional criteria
for a *chiral mirror*.

Inputs
------
* `Tσp, Tσm`   – σ⁺ and σ⁻ transmissions through **+z hemisphere**
* `Rσp, Rσm`   – σ⁺ and σ⁻ reflections through **−z hemisphere**

Optional
--------
* `Rσp_phase`, `Rσm_phase` – phase (rad) of complex reflection coeffs
                             for σ⁺ and σ⁻ (plane‑wave calc).  
* `dop_back`               – degree of polarisation of back‑scattered beam.
* Threshold kwargs let you tighten or loosen acceptance.

Returns
-------
`NamedTuple` containing  

* the eight flux coefficients `Tσ⁺, Tσ⁻, Rσ⁺, Rσ⁻`,  
* circular‑dichroic ratios `CD_R, CD_T`,  contrast `C_R`,  
* booleans `is_chiral, pass_dop, pass_phase`.
"""
function chiral_mirror_metrics(Tσp, Tσm, Rσp, Rσm;
                               Rσp_phase = nothing, Rσm_phase = nothing,
                               dop_back  = nothing,
                               thresh_CD::Real = 0.9,
                               thresh_T::Real  = 0.05,
                               thresh_DOP::Real = 0.99,
                               thresh_phase::Real = π/4,
                               kind::Symbol = :product,
                               w_T::Real = 2.0,
                               α::Real = 20.0,)

    CD_R = (Rσp - Rσm) / (Rσp + Rσm + eps())   # eps to avoid /0
    CD_T = (Tσp - Tσm) / (Tσp + Tσm + eps())

    C_R  = max(Rσp, Rσm) / (min(Rσp, Rσm) + eps())

    # --- criteria -------------------------------------------------------
    # main chiral mirror window: high CD_R and low transmission
    is_chiral = abs(CD_R) ≥ thresh_CD && (Tσp + Tσm) ≤ thresh_T
    # if !is_chiral
    #     CD_R =0.0
    # end

    Ttot = Tσp + Tσm

    obj = if kind === :product
        (1 - Ttot) * abs(CD_R)
    elseif kind === :linear
        max(abs(CD_R) - w_T*Ttot, 0)               # never negative
    elseif kind === :sigmoid
        abs(CD_R) / (1 + exp(α*(Ttot - thresh_T)))
    else
        error("kind must be :product, :linear or :sigmoid")
    end

    # optional: back polarisation purity
    pass_dop   = dop_back === nothing || dop_back ≥ thresh_DOP

    # optional: opposite phase shift (≈ π) between helicities
    pass_phase = true
    if !(Rσp_phase === nothing || Rσm_phase === nothing)
        phase_diff = abs(mod(Rσp_phase - Rσm_phase + π, 2π) - π)  # wrap
        pass_phase = phase_diff ≥ thresh_phase
    end

    return (Tσp = Tσp,  Tσm = Tσm,
            Rσp = Rσp,  Rσm = Rσm,
            CD_R = CD_R, CD_T = CD_T, C_R = C_R, obj=obj,
            is_chiral = is_chiral && pass_dop && pass_phase,
            pass_dop = pass_dop,
            pass_phase = pass_phase)
end

# ---------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------
# @inline normed(v) = v / norm(v)

# """
#     helicity_basis(ŝ) -> (e₊, e₋)

# Return the right‑ and left‑hand circular unit vectors **transverse** to the
# propagation direction `ŝ` (|ŝ| = 1).  Formula:

#     e1 = ẑ × ŝ            (or x̂ if ŝ ∥ ẑ)
#     e2 = ŝ × e1
#     e± = (e1 ± i e2)/√2
# """
# function helicity_basis(ŝ::SVector{3})
#     # pick a stable transverse axis
#     if abs(ŝ[3]) < 0.999
#         e1 = normed(cross(SVector(0.0,0.0,1.0), ŝ))
#     else
#         e1 = SVector(1.0, 0.0, 0.0)           # beam along z → take x̂
#     end
#     e2 = cross(ŝ, e1)                        # already unit
#     e_plus  = (e1 + im*e2) / √2
#     e_minus = (e1 - im*e2) / √2
#     return e_plus, e_minus
# end

# # -------------------------------------------------------------------
# # 1.  Phase of the σ⁺ / σ⁻ reflection coefficients  (normal incidence)
# # -------------------------------------------------------------------
# """
#     reflection_phases(inc_wave, E, coll, σ;
#                       k = 2π, R = 10) -> (φσ⁺, φσ⁻)

# Evaluate the scattered field at the backward direction **(0,0,-R)** and
# return the phases (in rad, wrapped to ±π) of the complex reflection
# coefficients for σ⁺ and σ⁻, normalised to the incident amplitude.

# Works for normal‑incidence beams (plane or broad Gaussian).  For oblique
# incidence replace the two `rb`/`rf` vectors accordingly.
# """
# function reflection_phases(inc_wave, E, coll, σ; k::Real = 2π, R::Real = 10.0)
#     # helicity unit vectors (x,y,z order)
#     e_plus  = SVector(1,  im, 0) / √2
#     e_minus = SVector(1, -im, 0) / √2

#     # +z direction → incident reference amplitude
#     rf = SVector(0.0, 0.0,  R)
#     Ei = inc_wave(rf, E)
#     Ei_p, Ei_m = dot(Ei, e_plus), dot(Ei, e_minus)

#     # −z direction → reflected scattered field
#     rb   = SVector(0.0, 0.0, -R)
#     Esc  = field.scattered_field(rb, coll, σ, k)
#     Er_p, Er_m = dot(Esc, e_plus), dot(Esc, e_minus)

#     φp = angle(Er_p / Ei_p)
#     φm = angle(Er_m / Ei_m)
#     return φp, φm
# end
# """
#     reflection_phases(inc_wave, E, coll, σ;
#                       kin = SVector(0,0,1), k = 2π, R = 10)

# Phase (rad) of σ⁺ / σ⁻ reflection coefficients for **arbitrary incidence**.

# * `kin` – incident *wavevector* **direction** (need not be unit length);
#           specify as `SVector{3}` or `(θ,φ)` tuple (rad).
# * `k`   – vacuum wavenumber.
# * `R`   – far‑field evaluation radius.

# The function evaluates the *scattered* field in the exact specular
# direction (kspec = kin reflected at z=0) and divides by the amplitude of
# the incident helicity on the forward side.
# """
# function reflection_phases(inc_wave, E, coll, σ;
#                            kin = SVector(0.0,0.0,1.0),
#                            k::Real = 2π,
#                            R::Real = 10.0)

#     # handle (θ,φ) input
#     if !(kin isa SVector)
#         θ, φ = kin
#         kin  = SVector(sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ))
#     end
#     ŝ_in  = normed(kin)
#     @assert ŝ_in[3] > 0 "Incident direction must have +z component (coming from –z)"

#     # specular reflection: flip z component
#     ŝ_ref = SVector(ŝ_in[1], ŝ_in[2], -ŝ_in[3])

#     # helicity bases for forward & backward directions
#     eplus_in,  eminus_in  = helicity_basis(ŝ_in)
#     eplus_out, eminus_out = helicity_basis(ŝ_ref)

#     # field evaluation points (far field)
#     rf = R * ŝ_in         # forward
#     rb = R * ŝ_ref        # backward

#     Ei  = inc_wave(rf, E)                       # incident field amplitude
#     Esc = field.scattered_field(rb, coll, σ, k) # reflected scattered field

#     # project onto helicity bases
#     Ei_p,  Ei_m  = dot(Ei,  eplus_in),  dot(Ei,  eminus_in)
#     Er_p,  Er_m  = dot(Esc, eplus_out), dot(Esc, eminus_out)

#     φp = angle(Er_p / Ei_p)
#     φm = angle(Er_m / Ei_m)

#     # wrap to (−π, π]
#     φp = mod(φp + π, 2π) - π
#     φm = mod(φm + π, 2π) - π
#     return φp, φm
# end


# # -------------------------------------------------------------------
# # 2.  Degree‑of‑polarisation of the back‑scattered beam
# # -------------------------------------------------------------------
# """
#     dop_back_fibonacci(inc_wave, E, coll, σ;
#                        k = 2π, R = 10, Ndirs = 20_000) -> DOP

# Integrate the Stokes parameters of the **scattered** field over the
# backward (−z) hemisphere using `Ndirs` Fibonacci directions and return the
# overall degree of polarisation (0 ≤ DOP ≤ 1).
# """
# function dop_back_fibonacci(inc_wave, E, coll, σ;
#                             k::Real = 2π, R::Real = 10.0, Ndirs::Int = 20_000)

#     θ, φ = fibonacci_angles(Ndirs)                  # forward hemi only
#     ΔΩ   = 2π / Ndirs                               # equal solid angle

#     Itot = Qtot = Utot = Vtot = 0.0

#     for (th, ph) in zip(θ, φ)
#         sinθ, cosθ = sincos(th);  sinφ, cosφ = sincos(ph)

#         r̂ = SVector(sinθ*cosφ, sinθ*sinφ, -cosθ)    # -z direction
#         rb = R * r̂

#         Esc = field.scattered_field(rb, coll, σ, k)
#         Ex, Ey = Esc[1], Esc[2]

#         I = abs2(Ex) + abs2(Ey)
#         Q = abs2(Ex) - abs2(Ey)
#         U = 2*real(Ex*conj(Ey))
#         V = -2*imag(Ex*conj(Ey))

#         Itot += I * ΔΩ
#         Qtot += Q * ΔΩ
#         Utot += U * ΔΩ
#         Vtot += V * ΔΩ
#     end

#     return Itot ≈ 0 ? 0.0 : sqrt(Qtot^2 + Utot^2 + Vtot^2) / Itot
# end


end

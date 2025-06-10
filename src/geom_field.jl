module GeomField

using LinearAlgebra, AtomicArrays, StaticArrays, Base.Threads
export build_fourlevel_system, omega_1d_triplet, omega_2d_triplet, bands_GXMG

"""
    NonlocalArrays.build_fourlevel_system(; a=0.2, Nx=4, Ny=4,
                                          delta_val=0.2,
                                          POL="R",
                                          amplitude=0.02,
                                          angle_k=[π/6, 0.0])

Builds a four-level atomic system consisting of:

- A rectangular atomic array
- Polarizations and decay rates
- An incident electric field
- Rabi frequencies calculated from the field

# Keyword Arguments

- `a`: Lattice spacing (default 0.2)
- `Nx`, `Ny`: Number of atoms along x and y axes
- `delta_val`: Detuning value for all atoms except the first
- `gamma`: Spontaneous decay rate of all three transitions
- `POL`: "R" or "L" circular polarization
- `amplitude`: Electric field amplitude
- `field_func`: Shape of the incident beam
- `angle_k`: Incident field direction `[θ, φ]`

# Returns

- `coll`: `FourLevelAtomCollection`
- `field`: `EMField` structure
- `field_func`: function used to evaluate the field at a point
- `OmR`: Rabi frequencies at each atom position
"""
function build_fourlevel_system(; a=0.2, Nx=4, Ny=4,
                                delta_val=0.0,
                                gamma=0.25,
                                POL="R",
                                amplitude=0.02,
                                field_func=AtomicArrays.field.gauss,
                                angle_k=[0.0, 0.0])
    # TODO: adjust for more generosity
    # Compute positions
    positions = AtomicArrays.geometry.rectangle(a, a; Nx=Nx, Ny=Ny,
                    position_0 = [(-Nx/2 + 0.5)*a, (-Ny/2 + 0.5)*a, 0.0])
    N = length(positions)

    # Build polarization and decay profile
    pols = AtomicArrays.polarizations_spherical(N)
    gam  = [AtomicArrays.gammas(gamma)[m] for m in 1:3, j in 1:N]
    deltas = [(i == 1) ? delta_val : delta_val for i in 1:N]

    # Build atomic collection
    coll = AtomicArrays.FourLevelAtomCollection(positions;
                    deltas=deltas, polarizations=pols, gammas=gam)

    # Create field
    k_mod = 2π
    polarisation = POL == "R" ? [1.0, -1.0im, 0.0] : [1.0, 1.0im, 0.0]
    waist_radius = 0.3 * a * sqrt(Nx * Ny)
    field = AtomicArrays.field.EMField(amplitude, k_mod, angle_k, polarisation;
                    position_0=[0.0, 0.0, 0.0], waist_radius)

    # Compute Rabi frequency at atom positions
    OmR = AtomicArrays.field.rabi(field, field_func, coll)

    return coll, field, field_func, OmR
end
"""
    build_fourlevel_system(params::Dict)

Overload that takes a dictionary of parameters and forwards them as keyword arguments.
Supports both POL and POLARIZATION, and angle_k or anglek.
"""
function build_fourlevel_system(params::Dict{String,Any})
    # Resolve aliases
    POL = get(params, "POL", get(params, "POLARIZATION", "R"))
    angle_k = get(params, "angle_k", get(params, "anglek", [0.0, 0.0]))

    # Forward values using the original function
    return build_fourlevel_system(; 
        a = get(params, "a", 0.2),
        Nx = get(params, "Nx", 4),
        Ny = get(params, "Ny", 4),
        delta_val = get(params, "delta_val", get(params, "deltas", 0.0)),
        gamma = get(params, "gamma", 0.25),
        POL = POL,
        amplitude = get(params, "amplitude", 0.02),
        field_func = get(params, "field_func", AtomicArrays.field.gauss),
        angle_k = angle_k,
    )
end

# -------------------------------------------------------------------
#  Dispersion for a Zeeman triplet (m = −1,0,+1) in periodic arrays
#  Paste this after the existing code and 'using .interaction'
# -------------------------------------------------------------------
"""
    omega_1d_triplet(k; a, μ, γ, Δ, ωc = 0.0, Nmax = 250)

Collective dispersion **matrix** for an *infinite 1-D chain* of atoms.

Arguments  
* `k`     – Bloch wave number (same units as 1/a).  
* `a`     – lattice constant.  
* `μ`     – `Vector{SVector{3,T}}` (length 3) giving dipole orientation
            of the m = −1,0,+1 transitions.  
* `γ`     – `Vector{T}` (length 3) single-atom decay rates Γₘ.  
* `Δ`     – `Vector{T}` (length 3) Zeeman detunings (ωₘ − ωc).  
* `ωc`    – reference / carrier frequency (adds ωc ⋅ 𝟙).  
* `Nmax`  – truncation: keep |n| ≤ Nmax in the lattice sum.

Returns a **3 × 3 real matrix** ωₘₘ′(k).
"""
function omega_1d_triplet(k::Real, a::Real,
                      μ::Vector{<:AbstractVector},
                      γ::AbstractVector,
                      Δ::AbstractVector;
                      ωc::Real = 0.0, Nmax::Int = 250)

    @assert length(μ) == 3 == length(γ) == length(Δ)
    W = Diagonal(ωc .+ Δ) |> Matrix      # start with on-site Zeeman terms

    # helper: Ω(n a) for any pair (m,m′)
    Ω(n, m, mp) = interaction.Omega([0.0, 0.0, 0.0],
                                    [n*a, 0.0, 0.0],
                                    μ[m], μ[mp], γ[m], γ[mp])

    for m in 1:3, mp in 1:3
        Σ = zero(eltype(γ))
        for n = 1:Nmax
            Σ += 2 * Ω(n, m, mp) * cos(k*n*a)   # even in n → ×2
        end
        W[m, mp] += Σ
    end
    return W
end


# Safe 3×3 eigenvalue getter for either SMatrix or Matrix
function _eigvals3(W)
    if ishermitian(W)                    # real-symmetric or Hermitian
        return eigvals(Hermitian(W))
    else                                 # non-Hermitian → convert
        return eigvals(Matrix(W))
    end
end

# ---------------------------------------------------------------------
# _precompute_pair_table!   (internal helper)
# ---------------------------------------------------------------------
"""
    _precompute_pair_table!(Rxy, tab, μ, γ, pairfun)

Fill `tab[i,3(m-1)+mp] = pairfun(Rᵢ, m, mp)` for every lattice vector
`Rᵢ` in `Rxy` (size 2×nR) using the supplied `interaction` function
(`Omega` or `Gamma`).  Works in-place, no allocations.
"""
function _precompute_pair_table!(Rxy::AbstractMatrix,  # 2 × nR
                                 tab::AbstractMatrix,  # nR × 9
                                 μ::Vector, γ::Vector,
                                 pairfun::F)           where F<:Function
    nR = size(Rxy, 2)
    @inbounds for i in 1:nR
        rx, ry = Rxy[1,i], Rxy[2,i]
        for m in 1:3, mp in 1:3
            tab[i, 3*(m-1)+mp] =
                pairfun([0.,0.,0.],
                        [rx, ry, 0.],
                        μ[m], μ[mp], γ[m], γ[mp])
        end
    end
    tab                                   # return the filled table
end

# ---------------------------------------------------------------------
# Shared lattice-vector generator
# ---------------------------------------------------------------------
function _build_Rxy(a::Real, Nmax::Int)
    Rx = Float64[];  Ry = Float64[]
    for nx in -Nmax:Nmax, ny in -Nmax:Nmax
        (nx==0 && ny==0) && continue
        push!(Rx, a*nx);  push!(Ry, a*ny)
    end
    hcat(Rx, Ry)'                          # 2 × nR
end


"""
    precompute_Omega_table(a, μ, γ; Nmax=60)

Return `(Rxy, Ωtab)` with Ω values.
"""
function precompute_Omega_table(a, μ, γ; Nmax=60)
    Rxy  = _build_Rxy(a, Nmax)
    Ωtab = Matrix{ComplexF64}(undef, size(Rxy,2), 9)
    _precompute_pair_table!(Rxy, Ωtab, μ, γ, interaction.Omega)
    return Rxy, Ωtab
end


"""
    precompute_Gamma_table(a, μ, γ; Nmax=60)

Return `(Rxy, Γtab)` with Γ values.
"""
function precompute_Gamma_table(a, μ, γ; Nmax=60)
    Rxy  = _build_Rxy(a, Nmax)
    Γtab = Matrix{ComplexF64}(undef, size(Rxy,2), 9)
    _precompute_pair_table!(Rxy, Γtab, μ, γ, interaction.Gamma)
    return Rxy, Γtab
end

"""
    precompute_Omega_Gamma_tables(a, μ, γ; Nmax=60)

Return `(Rxy, Ωtab, Γtab)` with *both* pair tables sharing the
same lattice-vector list.
"""
function precompute_Omega_Gamma_tables(a, μ, γ; Nmax=60)
    Rxy  = _build_Rxy(a, Nmax)
    nR   = size(Rxy,2)
    Ωtab = Matrix{ComplexF64}(undef, nR, 9)
    Γtab = Matrix{ComplexF64}(undef, nR, 9)

    _precompute_pair_table!(Rxy, Ωtab, μ, γ, interaction.Omega)
    _precompute_pair_table!(Rxy, Γtab, μ, γ, interaction.Gamma)

    return Rxy, Ωtab, Γtab
end


"""
    omega_2d_triplet_fast(kx, ky, R, Ωtab, Δ; ωc = 0.0)

Collective dispersion **matrix** for an *infinite 2-D square lattice*
(period `a` along x and y).

Arguments mirror `ω_1d_triplet`; the sum now runs over
‖n‖₂ ≤ Nmax (excluding n = 0).

Returns a 3 × 3 matrix ωₘₘ′(k).

Vectorised, allocation-free version.  `R, Ωtab` obtained from
`precompute_Ω_table`.  Returns an `SMatrix{3,3,Float64}`.
"""
@inline function omega_2d_triplet_fast(kx::Real, ky::Real,
                                       Rxy ::AbstractMatrix,   # 2 × nR
                                       Ωtab::AbstractMatrix,   # nR × 9
                                       Δ   ::AbstractVector{<:Real};
                                       ωc  ::Real = 0.0)

    # element-wise k·R  (nR-vector, zero allocations)
    θ = kx .* @view(Rxy[1, :]) .+ ky .* @view(Rxy[2, :])
    c = cos.(θ)                               # nR-vector

    tmp9 = Ωtab' * c                          # 9-vector (BLAS gemv)
    W    = SMatrix{3,3}(tmp9)                 # static 3×3

    return W .+ Diagonal(ωc .+ Δ)
end
# ------------------------------------------------------------------
# Γ-block  (collective radiative decay)
# ------------------------------------------------------------------
"""
    gamma_2d_triplet_fast(kx, ky, Rxy, Γtab, γ)

Collective decay **matrix** for an *infinite 2-D square lattice*
(period `a` along x and y).

Arguments mirror `ω_1d_triplet`; the sum now runs over
‖n‖₂ ≤ Nmax (excluding n = 0).

Returns a 3 × 3 matrix Γₘₘ′(k).
"""
@inline function gamma_2d_triplet_fast(kx::Real, ky::Real,
                                       Rxy ::AbstractMatrix,   # 2 × nR
                                       Γtab::AbstractMatrix,   # nR × 9
                                       γ   ::AbstractVector{<:Real})

    θ = kx .* @view(Rxy[1,:]) .+ ky .* @view(Rxy[2,:])
    c = cos.(θ)                                         # odd part

    tmp9 = Γtab' * c
    Γk   = SMatrix{3,3}(tmp9) .+ Diagonal(γ)

    return Γk                                           # real-symmetric
end

# --------------------------------------------------------------------
# Full complex Bloch block  Ω(k) – i Γ(k)/2
# --------------------------------------------------------------------
@inline function omega_gamma_2d_triplet_fast(kx::Real, ky::Real,
                                             Rxy , Ωtab , Γtab ,
                                             Δ::AbstractVector{<:Real},
                                             γ::AbstractVector{<:Real};
                                             ωc::Real = 0.0)

    θ   = kx .* @view(Rxy[1,:]) .+ ky .* @view(Rxy[2,:])
    c   = cos.(θ)

    Ωk  = SMatrix{3,3}(Ωtab' * c)                    # BLAS gemv
    Γk  = SMatrix{3,3}(Γtab' * c) .+ Diagonal(γ)     # Γ(k)

    return Ωk .- 0.5im*Γk .+ Diagonal(ωc .+ Δ)       # 3×3 ComplexF64
end



"""
    omega_2d_triplet(kx, ky; a, μ, γ, Δ, ωc = 0.0, Nmax = 60)

Collective dispersion **matrix** for an *infinite 2-D square lattice*
(period `a` along x and y).

Arguments mirror `ω_1d_triplet`; the sum now runs over
‖n‖₂ ≤ Nmax (excluding n = 0).

Returns a 3 × 3 real matrix ωₘₘ′(k).
"""
function omega_2d_triplet(kx::Real, ky::Real, a::Real,
                      μ::Vector{<:AbstractVector},
                      γ::AbstractVector,
                      Δ::AbstractVector;
                      ωc::Real = 0.0, Nmax::Int = 60)

    @assert length(μ) == 3 == length(γ) == length(Δ)
    W = Diagonal(ωc .+ Δ) |> Matrix

    Ω(nx, ny, m, mp) = interaction.Omega([0.0, 0.0, 0.0],
                                         [nx*a, ny*a, 0.0],
                                         μ[m], μ[mp], γ[m], γ[mp])

    for m in 1:3, mp in 1:3
        Σ = zero(eltype(γ))
        for nx = -Nmax:Nmax, ny = -Nmax:Nmax
            (nx == 0 && ny == 0) && continue
            Σ += Ω(nx, ny, m, mp) *
                  cos(kx*nx*a + ky*ny*a)
        end
        W[m, mp] += Σ
    end
    return W
end

"""
    bloch_3x3_H(H_eff, r, kx, ky) -> Matrix{ComplexF64}(3,3)

Compact 3 × 3 Bloch Hamiltonian from a finite-cluster matrix.

* `H_eff` — 3N × 3N effective Hamiltonian (one Zeeman triplet per site).
* `r`     — length-N vector of in-plane positions (same units as *k*).
* `kx,ky` — Bloch-vector components.

Formula  
`Wₘₘ′(k⃗) = (1/N) ∑_{j,l} H_eff[(j,m),(l,m′)] · e^{-i k⃗·(rⱼ−rₗ)}`

Returns a 3 × 3 `ComplexF64` matrix; eigenvalues are the three bands at
`(kx,ky)`.  *O(N²)* operations.
"""
@inline function bloch_3x3_H(H_eff::AbstractMatrix,
                           r::Vector{SVector{2,Float64}},
                           kx::Real, ky::Real)
    N = length(r)
    W = zeros(ComplexF64, 3, 3)

    @inbounds for j in 1:N, l in 1:N
        phase = exp(-im*(kx*(r[j][1]-r[l][1]) +
                         ky*(r[j][2]-r[l][2])))
        baseJ = 3*(j-1);  baseL = 3*(l-1)

        for m in 1:3, mp in 1:3
            W[m, mp] += H_eff[baseJ+m, baseL+mp] * phase
        end
    end
    return W / N
end

#############################
# Generic band-structure core
#############################

"""
    path_bands(knodes; Nk=200, compute_W, n_bands=3,
               threads=true, keep_k=false)

Diagonalise `compute_W(kx,ky)` on the broken-line path
connecting `knodes = [(kx₁,ky₁),(kx₂,ky₂), …]`.

Returns either

* `ωmat               :: Matrix{Float64}(n_bands, npts)`  or
* `(ωmat, s)          # s = cumulative path length`

depending on `keep_k`.
"""
function path_bands(knodes::AbstractVector{<:Tuple};
                    Nk::Int           = 200,
                    compute_W::F      where F<:Function,
                    n_bands::Int      = 3,
                    threads::Bool     = Threads.nthreads()>1,
                    keep_k::Bool      = false,
                    return_gamma::Bool = false)

    # -- build the k-path -------------------------------------------------
    segment(k1, k2) = let
        (kx1, ky1), (kx2, ky2) = k1, k2
        t = range(0.0, 1.0, Nk+1)[1:end-1]
        collect(zip(kx1 .+ t.*(kx2-kx1), ky1 .+ t.*(ky2-ky1)))
    end
    kpath = reduce(vcat, (segment(knodes[i],knodes[i+1])
                          for i in 1:length(knodes)-1))
    npts  = length(kpath)

    ωmat  = Matrix{Float64}(undef, n_bands, npts)
    Γmat  = return_gamma ? Matrix{Float64}(undef, n_bands, npts) : nothing
    s     = keep_k      ? Vector{Float64}(undef, npts)          : nothing

    work = idx->begin
        kx, ky  = kpath[idx]
        λ       = _eigvals3(compute_W(kx,ky))
        λ_sorted = sort(λ; by = real)
        ωmat[:,idx] .= real.(λ_sorted)
        return_gamma && (Γmat[:,idx] .= -2*imag.(λ_sorted))
    end

    if threads
        Threads.@threads for idx in 1:npts
            work(idx)
        end
    else
        for idx in 1:npts work(idx) end
    end

    if keep_k
        s[1] = 0.0
        for i in 2:npts
            s[i] = s[i-1] + hypot(kpath[i][1]-kpath[i-1][1],
                                  kpath[i][2]-kpath[i-1][2])
        end
    end

    return return_gamma ?
           (keep_k ? (ωmat, Γmat, s) : (ωmat, Γmat)) :
           (keep_k ? (ωmat, s)       :  ωmat)
end


"""
    grid_bands(kx_vec, ky_vec; compute_W, n_bands=3,
               threads=true, keep_k=false)

Diagonalise `compute_W(kx,ky)` on a rectangular grid.

Returns either

* `ωbands[b, ix, iy]`  (`n_bands × Nx × Ny`)  or
* `(ωbands, kx_vec, ky_vec)`
"""
function grid_bands(kx_vec::AbstractVector,
                    ky_vec::AbstractVector;
                    compute_W::F      where F<:Function,
                    n_bands::Int      = 3,
                    threads::Bool     = Threads.nthreads()>1,
                    keep_k::Bool      = false,
                    return_gamma::Bool = false)

    Nx, Ny  = length(kx_vec), length(ky_vec)
    ωbands  = Array{Float64}(undef, n_bands, Nx, Ny)
    Γbands  = return_gamma ? similar(ωbands) : nothing

    work = (ix,iy)->begin
        λ = _eigvals3(compute_W(kx_vec[ix], ky_vec[iy]))
        λ_sorted = sort(λ; by = real)
        ωbands[:,ix,iy] .= real.(λ_sorted)
        return_gamma && (Γbands[:,ix,iy] .= -2*imag.(λ_sorted))
    end

    if threads
        Threads.@threads for idx in 1:Nx*Ny
            ix = (idx-1) % Nx + 1
            iy = (idx-1) ÷ Nx + 1
            work(ix,iy)
        end
    else
        for iy in 1:Ny, ix in 1:Nx
            work(ix,iy)
        end
    end

    return return_gamma ?
           (keep_k ? (ωbands, Γbands, kx_vec, ky_vec)
                    : (ωbands, Γbands)) :
           (keep_k ? (ωbands, kx_vec, ky_vec)
                    :  ωbands)
end

#######################################
# (a) Infinite 2-D square-lattice model
#######################################
function bands_GXMG(a, μ, γ, Δ;
                    ωc=0.0, Nmax=60,
                    Nk=200, keep_k=false,
                    threads=true, return_gamma=false)

    if return_gamma
        R, Ωtab, Γtab = precompute_Omega_Gamma_tables(a, μ, γ; Nmax)
        compute_W = (kx,ky)->omega_gamma_2d_triplet_fast(
                              kx,ky, R,Ωtab,Γtab, Δ,γ; ωc)
    else
        R, Ωtab = precompute_Omega_table(a, μ, γ; Nmax)
        compute_W = (kx,ky)->omega_2d_triplet_fast(
                              kx,ky, R,Ωtab, Δ; ωc)
    end

    knodes = [(π/a,π/a), (0.0,0.0), (π/a,0.0), (π/a,π/a)]
    path_bands(knodes; Nk, compute_W,
               threads, keep_k, return_gamma)
end



########################################
# (b) Empirical Bloch matrix from H_eff
########################################
function bands_GXMG_from_H(H_eff, r, a;
                           Nk=200, keep_k=false, threads=true,
                           return_gamma=false)

    compute_W = (kx,ky)->bloch_3x3_H(H_eff, r, kx, ky)
    knodes = [(0.0,0.0), (π/a,0.0), (π/a,π/a), (0.0,0.0)]
    path_bands(knodes; Nk, compute_W, threads, keep_k, return_gamma)
end

############################################
# Universal rectangular-grid band calculator
############################################
"""
    bands_2d_grid(;           # ← *all arguments are keyword* on purpose
        # ---------- model specification --------------------------------
        a         = nothing,  μ = nothing, γ = nothing, Δ = nothing,
        ωc        = 0.0,      Nmax = 60,                       # ↳ 2-D infinite model
        H_eff     = nothing,  r = nothing,                    # ↳ empirical Bloch model
        compute_W = nothing,                                  # ↳ fully custom closure
        # ---------- k-grid control -------------------------------------
        Nkx   = 201, Nky = 201, fullBZ = false,
        kx_vec = nothing, ky_vec = nothing,                   # ↳ supply explicit grids
        # ---------- misc -----------------------------------------------
        n_bands = 3,  threads = Threads.nthreads()>1, keep_k=false)

Compute the lowest `n_bands` collective modes ω₁,ω₂, … on a
rectangular (kx,ky) grid.

Three mutually exclusive ways to define the Bloch matrix `W(kx,ky)`:

* **(i) Infinite 2-D square lattice**  
  pass   `a, μ, γ, Δ`   (and optionally `ωc, Nmax`).

* **(ii) Empirical Bloch matrix**  
  pass   `H_eff, r`   where `H_eff` is a 3 N×3 N finite-cluster matrix
  and `r` an N-vector of in-plane atomic coordinates (in *units of a*).

* **(iii) Fully custom**  
  pass a closure `compute_W = (kx,ky)-> …` that returns a
  3 × 3 (real-symmetric or Hermitian) matrix.

If you provide explicit `kx_vec, ky_vec`, they override `Nkx, Nky,
fullBZ`.  For custom `compute_W`, grids *must* be given explicitly or
through `kx_vec, ky_vec` (or `a` if you still want the default range).

Return value
------------

* `ωbands[b, ix, iy]`  (`n_bands × Nkx × Nky`)  
  or `(ωbands, kx_vec, ky_vec)` when `keep_k=true`.
"""
function bands_2d_grid(; a=nothing, μ=nothing, γ=nothing, Δ=nothing,
                          ωc=0.0, Nmax=60,
                          H_eff=nothing, r=nothing,
                          compute_W=nothing,
                          Nkx=201, Nky=201, fullBZ=false,
                          kx_vec=nothing, ky_vec=nothing,
                          n_bands=3, threads=Threads.nthreads()>1,
                          keep_k=false, return_gamma=false)

    if compute_W === nothing
        if H_eff !== nothing && r !== nothing
            compute_W = (kx,ky)->bloch_3x3_H(H_eff,r,kx,ky)

        elseif a!==nothing && μ!==nothing && γ!==nothing && Δ!==nothing
            if return_gamma
                R,Ωtab,Γtab = precompute_Omega_Gamma_tables(a,μ,γ;Nmax)
                compute_W = (kx,ky)->omega_gamma_2d_triplet_fast(
                                      kx,ky,R,Ωtab,Γtab,Δ,γ;ωc)
            else
                R,Ωtab = precompute_Omega_table(a,μ,γ;Nmax)
                compute_W = (kx,ky)->omega_2d_triplet_fast(
                                      kx,ky,R,Ωtab,Δ;ωc)
            end
        else
            error("Supply (a,μ,γ,Δ) or (H_eff,r) or compute_W.")
        end
    end

    if kx_vec===nothing || ky_vec===nothing
        a===nothing && error("Need `a` to build default grid.")
        kmin,kmax = fullBZ ? (-π/a,π/a) : (0.0,π/a)
        kx_vec = range(kmin,kmax,Nkx)
        ky_vec = range(kmin,kmax,Nky)
    else
        Nkx,Nky = length(kx_vec), length(ky_vec)
    end

    grid_bands(kx_vec,ky_vec;
               compute_W, n_bands, threads,
               keep_k, return_gamma)
end

end
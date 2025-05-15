module GeomField

using LinearAlgebra, AtomicArrays
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
        W[m, mp] += real(Σ)
    end
    return W
end

function bands_GXMG(a::Real,
                    μ::Vector{<:AbstractVector},
                    γ::AbstractVector,
                    Δ::AbstractVector;
                    ωc   ::Real = 0.0,
                    Nmax ::Int  = 60,
                    Nk   ::Int  = 200,
                    keep_k::Bool = false)

    # High-symmetry k-points (in 1/a units)
    Γ = (0.0, 0.0)
    X = (π/a, 0.0)
    M = (π/a, π/a)
    knodes = (Γ, X, M, Γ)

    # helper: linear interpolation between two points
    function segment(k1, k2)
        (kx1, ky1), (kx2, ky2) = k1, k2
        t = range(0.0, 1.0, Nk+1)[1:end-1]  # exclude endpoint
        kx = kx1 .+ t .* (kx2 - kx1)
        ky = ky1 .+ t .* (ky2 - ky1)
        return collect(zip(kx, ky))
    end

    # build full path
    kpath = vcat([segment(knodes[i], knodes[i+1]) for i in 1:3]...)  # 3*Nk pts
    npts  = length(kpath)

    # allocate band matrix
    ωmat = Matrix{Float64}(undef, 3, npts)

    s = zeros(Float64, npts)   # cumulative distance along the path
    prev = first(kpath)
    for (idx, (kx, ky)) in enumerate(kpath)
        # diagonalise 3×3 matrix
        ωmat[:, idx] = real(sort(eigvals(
            omega_2d_triplet(kx, ky, a, μ, γ, Δ;
                                  ωc = ωc, Nmax = Nmax))))
        # path length
        dx = hypot(kx - prev[1], ky - prev[2])
        s[idx] = (idx == 1) ? 0.0 : s[idx-1] + dx
        prev = (kx, ky)
    end
    return keep_k ? (ωmat, s) : ωmat
end



end
module GeomField

using LinearAlgebra, AtomicArrays
export build_fourlevel_system

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

end
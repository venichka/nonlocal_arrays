module NonlocalArrays

include("algebra.jl")
include("correlations.jl")
include("geom_field.jl")
include("scattering.jl")
include("io_utils.jl")
include("plotting.jl")
include("param_utils.jl")

using .Algebra
using .Correlations
using .GeomField
using .Scattering
using .IOUtils
using .Plotting
using .ParamUtils

# Re‑export public symbols (one‑liners keep the file short)
for M in (Algebra, Correlations, GeomField,
          Scattering, IOUtils, Plotting, ParamUtils)
    for name in names(M; all = false, imported = false)
        @eval export $name
    end
end
end # module NonlocalArrays

module ParamUtils

export params_to_vars!, filter_results, find_state


"""
    params_to_vars!(d::Dict; make_tuple=false, mod=Main)

From a parameter dictionary `d` create variables in module `mod`
(or return a `NamedTuple` when `make_tuple=true`).

* Keys of the form `"nX"` tell you the sweep length of `"X"`.
* Unswept parameters         → scalar (or tuple as-is).
* Swept numeric tuple (a,b)  → `range(a, b; length = nX)`.
* `POLARIZATION`             → `Vector` of strings.
"""
function params_to_vars!(d::Dict{String,Any};
                         make_tuple::Bool=false,
                         mod::Module=Main)

    out = Dict{Symbol,Any}()

    for (key, val) in d
        startswith(key, 'n') && continue          # skip the counter keys

        nkey    = "n$(key)"
        swept   = haskey(d, nkey) && d[nkey] > 1
        sym     = Symbol(lowercase(key))

        value = if key == "POLARIZATION"
            collect(val)                           # always vector
        elseif swept
            if isa(val, Tuple) && length(val)==2 && all(isa.(val, Number))
                range(val[1], val[2]; length=d[nkey])
            else
                collect(val)                       # non-numeric sweep
            end
        else
            val                                   # scalar / unswept tuple
        end

        out[sym] = value
    end

    if make_tuple
        return NamedTuple(out)
    else
        # Inject into the chosen module with Core.eval (global scope)
        for (sym, v) in out
            Core.eval(mod, :( $sym = $v ))
        end
        return nothing
    end
end

"""
    filter_results(results, selectors) -> Dict{Dict{String,Any},Any}

Return a subset of the sweep dictionary `results` that satisfies every rule
in `selectors`.

`results`   – Dict{Dict{String,Any},Any} (keys = parameter sets).  
`selectors` – Dict{String,Any} whose values define matching rules:

| selector type             | pass condition                                     |
|---------------------------|----------------------------------------------------|
| scalar (Number, String…) | `params[key] == selector`                           |
| AbstractArray / `Set`    | `in(params[key], selector)`                         |
| `UnitRange`, `StepRange` | same as array (`in`)                                |
| `Function`               | `selector(params[key])` returns `true`             |

All rules are **AND‑combined**.
"""
function filter_results(results::AbstractDict{<:AbstractDict},
                        selectors::AbstractDict)

    matches(key, rule, params) = begin
        val = get(params, key, nothing)

        if rule isa Function
            rule(val)

        elseif rule isa AbstractArray || rule isa Set
            if val isa AbstractArray                 # ← both arrays: compare
                val == rule
            else                                     # ← scalar vs collection
                in(val, rule)
            end

        else                                         # scalar / numeric range
            val == rule
        end
    end

    Dict( p => r for (p,r) in results
         if all(matches(k,v,p) for (k,v) in selectors) )
end


"""
    find_state(results, fixed_params; all_states = false, throw_on_missing = true)

Search the sweep `results` (a `Dict` whose keys are parameter dictionaries)
for every entry whose parameter set contains **all** of
`fixed_params`.  

Returns  

* `state, params`               – first match         (`all_states = false`)  
* `[(state₁, params₁), …]`      – vector of *all*     (`all_states = true`)  
* `nothing`                     – when no match found and
                                  `throw_on_missing = false`.

Raises an error otherwise.
"""
function find_state(results::AbstractDict, fixed_params::AbstractDict;
                    all_states::Bool = false,
                    throw_on_missing::Bool = true)

    matches(p) = all(get(p, k, nothing) == v for (k, v) in fixed_params)

    if all_states
        out = [(state, params) for (params, state) in results if matches(params)]
        return !isempty(out) ? out :
               (throw_on_missing ? error("No matching state") : nothing)
    else
        for (params, state) in results
            matches(params) && return state, params
        end
        return throw_on_missing ? error("No matching state") : nothing
    end
end

end

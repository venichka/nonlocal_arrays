module IOUtils

using CSV, Printf, Serialization
export save_result, load_all_results, parse_result_filename,
       save_sweep, load_sweep, parse_sweep_filename,
       get_parameters_csv

# Saving serialization

"""
    NonlocalArrays.save_result(path, result, N, B_z, pol, amplitude, a)

Serialize a result to a file with a descriptive name.
"""
function save_result(path::String, result,
                     N::Int, B_z::Real,
                     pol::String, amplitude::Float64, a::Float64)
    filename = joinpath(path,
        @sprintf("result_N%d_Bz%.4f_%s_E%.4f_a%.3f.bson", N, B_z, pol, amplitude, a))
    serialize(filename, result)
    return filename
end

"""
    NonlocalArrays.load_all_results(path)

Deserialize all .bson result files in the directory.
"""
function load_all_results(path::String)
    files = filter(f -> endswith(f, ".bson"), readdir(path; join=true))
    return [deserialize(file) for file in sort(files)]
end

"""
    NonlocalArrays.parse_result_filename(filename::String)

Extract parameters (N, B_z, polarization, amplitude, a) from filename string.
"""
function parse_result_filename(filename::String)
    pattern = r"result_N(\d+)_Bz([-\d.]+)_([RL])_E([\d.]+)_a([\d.]+)\.bson"
    m = match(pattern, basename(filename))
    isnothing(m) && error("Invalid filename format: $filename")

    N        = parse(Int, m.captures[1])
    B_z      = parse(Float64, m.captures[2])
    pol      = m.captures[3]
    E0       = parse(Float64, m.captures[4])
    a_val    = parse(Float64, m.captures[5])

    return (; N, B_z, POLARIZATION = pol, amplitude = E0, a = a_val)
end


"""
    NonlocalArrays.save_sweep(path::String, data;
               description::String = "results", kwargs...)

Save any `data` object with a filename auto-generated from key-value pairs in `kwargs`,
plus a short description. Supports Float, Tuple, Vector, and String keyword arguments.

Examples:
save_sweep(PATH_DATA, data; description="states", Bz=(0.1, 0.2), N=100, POL="R", a=0.2)

Creates a file like:
    states_Bz_0.1_to_0.2_N_100_POL_R_a_0.2.bson

Or parameters can be passed as a single dict:
sweep_params = Dict(
    "a"            => (0.2, 0.25),
    "Bz"          => collect(range(-0.2, 0.2, length=3)),
    "deltas"       => (0.2, 0.3),
    "POLARIZATION" => ("R", "L"),
    "amplitude"    => (0.02, 0.03),
    "anglek"      => [[π/6, 0.0], [π/4, 0.1]]
)
savepath = save_sweep_0(PATH_DATA, data;
                      description = "steady-states-sweep",
                      sweep_params = sweep_params)
"""
function save_sweep(path::String, data;
                    description::String = "results", kwargs...)
    mkpath(path)
    parts = [description]

    # 1) Process sweep_params
    if haskey(kwargs, :sweep_params)
        for (key, val) in pairs(kwargs[:sweep_params])
            k = string(key)

            # --- Tuple of two numeric vectors (e.g. ([π/6,0],[π/4,0.1])) ---
            if isa(val, Tuple) && length(val) == 2 &&
               isa(val[1], AbstractVector{<:Number}) &&
               isa(val[2], AbstractVector{<:Number})
                minv, maxv = val[1], val[2]
                s1 = "[" * join([ @sprintf("%.3f", x) for x in minv ], ",") * "]"
                s2 = "[" * join([ @sprintf("%.3f", x) for x in maxv ], ",") * "]"
                push!(parts, "$(k)_$(s1)_to_$(s2)")
                push!(parts, "n$(k)_2")

            # --- Vector of vectors of numbers (e.g. [[π/6,0],[π/4,0.1],...]) ---
            elseif isa(val, AbstractVector) &&
                   all(x -> isa(x, AbstractVector{<:Number}), val)
                minv, maxv = val[1], val[end]
                s1 = "[" * join([ @sprintf("%.3f", x) for x in minv ], ",") * "]"
                s2 = "[" * join([ @sprintf("%.3f", x) for x in maxv ], ",") * "]"
                push!(parts, "$(k)_$(s1)_to_$(s2)")
                push!(parts, "n$(k)_$(length(val))")

            # — tuple of strings or chars —
            elseif isa(val, Tuple) && all(x->isa(x,AbstractString)||isa(x,Char), val)
                strs = string.(val)
                sweep = join(strs, "_to_")
                push!(parts, "$(k)_$(sweep)", "n$(k)_$(length(val))")

            # — vector of strings or chars —
            elseif isa(val, AbstractVector) && all(x->isa(x,AbstractString)||isa(x,Char), val)
                strs = string.(val)
                if length(val) > 1
                    sweep = join(strs, "_to_")
                    push!(parts, "$(k)_$(sweep)", "n$(k)_$(length(val))")
                else
                    push!(parts, "$(k)_$(strs[1])")
                end

            # - integer scalar -
            elseif isa(val, Int)
                push!(parts, "$(k)_$(val)")

            # — float scalar —
            elseif isa(val, AbstractFloat)
                push!(parts, @sprintf("%s_%.3f", k, val))

            # — string scalar —
            elseif isa(val, AbstractString) || isa(val, Char)
                push!(parts, "$(k)_$(val)")

            # — numeric vector of Numbers —
            elseif isa(val, AbstractVector) && all(x->isa(x,Number), val)
                if length(val) > 2
                    push!(parts, @sprintf("%s_%.3f_to_%.3f", k, minimum(val), maximum(val)))
                    push!(parts, "n$(k)_$(length(val))")
                elseif length(val) == 2
                    s = "[" * join([ @sprintf("%.3f", x) for x in val ], ",") * "]"
                    push!(parts, "$(k)_$s")
                else
                    x = val[1]
                    if isa(x, Int)
                        push!(parts, "$(k)_$(x)")
                    else
                        push!(parts, @sprintf("%s_%.3f", k, x))
                    end
                end

            # — range of numbers —
            elseif isa(val, AbstractRange)
                arr = collect(val)
                push!(parts, @sprintf("%s_%.3f_to_%.3f", k, minimum(arr), maximum(arr)))
                push!(parts, "n$(k)_$(length(arr))")

            # — tuple of two numbers —
            elseif isa(val, Tuple) && length(val)==2 && all(x->isa(x,Number), val)
                push!(parts, @sprintf("%s_%.3f_to_%.3f", k, val[1], val[2]))
                push!(parts, "n$(k)_2")

            else
                error("Unsupported sweep_params[$k] = $val of type $(typeof(val))")
            end
        end
    end

    # 2) Process other kwargs
    for (key, val) in pairs(kwargs)
        if key === :sweep_params; continue; end
        k = string(key)
        strval = if isa(val, Int)
            "$(k)_$(val)"
        elseif isa(val, AbstractFloat)
            @sprintf("%s_%.3f", k, val)
        elseif isa(val, AbstractString)
            "$(k)_$(val)"
        elseif isa(val, AbstractVector) && all(x->isa(x,Number), val)
            if length(val) > 2
                @sprintf("%s_%.3f_to_%.3f", k, minimum(val), maximum(val))
            elseif length(val)==2
                s = "[" * join([ @sprintf("%.3f", x) for x in val ], ",") * "]"
                "$(k)_$s"
            else
                x = val[1]
                isa(x,Int) ? "$(k)_$(x)" : @sprintf("%s_%.3f", k, x)
            end
        elseif isa(val, Tuple) && length(val)==2 && all(x->isa(x,Number), val)
            @sprintf("%s_%.3f_to_%.3f", k, val[1], val[2])
        else
            error("Unsupported kwarg $k => $val of type $(typeof(val))")
        end
        push!(parts, strval)
    end

    # 3) Save
    filename = join(parts, "_") * ".bson"
    filepath = joinpath(path, filename)
    Serialization.serialize(filepath, data)
    return filepath
end

"""
    load_sweep(filepath::AbstractString) -> (data, params)

Reads the .bson file at `filepath`, deserializes the contents back into
`data`, and parses `filepath`’s filename to recover its sweep parameters.
"""
function load_sweep(filepath::AbstractString)
    data   = Serialization.deserialize(filepath)
    params = parse_sweep_filename(filepath)
    return data, params
end

"""
    NonlocalArrays.parse_sweep_filename(filename::String) -> Dict{String, Any}

Parses filenames like:
"steady_states_Bz_-0.200_to_0.200_num_vals_36_N_100_POL_R_E0_0.020_a_0.200_geometry_rect.bson"
into a dictionary of parameter names and values.
"""
function parse_sweep_filename(filename::String)::Dict{String, Any}
    # Remove directory components and file extension.
    base = splitext(basename(filename))[1]
    # Split the base name on underscores.
    tokens = split(base, '_')
    
    # By convention, discard the first token (the file prefix).
    if length(tokens) > 1
        tokens = tokens[2:end]
    end

    params = Dict{String, Any}()
    i = 1
    while i <= length(tokens)
        # If there is no value token, break.
        if i == length(tokens)
            break
        end

        key = tokens[i]
        # Check if a range is indicated by the literal token "to"
        if i + 2 <= length(tokens) && tokens[i+2] == "to" && i + 3 <= length(tokens)
            part1 = tokens[i+1]
            part2 = tokens[i+3]
            # If both parts are enclosed in brackets, parse them as vectors.
            if startswith(strip(part1), "[") && endswith(strip(part1), "]") &&
               startswith(strip(part2), "[") && endswith(strip(part2), "]")
                params[key] = (parse_vector(part1), parse_vector(part2))
            else
                # Otherwise, try to parse both parts as numbers.
                n1 = tryparse(Float64, part1)
                n2 = tryparse(Float64, part2)
                if n1 !== nothing && n2 !== nothing
                    params[key] = (n1, n2)
                else
                    # Otherwise, treat them as strings.
                    params[key] = (part1, part2)
                end
            end
            i += 4
        else
            # Single value: attempt to parse as Int first, then Float; if that fails, leave as String.
            val_token = tokens[i+1]
            iv = tryparse(Int, val_token)
            if iv !== nothing
                params[key] = iv
            else
                fv = tryparse(Float64, val_token)
                if fv !== nothing
                    params[key] = fv
                else
                    params[key] = val_token
                end
            end
            i += 2
        end
    end
    return params
end


# Function to retrieve parameters from the CSV file based on specified fields
function get_parameters_csv(csv_file, state, N, geometry, detuning_symmetry, direction)
    # Read the CSV file into a DataFrame
    df = CSV.read(csv_file, DataFrame)

    # Filter the DataFrame based on the specified fields
    filtered_df = filter(row -> row.State_proj_max == state && row.N == N && row.geometry == geometry &&
                                row.detuning_symmetry == detuning_symmetry && row.Direction == direction, df)

    # Check if any rows match the criteria
    if nrow(filtered_df) == 0
        println("No matching parameters found.")
        return nothing
    end

    # Extract the desired parameters
    a = filtered_df.a[1]
    E₀ = filtered_df.E₀[1]
    Δ_params = zeros(Float64, N)
    for i in 1:N
        Δ_params[i] = filtered_df[!, Symbol("Δ_$i")][1]
    end

    return Dict("a" => a, "E_0" => E₀, "Δ_vec" => Δ_params)
end

# ------- Helper functions -------

# Helper function: parse a vector string like "[0.524,0.000]" into a vector of Float64.
function parse_vector(s::AbstractString)
    # Remove any leading/trailing whitespace and square brackets.
    s = strip(s, ['[', ']'])
    # Split on comma.
    parts = split(s, ',')
    # Parse each part as Float64, rounding via @sprintf can be done later if needed.
    return [parse(Float64, strip(p)) for p in parts]
end

end

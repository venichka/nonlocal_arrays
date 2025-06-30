module Plotting

using CairoMakie, AtomicArrays, LinearAlgebra
import ..GeomField: build_fourlevel_system
export plot_atoms_with_field,
       plot_sweep_quantity, plot_sweep_multicurve, plot_sweep_heatmap,
       field_intensity_map

# Function to extract positions from the FourLevelAtomCollection and plot them
function plot_atoms_with_field(coll::AtomicArrays.FourLevelAtomCollection, field::AtomicArrays.field.EMField)
    CairoMakie.activate!()
    # Extract atom positions
    xs = [atom.position[1] for atom in coll.atoms]
    ys = [atom.position[2] for atom in coll.atoms]
    zs = [atom.position[3] for atom in coll.atoms]

    # Create a new figure and add a 3D axis.
    fig = Figure(size = (800, 600))
    ax = Axis3(fig[1, 1];
        title = "3D Array of Atoms with Field Vectors",
        xlabel = "x", ylabel = "y", zlabel = "z",
        perspectiveness = 0.8
    )

    # Plot atom positions
    scatter!(ax, xs, ys, zs; markersize = 10, color = :blue)

    # Field origin
    field_origin = Point3f(field.position_0...)

    # Arrow scaling
    k_scale = 1.0
    pol_scale = 1.0

    # k-vector arrow
    k_arrow = Vec3f(normalize(field.k_vector)...) * k_scale
    arrows!(ax, [field_origin], [k_arrow]; linewidth = 0.02, arrowsize = 0.04, color = :red)
    text!(ax, "k-vector"; position = field_origin + k_arrow, align = (:center, :bottom), color = :red)

    # polarization arrow
    pol_arrow = Vec3f(real.(field.polarisation)...) * pol_scale
    pol_arrow_im = Vec3f(imag(field.polarisation)...) * pol_scale
    arrows!(ax, [field_origin], [pol_arrow]; linewidth = 0.02, arrowsize = 0.04, color = :green)
    arrows!(ax, [field_origin], [pol_arrow_im]; linewidth = 0.02, arrowsize = 0.04, color = :green)
    text!(ax, "polarization"; position = field_origin + pol_arrow, align = (:center, :top), color = :green)

    return fig
end

"""
    plot_sweep_quantity(results, quantity_func, xparam; fixed_params=Dict(), label_func=identity, sort_x=true)

Plot a single quantity versus one swept parameter.
"""
function plot_sweep_quantity(results::Union{Dict{Dict{String,Any}, Any},
                                              Dict{Dict{String,Any}, AtomicArrays.fourlevel_meanfield.ProductState{Int64, Float64}}},
                             quantity_func::Function,
                             xparam::String;
                             fixed_params::Dict{String,Any}=Dict(),
                             label_func::Function=identity,
                             sort_x::Bool=true,
                             ylabel="",
                             return_data=false)
    xs, ys = Float64[], Float64[]
    for (params, result) in results
        match = all(get(params, k, nothing) == v for (k, v) in fixed_params)
        if match && haskey(params, xparam)
            push!(xs, params[xparam])
            push!(ys, quantity_func(result, params))
        end
    end
    if sort_x
        ord = sortperm(xs)
        xs, ys = xs[ord], ys[ord]
    end
    if return_data
        return xs, ys
    else
        fig = Figure()
        ax = Axis(fig[1, 1], xlabel = xparam, ylabel = ylabel, title = "Sweep: $xparam")
        lines!(ax, xs, ys, label = label_func(fixed_params))
        axislegend(ax)
        return ax, fig
    end
end

"""
    plot_sweep_multicurve(results, quantity_func, xparam, curve_param; fixed_params=Dict())

Plot multiple curves with varying `curve_param`, each curve showing quantity vs `xparam`.
"""
function plot_sweep_multicurve(results::Union{Dict{Dict{String,Any}, Any},
                                              Dict{Dict{String,Any}, AtomicArrays.fourlevel_meanfield.ProductState{Int64, Float64}}},
                                quantity_func::Function,
                                xparam::String,
                                curve_param::String;
                                fixed_params::Dict{String,Any}=Dict(),
                                ylabel::String="")
    grouped = Dict{Any, Vector{Tuple{Float64, Float64}}}()
    for (params, result) in results
        match = all(get(params, k, nothing) == v for (k, v) in fixed_params)
        if match && haskey(params, xparam) && haskey(params, curve_param)
            xval = params[xparam]
            curveval = params[curve_param]
            yval = quantity_func(result, params)
            push!(get!(grouped, curveval, []), (xval, yval))
        end
    end
    unzip(pairs::Vector{<:Tuple}) = map(x -> getindex.(pairs, x), 1:2)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = xparam, ylabel = ylabel, title = "$xparam vs $curve_param")
    for (curveval, data) in grouped
        xs, ys = unzip(data)
        ord = sortperm(xs)
        lines!(ax, xs[ord], ys[ord], label = "$curve_param = $curveval")
    end
    axislegend(ax)
    return ax, fig
end

"""
    plot_sweep_heatmap(results, quantity_func, xparam, yparam;
                       fixed_params = Dict(),
                       zlabel       = "",
                       figure_kw    = (;),      # kwargs → Figure()
                       axis_kw      = (;),      # kwargs → Axis()
                       heatmap_kw   = (;),      # kwargs → heatmap!
                       colorbar_kw  = (;),      # kwargs → Colorbar()
                       logx = false, logy = false)

Generate a heat-map of `quantity_func(result, params)` vs. two swept
parameters stored in `results::Dict`.

# Keyword interface
* `figure_kw`   – pass any `Figure` attributes (e.g. `resolution = (800,600)`).
* `axis_kw`     – Axis attributes (`title`, `xlabel`, `ylabel`, `xscale = log10`, …).
* `heatmap_kw`  – Heatmap attributes (`colormap = :viridis`,
                  `colorrange = (-1,1)`, `nan_color = RGBAf0(0,0,0,0)`, …).
* `colorbar_kw` – Colorbar attributes (`label = "my z"`, `ticks`, …).
* `logx|logy`   – quick flags for log scaling of axes.

Returns `(fig, xgrid, ygrid, Z)`.
"""
function plot_sweep_heatmap(results::Union{Dict{Dict{String,Any}, Any},
                                              Dict{Dict{String,Any}, AtomicArrays.fourlevel_meanfield.ProductState{Int64, Float64}}},
                             quantity_func::Function,
                             xparam::String,
                             yparam::String;
                             fixed_params::Dict{String,Any}=Dict(),
                             figure_kw::NamedTuple  = (;),
                             axis_kw::NamedTuple    = (;),
                             heatmap_kw::NamedTuple = (;),
                             colorbar_kw::NamedTuple = (;),
                             logx::Bool = false,
                             logy::Bool = false,
                             data::Union{Nothing,Tuple} = nothing)

    # ---------------- Aggregate data ------------------------------------
    if data === nothing
        vals  = Dict{Tuple{Float64,Float64},Float64}()
        xvals = Float64[]
        yvals = Float64[]

        for (params, result) in results
            match = all(get(params, k, nothing) == v for (k,v) in fixed_params)
            if match && haskey(params, xparam) && haskey(params, yparam)
                x = Float64(params[xparam])
                y = Float64(params[yparam])
                push!(xvals, x);  push!(yvals, y)
                vals[(x,y)] = quantity_func(result, params)
            end
        end

        xgrid = sort(unique(xvals))
        ygrid = sort(unique(yvals))
        Z     = [get(vals, (x,y), NaN) for y in ygrid, x in xgrid]
    else
        xgrid, ygrid, Z = data
    end

    # ---------------- Figure & axis -------------------------------------
    fig = Figure(; figure_kw...)
    default_title = "Heatmap: $yparam vs $xparam"
    axis_kw = merge( (; xlabel=xparam, ylabel=yparam, title=default_title), axis_kw)

    # log scales if requested
    if logx; axis_kw = merge(axis_kw, (; xscale=log10));  end
    if logy; axis_kw = merge(axis_kw, (; yscale=log10));  end

    ax = Axis(fig[1,1]; axis_kw...)

    # ---------------- Heatmap + colorbar --------------------------------
    hmap = heatmap!(ax, xgrid, ygrid, Z'; heatmap_kw...)
    Colorbar(fig[1,2], hmap; colorbar_kw...)

    return fig, xgrid, ygrid, Z
end

"""
    field_intensity_map(ax_E, ax_I, p. result; plane="xz",
                         npoints=100, scale=3.5)

Draw maps of **Re/|E|** and **Intensity** on the chosen plane
(`"xy"`, `"xz"`, `"yz"`).

* `p`      – parameter Dict (contains `Nx`, `Ny`, `a`, …)
* `result` – steady-state ProductState
* `ax_E`   – 3×2 `Axis` grid (Re/Abs of Ex,Ey,Ez)
* `ax_I`   – single `Axis` for total intensity
"""
function field_intensity_map(; p::Dict, 
                               result,
                               plane::String="xz",
                               inc_field_profile::Function,
                               npoints=100, scale=3.5,
                               data::Union{Nothing, NamedTuple}=nothing)
    # build system -----------------------------------------------------
    coll, Ein, field_func, _ = build_fourlevel_system(
                                merge(p, Dict("field_func"=>inc_field_profile)))
    if data === nothing

        σ  = AtomicArrays.fourlevel_meanfield.sigma_matrices([result], 1)[1]

        Nx, Ny, a = p["Nx"], p["Ny"], p["a"]

        # grid -------------------------------------------------------------
        lim          = Nx * a/2 * scale
        coord        = range(-lim, lim; length=npoints)

        # choose varying / fixed coordinates ------------------------------
        fixed_val, set_r!, atom_coords, label1, label2 = plane == "xy" ? (
            Nx*a/2+0.2,
            (r,x,y)->(r[1]=x; r[2]=y; r[3]=fixed_val), 
            ([atom.position[1] for atom in coll.atoms],
             [atom.position[2] for atom in coll.atoms]), "x","y") :
        plane == "xz" ? ((isodd(Nx) ? a/2 : 0.0),
            (r,x,z)->(r[1]=x; r[2]=fixed_val; r[3]=z), 
            ([atom.position[1] for atom in coll.atoms],
             [atom.position[3] for atom in coll.atoms]),"x","z") :
        plane == "yz" ? (Nx*a/2+0.2,
            (r,y,z)->(r[1]=fixed_val; r[2]=y; r[3]=z),
            ([atom.position[2] for atom in coll.atoms],
             [atom.position[3] for atom in coll.atoms]), "y","z") :
        error("plane must be xy, xz or yz")

        nx = ny = npoints
        ReE, AbsE = [zeros(nx,ny) for _ in 1:3], [zeros(nx,ny) for _ in 1:3]
        Igrid     = zeros(nx,ny)
        Igrid_sc  = zeros(nx,ny)

        # main loop (threaded) --------------------------------------------
        Threads.@threads for j in 1:ny
            r = zeros(3)
            for i in 1:nx
                x1, x2 = coord[i], coord[j]
                set_r!(r, x1, x2)               # fill r depending on plane

                Esc  = AtomicArrays.field.scattered_field(r, coll, σ)/√3
                Etot = field_func(r, Ein) + Esc
                I_sc    = norm(Esc)^2 / abs2(Ein.amplitude)
                I_tot    = norm(Etot)^2 / abs2(Ein.amplitude)

                ReE[1][i,j] = real(Esc[1]); AbsE[1][i,j] = abs(Esc[1])
                ReE[2][i,j] = real(Esc[2]); AbsE[2][i,j] = abs(Esc[2])
                ReE[3][i,j] = real(Esc[3]); AbsE[3][i,j] = abs(Esc[3])
                Igrid[i,j]  = I_tot
                Igrid_sc[i,j]  = I_sc
            end
        end
    else
        coord = data.x
        ReE = data.reE
        AbsE = data.absE
        Igrid = data.I_tot
        Igrid_sc = data.I_sc
    end

    # ---- plotting ----------------------------------------------------
    fig    = Figure(size = (900,1100))
    ax_E = [Axis(fig[i,j]; aspect=DataAspect()) for i in 1:3, j in 1:2]
    ax_I   = [Axis(fig[4,j]; aspect=DataAspect()) for j in 1:2]
    comps = ("E_x","E_y","E_z")
    for k in 1:3
        cf_re = contourf!(ax_E[k,1], coord, coord, ReE[k], colormap=:plasma,
                  levels=10);  ax_E[k,1].title = "Re("*comps[k]*")"
        scatter!(ax_E[k,1], atom_coords[1], atom_coords[2], markersize = 8, color = :white)
        Colorbar(fig[k, 3], cf_re, width = 15, height = Relative(1))
        cf_abs= contourf!(ax_E[k,2], coord, coord, AbsE[k], colormap=:plasma,
                  levels=10);  ax_E[k,2].title = "Abs("*comps[k]*")"
        scatter!(ax_E[k,2], atom_coords[1], atom_coords[2], markersize = 8, color = :white)
        Colorbar(fig[k, 4], cf_abs, width = 15, height = Relative(1))
    end
    for j in 1:2
        I = (j == 1) ? Igrid : Igrid_sc
        cf_in = contourf!(ax_I[j], coord, coord, I, 
                          colormap=:plasma, levels=40)
        scatter!(ax_I[j], atom_coords[1], atom_coords[2], markersize = 8, color = :white)
        Colorbar(fig[4, j+2], cf_in, width = 15, height = Relative(1))
        ax_I[j].title = "Intensity"
        ax_I[j].xlabel, ax_I[j].ylabel = label1, label2
    end
    return fig, ax_E, ax_I, (x=coord, y=coord, reE=ReE, absE=AbsE, 
                               I_tot=Igrid, I_sc=Igrid_sc)
end


end

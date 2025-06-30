# 4levele_data_processing_updated.jl
begin
    using Pkg
    Pkg.activate(pwd()[end-14:end] == "nonlocal_arrays" ? "." : "../")
end

# ── dependencies ------------------------------------------------------
using CairoMakie, GLMakie, LinearAlgebra, FFTW
using QuantumOptics, AtomicArrays, NonlocalArrays, BenchmarkTools
using StaticArrays

# ── 1. CONFIGURATION ──────────────────────────────────────────────────
"""
All user-tuneable knobs live here.
"""
Base.@kwdef mutable struct Config
    # --- I/O -----------------------------------------------------------
    data_file::String = joinpath("/Users/nikita/Documents",#"..", "Data",
        "steady-states-sweep_deltas_-1.800_to_1.800_ndeltas_100_\
         anglek_[0.000,0.000]_to_[0.000,0.000]_nanglek_1_\
         Nx_8_Ny_8_POLARIZATION_R_to_L_nPOLARIZATION_2_\
         amplitude_0.020_to_0.200_namplitude_4_\
         a_0.250_to_1.000_na_100_Bz_0.000_to_0.200_nBz_3.bson")

    save_figs::Bool  = false
    figs_dir::String = "../Figs"

    # --- sweep sub-selection ------------------------------------------
    fixed::Dict{String,Any} = Dict(
        "a"           => 0.3181818181818182,           # lattice const
        "deltas"      => 0.20,
        "Bz"          => 0.20,           # Zeeman field
        "amplitude"   => 0.02,           # drive
        "anglek"      => [0.0, 0.0],     # incidence
        "Nx"          => 8,
        "Ny"          => 8,
        "POLARIZATION"=> "R",
    )

    vary_x::String  = "deltas"           # x-axis in 1-D plots
    vary_y::String  = "a"                # 2-D heat-map second axis

    # --- physics -------------------------------------------------------
    field_profile   = AtomicArrays.field.gauss
    zlim::Float64   = 30.0
    surface         = :hemisphere        # or :plane

    # --- plotting tweaks ----------------------------------------------
    Nk_path::Int    = 500                # resolution Γ–X–M–Γ
    Nk_grid::Int    = 181                # k-grid per axis
end

cfg = Config()         # <<<<<<  CHANGE ONLY HERE  <<<<<<

# ── 2. HELPERS ────────────────────────────────────────────────────────
const PATH_FIGS = cfg.figs_dir

"""
    @timed expr

Execute `expr`, log how long it took, and return its value.
"""
macro timed(ex)
    quote
        _t0  = time()
        _val = $(esc(ex))

        file = String(@__FILE__)      # Symbol  →  String
        line = @__LINE__

        @info "$(basename(file)):$line finished in $(round(time() - _t0, digits=2)) s"
        _val
    end
end


"""
Convert the BSON file into `(results, params_dict, pars)` and cache it.
"""
function load_sweep_cached(file::String)
    results, params = NonlocalArrays.load_sweep(file)
    pars            = NonlocalArrays.params_to_vars!(params; make_tuple=true)
    return results, params, pars
end

"""
`filter_by_fixed(results; fixed)` – keep only entries matching `fixed`.
"""
filter_by_fixed(res; fixed=Dict()) =
    NonlocalArrays.filter_results(res, fixed)

# --- physics post-processors -----------------------------------------

_tr_coeff(result, p; reflection=false) = let
    coll, field, _ = NonlocalArrays.build_fourlevel_system(
                         merge(p, Dict("field_func"=>cfg.field_profile)))
    σ = AtomicArrays.fourlevel_meanfield.sigma_matrices([result],1)[1]
    TR = AtomicArrays.field.transmission_reflection(field, coll, σ;
            beam=:gauss, surface=cfg.surface, samples=400, zlim=cfg.zlim,
            size=[5,5])
    (reflection) ? TR[2] : TR[1]
end

transmission(result, p) = _tr_coeff(result, p; reflection=false)
reflection(result, p) = _tr_coeff(result, p; reflection=true)

mirror_metric(result, p) = let
    coll, field, _ = NonlocalArrays.build_fourlevel_system(
                         merge(p, Dict("field_func"=>cfg.field_profile)))
    σ = AtomicArrays.fourlevel_meanfield.sigma_matrices([result],1)[1]
    TR = NonlocalArrays.transmission_reflection_new(field, coll, σ;
            beam=:gauss, surface=:plane, samples=40, zlim=cfg.zlim,
            size=(2,2), return_helicity=true, return_powers=false)
    NonlocalArrays.chiral_mirror_metrics(
        TR.T_sigma_plus, TR.T_sigma_minus,
        TR.R_sigma_plus, TR.R_sigma_minus).obj
end

"""
    plot_sweep_multicurve(results, quantity, xparam, curve_param;
                          fixed_params = nothing, ylabel = "")

Plot `quantity(result, params)` versus `xparam`, drawing one curve
for every distinct value of `curve_param`.

If `fixed_params` is

* a `Dict`  → only entries matching that dictionary are used;
* `nothing` → the function detects parameters that are *constant* across
  the whole sweep (excluding `xparam` and `curve_param`) and uses those
  as the implicit fixed slice.
"""
function plot_sweep_multicurve_new(
        results      :: AbstractDict,
        quantity     :: Function,
        xparam       :: AbstractString,
        curve_param  :: AbstractString;
        fixed_params :: Dict{String,Any},
        ylabel       :: AbstractString = "")

    # ── 0. validate input & build constant slice once ──────────────────
    isempty(fixed_params) && error("`fixed_params` must not be empty.")

    # keep a *copy* so we don't mutate the caller's dictionary
    const_slice = Dict(kv for kv in fixed_params     # drop xparam/curve_param
                       if first(kv) ∉ (xparam, curve_param))

    fixed_keys  = collect(keys(const_slice))         # faster inside loop

    # ── 1. bucket by `curve_param` ─────────────────────────────────────
    curves = Dict{Any, Vector{Tuple{Float64,Float64}}}()

    # for (p, r) in results
    pairs_vec = collect(results)
    for i in eachindex(pairs_vec)
        p, r = pairs_vec[i]              # destructure the Pair
        # fast path: reject as soon as one key mismatches
        @inbounds begin
            ok = true
            for k in fixed_keys
                if get(p, k, nothing) != const_slice[k]
                    ok = false
                    break
                end
            end
            ok || continue
        end

        x = Float64(p[xparam])
        c = p[curve_param]

        # quantity can use the *original* parameter dict `p`
        y = quantity(r, p)

        push!(get!(curves, c, Vector{Tuple{Float64,Float64}}()), (x, y))
    end

    isempty(curves) && error("No data matched the supplied fixed_params.")

    # ── 2. plot ────────────────────────────────────────────────────────
    fig = Figure()
    ax  = Axis(fig[1, 1];
               xlabel = xparam,
               ylabel = ylabel,
               title  = "$xparam vs $curve_param")

    for (c, pairs) in sort!(collect(curves); by = first)
        xs = first.(pairs)
        ys = last.(pairs)
        ord = sortperm(xs)
        lines!(ax, xs[ord], ys[ord]; label = "$curve_param = $c")
    end

    axislegend(ax)
    return ax, fig
end



# ── 3. WORKFLOW ───────────────────────────────────────────────────────
CairoMakie.activate!()
@timed begin
results, params_dict, pars = load_sweep_cached(cfg.data_file)
sliced_dict = Dict(kv for kv in cfg.fixed
                   if first(kv) ∉ (cfg.vary_x, cfg.vary_y))
sel = filter_by_fixed(results; fixed=sliced_dict)

@info "Found $(length(sel)) states for the chosen slice."
end

begin
# --- 1-D curves -------------------------------------------------------
x, y  = plot_sweep_quantity(sel, transmission, cfg.vary_x;
                            fixed_params=Dict(kv for kv in cfg.fixed
                   if first(kv) != cfg.vary_x), return_data=true)
end

@timed begin
# --- multi-curve example (polarisation) ------------------------------
par1 = cfg.vary_x
par2 = "POLARIZATION"
sliced_dict_mult = Dict(kv for kv in cfg.fixed
                   if first(kv) ∉ (par1, par2))
ax_T, fig_T = plot_sweep_multicurve(results, reflection, par1, par2;
                                    fixed_params=sliced_dict_mult)
end

begin
# --- 2-D heat map -----------------------------------------------------
xv = sort(unique(p[cfg.vary_x] for p in keys(sel)))
yv = sort(unique(p[cfg.vary_y] for p in keys(sel)))
Z = [ let p = merge(cfg.fixed,
                    Dict(cfg.vary_x => a,
                         cfg.vary_y => b))
          haskey(sel, p) ? mirror_metric(sel[p], p) : NaN
     end
     for b in yv, a in xv ]   # <-- (row, column) order

fig_H = Figure(); Axis(fig_H[1,1]; xlabel=cfg.vary_x, ylabel=cfg.vary_y)
heatmap!(xv, yv, Z'; colormap=:plasma, colorrange=(0,1))
end

begin
# --- band structure along Γ–X–M–Γ ------------------------------------
μ  = [AtomicArrays.fourlevel_misc.polarizations_spherical()[i,:] for i=1:3]
γ  = AtomicArrays.fourlevel_misc.gammas(0.25)
Δ  = [-1,0,1] .* cfg.fixed["Bz"]
a  = cfg.fixed["a"]

ω, γk, s = NonlocalArrays.bands_GXMG(a, μ, γ, Δ;
              Nmax=200, Nk=cfg.Nk_path, keep_k=true, return_gamma=true)

fig_B = Figure(size=(800,400))
axω   = Axis(fig_B[1,1]; ylabel="ω", xticks=(s[[1,cfg.Nk_path+1,2cfg.Nk_path+1,end]],
                                            ["M","Γ","X","M"]))
axγ   = Axis(fig_B[2,1]; ylabel="γ", xticklabelsvisible=false)
for b in 1:3
    lines!(axω, s, ω[b,:]; linewidth=2)
    lines!(axγ, s, γk[b,:]; linewidth=2)
end
vlines!(axω, s[[1,cfg.Nk_path+1,2cfg.Nk_path+1,end]]; color=:grey, linestyle=:dash)
vlines!(axγ, s[[1,cfg.Nk_path+1,2cfg.Nk_path+1,end]]; color=:grey, linestyle=:dash)
end

begin
# Field plots: Re, Abs, Intensity
# choose a slice & state ------------------------------------------------
plane = "xz"
p_fixed = merge(cfg.fixed, Dict("POLARIZATION" => "R",
                                "deltas"       => pars.deltas[11]))
result, _ = find_state(results, p_fixed)

fig_F = NonlocalArrays.field_intensity_map(; p=p_fixed, result=result,
                                             inc_field_profile=cfg.field_profile)[1]

fig_F
end



# ---- save / display --------------------------------------------------
cfg.save_figs && foreach((name,fig)->save(joinpath(PATH_FIGS,name*".pdf"),fig),
     [("curve_T",fig_T), ("heatmap_metric",fig_H), ("bands",fig_B)])

# fig_T, fig_H, fig_B   # show in REPL / Pluto / VSCode

fig_T
fig_H
fig_B

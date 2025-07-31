# 4levele_data_processing_updated.jl
begin
    using Pkg
    Pkg.activate(pwd()[end-14:end] == "nonlocal_arrays" ? "." : "../")
end

# ── dependencies ------------------------------------------------------
using CairoMakie, GLMakie, LinearAlgebra, FFTW
using QuantumOptics, AtomicArrays, NonlocalArrays, BenchmarkTools
using ProgressMeter
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
        "deltas"      => 0.16363636363636364,#0.12727272727272726,#0.05454545454545454,
        "Bz"          => 0.2,           # Zeeman field
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

"""
    chiral_plane_map(results; vary   = ["deltas","a"],
                               fixed  :: Dict,
                               pars   :: NamedTuple,
                               profile = AtomicArrays.field.gauss,
                               surface = :plane,
                               zlim    = 50.0,
                               samples = 400)   -> (HP, CD, χ, xs, ys)

Compute helicity–preserving (HP), circular-dichroism (CDᵣ) and their
product X = HP⋅CDᵣ on a rectangular grid of 1–3 varying parameters.

`vary` is an ordered vector of parameter names (`String`s) you want to
scan (∈ `pars`), e.g. `["deltas"]`, `["deltas","a"]`, or
`["deltas","a","Bz"]`.

`fixed` **must** include every other parameter present in `results`
(typical slice dictionary).  The function returns
HP, CD, χ, axes...

* `HP`, `CD`, `χ` – `Float64` arrays with one dimension per entry in
  `vary` (lengths inherited from `pars` fields).
* `axes` – the vectors that label each axis (`xs`, `ys`, `zs`).

A threaded sweep is used; progress is shown with `ProgressMeter`.
"""
function chiral_plane_map(results :: AbstractDict;
                          vary    :: Vector{String},
                          fixed   :: Dict{String,Any},
                          pars    :: NamedTuple,
                          profile          = AtomicArrays.field.gauss,
                          surface::Symbol  = :plane,
                          zlim::Real       = 50.0,
                          samples::Integer = 400)

    nv = length(vary)

    # build axis vectors -------------------------------------------------
    axes = map(vary) do p
        if p == "Bz"
            p = "bz"
        elseif p == "Nx"
            p = "nx"
        elseif p == "Nx"
            p = "ny"
        end
        getfield(pars, Symbol(p)) |> collect   # ensure vector
    end

    dims = map(length, axes)
    HP  = fill(NaN, dims...)
    CD  = similar(HP)
    X   = similar(HP)

    # fixed slice (remove possibly present vary keys):
    const_slice = Dict(kv for kv in fixed if first(kv) ∉ vary)

    # progress bar -------------------------------------------------------
    tot = prod(dims)
    prog = Progress(tot; desc = "chiral-map")

    Threads.@threads for idx in CartesianIndices(HP)
        # thread-local parameter dict -----------------------------------
        p = copy(const_slice)
        for (i, key) in enumerate(vary)
            p[key] = axes[i][idx[i]]
        end

        # look up two polarisations ------------------------------------
        pR   = merge(p, Dict("POLARIZATION"=>"R"))
        pL   = merge(p, Dict("POLARIZATION"=>"L"))
        rR   = get(results, pR, nothing)
        rL   = get(results, pL, nothing)
        (rR===nothing || rL===nothing) && (next!(prog); continue)

        # build systems only once per polarisation ---------------------
        build_sys = (pr, res)->begin
            coll, field, _ = NonlocalArrays.build_fourlevel_system(
                                 merge(pr, Dict("field_func"=>profile)))
            σ = AtomicArrays.fourlevel_meanfield.sigma_matrices([res],1)[1]
            NonlocalArrays.transmission_reflection_new(field, coll, σ;
                beam=:gauss, surface=surface,
                samples=samples, zlim=zlim, size=(2,2),
                return_helicity=true, return_powers=false)
        end
        TR_R = build_sys(pR, rR)
        TR_L = build_sys(pL, rL)

        metr = NonlocalArrays.chiral_mirror_metrics_new(
                [TR_R.R_sigma_minus TR_R.R_sigma_plus;
                 TR_L.R_sigma_minus TR_L.R_sigma_plus])

        HP[idx] = metr.HP
        CD[idx] = metr.CD_R
        X[idx]  = HP[idx] * CD[idx]

        next!(prog)
    end
    return HP, CD, X, axes...
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


# chiral mirror metrics
HP, CD, chi, xs, ys = chiral_plane_map(
        results;
        vary  = ["deltas", "a"],     # one or two or three parameters
        fixed = Dict(k=>v for (k,v) in cfg.fixed if k ≠ "POLARIZATION"),
        pars  = pars,               # the NamedTuple from params_to_vars!
        surface=:plane,
        samples=20,
        zlim  = cfg.zlim)

let 
fig = Figure()
default_title = "Heatmap: $(cfg.vary_x) vs $(cfg.vary_y)"
axis_kw = (; xlabel=cfg.vary_x, ylabel=cfg.vary_y, title=default_title)

ax = Axis(fig[1,1]; axis_kw...)

hmap = heatmap!(ax, xs, ys, chi; colormap=:plasma)
vlines!(cfg.fixed["deltas"]; linestyle=:dash, color=:black)
Colorbar(fig[1,2], hmap; )
fig
end

let
fig_HP = Figure()
ax_HP = Axis(fig_HP[1, 1], xlabel = cfg.vary_x, ylabel = "HP", title = "Sweep: $(cfg.vary_x)")
lines!(ax_HP, xs, chi)
fig_HP
end


begin
# --- 1-D curves -------------------------------------------------------
x, y  = plot_sweep_quantity(sel, transmission, cfg.vary_x;
                            fixed_params=Dict(kv for kv in cfg.fixed
                   if first(kv) != cfg.vary_x), return_data=true)
end

@timed begin
# --- multi-curve example (polarisation) ------------------------------
par1 = cfg.vary_y
par2 = "POLARIZATION"
sliced_dict_mult = Dict(kv for kv in cfg.fixed
                   if first(kv) ∉ (par1, par2))
ax_T, fig_T = plot_sweep_multicurve(results, reflection, par1, par2;
                                    fixed_params=sliced_dict_mult)
fig_T
end

begin
# --- 2-D heat map -----------------------------------------------------
xv = sort(unique(p[cfg.vary_x] for p in keys(sel)))
yv = sort(unique(p[cfg.vary_y] for p in keys(sel)))
Z = [ let p = merge(cfg.fixed,
                    Dict(cfg.vary_x => a,
                         cfg.vary_y => b))
          haskey(sel, p) ? reflection(sel[p], p) : NaN
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
fig_B
end

begin
# Field plots: Re, Abs, Intensity
# choose a slice & state ------------------------------------------------
plane = "xy"
p_fixed = merge(cfg.fixed, Dict( # put here any params you want to modify
                                "POLARIZATION" => "R",
                                # "deltas"       => pars.deltas[11]
                                ))
result, _ = find_state(results, p_fixed)

fig_F, _, _, data = NonlocalArrays.field_intensity_map(; p=p_fixed, 
                                             result=result,
                                             plane=plane,
                                             plane_pos=(-3.4, 0.5, 0.5),
                                             scale=1.0,
                                             inc_field_profile=cfg.field_profile,
                                             plot=true)

fig_F
end

# field polarization study
begin
E_data = data.reE .+ 1im*data.imE
stokes_params = NonlocalArrays.stokes.(E_data[1], E_data[2])
v_obj = zeros(size(stokes_params)...)
u_obj = zeros(size(stokes_params)...)
q_obj = zeros(size(stokes_params)...)
field_obj = Array{Vector}(undef, size(stokes_params)...)
for i in eachindex(stokes_params[:, 1]), j in eachindex(stokes_params[1,:])
    # println((stokes_params[i,j].I, stokes_params[i,j].V)./stokes_params[i,j].I)
    v_obj[i,j] = stokes_params[i,j].V/stokes_params[i,j].I
    u_obj[i,j] = stokes_params[i,j].U/stokes_params[i,j].I
    q_obj[i,j] = stokes_params[i,j].Q/stokes_params[i,j].I
    field_obj[i,j] = [E_data[1][i,j], E_data[2][i,j], E_data[3][i,j]]
end
end

let 
    objs = [q_obj, u_obj, v_obj]
    fig = Figure(;size=(900,900)) 
    for i in eachindex(objs)
        Axis(fig[i,1]; aspect=DataAspect(),xlabel=string(plane[1]),
                                           ylabel=string(plane[2]))
        hmap = heatmap!(data.x, data.y, objs[i];
                        colormap=:seismic, colorrange=(-1,1))
        Colorbar(fig[i,2], hmap; )
    end
    Axis(fig[2,3]; aspect=DataAspect(),xlabel=string(plane[1]),
                                       ylabel=string(plane[2]))
    hmap = heatmap!(data.x, data.y, data.I_sc;
                    colormap=:gist_heat, colorrange=(0,0.25))
    Colorbar(fig[2,4], hmap; )
    fig
end

# arrow plot
let
    fig = Figure(size = (800, 800))
    ax = Axis(fig[1, 1], backgroundcolor = "black")
    # xs = data.x
    # ys = data.y
    range_i = 1:2:100
    xs = [data.x[i] for i in range_i]
    ys = [data.y[i] for i in range_i]
    # explicit method
    # us = [real(field_obj[i,j][1]) for i in eachindex(xs), j in eachindex(ys)]
    # vs = [real(field_obj[i,j][2]) for i in eachindex(xs), j in eachindex(ys)]
    us = [real(field_obj[i,j][1]) for i in range_i, j in range_i]
    vs = [real(field_obj[i,j][2]) for i in range_i, j in range_i]
    strength = vec(sqrt.(us .^ 2 .+ vs .^ 2))
    arrows!(ax, xs, ys, us, vs, lengthscale = 50.5, color = strength)
    fig
end


# ---- save / display --------------------------------------------------
cfg.save_figs && foreach((name,fig)->save(joinpath(PATH_FIGS,name*".pdf"),fig),
     [("curve_T",fig_T), ("heatmap_metric",fig_H), ("bands",fig_B)])

# fig_T, fig_H, fig_B   # show in REPL / Pluto / VSCode

fig_T
fig_H
fig_B

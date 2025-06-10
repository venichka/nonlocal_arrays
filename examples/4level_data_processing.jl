# TODO: 
# 1. Check R for pars.a[151, 187]

begin
    using Pkg
    Pkg.activate(pwd()[end-14:end] == "nonlocal_arrays" ? "." : "../")
end

using CairoMakie, GLMakie, LinearAlgebra, FFTW
using QuantumOptics
using AtomicArrays
using NonlocalArrays
using BenchmarkTools
using StaticArrays


const PATH_DATA = "../Data/"
const PATH_FIGS = "../Figs/"

# path_bson_file = "/Users/nikita/Documents/steady-states-sweep_deltas_-1.800_to_1.800_ndeltas_200_anglek_[0.000,0.000]_to_[0.000,0.000]_nanglek_1_Nx_6_Ny_6_POLARIZATION_R_to_L_nPOLARIZATION_2_amplitude_0.020_a_0.100_to_1.000_na_200_Bz_0.000_to_0.200_nBz_3.bson"
path_bson_file = "/Users/nikita/Documents/steady-states-sweep_deltas_-1.800_to_1.800_ndeltas_100_anglek_[0.000,0.000]_to_[0.000,0.000]_nanglek_1_Nx_8_Ny_8_POLARIZATION_R_to_L_nPOLARIZATION_2_amplitude_0.020_to_0.200_namplitude_4_a_0.250_to_1.000_na_100_Bz_0.000_to_0.200_nBz_3.bson"



results, params_dict = load_sweep(path_bson_file)

pars = params_to_vars!(params_dict; make_tuple=true)
pars.ny

fixed_params_mult = Dict(
    "a" => pars.a[10],
    # "a" => pars.a[121],
    # "deltas" => 0.0,
    "Bz" => 0.2,
    "amplitude" => 0.02,
    "anglek" => pars.anglek[1],
    "Nx" => pars.nx,
    "Ny" => pars.ny,
)
fixed_params_mult_1 = Dict(
    # "a" => pars.a[125],
    "a" => pars.a[10],
    # "deltas" => 0.0,
    # "Bz" => 0.2,
    "POLARIZATION" => "R",
    "amplitude" => 0.02,
    "anglek" => pars.anglek[1],
    "Nx" => pars.nx,
    "Ny" => pars.ny,
)

fixed_params_mult_2 = Dict(
    # "a" => pars.a[24],
    # "deltas" => pars.deltas[111],
    "Bz" => 0.2,
    # "POLARIZATION" => "R",
    "amplitude" => 0.02,
    "anglek" => pars.anglek[1],
    "Nx" => pars.nx,
    "Ny" => pars.ny,
)

fixed_params_heat = Dict(
    # "a" => pars.a[24],
    # "deltas" => 0.0,
    "Bz" => 0.2,
    "POLARIZATION" => "R",
    "amplitude" => 0.2,
    "anglek" => pars.anglek[1],
    "Nx" => pars.nx,
    "Ny" => pars.ny,
)

results_trunc = filter_results(results, Dict("a" => [pars.a[2*i-1] for i=1:100],
                                             "deltas" => [pars.deltas[2*i-1] 
                                                          for i = 1:100]))
results

transmission_func = (result, params) -> begin
    coll, field, field_func, _ = build_fourlevel_system(merge(params, Dict("field_func" => AtomicArrays.field.gauss)))
    sigmas_m = AtomicArrays.fourlevel_meanfield.sigma_matrices([result], 1)[1]
    zlim = 30.0
    coefs = AtomicArrays.field.transmission_reflection(field, coll, sigmas_m;
                                                 beam=:gauss,
                                                 surface=:hemisphere,
                                                #  polarization=[1,1im,0]/sqrt(2),
                                                 samples=400,
                                                 zlim=zlim,
                                                 size=[5.0,5.0])
    coefs[2] #+ coefs[2]
end

mirror_identify_func = (result, params) -> begin
    coll, field, field_func, _ = build_fourlevel_system(merge(params, Dict("field_func" => AtomicArrays.field.gauss)))
    sigmas_m = AtomicArrays.fourlevel_meanfield.sigma_matrices([result], 1)[1]
    zlim = 300.0
    TR = transmission_reflection_new(field, coll, sigmas_m;
                                     beam=:gauss,
                                     surface=:plane,
                                     size=(2.0,2.0),
                                     samples=40,
                                     zlim=zlim,
                                     return_helicity=true,
                                     return_powers=false)
    metrics = NonlocalArrays.chiral_mirror_metrics(TR.T_sigma_plus,
                                                   TR.T_sigma_minus, 
                                                   TR.R_sigma_plus,
                                                   TR.R_sigma_minus,
                                                   thresh_T=0.1,
                                                   kind=:product,
                                                   )
    metrics.obj
end

CairoMakie.activate!()

fig = plot_sweep_quantity(results, transmission_func, "deltas";
    fixed_params=merge(fixed_params_mult, Dict("POLARIZATION" => "R")),
    ylabel="Transmission")

plot_sweep_multicurve(results,
                                transmission_func,
                                # mirror_identify_func,
                                "deltas",
                                # "a",
                                "POLARIZATION";
                                fixed_params=fixed_params_mult,
                                ylabel="Reflection")

plot_sweep_multicurve(results,#_trunc,
                                # mirror_identify_func,
                                transmission_func,
                                "deltas",
                                "Bz";
                                fixed_params=fixed_params_mult_1,
                                ylabel="Reflection")

fig2, x2, y2, z2 = plot_sweep_heatmap(results,#_trunc,
                #    transmission_func,
                   mirror_identify_func,
                   "deltas",
                   "a";
                   fixed_params=fixed_params_heat,
                #    data=(x,y,z),
                   figure_kw=(size=1.5.*(600,450),),
                   axis_kw=(aspect = AxisAspect(1),
                            limits=(pars.deltas[1], pars.deltas[end],
                                    pars.a[1],pars.a[end]),
                            xlabel=L"\Delta",
                            title="Objective: a vs Δ, σ⁻",
                            xlabelsize=20,
                            ylabelsize=20,
                            xticklabelsize = 18,
                            yticklabelsize = 18,
                            titlesize=24,
                                    ),
                   heatmap_kw=(colormap=:plasma,
                               colorrange=(0,1)),
                   colorbar_kw=(label="Objective and chirality",))
fig
fig1
fig2
# save(PATH_FIGS*"pres_mirror_identify.pdf", fig, px_per_unit=4)

# Plots for presentation
let
    fig_p1 = plot_sweep_multicurve(results,
                                    transmission_func,
                                    # mirror_identify_func,
                                    "deltas",
                                    # "a",
                                    "POLARIZATION";
                                    fixed_params=fixed_params_mult,
                                    ylabel="Transmission")
    ax = fig_p1.content[1]
    ax.xlabel = L"(\omega - \omega_0) / \omega_0"
    ax.ylabel = "Reflection"
    ax.title = L"B_z = 0.2"

    ax.xlabelsize = 18
    ax.ylabelsize = 18
    ax.titlesize  = 20
    ax.xticklabelsize = 14
    ax.yticklabelsize = 14
    # save(PATH_FIGS*"pres_reflection.pdf", fig_p1, px_per_unit=4)
    fig_p1
end



@btime NonlocalArrays.transmission_reflection_new(E_in, coll, sigmas_m; beam=:gauss, samples=400, zlim=20.0, return_helicity=true, return_powers=true)
@btime field.transmission_reflection(E_in, coll, sigmas_m; beam=:gauss,
surface=:hemisphere,
samples=400,
zlim=20.0,
size=[5.0,5.0])

test = stokes(1.0+im, 1.0-im)
test.I
test000 = NonlocalArrays.chiral_mirror_metrics(test00[1], test00[2], test00[3], test00[4], test00[5])

NonlocalArrays.Scattering.helicity_basis(-normalize(E_in.k_vector))

begin
    # Options: "xy", "xz", "yz"
    plane = "xz"   # change this to "xz" or "yz" as desired



    coll, E_in, field_func, _ = build_fourlevel_system(merge(fixed_params_mult, Dict("field_func" => AtomicArrays.field.gauss)))

    result_0, params_0 = find_state(results, merge(fixed_params_mult, Dict("POLARIZATION" => "R", "deltas" => pars.deltas[11])))

    sigmas_m = AtomicArrays.fourlevel_meanfield.sigma_matrices([result_0], 1)[1]
    smm_s = AtomicArrays.fourlevel_meanfield.smm(result_0)

    Nx = fixed_params_mult["Nx"]
    Ny = fixed_params_mult["Ny"]
    a = fixed_params_mult["a"]

    # === Define grid parameters ===
    n_points = 100
    factor_scale = 3.5
    grid_min, grid_max = -Nx*a/2*factor_scale, Nx*a/2*factor_scale
    coord1_range = range(grid_min, grid_max; length=n_points)
    coord2_range = range(grid_min, grid_max; length=n_points)

    # Depending on the plane, choose the two varying coordinates and fix the third:
    if plane == "xy"
        fixed_value = Nx*a/2 + 0.2                           # fixed z
        fixed_index = 3
        label1, label2, fixed_label = "x", "y", "z"
    elseif plane == "xz"
        fixed_value = (mod(Nx, 2) == 0) ? 0.0 : a/2          # fixed y
        fixed_index = 2
        label1, label2, fixed_label = "x", "z", "y"
    elseif plane == "yz"
        fixed_value = Nx*a/2 + 0.2                           # fixed x
        fixed_index = 1
        label1, label2, fixed_label = "y", "z", "x"
    else
        error("Unknown plane. Choose 'xy', 'xz', or 'yz'.")
    end

    nx, ny = length(coord1_range), length(coord2_range)

    # === Preallocate arrays to store field projections ===
    Re_field_x = zeros(nx, ny)
    Abs_field_x = zeros(nx, ny)
    Re_t_field_x = zeros(nx, ny)
    Abs_t_field_x = zeros(nx, ny)
    Re_field_y = zeros(nx, ny)
    Abs_field_y = zeros(nx, ny)
    Re_t_field_y = zeros(nx, ny)
    Abs_t_field_y = zeros(nx, ny)
    Re_field_z = zeros(nx, ny)
    Abs_field_z = zeros(nx, ny)
    Re_t_field_z = zeros(nx, ny)
    Abs_t_field_z = zeros(nx, ny)

    I_total = zeros(nx, ny)

    # === Update atom positions according to the selected plane ===
    # For "xy": use (x,y), for "xz": use (x,z), for "yz": use (y,z)
    if plane == "xy"
        atom_coord1 = [atom.position[1] for atom in coll.atoms]
        atom_coord2 = [atom.position[2] for atom in coll.atoms]
    elseif plane == "xz"
        atom_coord1 = [atom.position[1] for atom in coll.atoms]
        atom_coord2 = [atom.position[3] for atom in coll.atoms]
    elseif plane == "yz"
        atom_coord1 = [atom.position[2] for atom in coll.atoms]
        atom_coord2 = [atom.position[3] for atom in coll.atoms]
    end

    atom_x = [atom.position[1] for atom in coll.atoms]
    atom_y = [atom.position[2] for atom in coll.atoms]
    atom_z = [atom.position[3] for atom in coll.atoms]

    # === Compute the scattered field on the grid ===
    # In each case, r is built by inserting the two grid coordinates into the proper indices.
    for (i, a) in enumerate(coord1_range)
        for (j, b) in enumerate(coord2_range)
            r = zeros(3)
            # Set the two varying coordinates.
            if plane == "xy"
                r[1] = a; r[2] = b; r[3] = fixed_value
            elseif plane == "xz"
                r[1] = a; r[2] = fixed_value; r[3] = b
            elseif plane == "yz"
                r[1] = fixed_value; r[2] = a; r[3] = b
            end
            # Compute the scattered field at r using the provided function.
            # It is assumed that sigmas_m is defined appropriately.
            E = AtomicArrays.field.scattered_field(r, coll, sigmas_m)/sqrt(3)
            E_t = field_func(r, E_in) + E
            I_t = (norm(E_t)^2 / abs(E_in.amplitude)^2)
            
            # Save the real and absolute values for each Cartesian component.
            Re_field_x[i, j] = real(E[1])
            Abs_field_x[i, j] = abs(E[1])
            Re_field_y[i, j] = real(E[2])
            Abs_field_y[i, j] = abs(E[2])
            Re_field_z[i, j] = real(E[3])
            Abs_field_z[i, j] = abs(E[3])
            Re_t_field_x[i, j] = real(E_t[1])
            Abs_t_field_x[i, j] = abs(E_t[1])
            Re_t_field_y[i, j] = real(E_t[2])
            Abs_t_field_y[i, j] = abs(E_t[2])
            Re_t_field_z[i, j] = real(E_t[3])
            Abs_t_field_z[i, j] = abs(E_t[3])
            I_total[i, j] = I_t
        end
    end
end

plot_atoms_with_field(coll, E_in)

AtomicArrays.field.transmission_reflection(E_in, coll, sigmas_m;
                                                 beam=:gauss,
                                                 surface=:hemisphere,
                                                 polarization=[1.0, 1im, 0]/sqrt(2),
                                                 samples=400,
                                                 zlim=50,
                                                 size=[5.0,5.0])

transmission_reflection_new(E_in, coll, sigmas_m;
                                                 beam=:gauss,
                                                 surface=:plane,
                                                 samples=40,
                                                 zlim=5,
                                                 size=(5.0,5.0),
                                                 return_helicity=true,
                                                 return_powers=false)


f_test = (E, k) -> dot(normalize(imag(cross(conj(E), E))), normalize(k))
test = field.total_field(field_func, [0.0,0.0,-5.0], E_in, coll, sigmas_m)
test = field.scattered_field([0.0,0.0,-5.0], coll, sigmas_m)
test_00 = field_func([0.0,0.0,5.0], E_in)
stokes(test[1], test[2])
stokes(test)

f_test(test, [0.0,0.0,-5.0])
f_test(test_00, [0.0,0.0,5.0])
E_in.polarisation
test

NonlocalArrays.Scattering.reflection_phases(field_func, E_in, coll, sigmas_m; R=5.0)



# Plots
# fields
let
    CairoMakie.activate!()
    fig = Figure(size = (900, 1100))

    Re_x = Re_t_field_x
    Abs_x = Abs_t_field_x
    Re_y = Re_t_field_y
    Abs_y = Abs_t_field_y
    Re_z = Re_t_field_z
    Abs_z = Abs_t_field_z

    nlevels = 10

    # --- Row 1: x-component (Eₓ) ---
    ax1 = Axis(fig[1, 1], title = "Re(Eₓ)", xlabel = label1, ylabel = label2)
    hm1 = contourf!(ax1, coord1_range, coord2_range, Re_x, colormap = :plasma, levels=nlevels)
    scatter!(ax1, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[1, 2], hm1, width = 15, height = Relative(1))

    ax2 = Axis(fig[1, 3], title = "Abs(Eₓ)", xlabel = label1, ylabel = label2)
    hm2 = contourf!(ax2, coord1_range, coord2_range, Abs_x, colormap = :plasma, levels=nlevels)
    scatter!(ax2, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[1, 4], hm2, width = 15, height = Relative(1))

    # --- Row 2: y-component (Eᵧ) ---
    ax3 = Axis(fig[2, 1], title = "Re(Eᵧ)", xlabel = label1, ylabel = label2)
    hm3 = contourf!(ax3, coord1_range, coord2_range, Re_y, colormap = :plasma, levels=nlevels)
    scatter!(ax3, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[2, 2], hm3, width = 15, height = Relative(1))

    ax4 = Axis(fig[2, 3], title = "Abs(Eᵧ)", xlabel = label1, ylabel = label2)
    hm4 = contourf!(ax4, coord1_range, coord2_range, Abs_y, colormap = :plasma, levels=nlevels)
    scatter!(ax4, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[2, 4], hm4, width = 15, height = Relative(1))

    # --- Row 3: z-component (E_z) ---
    ax5 = Axis(fig[3, 1], title = "Re(E_z)", xlabel = label1, ylabel = label2)
    hm5 = contourf!(ax5, coord1_range, coord2_range, Re_z, colormap = :plasma, levels=nlevels)
    scatter!(ax5, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[3, 2], hm5, width = 15, height = Relative(1))

    ax6 = Axis(fig[3, 3], title = "Abs(E_z)", xlabel = label1, ylabel = label2)
    hm6 = contourf!(ax6, coord1_range, coord2_range, Abs_z, colormap = :plasma, levels=nlevels)
    scatter!(ax6, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[3, 4], hm6, width = 15, height = Relative(1))

    # save(PATH_FIGS*"4level_E_t_a"*string(a)*"_"*POLARIZATION*"_theta"*string(round(field.angle_k[1], digits=2))*"_N"*string(N)*".pdf", fig, px_per_unit=4)
    fig
end

let
    CairoMakie.activate!()
    fig = Figure(size = (800, 700))

    nlevels = 50

    # --- Row 1: x-component (Eₓ) ---
    ax1 = Axis(fig[1, 1], title = "Intensity, E₀ = "*string(round(E_in.amplitude, digits=3)), xlabel = label1, ylabel = label2,
               titlesize = 24, xlabelsize = 20, ylabelsize = 20,
               xticklabelsize = 18, yticklabelsize = 18)
    hm1 = contourf!(ax1, coord1_range, coord2_range, I_total,
                    colormap = :plasma, 
                    levels = range(0, 1.0*maximum(I_total), nlevels), 
                    # colorscale=log10,
                    )
    scatter!(ax1, atom_coord1, atom_coord2, markersize = 8, color = :white)
    Colorbar(fig[1, 2], hm1, label="I/|E₀|²",
             labelsize = 20, ticklabelsize = 18,
             width = 15, height = Relative(1))

    # save(PATH_FIGS*"4level_I_t_a"*string(a)*"_"*POLARIZATION*"_theta"*string(round(field.angle_k[1], digits=2))*"_N"*string(N)*".pdf", fig, px_per_unit=4)
    fig
end


# Dispersion curves
begin
    # Dipole vectors for σ⁻, π, σ⁺  (example: quantization along ẑ)
    μs = [AtomicArrays.fourlevel_misc.polarizations_spherical()[i,:] for i = 1:3]

    γs = AtomicArrays.fourlevel_misc.gammas(0.25)            # set identical linewidths
    B_z = 0.2
    Δs = [B_z*m for m = -1:1]            # Zeeman splittings
    a  = 0.2                      # lattice constant in λ units

    kvals = range(-π/a, π/a; length=300)
    bands = [eigvals(omega_1d_triplet(k, a, μs, γs, Δs))
            for k in kvals]        # three dispersion branches

            # Convert to array of shape (3, Nk) for plotting
    ωmat = reduce(hcat, bands)                  # each column is k-point
end

CairoMakie.activate!()
let
    f  = Figure(size = (800, 500))
    ax = Axis(f[1, 1],
              xlabel = "k·a / π",
              ylabel = "ω / Γ",
              xlabelsize=20,
              ylabelsize=20,
              xticklabelsize = 18,
              yticklabelsize = 18,
              titlesize=24,
              xticks = -1:0.5:1,
              title=L"B_z = 0.2",
            #   yticks = nothing
              )  # or `yticks = nothing` also works

    # plot each band
    colors = [:dodgerblue, :crimson, :seagreen]
    for b in 1:3
        lines!(ax, collect(kvals .* a / π), ωmat[b,:], linewidth = 2, color = colors[b], label = "band $b")
    end

    axislegend(ax; position = :rb, labelsize = 18)
    # save(PATH_FIGS*"pres_dispersion_a"*string(a)*"_Bz"*string(round(B_z, digits=2))*".pdf", f, px_per_unit=4)
    f
end

ωbands, γbands, s = bands_GXMG(a, μs, γs, Δs; Nk = 500, keep_k = true, Nmax=200, threads=true, return_gamma=true)

let
    fig = Figure(size = (800, 450))

    colors = [:dodgerblue, :crimson, :seagreen]
    for i = 1:2
        ax  = Axis(fig[i, 1];
                xlabel = "Γ   X   M   Γ",
                ylabel = (i == 1) ? "ω / Γ" : "γ",
                xticklabelrotation = 0,
                xlabelsize = 15,
                ylabelsize = 14,
                xticks=([0.0, maximum(s)/3, 2*maximum(s)/3, s[end]], ["Γ", "X", "M", "Γ"]),
                #    yticks = :none
                )
        for b = 1:3
            lines!(ax, s, 
                   (i==1) ? ωbands[b, :] : γbands[b, :],
                   color = colors[b], linewidth = 2)
        end
    # vertical guide lines at X and M
    vlines!(ax, [maximum(s)/3, 2*maximum(s)/3], color = :gray, linestyle = :dash)
    end

    # save(PATH_FIGS*"pres_bands_GXMG_B_0.0.pdf", fig)
    fig
end


ωbands, γbands, kx_vec, ky_vec = GeomField.bands_2d_grid(; a=a, μ=μs, γ=γs, Δ=Δs, 
                                                 Nkx=181, Nky=181, Nmax=60,
                                                 fullBZ=true, keep_k=true,
                                                 return_gamma=true)
γbands
let
    # ── visual settings ──────────────────────────────────────────────
    cmap   = :inferno
    titles = ["Band 1", "Band 2", "Band 3"]

    # Makie wants (x, y) indexing, so transpose iy/ix slice
    band_data = (ωbands[1, :, :]',
                 ωbands[2, :, :]',
                 ωbands[3, :, :]')
    decay_data = (γbands[1, :, :]',
                  γbands[2, :, :]',
                  γbands[3, :, :]')

    # Figure: 1 row, 6 columns  (Axis | Colorbar) × 3
    fig = Figure(size = (1500, 2*360))
    for i in 1:2, b in 1:3
        col_ax  = 2b - 1             #   1,3,5
        col_cb  = 2b                 #   2,4,6

        ax = Axis(fig[i, col_ax];
                  xlabel = "kₓ (π·a⁻¹)",
                  ylabel = "kᵧ (π·a⁻¹)",
                  title  = (i==1) ? titles[b] : "",
                  aspect = DataAspect())

        hm = heatmap!(ax, a/pi*kx_vec, a/pi*ky_vec, 
                      (i==1) ? band_data[b] : decay_data[b];
                      colormap = cmap)

        Colorbar(fig[i, col_cb], hm;
                 label = (i==1) ? "ω(k) (rad·s⁻¹)" : "γ(k) (rad·s⁻¹)")
    end

    fig                               # display; call save("bands2d.png", fig) if desired
end

GLMakie.activate!()
let
    # axes & figure ----------------------------------------------------
    fig = Figure(size = (900, 700))
    ax  = Axis3(fig[1, 1];
                xlabel = "kₓ  (π · a⁻¹)",
                ylabel = "kᵧ  (π · a⁻¹)",
                zlabel = "ω(k)  (rad ⁄ s)",
                title  = "Collective dispersion for 2-D square array")

    # convenience: unpack the three Nkx×Nky matrices
    band1 = ωbands[1, :, :]        # dims (Nkx, Nky)
    band2 = ωbands[2, :, :]
    band3 = ωbands[3, :, :]

    # surface plots (transparent & shading-off so they don’t self-occlude too much)
    surface!(ax, a/pi*kx_vec, a/pi*ky_vec, band1;
             colormap = :batlow,
            #  colormap=[:blue2],
             transparency = true, alpha = 0.7,
             shading = NoShading, label = "Band 1")
    surface!(ax, a/pi*kx_vec, a/pi*ky_vec, band2;
             colormap = :bilbao, 
            #  colormap=[:orange3],
             transparency = true, alpha = 0.7,
             shading = NoShading, label = "Band 2")
    surface!(ax, a/pi*kx_vec, a/pi*ky_vec, band3;
             colormap = :tokyo, 
            #  colormap=[:seagreen3],
             transparency = true, alpha = 0.7,
             shading = NoShading, label = "Band 3")

    axislegend(ax; position = :lt)

    fig                    # displays (or `save("bands3D.png", fig)` )
end

# Compute bands using quasimomentum
fixed_params_bands = Dict(
    "a" => 0.1,
    "deltas" => 0.0,
    "Bz" => 0.0,
    "amplitude" => 0.0,
    "anglek" => [0.0,0.0],
    "Nx" => 10,
    "Ny" => 10,
)
begin
    M_exc = 1
    a_bands = fixed_params_bands["a"]
    Nx = fixed_params_bands["Nx"]
    Ny = fixed_params_bands["Ny"]
    sublevels_m = [-1, 0, 1]
    B_z = fixed_params_bands["Bz"]

    coll_bands, _, _, _ = build_fourlevel_system(merge(fixed_params_bands,
                                Dict("field_func" => AtomicArrays.field.gauss)))
    OmegaTensor = AtomicArrays.OmegaTensor_4level(coll_bands)
    GammaTensor = AtomicArrays.GammaTensor_4level(coll_bands)
   
    N_tot = Nx * Ny
    b_reduced = AtomicArrays.ReducedAtomBasis(N_tot, M_exc)

    # spsm = [AtomicArrays.reducedsigmaplus(b_reduced, i, m)*
    #         AtomicArrays.reducedsigmaminus(b_reduced, j, mp)
    #         for i = 1:N_tot, j = 1:N_tot, m = -1:1, mp = -1:1]
    # H_eff = sum((coll_bands.atoms[i].delta + sublevels_m[m] * B_z)*
    #             spsm[i,i,m,m] for i=1:N_tot, m=1:3) + 
    #         sum((OmegaTensor[i,j,m,mp] - 0.5im*GammaTensor[i,j,m,mp])*
    #              spsm[i, j, m, mp]
    #               for i=1:N_tot, j=1:N_tot, m=1:3, mp=1:3)

    H_eff = sum((coll_bands.atoms[i].delta + sublevels_m[m] * B_z)*
                AtomicArrays.reducedsigmaplus(b_reduced, i, sublevels_m[m])*
                AtomicArrays.reducedsigmaminus(b_reduced, i, sublevels_m[m]) 
                for i=1:N_tot, m=1:3) + 
            sum((OmegaTensor[i,j,m,mp] - 0.5im*GammaTensor[i,j,m,mp])*
                 AtomicArrays.reducedsigmaplus(b_reduced, i, sublevels_m[m])*
                 AtomicArrays.reducedsigmaminus(b_reduced, j, sublevels_m[mp])
                  for i=1:N_tot, j=1:N_tot, m=1:3, mp=1:3)
end

Gf = AtomicArrays.interaction.GreenTensor(-(coll_bands.atoms[15].position - coll_bands.atoms[4].position), 2*pi)
- coll_bands.polarizations[1,:,4]' * real(Gf) * coll_bands.polarizations[3,:,15] / 16
coll_bands.polarizations[1,:,4]' * imag(Gf) * coll_bands.polarizations[3,:,15] / 8
OmegaTensor[4,15,1,3]
GammaTensor[4,15,1,3]
maximum(imag(OmegaTensor))
var = AtomicArrays.reducedsigmaplus(b_reduced, 1, -1)*
                 AtomicArrays.reducedsigmaminus(b_reduced, 1, 1)
dense(var).data

Rt, Ωtab = NonlocalArrays.GeomField.precompute_Omega_table(a, μs, γs; Nmax=60)
Rt, Γtab = NonlocalArrays.GeomField.precompute_Gamma_table(a, μs, γs; Nmax=60)

NonlocalArrays.GeomField.omega_2d_triplet_fast(0.0, 0.0, Rt, Ωtab, Δs)
W_test = NonlocalArrays.GeomField.gamma_2d_triplet_fast(2.5, 1.0, Rt, Ωtab, γs)
sort!(eigvals(W_test), by=real)

eigvals(Matrix(NonlocalArrays.GeomField.omega_gamma_2d_triplet_fast(2.1,2.1,
                            Rt , Ωtab , Γtab ,
                            Δs,
                            γs)))
Γtab
GammaTensor[1,4,1,1]
# --------------------------------------------------------------------
# 1.   Build 3×3 Bloch block from H_eff and known positions
# --------------------------------------------------------------------
@inline function bloch_3x3(H_eff::AbstractMatrix,
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

# --------------------------------------------------------------------
# 2.   Bands along Γ–X–M–Γ using *provided* positions
# --------------------------------------------------------------------
function bands_GXMG_from_H(H_eff::AbstractMatrix,
                           r::Vector{SVector{2,Float64}},
                           a::Real;                     # lattice period (for k-path)
                           Nk::Int=200,
                           threads::Bool=true)

    Γ = (0.0, 0.0);      X = (π/a, 0.0);      M = (π/a, π/a)
    knodes = (Γ, X, M, Γ)

    segment(k1, k2) = let
        (kx1, ky1), (kx2, ky2) = k1, k2
        t = range(0.0, 1.0, Nk+1)[1:end-1]
        collect(zip(kx1 .+ t.*(kx2-kx1), ky1 .+ t.*(ky2-ky1)))
    end

    kpath = vcat([segment(knodes[i], knodes[i+1]) for i in 1:3]...)
    npts  = length(kpath)

    ωmat = Matrix{Float64}(undef, 3, npts)
    s    = Vector{Float64}(undef, npts)

    if threads && Threads.nthreads()>1
        Threads.@threads for idx in 1:npts
            kx, ky = kpath[idx]
            Wk = bloch_3x3(H_eff, r, kx, ky)
            ωmat[:, idx] .= sort(real(eigvals(Wk)))
        end
    else
        for idx in 1:npts
            kx, ky = kpath[idx]
            Wk = bloch_3x3(H_eff, r, kx, ky)
            ωmat[:, idx] = sort(real(eigvals(Wk)))
        end
    end

    s[1] = 0.0
    for i in 2:npts
        s[i] = s[i-1] + hypot(kpath[i][1]-kpath[i-1][1],
                              kpath[i][2]-kpath[i-1][2])
    end

    return ωmat, s
end

# --------------------------------------------------------------------
# 3.   Usage with your existing objects
# --------------------------------------------------------------------
# • H_eff  – already built in your block
# • coll_bands – already built; provides the atomic coordinates

# Extract (x,y) coordinates once
r_xy = [ SVector(coll_bands.atoms[i].position[1:2]...) for i in eachindex(coll_bands.atoms) ]
# Compute bands
bands, decay, kdist = NonlocalArrays.GeomField.bands_GXMG_from_H(dense(H_eff).data, r_xy, fixed_params_bands["a"];
                                 Nk = 250, threads = true, keep_k=true)
bands, kdist = bands_GXMG_from_H(dense(H_eff).data, r_xy, fixed_params_bands["a"];
                                 Nk = 250, threads = true)

# Optional quick plot
let
    fig = Figure(size = (800, 450))
    ax  = Axis(fig[1, 1];
            xlabel = "Γ   X   M   Γ",
            ylabel = "ω / Γ",
            xticklabelrotation = 0,
            xlabelsize = 15,
            ylabelsize = 14,
            xticks=([0.0, maximum(kdist)/3, 2*maximum(kdist)/3, kdist[end]], ["Γ", "X", "M", "Γ"]),
            #    yticks = :none
            )

    colors = [:dodgerblue, :crimson, :seagreen]
    for b in 1:3
        lines!(ax, kdist, bands[b, :], color = colors[b], linewidth = 2)
    end

    # vertical guide lines at X and M
    vlines!(ax, [maximum(kdist)/3, 2*maximum(kdist)/3], color = :gray, linestyle = :dash)
    # save(PATH_FIGS*"pres_bands_GXMG_B_0.0.pdf", fig)
    fig
end






















begin
    λ, states = eigenstates(H_eff; warning=false)
    γ = -2 .* imag.(λ)
    s_ind_max = sortperm(γ, rev=true)[1]
    s_ind = sortperm(γ)[1]
    ψ = states[s_ind]
    ψ_max = states[s_ind_max]
end


# fft approach
states
begin
    q_perp_x = [-π/a_bands + 2*π*(i-1) / a_bands/(Nx) for i=1:Nx]
    states_2D = [[states[i].data[(Nx)*(j-1) + (k-1)%(Ny) + 1]
                for j=1:Nx, k=1:Ny] for i = 1:N_tot]
    states_2D_fft = [sqrt(2)*fftshift(fft(states_2D[i]))/sqrt(Nx*Ny) for i=1:N]

    q_quasi_fft = [sum([sum([abs(states_2D_fft[i][j,k])^2*
                            norm([q_perp_x[j], q_perp_x[k]])
                            for k = 1:Ny]) for j=1:Nx])
                for i = 1:N]

    s_ind = sortperm(q_quasi)[1]

    s_ind = sortperm(γ, rev=false)[1]
    q_quasi[s_ind]
    γ[s_ind]

    ψ = states[s_ind]
end







# Tests
var1, var2 = GeomField.precompute_Ω_table(a, μs, γs; Nmax=60)
var1
var2

GeomField.omega_2d_triplet_fast(1.0, -1.0,
                      var1,
                      var2,
                      Δs;
                      ωc=0.0)
GeomField.omega_2d_triplet(1.0, -1.0, a,
                      μs,
                      γs,
                      Δs;
                      ωc = 0.0, Nmax = 60)
# TODO: 
# 1. Check R for pars.a[151, 187]

begin
    using Pkg
    Pkg.activate(pwd()[end-14:end] == "nonlocal_arrays" ? "." : "../")
end

using CairoMakie, GLMakie, LinearAlgebra, FFTW
using AtomicArrays
using NonlocalArrays
using BenchmarkTools


const PATH_DATA = "../Data/"
const PATH_FIGS = "../Figs/"

path_bson_file = "/Users/nikita/Documents/steady-states-sweep_deltas_-1.800_to_1.800_ndeltas_200_anglek_[0.000,0.000]_to_[0.000,0.000]_nanglek_1_Nx_6_Ny_6_POLARIZATION_R_to_L_nPOLARIZATION_2_amplitude_0.020_a_0.100_to_1.000_na_200_Bz_0.000_to_0.200_nBz_3.bson"



results, params_dict = load_sweep(path_bson_file)

pars = params_to_vars!(params_dict; make_tuple=true)
pars.ny

fixed_params_mult = Dict(
    "a" => pars.a[187],
    # "deltas" => 0.0,
    "Bz" => 0.2,
    "amplitude" => 0.02,
    "anglek" => [0.0, 0.0],
    "Nx" => 6,
    "Ny" => 6,
)
fixed_params_mult_1 = Dict(
    "a" => pars.a[125],
    # "deltas" => 0.0,
    # "Bz" => 0.2,
    "POLARIZATION" => "R",
    "amplitude" => 0.02,
    "anglek" => [0.0, 0.0],
    "Nx" => 6,
    "Ny" => 6,
)

fixed_params_mult_2 = Dict(
    # "a" => pars.a[24],
    "deltas" => pars.deltas[111],
    "Bz" => 0.2,
    # "POLARIZATION" => "R",
    "amplitude" => 0.02,
    "anglek" => [0.0, 0.0],
    "Nx" => 6,
    "Ny" => 6,
)

fixed_params_heat = Dict(
    # "a" => pars.a[24],
    # "deltas" => 0.0,
    "Bz" => 0.2,
    "POLARIZATION" => "L",
    "amplitude" => 0.02,
    "anglek" => [0.0, 0.0],
    "Nx" => 6,
    "Ny" => 6,
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
                                                 polarization=[1,1im,0]/sqrt(2),
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
                                                   kind=:sigmoid,
                                                   )
    metrics.obj
end

CairoMakie.activate!()

fig = plot_sweep_quantity(results, transmission_func, "deltas";
    fixed_params=merge(fixed_params_mult, Dict("POLARIZATION" => "R")),
    ylabel="Transmission")

plot_sweep_multicurve(results_trunc,
                                transmission_func,
                                # mirror_identify_func,
                                # "deltas",
                                "a",
                                "POLARIZATION";
                                fixed_params=fixed_params_mult_2,
                                ylabel="Reflection")

plot_sweep_multicurve(results_trunc,
                                # mirror_identify_func,
                                transmission_func,
                                "deltas",
                                "Bz";
                                fixed_params=fixed_params_mult_1,
                                ylabel="Reflection")

fig, x, y, z = plot_sweep_heatmap(results_trunc,
                #    transmission_func,
                   mirror_identify_func,
                   "deltas",
                   "a";
                   fixed_params=fixed_params_heat,
                #    data=(x,y,z),
                   figure_kw=(size=1.5.*(600,450),),
                   axis_kw=(aspect = AxisAspect(1),
                            limits=(pars.deltas[1], pars.deltas[end],
                                    pars.a[1],pars.a[end])),
                   heatmap_kw=(colormap=:plasma,
                               colorrange=(0,1)),
                   colorbar_kw=(label="Transmission",))
fig



filter_results(results, Dict(
    "a" => pars.a[24],
    "deltas" => pars.deltas[1],
    "Bz" => 0.2,
    "POLARIZATION" => "L",
    "amplitude" => 0.02,
    "anglek" => [0.0, 0.0],
    "Nx" => 6,
    "Ny" => 6,
))
results


@btime NonlocalArrays.transmission_reflection_new(E_in, coll, sigmas_m; beam=:gauss, samples=400, zlim=20.0, return_helicity=true, return_powers=true)
@btime field.transmission_reflection(E_in, coll, sigmas_m; beam=:gauss,
surface=:hemisphere,
samples=400,
zlim=20.0,
size=[5.0,5.0])

test = stokes(1.0+im, 1.0-im)
test.I
test000 = NonlocalArrays.chiral_mirror_metrics(test00[1], test00[2], test00[3], test00[4], test00[5])

begin
    # Options: "xy", "xz", "yz"
    plane = "xz"   # change this to "xz" or "yz" as desired



    coll, E_in, field_func, _ = build_fourlevel_system(merge(fixed_params_mult, Dict("field_func" => AtomicArrays.field.gauss)))

    result_0, params_0 = find_state(results, merge(fixed_params_mult, Dict("POLARIZATION" => "R", "deltas" => pars.deltas[111])))

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

AtomicArrays.field.transmission_reflection(E_in, coll, sigmas_m;
                                                 beam=:gauss,
                                                 surface=:hemisphere,
                                                 polarization=[1.0, 1im, 0]/sqrt(2),
                                                 samples=400,
                                                 zlim=50,
                                                 size=[5.0,5.0])

test = field.total_field(field_func, [0.0,0.0,5.0], E_in, coll, sigmas_m)
test = field.scattered_field([1.0,1.0,-5.0], coll, sigmas_m)
stokes(test[1], test[2])

test_0 = helicity_flux_fibonacci(field_func, E_in, coll, sigmas_m;
                                 R=5.0, Ndirs=400)

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

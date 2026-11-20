"""
    ko-ko_var.jl

    This code implements the model described in Capolupo et al., 
"Tissue swelling and local mechanosensation drive crypt convergence in intestinal development."

Vertex-based Monte Carlo model of an epithelial monolayer containing two
crypts. Crypt cells adjust their apical and basal tensions in response to cell
width and curvature; villus cells do not sense curvature and retain their
polarized tension balance.

The simulation corresponds to a line of cells, with the crypts in the center and 
villus cells on either side. Boundary conditions are fixed.
For each simulation, geometry is optimized for a given level of swelling, but 
crypt cells update their cytoskeletal tensions.

The myosin asymmetry between the two crypts can be adjusted through the `dg01` and `dg02` parameters in `main()`.
In the default setting of this code, the myosin levels are fixed, corresponding to the cPL2KO in vivo conditions.
Alternatively, cPLA2 can be turned on by changing the flags "cpla2_crypt1" and "cpla2_crypt2" to true, which allows
mechanosensation in either crypt, which can then homogenize the myosin
This can implement the CRL and mosaic KO-WT conditions.

This code runs the simulation for different swelling levels of the villus cells,
 which are set by the `cell_vol_villus` parameter in `main()`. 
The tested values are in the range 1.4-2.0 (in units of 10 μm^2), 
expressed as the cross section of a single villus cell (see the paper).
At the peak of swelling, experimental values are around 2.0.

Running this file removes the previous `multi_output/` directory, then runs
ten simulations over the configured villus-cell areas. Results are written to
`multi_output/` beside this script. 

All lengths and areas are expressed in units of 10 μm and (10 μm)^2,
respectively. Requires `CircularArrays`, `Images`, and `Plots`; `Random` is
part of the Julia standard library.
"""

using Random
using Plots
using Images
using CircularArrays

const T = 200001  # Maximum simulation time.
const dt = 1.0    # Time increment.
const plotstep = 1000
const Ncrypt = 12
const Nvillus = 12
const N = 2*Ncrypt + 2*Nvillus  # Number of vertices.
const wvillus = 0.8
const Lwall = wvillus*Nvillus
const sticky_walls = 0.
const open_neck = 2.0
const nTA = 3
const Time_2nd_crypt = 1

const extra_N = 7  # Replicated cells across the periodic boundary for plotting.

l_c = 0.65  # Cells narrower than this become more apically constricted.
Ga = 1.0    # Apical tension.
Gb = 1.0    # Basal tension.
Gl = 1.0    # Lateral tension.
d_Ga = 0.8  # Maximum tension increase; Ga + Gb remains constant.
d_Ga2 = 0.0

const Gav = 1.
const Gbv = 1.


const cell_vol = 0.6  # Cell area in (10 μm)^2; typical values are 0.8 in crypts and 1.5-1.7 in villi.
effective_villus_zeta = -1.2
cell_vol_villus = ((Gav+Gbv) - effective_villus_zeta)/Gl*wvillus^2

const wcrypt = sqrt(Gl/(Ga + Gb)*cell_vol)*0.9
const wall_distance = Ncrypt*wcrypt
const Ltot = 2*Lwall + 2*wall_distance

const min_length = 0.25
const temp = 0.001  # Metropolis temperature.

hcrypt = cell_vol/wcrypt
curv0 = 0.25*hcrypt 
# Alternate response signatures retained for analyses that call them directly.
Ga_fct(l, sc) = Ga + sc * d_Ga * tanh(l_c/l)
Gb_fct(l, sc) = Gb - sc * d_Ga * tanh(l_c/l)

Ga_fct(l, sc, curv) = max(Ga - sc * d_Ga * tanh(5 * (l - l_c) / l_c) + d_Ga2 * tanh(4 * curv) * (curv < 0 ? 1 : 0), 0.1)
Gb_fct(l, sc, curv) = max(Gb + sc * d_Ga * tanh(5 * (l - l_c) / l_c) - d_Ga2 * tanh(4 * curv) * (curv < 0 ? 1 : 0), 0.1)

# Update crypt tensions from the initial myosin asymmetry and local cell width.
Ga_fct(dg0, l, sc, curv) = max(Ga + dg0/2 - (sc > 0) * (d_Ga - dg0)/2 * tanh(5 * (l - l_c) / l_c) * (l < l_c ? 1 : 0) + 0*d_Ga2/2 * tanh(curv / curv0) * (curv < 0 ? 1 : 0), 0.1)
Gb_fct(dg0, l, sc, curv) = max(Gb - dg0/2 + (sc > 0) * (d_Ga - dg0)/2 * tanh(5 * (l - l_c) / l_c) * (l < l_c ? 1 : 0) - 0*d_Ga2/2 * tanh(curv / curv0) * (curv < 0 ? 1 : 0), 0.1)

const tau_myosin = 100  # Myosin relaxation timescale.

# Volume-constraint penalty strength.
const K = 100.0

function dg_grad(dg)
    dgmax = 0.8
    dgmin = 0.2
    abmax = (1 + dgmax/2)/(1 - dgmax/2)
    abmin = (1 + dgmin/2)/(1 - dgmin/2)
    ab = (1 + dg/2)/(1 - dg/2)
    x = max(min((ab - abmin)/(abmax - abmin),1),0)
    hi_color = RGB(249/255,179/255,54/255)
    lo_color = RGB(51/255,139/255,204/255)
    color = hi_color*x + lo_color*(1-x)
    return color
end

# Compute the area enclosed by four vertices, accounting for periodic wrapping.
function cell_volume(apical_i,apical_im,basal_i,basal_im)
    if abs(apical_i[1]-apical_im[1]) > 10
        if apical_i[1] > apical_im[1]
            ap_im = apical_im[:]
            ap_im[1] = ap_im[1] + Ltot
            bas_im = basal_im[:]
            bas_im[1] = bas_im[1] + Ltot
            area = 0.5 * abs((apical_i[1] * basal_i[2] - apical_i[2] * basal_i[1]) + 
                    (basal_i[1] * basal_im[2] - basal_i[2] * basal_im[1]) + 
                    (bas_im[1] * ap_im[2] - bas_im[2] * ap_im[1]) + 
                    (ap_im[1] * apical_i[2] - ap_im[2] * apical_i[1]))
        else
            ap_i = apical_i[:]
            bas_i = basal_i[:]
            bas_i[1] = bas_i[1] + Ltot
            ap_i[1] = ap_i[1] + Ltot
            area = 0.5 * abs((ap_i[1] * bas_i[2] - ap_i[2] * bas_i[1]) + 
                    (bas_i[1] * basal_im[2] - bas_i[2] * basal_im[1]) + 
                    (basal_im[1] * apical_im[2] - basal_im[2] * apical_im[1]) + 
                    (apical_im[1] * ap_i[2] - apical_im[2] * ap_i[1]))
        end
    else
        area = 0.5 * abs((apical_i[1] * basal_i[2] - apical_i[2] * basal_i[1]) + 
                    (basal_i[1] * basal_im[2] - basal_i[2] * basal_im[1]) + 
                    (basal_im[1] * apical_im[2] - basal_im[2] * apical_im[1]) + 
                    (apical_im[1] * apical_i[2] - apical_im[2] * apical_i[1]))
    end
    return area
end

# Compute the shortest distance between two points on the periodic domain.
function distance(A,B)
    d = sqrt(min((B[1] - A[1])^2,(abs(B[1] - A[1])-Ltot)^2) + (B[2] - A[2])^2)
    return d
end

function energy_cell(apical,basal,Ga,Gb,Gl,K,target_volume, flag_accept)
    # Sum lateral, apical, basal, and optional wall-contact energies.
    E = 0
    im = 1
    i = 2

    length_l1 = @views distance(apical[im,:],basal[im,:])
    length_l2 = @views distance(apical[i,:],basal[i,:])
    length_a1 = @views distance(apical[im,:],apical[i,:])
    length_b1 = @views distance(basal[im,:],basal[i,:])

    E += Gl * (length_l1 + length_l2)/2
    E += Ga * (length_a1)
    E += Gb * (length_b1)

    if abs(basal[i,1]) > open_neck
        E += sticky_walls*(basal[i,2])^2
    end

    if abs(basal[im,1]) > open_neck
        E += sticky_walls*(basal[im,2])^2
    end

    if length_l1 < min_length || length_a1 < min_length || length_b1 < min_length
        flag_accept = false
    else
        flag_accept = true
    end
    return E
end

function penalty_volume(apical,basal,K,target_vol)
    E = 0
    i = 2
    im = 1

    area = @views cell_volume(apical[i,:],apical[im,:],basal[i,:],basal[im,:])

    E += K * (area - target_vol)^2

    return E
end

function asp_ratio(A,B,C,D,w0) # A/B are apical vertices; C/D are basal vertices.
    ap = @views distance(A,B)
    bas = @views distance(C,D)
    w = (ap+bas)/2
    epsilon = w/w0

    return epsilon
end

"""
    simulation(dg01, dg02, zeta, parameters)

Run one Monte Carlo realization for two crypts with initial myosin
asymmetries `dg01` and `dg02`. The `parameters` tuple contains the simulation
settings; output data and PDF snapshots are written under
`multi_output/<villus_cell_area>/`.
"""
function simulation(dg01, dg02, zeta, (T, dt, plotstep, N, cell_vol, cell_vol_villus, min_length, temp, Ga, Gb, Gl, l_c, d_Ga, d_Ga2, tau_myosin, K), cpla2_crypt1, cpla2_crypt2)
    N_vertex = N 
    wflat_crypt = sqrt(cell_vol * Gl / (Ga + Gb))
    wflat_villus = sqrt(cell_vol_villus * Gl / (Ga + Gb))

    zetaprint = round(cell_vol_villus * 1000) / 1000
    foldername = "./multi_output/$(zetaprint)"

    mkpath(foldername)
    mkpath("$(foldername)/images")
    mkpath("$(foldername)/data")

    # Initialize output series.
    neck1 = Float64[]
    length1_vals = Float64[]
    neck2 = Float64[]
    length2_vals = Float64[]

    # Initialize apical and basal vertex coordinates.
    apical = CircularArray(zeros(N_vertex, 2))    # (x,y)
    basal = CircularArray(zeros(N_vertex, 2))
    apical_new = CircularArray(zeros(N_vertex, 2))
    basal_new = CircularArray(zeros(N_vertex, 2))

    Ga_all = CircularArray(zeros(N_vertex)) .+ Gav
    Gb_all = CircularArray(zeros(N_vertex)) .+ Gbv
    crypt_flag = CircularArray(zeros(N_vertex))
    list_of_volumes = CircularArray(zeros(N_vertex))

    aspect_ratios = CircularArray(zeros(N_vertex))

    target_cell_area = CircularArray(zeros(N_vertex)) .+ cell_vol

    changes = zeros(Int(T/dt))
    av_diff = zeros(Int(T/dt))

    E_i = 0.0
    E_f = 0.0

    step = 0.01  # Monte Carlo displacement size.

    # Initialize the epithelium and label crypt cells.
    crypt_flag[Nvillus + 1 + nTA : Nvillus + Ncrypt - nTA] .= 1

    list_of_volumes[1:end] .= cell_vol_villus
    list_of_volumes[Nvillus + 1 : Nvillus + Ncrypt] .= cell_vol
    list_of_volumes[2*Nvillus + Ncrypt + 1 : end] .= cell_vol

    is_villus = list_of_volumes.*0 .+ 1
    is_villus[Nvillus + 2 : Nvillus + Ncrypt] .= 0
    is_villus[2*Nvillus + Ncrypt + 2 : end] .= 0



    hvillus = cell_vol_villus/wvillus

    basal[1 : Nvillus+1,1] = wvillus.*(0:Nvillus)
    basal[Nvillus+2 : Nvillus + Ncrypt + 1,1] = basal[Nvillus+1,1] .+ wcrypt.*(1:Ncrypt)
    basal[Nvillus + Ncrypt + 2 : 2*Nvillus + Ncrypt + 1,1] = basal[Nvillus+Ncrypt+1,1] .+ wvillus.*(1:Nvillus)
    basal[2*Nvillus + Ncrypt + 2 : end,1] = basal[2*Nvillus + Ncrypt + 1,1] .+ wcrypt.*(1:Ncrypt-1)

    apical[1:end, 1] = basal[1:end, 1]

    basal[1:end, 2] .= 0

    hcrypt = cell_vol/wcrypt
    apical[1:end, 2] .= hvillus
    apical[Nvillus + 2:Nvillus + Ncrypt, 2] .= hcrypt
    apical[2*Nvillus + Ncrypt + 2:end, 2] .= hcrypt

    Lmax = basal[end,1] - basal[1,1]
    Lmirror = Lmax + wcrypt
    index_center1 = Integer(Nvillus + Ncrypt/2)
    index_center2 = Integer(Nvillus + Ncrypt + Nvillus + Ncrypt/2)

    Lcrypt = wall_distance

    apical_new[:,:] = apical[:,:]

    basal_new[:,:] = basal[:,:]

    vapical = CircularArray(@view(apical[:,:]))
    vbasal = CircularArray(@view(basal[:,:]))
    vapical_new = CircularArray(@view(apical_new[:,:]))
    vbasal_new = CircularArray(@view(basal_new[:,:]))
    vvol = CircularArray(@view(list_of_volumes[:]))
    vGa = CircularArray(@view(Ga_all[:]))
    vGb = CircularArray(@view(Gb_all[:]))

    # Save the initial monolayer configuration.
    open("$(foldername)/data/monolayer_i.txt", "w") do f
        for i in 1:N
            write(f, "$(apical[i, 1]) $(apical[i, 2]) $(basal[i, 1]) $(basal[i, 2])\n")
        end
    end

    t = 0.0

    av_lat = 0.0
    nattempts = 0
    nchanges = 0
    flag_curv = 0.0

    E_tot = 0.0  # Accumulated energy change.

    while t < T

        if t == Time_2nd_crypt
            crypt_flag[2*Nvillus + Ncrypt + 1 + nTA : 2*Nvillus + 2*Ncrypt - nTA] .= 2
        end

        # Compute the mean apical-to-basal edge length.
        av_lat = 0.0
        for i in 1:N_vertex
            av_lat += sqrt((apical[i, 1] - basal[i, 1])^2 + (apical[i, 2] - basal[i, 2])^2) / N_vertex
        end
        
        for tryN in 1:N_vertex
            center_crypt1 = apical[index_center1, 1]
            center_crypt2 = apical[index_center2, 1]

            if t < Time_2nd_crypt
                i = rand(1:N_vertex - Ncrypt)  # Select a vertex at random.
            else
                i = rand(1:N_vertex)
            end
            flag_accept = true  # Reject moves that create an edge that is too short.
            
            if 1 <= i <= N_vertex  # Update each selected vertex.
                ip = i + 1
                im = i - 1

                # Relax crypt-cell tensions toward their width-dependent targets.
                length_l_temp = distance(vapical[i,:],vbasal[i,:])
                length_a2 = distance(vapical[i,:],vapical[ip,:])
                length_a1 = distance(vapical[i,:],vapical[im,:])
                length_b2 = distance(vbasal[i,:],vbasal[ip,:])
                length_b1 = distance(vbasal[i,:],vbasal[im,:])

                l_pack1 = (length_b1 + length_a1) / 2
                l_pack2 = (length_b2 + length_a2) / 2
                sc1 = crypt_flag[im]
                sc2 = crypt_flag[ip]
                curv1 = (length_b1 - length_a1) / (length_b1 + length_a1)
                curv2 = (length_b2 - length_a2) / (length_b2 + length_a2)

                if crypt_flag[i] > 0
                    if crypt_flag[i] == 1
                        sc1 = cpla2_crypt1 ? 1 : 0
                    else
                        sc2 = cpla2_crypt2 ? 1 : 0
                    end
                    dg0 = dg01 * (crypt_flag[i] == 1) + dg02 * (crypt_flag[i] == 2)
                    Ga_all[i] += (Ga_fct(dg0, l_pack2, sc2, curv2) - Ga_all[i]) * dt / tau_myosin
                    Gb_all[i] += (Gb_fct(dg0, l_pack2, sc2, curv2) - Gb_all[i]) * dt / tau_myosin
                end

                if crypt_flag[im] > 0  
                    if crypt_flag[im] == 1
                        sc1 = cpla2_crypt1 ? 1 : 0
                    else
                        sc1 = cpla2_crypt2 ? 1 : 0
                    end
                    dg0 = dg01 * (crypt_flag[im] == 1) + dg02 * (crypt_flag[im] == 2)
                    Ga_all[im] += (Ga_fct(dg0, l_pack1, sc1, curv1) - Ga_all[im]) * dt / tau_myosin
                    Gb_all[im] += (Gb_fct(dg0, l_pack1, sc1, curv1) - Gb_all[im]) * dt / tau_myosin
                end

                # Compute the energy before moving the selected vertex.
                E_i = 0
                E_i += energy_cell(vapical[i:ip,:],vbasal[i:ip,:],vGa[i],vGb[i],Gl,K,vvol[i], flag_accept) 
                E_i += energy_cell(vapical[im:i,:],vbasal[im:i,:],vGa[im],vGb[im],Gl,K,vvol[im], flag_accept)
                E_i += penalty_volume(vapical[i:ip,:],vbasal[i:ip,:],K,vvol[i])
                E_i += penalty_volume(vapical[im:i,:],vbasal[im:i,:],K,vvol[im])
                
                
                # Propose a random apical-vertex displacement.
                angle = 2 * π * rand()
                x_temp = apical[i, 1] + step * cos(angle)
                y_temp = apical[i, 2] + step * sin(angle)

                apical_new[i, 1] = x_temp
                apical_new[i, 2] = y_temp

                # Compute the energy after the proposed displacement.
                E_f = 0
                E_f += energy_cell(vapical_new[i:ip,:],vbasal_new[i:ip,:],vGa[i],vGb[i],Gl,K,vvol[i], flag_accept) 
                E_f += energy_cell(vapical_new[im:i,:],vbasal_new[im:i,:],vGa[im],vGb[im],Gl,K,vvol[im], flag_accept)
                E_f += penalty_volume(vapical_new[i:ip,:],vbasal_new[i:ip,:],K,vvol[i])
                E_f += penalty_volume(vapical_new[im:i,:],vbasal_new[im:i,:],K,vvol[im])
                
                
                nattempts += 1
                
                # Accept or reject the move using the Metropolis criterion.
                accept = false
                
                if E_f < E_i
                    if flag_accept # The move must preserve the minimum edge length.
                        accept = true
                    end
                else
                    if rand() < exp(-(E_f - E_i) / temp)
                        if flag_accept
                            accept = true
                        end
                    end
                end

                sym_flag = true
                if i != index_center1 && i != index_center2
                    if (apical_new[i,1] - center_crypt1)*(i-index_center1) < 0
                        sym_flag = false
                    end
                    if (apical_new[i,1] - center_crypt2)*(i-index_center2) < 0
                        sym_flag = false
                    end
                end

                if i == index_center1 || i == index_center2
                    sym_flag = true
                end

                
                if accept && sym_flag
                    apical[i, 1] = apical_new[i, 1]
                    apical[i, 2] = apical_new[i, 2]
                    basal[i,:] = basal_new[i,:]
                    changes[Int(t/dt) + 1] += 1
                    av_diff[Int(t/dt) + 1] += E_i - E_f
                    E_tot += -E_i + E_f
                    nchanges += 1
                else
                    apical_new[i, 1] = apical[i, 1]
                    apical_new[i, 2] = apical[i, 2]
                    basal_new[i,:] = basal[i,:]
                end
                
                # Compute the energy before moving the basal vertex.

                E_i = 0
                E_i += energy_cell(vapical[i:ip,:],vbasal[i:ip,:],vGa[i],vGb[i],Gl,K,vvol[i], flag_accept) 
                E_i += energy_cell(vapical[im:i,:],vbasal[im:i,:],vGa[im],vGb[im],Gl,K,vvol[im], flag_accept)
                E_i += penalty_volume(vapical[i:ip,:],vbasal[i:ip,:],K,vvol[i])
                E_i += penalty_volume(vapical[im:i,:],vbasal[im:i,:],K,vvol[im])
                
                # Propose a random basal-vertex displacement.

                angle = 2 * π * rand()
                x_temp = basal[i, 1] + step * cos(angle)
                y_temp = basal[i, 2] + step * sin(angle)

                basal_new[i, 1] = x_temp
                if is_villus[i] == 0
                    basal_new[i, 2] = y_temp
                end

                E_f = 0
                E_f += energy_cell(vapical_new[i:ip,:],vbasal_new[i:ip,:],vGa[i],vGb[i],Gl,K,vvol[i], flag_accept) 
                E_f += energy_cell(vapical_new[im:i,:],vbasal_new[im:i,:],vGa[im],vGb[im],Gl,K,vvol[im], flag_accept)
                E_f += penalty_volume(vapical_new[i:ip,:],vbasal_new[i:ip,:],K,vvol[i])
                E_f += penalty_volume(vapical_new[im:i,:],vbasal_new[im:i,:],K,vvol[im])
                
                nattempts += 1
                
                # Accept or reject the basal move using the Metropolis criterion.
                accept = false

                if E_f < E_i
                    if flag_accept # The move must preserve the minimum edge length.
                        accept = true
                    end
                else
                    if rand() < exp(-(E_f - E_i) / temp)
                        if flag_accept
                            accept = true
                        end
                    end
                end
                
                if accept
                    basal[i, 1] = basal_new[i, 1]
                    basal[i, 2] = basal_new[i, 2]
                    changes[Int(t/dt) + 1] += 1
                    av_diff[Int(t/dt) + 1] += E_i - E_f
                    E_tot += -E_i + E_f
                    nchanges += 1
                else
                    basal_new[i, 1] = basal[i, 1]
                    basal_new[i, 2] = basal[i, 2]
                end
            end
        end
        
        # Save sampled geometry and time-series data.

        if t % plotstep == 0

            open(joinpath(foldername, "data", "aspect_ratios.txt"), "a") do f
                println(f,["$eps " for eps in aspect_ratios]...)
            end

            aspect_ratios[:] .= 0

            # Estimate the mean epithelial area from the apical and basal contours.
            tot_area = sum(0.5 * (apical[1:end-1,2] + apical[2:end,2]) .* (apical[2:end,1] - apical[1:end-1,1]) - 
                        0.5 * (basal[1:end-1,2] + basal[2:end,2]) .* (basal[2:end,1] - basal[1:end-1,1])) / (N_vertex-1)
            
            tot_area = tot_area / (N_vertex-1)
            # Compute the target mean cell area.
            tot_target_vol = sum(target_cell_area) / N_vertex
            nchanges = 0
            nattempts = 0
            
            open(joinpath(foldername, "data", "energy.txt"), "a") do f
                println(f, E_tot)
            end
            
            # Collect cell segments, including periodic replicas for display.
            lines = []
            for i in 1:N+extra_N
                if i > N
                    push!(lines, (vapical[i,:] .+ [Ltot; 0], vbasal[i,:] .+ [Ltot; 0]))
                else
                    push!(lines, (vapical[i,:], vbasal[i,:]))
                end
            end
            
            fig = plot(aspect_ratio=:equal, xlim=[0, Ltot + 5], ylim=[-20, 20])

            villus_color = RGB(217/255,132/255,129/255)

            # Color crypt cells by myosin asymmetry and villus cells uniformly.
            for i = 1:N-1
                if crypt_flag[i] <= 0
                    color_cell = villus_color
                else
                    dg = Ga_all[i] - Gb_all[i]
                    color_cell = dg_grad(dg)
                end
                x_vertices = [apical[i,1], apical[i+1,1], basal[i+1,1], basal[i,1]]
                y_vertices = [apical[i,2], apical[i+1,2], basal[i+1,2], basal[i,2]]
                plot!(fig, Shape(x_vertices, y_vertices), fillcolor=color_cell, fillalpha=1.0,linecolor=:transparent, label="")
            end

            x_vertices = [apical[N,1], apical[N+1,1]+Ltot, basal[N+1,1]+Ltot, basal[N,1]]
            y_vertices = [apical[N,2], apical[N+1,2], basal[N+1,2], basal[N,2]]
            plot!(fig, Shape(x_vertices, y_vertices), fillcolor=villus_color, fillalpha=1.0,linecolor=:transparent, label="")

            for i = N:(N+extra_N)
                if crypt_flag[i] <= 0
                    color_cell = villus_color
                else
                    dg = Ga_all[i] - Gb_all[i]
                    color_cell = dg_grad(dg)
                end
                x_vertices = [apical[i,1], apical[i+1,1], basal[i+1,1], basal[i,1]] .+ Ltot
                y_vertices = [apical[i,2], apical[i+1,2], basal[i+1,2], basal[i,2]]
                plot!(fig, Shape(x_vertices, y_vertices), fillcolor=color_cell, fillalpha=1.0,linecolor=:transparent, label="")
            end
            

            # Draw apical-to-basal cell segments.
            for line in lines
                plot!(fig, [line[1][1], line[2][1]], [line[1][2], line[2][2]], color=:black, legend=false)
            end

            # Draw the apical and basal contours, including periodic replicas.
            plot!(fig, apical[:,1], apical[:,2], legend=false)
            plot!(fig, basal[:,1], basal[:,2], legend=false)
            plot!(fig, apical[N_vertex:N_vertex+1,1] + [0; Ltot], apical[N_vertex:N_vertex+1,2], legend=false)
            plot!(fig, basal[N_vertex:N_vertex+1,1] + [0; Ltot], basal[N_vertex:N_vertex+1,2], legend=false)
            plot!(fig, apical[1:extra_N,1].+Ltot, apical[1:extra_N,2], legend=false)
            plot!(fig, basal[1:extra_N,1].+Ltot, basal[1:extra_N,2], legend=false)
            
            savefig(fig, joinpath(foldername, "images", "frame_$(Int(t)).pdf"))

            # Record neck widths and lengths for both crypts.
            i = Nvillus+nTA
            iend = Nvillus+Ncrypt-nTA+2
            neck_temp = abs(basal[i,1]-basal[iend,1])
            
            push!(neck1, neck_temp)
            push!(length1_vals, abs(minimum(basal[i:iend,2])))

            i = 2*Nvillus +  Ncrypt + nTA
            iend = 2*Nvillus + 2*Ncrypt - nTA + 2
            neck_temp = abs(basal[i,1]-basal[iend,1])
            
            push!(neck2, neck_temp)
            push!(length2_vals, abs(minimum(basal[i:iend,2])))
        end

        t += dt

    end

    # Write final per-cell and time-series data.
    open(joinpath(foldername, "data", "areasf.txt"), "w") do f
        for i in 2:N_vertex-1
            area = cell_volume(apical[i,:],apical[i-1,:],basal[i,:],basal[i-1,:])
            write(f, area, " ")
            write(f, "\n")
        end
    end

    open(joinpath(foldername, "data", "changes.txt"), "w") do f
        for i in 1:Int(T/dt)
            println(f, changes[i], ' ', av_diff[i])
        end
    end

    open(joinpath(foldername, "apical_tension.txt"), "w") do f
        for i in 1:N_vertex
            println(f, Ga_all[i])
        end
    end

    open(joinpath(foldername, "width.txt"), "w") do f
        for i in 1:N_vertex-1
            length_a2 = sqrt((apical[i,1]-apical[mod1(i+1,N_vertex),1])^2 + (apical[i,2]-apical[mod1(i+1,N_vertex),2])^2)
            length_b2 = sqrt((basal[i,1]-basal[mod1(i+1,N_vertex),1])^2 + (basal[i,2]-basal[mod1(i+1,N_vertex),2])^2)
            l_pack = (length_b2 + length_a2)/2
            println(f, l_pack)
        end
    end

    open(joinpath(foldername, "neck1.txt"), "w") do f
        for i in 1:length(neck1)
            println(f, neck1[i])
        end
    end

    open(joinpath(foldername, "length1.txt"), "w") do f
        for i in 1:length(length1_vals)
            println(f, length1_vals[i])
        end
    end

    open(joinpath(foldername, "neck2.txt"), "w") do f
        for i in 1:length(neck2)
            println(f, neck2[i])
        end
    end

    open(joinpath(foldername, "length2.txt"), "w") do f
        for i in 1:length(length2_vals)
            println(f, length2_vals[i])
        end
    end

end


cd(@__DIR__)
rm("multi_output"; force=true, recursive=true)
mkpath("multi_output")

zeta = 0.0  # External in-plane tension; positive values stretch the tissue.
nsimu = 10
villus_2D_area_v = range(1.4, 2.0, nsimu)
dg01 = 0.8
dg02 = 0.3

cpla2_crypt1 = false
cpla2_crypt2 = false

for k in 1:nsimu
    cell_vol_villus = villus_2D_area_v[k]
    parameters = (T, dt, plotstep, N, cell_vol, cell_vol_villus, min_length, temp, Ga, Gb, Gl, l_c, d_Ga, d_Ga2, tau_myosin, K)
    println("Launching simulation with villus cell area = ", cell_vol_villus)
    @time simulation(dg01, dg02, zeta, parameters, cpla2_crypt1, cpla2_crypt2)
end

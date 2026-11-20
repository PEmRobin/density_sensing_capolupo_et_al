
"""
    density_sensing.jl

Vertex-based Monte Carlo model of an epithelial monolayer containing one
crypt. Crypt cells adjust their apical and basal tensions in response to cell
width and curvature; villus cells do not sense curvature and retain their
polarized tension balance.

The simulation corresponds to a line of cells, with the crypt in the center and 
villus cells on either side. Boundary conditions are fixed.
For each simulation, geometry is optimized for a given level of swelling, but 
crypt cells update their cytoskeletal tensions.

The mechanosensitive process can be adjusted through the functions 
`Ga_fct(l, sc, curv)` and `Gb_fct(l, sc, curv)`, which return the
apical and basal tensions for a cell with width `l`, curvature `curv`, 
and sensing flag `sc`. The default functions implement the 
density-sensing model described in Capolupo et al., 
"Tissue swelling and local mechanosensation drive crypt convergence in intestinal development."

Alternative models (eg curvature sensing) can also be tested by adjusting that function.

This code runs the simulation for different swelling levels of the villus cells,
 which are set by the `cell_vol_villus` parameter in `main()`. 
The tested values are in the range 1.4-2.0 (in units of 10 μm^2), 
expressed as the cross section of a single villus cell (see the paper).
At the peak of swelling, experimental values are around 2.0.

Running this file removes the previous `multi_output/` directory, then runs
ten simulations over the configured villus-cell areas. Results are written to
`multi_output/` beside this script. 

Requires the Julia packages `Images`, `Plots`, and `LsqFit`. `Random` and
`Statistics` are part of the standard library.
"""

using Random
using Statistics
using Plots
using Images
using LsqFit

"""Maximum simulation time; all lengths and areas use units of 10 μm."""
const T = 200001
const dt = 1.0   # Time increment
const plotstep = 1000
const Ncrypt = 12
const Nvillus = 12
const N = Ncrypt + 2*Nvillus + 1      # Number of vertices
const wvillus = 0.8
const Lwall = wvillus*Nvillus
const sticky_walls = 0.
const open_neck = 2.0
const nTA = 3

Ga = 1.0   # Apical tension
Gb = 1.0   # Basal tension
Gl = 1.0   # Lateral tension

const Gav = 1.
const Gbv = 1.


const cell_vol = 0.6  # Cell area in (10 μm)^2; typical values are 0.8 in crypts and 1.5-1.7 in villi.
effective_villus_zeta = -0.9

const wcrypt = 0.6
const wall_distance = Ncrypt*wcrypt

const min_length = 0.25
const temp = 0.001  # Metropolis temperature

l_c = 0.6  # Cells narrower than this become more apically constricted.
d_Ga = 0.5  # Maximum myosin-tension adjustment; Ga + Gb remains constant.
d_Ga2 = 0.0  # Curvature-sensing strength; use 0.4 to enable the default response.

# Tension response functions of cell width and curvature.
Ga_fct(l, sc) = Ga + sc * d_Ga * tanh(l_c/l)
Gb_fct(l, sc) = Gb - sc * d_Ga * tanh(l_c/l)

hcrypt = cell_vol/wcrypt
curv0 = 0.25/hcrypt 
# Curvature-dependent response, applied only to positive curvature.
Ga_fct(l, sc, curv) = max(Ga - sc * d_Ga * tanh(5 * (l - l_c) / l_c) * (l < l_c ? 1 : 0) + d_Ga2 * tanh(curv / curv0) * (curv > 0 ? 1 : 0), 0.1)
Gb_fct(l, sc, curv) = max(Gb + sc * d_Ga * tanh(5 * (l - l_c) / l_c) * (l < l_c ? 1 : 0) - d_Ga2 * tanh(curv / curv0) * (curv > 0 ? 1 : 0), 0.1)

const tau_myosin = 100  # Relaxation timescale for myosin tension.

# Volume-constraint penalty strength.
const K = 100.0

# Compute area enclosed within 4 points
function cell_volume(apical_i,apical_im,basal_i,basal_im)
    area = 0.5 * abs((apical_i[1] * basal_i[2] - apical_i[2] * basal_i[1]) + 
                    (basal_i[1] * basal_im[2] - basal_i[2] * basal_im[1]) + 
                    (basal_im[1] * apical_im[2] - basal_im[2] * apical_im[1]) + 
                    (apical_im[1] * apical_i[2] - apical_im[2] * apical_i[1]))
    return area
end

function curvature(apical, basal)
    length_a = ((apical[1:end-1,1] - apical[2:end,1]).^2 + (apical[1:end-1,2] - apical[2:end,2]).^2).^(0.5)
    length_b = ((basal[1:end-1,1] - basal[2:end,1]).^2 + (basal[1:end-1,2] - basal[2:end,2]).^2).^(0.5)
    lpack = (length_a + length_b)/2
    hcell = cell_vol./lpack
    curv = 2*(length_b - length_a) ./ (length_b + length_a)./hcell
    return curv
end

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

function curvilinear(line)
    n = length(line[:,1])
    s_curv = zeros(n)
    for k in 2:n 
        s_curv[k] = s_curv[k-1] + sqrt((line[k,1] - line[k-1,1]).^2 + (line[k,2] - line[k-1,2]).^2)
    end
    s_curv[:] = s_curv[:] .- mean(s_curv)
    return s_curv
end

# Computes distance between 2 points
function distance(A,B)
    d = sqrt((B[1] - A[1])^2 + (B[2] - A[2])^2)
    return d
end

function energy_cell(apical,basal,Ga,Gb,Gl,K,target_volume, flag_accept)
    # energy of a cell
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
    simulation(zeta, parameters)

Run one Monte Carlo realization for the supplied cell-area and model
parameters. Time-series data, geometry snapshots, and plots are written under
`multi_output/<villus_cell_area>/`.
"""
function simulation(zeta, (T, dt, plotstep, N, cell_vol, cell_vol_villus, min_length, temp, Ga, Gb, Gl, l_c, d_Ga, d_Ga2, tau_myosin, K))
    N_vertex = N 
    wflat_crypt = sqrt(cell_vol * Gl / (Ga + Gb))
    wflat_villus = sqrt(cell_vol_villus * Gl / (Ga + Gb))

    zetaprint = round(cell_vol_villus * 1000) / 1000
    foldername = "./multi_output/$(zetaprint)"

    mkpath(foldername)
    mkpath("$(foldername)/images")
    mkpath("$(foldername)/data")

    # Initialize output series.
    neck = Float64[]
    length_vals = Float64[]
    Rlist = Float64[]

    # Initialize apical and basal vertex coordinates.
    apical = zeros(N_vertex, 2)    # (x,y)
    basal = zeros(N_vertex, 2)
    apical_new = zeros(N_vertex, 2)
    basal_new = zeros(N_vertex, 2)

    Ga_all = zeros(N_vertex) .+ Gav
    Gb_all = zeros(N_vertex) .+ Gbv
    crypt_flag = zeros(N_vertex)
    list_of_volumes = zeros(N_vertex)

    aspect_ratios = zeros(N_vertex)

    target_cell_area = zeros(N) .+ cell_vol

    changes = zeros(Int(T/dt))
    av_diff = zeros(Int(T/dt))

    E_i = 0.0
    E_f = 0.0

    step = 0.01  # Monte Carlo displacement size.

    # Initialize the epithelium and mark the active crypt cells.
    crypt_flag[Nvillus + 1 + nTA : Nvillus + Ncrypt - nTA] .= 1

    list_of_volumes[1:end] .= cell_vol_villus
    list_of_volumes[Nvillus + 1 : Nvillus + Ncrypt] .= cell_vol

    is_villus = list_of_volumes.*0 .+ 1
    is_villus[Nvillus + 2 : Nvillus + Ncrypt] .= 0

    hvillus = cell_vol_villus/wvillus

    basal[1:Nvillus+1,1] = -(Lwall .- wvillus.*(0:Nvillus)) .-wall_distance/2
    basal[Nvillus + Ncrypt + 1:end,1] = wvillus.*(0:Nvillus) .+ wall_distance/2

    apical[1:Nvillus+1,1] = -(Lwall .- wvillus.*(0:Nvillus)) .-wall_distance/2
    apical[Nvillus + Ncrypt + 1:end,1] = wvillus.*(0:Nvillus) .+ wall_distance/2

    basal[1:Nvillus+1,2] .= 0.0
    basal[Nvillus + Ncrypt + 1:end,2] .= 0.0

    apical[1:Nvillus+1,2] .= hvillus
    apical[Nvillus + Ncrypt + 1:end,2] .= hvillus

    Lcrypt = wall_distance
    X = range(0, Lcrypt, length=Ncrypt+1)
    hcrypt = cell_vol/wcrypt

    basal[Nvillus + 2:Nvillus + Ncrypt, 1] = wcrypt.*(1:Ncrypt-1) .- wall_distance/2
    basal[Nvillus + 2:Nvillus + Ncrypt, 2] .= 0.0
    apical[Nvillus + 2:Nvillus + Ncrypt, 1] = wcrypt.*(1:Ncrypt-1) .- wall_distance/2
    apical[Nvillus + 2:Nvillus + Ncrypt, 2] .= hcrypt


    apical_new[:,:] = apical[:,:]

    basal_new[:,:] = basal[:,:]

    vapical = @view(apical[:,:])
    vGa = @view(Ga_all[:])
    vGb = @view(Gb_all[:])
    vbasal = @view(basal[:,:])
    vapical_new = @view(apical_new[:,:])
    vbasal_new = @view(basal_new[:,:])
    vvol = @view(list_of_volumes[:])

    centerline = (apical + basal)./2

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

    i = Nvillus+nTA
    iend = Nvillus+Ncrypt-nTA+2
    neck_temp = abs(basal[i,1]-basal[iend,1])


    
    while t < T

        # Compute the mean apical-to-basal edge length.
        av_lat = 0.0
        for i in 1:N_vertex
            av_lat += sqrt((apical[i, 1] - basal[i, 1])^2 + (apical[i, 2] - basal[i, 2])^2) / N_vertex
        end
        
        for tryN in 1:N_vertex
            i = rand(1:N_vertex)  # Select a vertex at random.
            flag_accept = true  # Reject moves that create an edge that is too short.
            
            if 1 < i < N_vertex  # Update interior vertices only.
                ip = i + 1
                im = i - 1
                # Legacy periodic-boundary handling; inactive for this open-boundary model.
                if i == N_vertex
                    ip = 1
                end
                if i == 1
                    im = N_vertex
                end

                # Relax crypt-cell tensions toward their width/curvature targets.
                length_l_temp = distance(vapical[i,:],vbasal[i,:])
                length_a2 = distance(vapical[i,:],vapical[ip,:])
                length_a1 = distance(vapical[i,:],vapical[im,:])
                length_b2 = distance(vbasal[i,:],vbasal[ip,:])
                length_b1 = distance(vbasal[i,:],vbasal[im,:])

                l_pack1 = (length_b1 + length_a1) / 2
                l_pack2 = (length_b2 + length_a2) / 2
                sc1 = crypt_flag[im]
                sc2 = crypt_flag[ip]
                hcell1 = cell_vol./l_pack1
                hcell2 = cell_vol./l_pack2
                curv1 = (length_b1 - length_a1) / (length_b1 + length_a1) / hcell1
                curv2 = (length_b2 - length_a2) / (length_b2 + length_a2) / hcell2

                if crypt_flag[i] == 1
                    sc2 = 1
                    Ga_all[i] += (Ga_fct(l_pack2, sc2, curv2) - Ga_all[i]) * dt / tau_myosin
                    Gb_all[i] += (Gb_fct(l_pack2, sc2, curv2) - Gb_all[i]) * dt / tau_myosin
                end

                if crypt_flag[im] == 1 
                    sc1 = 1
                    Ga_all[im] += (Ga_fct(l_pack1, sc1, curv1) - Ga_all[im]) * dt / tau_myosin
                    Gb_all[im] += (Gb_fct(l_pack1, sc1, curv1) - Gb_all[im]) * dt / tau_myosin
                end

                # Compute the energy before moving the selected vertex.
                E_i = 0
                if i == 1
                    E_i += energy_cell(vapical[i:ip,:],vbasal[i:ip,:],vGa[i],vGb[i],Gl,K,vvol[i], flag_accept)
                    E_i += penalty_volume(vapical[i:ip,:],vbasal[i:ip,:],K,vvol[i])
                    E_i += zeta * basal[i, 1] 
                elseif i == N_vertex # Legacy endpoint branch; excluded by the interior-vertex condition above.
                    E_i += energy_cell(vapical[im:i,:],vbasal[im:i,:],vGa[im],vGb[im],Gl,K,vvol[im], flag_accept)
                    E_i += penalty_volume(vapical[im:i,:],vbasal[im:i,:],K,vvol[im])
                    E_i += - zeta * basal[i, 1] 
                else
                    E_i += energy_cell(vapical[i:ip,:],vbasal[i:ip,:],vGa[i],vGb[i],Gl,K,vvol[i], flag_accept) 
                    E_i += energy_cell(vapical[im:i,:],vbasal[im:i,:],vGa[im],vGb[im],Gl,K,vvol[im], flag_accept)
                    E_i += penalty_volume(vapical[i:ip,:],vbasal[i:ip,:],K,vvol[i])
                    E_i += penalty_volume(vapical[im:i,:],vbasal[im:i,:],K,vvol[im])
                end
                
                # Propose a random apical-vertex displacement.
                angle = 2 * π * rand()
                x_temp = apical[i, 1] + step * cos(angle)
                y_temp = apical[i, 2] + step * sin(angle)

                apical_new[i, 1] = x_temp

                if i!= N_vertex && i != 1
                    apical_new[i, 2] = y_temp
                end

                # Compute the energy after the proposed displacement.
                E_f = 0
                if i == 1
                    E_f += energy_cell(vapical_new[i:ip,:],vbasal_new[i:ip,:],vGa[i],vGb[i],Gl,K,vvol[i], flag_accept)
                    E_f += penalty_volume(vapical_new[i:ip,:],vbasal_new[i:ip,:],K,vvol[i])
                    E_f += zeta * basal_new[i, 1] 
                elseif i == N # Legacy endpoint branch; excluded by the interior-vertex condition above.
                    E_f += energy_cell(vapical_new[im:i,:],vbasal_new[im:i,:],vGa[im],vGb[im],Gl,K,vvol[im], flag_accept)
                    E_f += penalty_volume(vapical_new[im:i,:],vbasal_new[im:i,:],K,vvol[im])
                    E_f += - zeta * basal_new[i, 1] 
                else
                    E_f += energy_cell(vapical_new[i:ip,:],vbasal_new[i:ip,:],vGa[i],vGb[i],Gl,K,vvol[i], flag_accept) 
                    E_f += energy_cell(vapical_new[im:i,:],vbasal_new[im:i,:],vGa[im],vGb[im],Gl,K,vvol[im], flag_accept)
                    E_f += penalty_volume(vapical_new[i:ip,:],vbasal_new[i:ip,:],K,vvol[i])
                    E_f += penalty_volume(vapical_new[im:i,:],vbasal_new[im:i,:],K,vvol[im])
                end
                
                nattempts += 1
                
                # Accept or reject the move using the Metropolis criterion.
                accept = false
                
                if E_f < E_i
                    if flag_accept # checks that new geometry does not break conditions like lateral length > min_length
                        accept = true
                    end
                else
                    if rand() < exp(-(E_f - E_i) / temp)
                        if flag_accept
                            accept = true
                        end
                    end
                end

                sym_flag = false
                if apical_new[i,1]*(i-N/2) >= 0
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
                
                # Propose a basal-vertex displacement.

                E_i = 0
                if i == 1
                    E_i += energy_cell(vapical[i:ip,:],vbasal[i:ip,:],vGa[i],vGb[i],Gl,K,vvol[i], flag_accept)
                    E_i += penalty_volume(vapical[i:ip,:],vbasal[i:ip,:],K,vvol[i])
                    E_i += zeta * basal[i, 1] 
                elseif i == N # Legacy endpoint branch; excluded by the interior-vertex condition above.
                    E_i += energy_cell(vapical[im:i,:],vbasal[im:i,:],vGa[im],vGb[im],Gl,K,vvol[im], flag_accept)
                    E_i += penalty_volume(vapical[im:i,:],vbasal[im:i,:],K,vvol[im])
                    E_i += -zeta * basal[i, 1] 
                else
                    E_i += energy_cell(vapical[i:ip,:],vbasal[i:ip,:],vGa[i],vGb[i],Gl,K,vvol[i], flag_accept) 
                    E_i += energy_cell(vapical[im:i,:],vbasal[im:i,:],vGa[im],vGb[im],Gl,K,vvol[im], flag_accept)
                    E_i += penalty_volume(vapical[i:ip,:],vbasal[i:ip,:],K,vvol[i])
                    E_i += penalty_volume(vapical[im:i,:],vbasal[im:i,:],K,vvol[im])
                end
                
                angle = 2 * π * rand()
                x_temp = basal[i, 1] + step * cos(angle)
                y_temp = basal[i, 2] + step * sin(angle)

                basal_new[i, 1] = x_temp
                if is_villus[i]==0
                    basal_new[i, 2] = y_temp
                end

                E_f = 0
                if i == 1
                    E_f += energy_cell(vapical_new[i:ip,:],vbasal_new[i:ip,:],vGa[i],vGb[i],Gl,K,vvol[i], flag_accept)
                    E_f += penalty_volume(vapical_new[i:ip,:],vbasal_new[i:ip,:],K,vvol[i])
                    E_f += zeta * basal_new[i, 1] 
                elseif i == N_vertex # Legacy endpoint branch; excluded by the interior-vertex condition above.
                    E_f += energy_cell(vapical_new[im:i,:],vbasal_new[im:i,:],vGa[im],vGb[im],Gl,K,vvol[im], flag_accept)
                    E_f += penalty_volume(vapical_new[im:i,:],vbasal_new[im:i,:],K,vvol[im])
                    E_f += -zeta * basal_new[i, 1] 
                else
                    E_f += energy_cell(vapical_new[i:ip,:],vbasal_new[i:ip,:],vGa[i],vGb[i],Gl,K,vvol[i], flag_accept) 
                    E_f += energy_cell(vapical_new[im:i,:],vbasal_new[im:i,:],vGa[im],vGb[im],Gl,K,vvol[im], flag_accept)
                    E_f += penalty_volume(vapical_new[i:ip,:],vbasal_new[i:ip,:],K,vvol[i])
                    E_f += penalty_volume(vapical_new[im:i,:],vbasal_new[im:i,:],K,vvol[im])
                end
                
                nattempts += 1
                
                # Accept or reject the basal move using the Metropolis criterion.
                accept = false

                if E_f < E_i
                    if flag_accept # checks that new geometry does not break conditions like lateral length > min_length
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

            curvatures = curvature(vapical[Nvillus + 1 + nTA : Nvillus + Ncrypt - nTA,:],vbasal[Nvillus + 1 + nTA : Nvillus + Ncrypt - nTA,:])
            centerline = (apical[Nvillus + 1 + nTA : Nvillus + Ncrypt - nTA,:] + basal[Nvillus + 1 + nTA : Nvillus + Ncrypt - nTA,:])./2

            R_curv = 0
            theta = 0
            @. model(s,R) = centerline[Int((Ncrypt)/2 - nTA),2] + R[1] - R[1]*cos(s/R[1])
            @. model(s,R) = R[2] + R[1] - R[1]*cos(s/R[1])
            @. modelx(s,R) = R*sin(s/R)
            @. modely(y0,s,R) = y0 + R - R*cos(s/R)
            R0 = [Lcrypt, centerline[Int((Ncrypt)/2 - nTA),2]]
            s_curv = curvilinear(@view(centerline[:,:]))

            fit = curve_fit(model, s_curv, centerline[:,2], R0)
            R_curv = coef(fit)
            y0 = R_curv[2]
            R_curv = R_curv[1]
            theta = (s_curv[end] - s_curv[1])/2/R_curv

            open(joinpath(foldername, "data", "curvatures.txt"), "a") do f
                println(f, ["$c " for c in curvatures]...)
            end

            open(joinpath(foldername, "data", "centerline.txt"), "a") do f
                println(f, ["$c " for c in centerline]...)
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
            
            # Collect apical-to-basal segments for plotting.
            lines = NTuple{2,SubArray{Float64,1,Matrix{Float64},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true}}[]
            for i in 1:N
                push!(lines, (@view(apical[i,:]), @view(basal[i,:])))
            end
            
            fig = plot(aspect_ratio=:equal, xlim=[-20, 20], ylim=[-20, 20])

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
            
            # Draw apical-to-basal cell segments.
            for line in lines
                plot!(fig, [line[1][1], line[2][1]], [line[1][2], line[2][2]], color=:black, legend=false)
            end
            
            # Draw the apical and basal contours.
            plot!(fig, apical[:,1], apical[:,2], legend=false)
            plot!(fig, basal[:,1], basal[:,2], legend=false)

            savefig(fig, joinpath(foldername, "images", "frame_$(Int(t)).pdf"))

            # Record crypt neck width, crypt length, and fitted curvature radius.
            i = Nvillus+nTA
            iend = Nvillus+Ncrypt-nTA+2
            neck_temp = abs(basal[i,1]-basal[iend,1])
            
            push!(neck, neck_temp)
            push!(length_vals, abs(basal[i,2]-minimum(basal[:,2])))
            push!(Rlist, R_curv)
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
        for i in 1:N_vertex-1
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

    open(joinpath(foldername, "neck.txt"), "w") do f
        for i in 1:length(neck)
            println(f, neck[i])
        end
    end

    open(joinpath(foldername, "length.txt"), "w") do f
        for i in 1:length(length_vals)
            println(f, length_vals[i])
        end
    end

    open(joinpath(foldername, "Rcurv.txt"), "w") do f
        for i in 1:length(Rlist)
            println(f, Rlist[i])
        end
    end

end


cd(@__DIR__)
rm("multi_output"; force=true, recursive=true)
mkpath("multi_output")

zeta = 0.5  # External in-plane tension; positive values stretch the tissue.
nsimu = 10
villus_2D_area_v = range(1.4, 2.1,nsimu)

for k in 1:nsimu
    cell_vol_villus = villus_2D_area_v[k]
    parameters = (T, dt, plotstep, N, cell_vol, cell_vol_villus, min_length, temp, Ga, Gb, Gl, l_c, d_Ga, d_Ga2, tau_myosin, K)
    println("Launching simulation with villus cell area = ", cell_vol_villus)
    @time simulation(zeta, parameters)
end
"""
    organoid_caps_MC.jl

Snapshot Monte Carlo model of an organoid with two crypts and a central
villus, archived with Capolupo et al., "Tissue swelling and local
mechanosensation drive crypt convergence in intestinal development."

Running this file performs one simulation using the parameters in `main()`
and saves a PDF figure in the script directory. The default simulation uses
30,000,000 Metropolis proposals. Set `Random.seed!(seed)` before `main()` for
a repeatable run.

Myosin levels and swelling can be adjusted in `main()` by changing the values of `dg1`, `dg2`, and `swelling`. 

Requires the Julia packages `Images` and `Plots`; `Random` and `Printf` are
part of the standard library.
"""

using Random
using Printf
using Images
using Plots

"""Map the swelling fraction to the translucent fill color used in plots."""
function swell_grad(sw)
    hi_color = RGBA(151/255,177/255,221/255,0.)
    lo_color = RGBA(151/255,177/255,221/255,1.0)

    hi_color = RGBA(233/255,71/255,52/255,0.5)
    lo_color = RGBA(233/255,71/255,52/255,0.)
    color = hi_color*sw + lo_color*(1-sw)
    return color
end


"""Map apical myosin asymmetry to the blue-to-orange plot color scale."""
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


"""
    main()

Run the archived single-snapshot model with its hard-coded initial geometry,
myosin asymmetries, and swelling. Print the resulting branching indices and
save a PDF sketch in the current directory.
"""
function main(R_t0,R0,dg1,dg2)
    T = 30_000_000 # Metropolis proposals; reduce for exploratory runs.
    temp = 1.1 # Metropolis temperature; increase for exploratory runs.
    dR = 0.1
    dtheta = pi/100

    ga1 = 1.0 + dg1/2
    gb1 = 1.0 - dg1/2

    ga2 = 1.0 + dg2/2
    gb2 = 1.0 - dg2/2

    het_myosin = abs(dg1-dg2)/0.5

    gav = 1.0
    gbv = 1.0

    gav = 0.8
    gbv = 1.6

    gl = 1.0

    v_c = 700.0  #crypt cell vol
    v_v = 1.0e3 # villus cell vol, 1.0e3 prior to swelling
    N_v = 120.0 #villus cell number
    N_c = 30.0 # crypt cell number
  
    V_lumen_t0 = 4/3*pi*R_t0^3


    RM = 28
    Rm = 22
    swelling = 1-(R0 - Rm)/(RM - Rm)
    println("Swelling = ", swelling)
    println("Myo heterogeneity = ", het_myosin)
    V_lumen = 4/3*pi*R0^3

    v_v = v_v + (V_lumen_t0 - V_lumen)/N_v
    println(v_v)

    Omega_c(thetac) = 2*pi*(1 + cos(thetac))
    h_c(Rc,thetac) = N_c * v_c/Rc^2/Omega_c(thetac)

    Omega_v(thetav1,thetav2) = 4*pi - 2*pi*(1-cos(thetav1)) - 2*pi*(1-cos(thetav2))
    h_v(vv,Rv,tv1,tv2) = N_v*vv/Omega_v(tv1,tv2)/Rv^2

    t_v(Rv,Rc,tc) = asin(sin(tc)*Rc/Rv)

    Energy_crypt(ga,gb,Rc,tc) = Omega_c(tc)*Rc^2*(ga + gb) + Omega_c(tc)*Rc*h_c(Rc,tc)*(gb - ga) + 2*sqrt(N_c)*sqrt(Omega_c(tc))*Rc*h_c(Rc,tc)*gl
    Energy_villus(vv,Rv,tv1,tv2) = Omega_v(tv1,tv2)*Rv^2*(gav + gbv) + Omega_v(tv1,tv2)*Rv*h_v(vv,Rv,tv1,tv2)*(gbv - gav) + 2*sqrt(N_v)*sqrt(Omega_v(tv1,tv2))*Rv*h_v(vv,Rv,tv1,tv2)*gl

    Constraints_c(Rc,tc) = exp( -100*(Rc - h_c(Rc,tc)/2) ) + exp(-100*Rc) + exp(-100*h_c(Rc,tc)) + exp(-100*tc) + exp(-100*(pi-tc))
    Energy_tot(vv,ga1,gb1,ga2,gb2,Rc1,Rc2,tc1,tc2,Rv,tv1,tv2) = Energy_crypt(ga1,gb1,Rc1,tc1) + Energy_crypt(ga2,gb2,Rc2,tc2) + Energy_villus(vv,Rv,tv1,tv2) + Constraints_c(Rc1,tc1) + Constraints_c(Rc2,tc2)

    Organoid_volume(vv,Rc1,Rc2,tc1,tc2,Rv,tv1,tv2) = pi/3*(Rc1 - h_c(Rc1,tc1)/2)^3*(2 - cos(tc1))*(1 + cos(tc1))^2 + pi/3*(Rc2 - h_c(Rc2,tc2)/2)^3*(2 - cos(tc2))*(1 + cos(tc2))^2 + 4/3*pi*(Rv - h_v(vv,Rv,tv1,tv2)/2)^3 - pi/3*(Rv - h_v(vv,Rv,tv1,tv2)/2)^3*(2 + cos(tv1))*(1 - cos(tv1))^2 - pi/3*(Rv - h_v(vv,Rv,tv1,tv2)/2)^3*(2 + cos(tv2))*(1 - cos(tv2))^2

    K = 100

    Rc1 = R0
    Rc2 = R0
    tc1 = pi/3.
    tc2 = pi/3.
    Rv = R0
    tv1 = t_v(Rv,Rc1,tc1)
    tv2 = t_v(Rv,Rc2,tc2)
    hc1 = h_c(Rc1,tc1)
    hc2 = h_c(Rc2,tc2)
    hv = h_v(v_v,Rv,tv1,tv2)

    t = 1
    error_counter = 0.0

    while t <= T
        hc1 = h_c(Rc1,tc1)
        hc2 = h_c(Rc2,tc2)
        tv1 = t_v(Rv,Rc1,tc1)
        tv2 = t_v(Rv,Rc2,tc2)
        hv = h_v(v_v,Rv,tv1,tv2)
        E_i = Energy_tot(v_v,ga1,gb1,ga2,gb2,Rc1,Rc2,tc1,tc2,Rv,tv1,tv2) + K*(Organoid_volume(v_v,Rc1,Rc2,tc1,tc2,Rv,tv1,tv2) - V_lumen)^2 

        ## MC step
        Rc1_new = Rc1 + dR * randn()
        Rc2_new = Rc2 + dR * randn()
        Rv_new = Rv + dR * randn()
        tc1_new = tc1 + dtheta * randn()
        tc2_new = tc2 + dtheta * randn()
        tv1_new = tv1
        tv2_new = tv2
        flag_accept = false
        try
            tv1_new = t_v(Rv_new,Rc1_new,tc1_new)
            tv2_new = t_v(Rv_new,Rc2_new,tc2_new)
            flag_accept = true
        catch 
            error_counter += 1.0
            tv1_new = tv1
            tv2_new = tv2 
            flag_accept = false
        end
        E_f = Energy_tot(v_v,ga1,gb1,ga2,gb2,Rc1_new,Rc2_new,tc1_new,tc2_new,Rv_new,tv1_new,tv2_new) + K*(Organoid_volume(v_v,Rc1_new,Rc2_new,tc1_new,tc2_new,Rv_new,tv1_new,tv2_new) - V_lumen)^2

        if flag_accept && ((E_f < E_i) || rand() < exp(-(E_f - E_i)/temp))
            Rc1 = Rc1_new
            Rc2 = Rc2_new
            Rv = Rv_new
            tc1 = tc1_new
            tc2 = tc2_new
        else 
            Rc1_new = Rc1
            Rc2_new = Rc2
            Rv_new = Rv
            tc1_new = tc1 
            tc2_new = tc2
        end
        t = t + 1
    end

    # Compute the branching index from each crypt's neck-to-length geometry.
    neck1 = 2*Rc1*sin(tc1) + hc1
    length1 = Rc1*(1 + cos(tc1))
    neck2 = 2*Rc2*sin(tc2) + hc2
    length2 = Rc2*(1 + cos(tc2))

    BI1 = 5 - neck1/length1
    BI2 = 5 - neck2/length2

    avg_BI = (BI1+BI2)/2
    var_BI = abs(BI1 - BI2)
    println("Avg BI = ",avg_BI)
    println("BI var = ",var_BI)
    
    villus_color = RGB(217/255,132/255,129/255)
    c1_color = dg_grad(dg1)
    c2_color = dg_grad(dg2)
    sw_color = swell_grad(swelling)
    
    lwv = 1*hv
    lwc1 = 1*hc1
    lwc2 = 1*hc2
    tp = LinRange(tv1,pi-tv2,100)
    fig_sketch = plot(aspect_ratio=:equal, xlim=[-2*Rv, 2*Rv], ylim=[-2*Rv, 2*Rv])
    
    plot!(fig_sketch,Rv.*cos.(tp),Rv.*sin.(tp), linewidth = lwv, linecolor = villus_color, fillcolor=sw_color, fill=true)
    tm = LinRange(pi+tv2,2*pi-tv1,100)
    plot!(fig_sketch,Rv.*cos.(tm),Rv.*sin.(tm), linewidth = lwv, linecolor = villus_color, fillcolor=sw_color, fill=true)

    tn = LinRange(-(pi - tc1),pi-tc1,100)
    xO1 = Rv*cos(tv1) + Rc1*cos(tc1)
    xy = [Rc1.*cos.(tn) .+ xO1, Rc1.*sin.(tn)]
    plot!(fig_sketch,xy[1,:],xy[2,:], linewidth = lwc1, linecolor = c1_color, fillcolor=sw_color, fill=true)

    xO2 = -Rv*cos(tv2) - Rc2*cos(tc2)
    to = LinRange(tc2,2*pi-tc2,100)
    xy2 = [Rc2.*cos.(to) .+ xO2, Rc2.*sin.(to)]
    plot!(fig_sketch,xy2[1,:],xy2[2,:], linewidth = lwc2, linecolor = c2_color, fillcolor=sw_color, fill=true)
    plot!(fig_sketch,legend=false,grid=false,showaxis = false)

    dg_range = range(0.2, 0.8, length=10)
    color_array = [dg_grad(y) for y in dg_range]
    custom_cmap = cgrad(color_array)
    x_values = range(0.2, 0.8, length=10)
    y_values = [(1 + y/2)/(1 - y/2) for y in dg_range]
    z_values = repeat(dg_range, 1, length(x_values))

    p2 = heatmap(x_values,y_values,z_values,
                c=custom_cmap,
                colorbar=false,
                ylabel="Myosin apical/basal",
                xticks=false,
                
                framestyle=:box)


    # Show the swelling color scale beside the organoid sketch.
    sw_range = range(0., 1.0, length=10)
    color_array = [swell_grad(1-y) for y in sw_range]
    custom_cmap = cgrad(color_array)
    x_values = [y for y in sw_range]
    v1 = 4*pi/3*Rm^3/1e4
    v2 = 4*pi/3*RM^3/1e4
    y_values = range(v1, v2, length=10)
    z_values = repeat(sw_range, 1, length(x_values))

    psw = heatmap(x_values,y_values,z_values,
                c=custom_cmap,
                colorbar=false,
                ylabel="Lumen volume (10^4 μm^3)",
                xticks=false,
                
                framestyle=:box)


    l = @layout [a{0.8w} b{0.1w} c{0.1w}]
    fig_sketch = plot(fig_sketch, p2, psw, layout=l)

    display(fig_sketch)
    sw = @sprintf("%.2f", swelling)
    myo1 = @sprintf("%.2f", dg1)
    myo2 = @sprintf("%.2f", dg2)
    figpath = "sw"*sw*"_1myo"*myo1*"_2myo"*myo2*".pdf"
    savefig(fig_sketch, figpath)
    println("Number of errors = ", error_counter)
    println("hc1 = ", hc1, "; hc2 = ", hc2)
end

cd(@__DIR__)

# Physical parameters for the default simulation. Change these values to explore
# the effects of swelling and myosin asymmetry on crypt convergence.

# Lumen radius
R_t0 = 33.0 # Lumen radius before swelling and morphogenesis, in um
R0 = R_t0/1.5 # Lumen radius being simulated (lost volume compared to Rt0 is redistributed to villus cells), in um.
# 1.5 is the default value for swelling, 1.3 is no swelling (SGLT1i).

# Myosin levels of the two crypts
dg1 = 0.8 # default: 0.8, 'most mature crypt' in the organoid
dg2 = 0.4 # least mature crypt in the organoid, default: 0.4 if no catchup mechanism (cPLA2i)
# Apical myosin is ga = 1 + dg/2, basal myosin is gb = 1 - dg/2.

main(R_t0,R0,dg1,dg2)

"""
    CRL_timecourse.jl

Code archive associated with Capolupo et al., "Tissue swelling and local mechanosensation drive 
crypt convergence in intestinal development."



Time-course model of an organoid with two crypts and a central
villus. Running this file creates PDF frames and BI time-course data in a
timestamped `CRL/` directory next to the script.

Mechanosensation can be turned on or off with the `cpla2` boolean in the `main` function. 
The `dg10` and `dg20` parameters control the initial apical myosin asymmetry between the two crypts. 
The time course is run for a fixed number of swelling steps (given by N_step), and the myosin asymmetry is 
updated at each step based on the current crypt neck width. Swelling is controlled by the `sw_v` 
array, which sets the evolution of the lumen volume over time. The simulation uses a Metropolis 
Monte Carlo relaxation to find the equilibrium geometry (T equilibration steps) at each swelling step.

Requires the Julia packages `Images`, `Plots`, and `DelimitedFiles`. The
optional `create_video_from_pdfs` helper also requires `pdftoppm` and `ffmpeg`
(or ImageMagick's `convert` as a PDF-conversion fallback).

The simulation uses random proposals; call `Random.seed!(seed)` before `main`
when a repeatable run is needed.


"""

using Random
using Printf
using Images
using Dates
using DelimitedFiles
using Plots

"""Create an MP4 from numerically ordered PDF frames in `foldername`."""
function create_video_from_pdfs(foldername, output_file="output.mp4"; fps=10)
    # Get all PDF files and sort them numerically
    pdf_files = filter(f -> endswith(f, ".pdf"), readdir(foldername))
    
    sort!(pdf_files, by = f -> begin
        m = match(r"(\d+)", f)
        m !== nothing ? parse(Int, m.captures[1]) : 0
    end)
    
    # Create temporary directory for PNGs
    temp_dir = mktempdir()
    
    println("Converting PDFs to PNGs...")
    for (i, pdf) in enumerate(pdf_files)
        pdf_path = joinpath(foldername, pdf)
        png_path = joinpath(temp_dir, "frame_$(lpad(i, 5, '0')).png")
        
        # Convert PDF to PNG (requires pdftoppm or imagemagick)
        try
            run(`pdftoppm -png -singlefile -r 600 $pdf_path $(png_path[1:end-4])`)
        catch
            # Fallback to imagemagick if pdftoppm not available
            run(`convert -density 150 $pdf_path $png_path`)
        end
    end
    
    println("Creating video...")
    # Now create video from PNGs
    run(`ffmpeg -framerate $fps -i $(temp_dir)/frame_%05d.png -pix_fmt yuv420p $foldername/$output_file`)
    
    # Cleanup
    rm(temp_dir, recursive=true)
    
    println("Video created: $output_file")
end


"""Map a branching-index value to the model's blue-to-orange color scale."""
function BI_grad(BI)
    BImax = 4.0
    BImin = 2.0
    xBI = max(min((BI - BImin)/(BImax - BImin),1),0)
    hiBI_color = RGB(249/255,179/255,54/255)
    loBI_color = RGB(51/255,139/255,204/255)
    BI_color = hiBI_color*xBI + loBI_color*(1-xBI)
    return BI_color
end

"""Map apical myosin asymmetry to the blue-to-orange color scale."""
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

function swell_grad(sw)
    hi_color = RGBA(151/255,177/255,221/255,0.)
    lo_color = RGBA(151/255,177/255,221/255,1.0)

    hi_color = RGBA(233/255,71/255,52/255,0.5)
    lo_color = RGBA(233/255,71/255,52/255,0.)
    color = hi_color*sw + lo_color*(1-sw)
    return color
end


"""
    minimizer(T, dg1, dg2, swelling, Rc1, Rc2, Rv, tc1, tc2, filename)

Run a Metropolis Monte Carlo relaxation for `T` iterations, then save a sketch
of the averaged geometry to `filename`. Returns both branching indices and
the averaged geometric state.
"""
function minimizer(T,dg1,dg2,swelling,Rc1,Rc2,Rv,tc1,tc2,filename)
    temp = .1
    dR = 0.1
    dtheta = pi/100

    ga1 = 1.0 + dg1/2
    gb1 = 1.0 - dg1/2

    ga2 = 1.0 + dg2/2
    gb2 = 1.0 - dg2/2

    #new values
    gav = 0.8
    gbv = 1.6

    gl = 1.0

    v_c = 700.0  #crypt cell vol
    v_v = 1.0e3 # villus cell vol
    N_v = 120.0 #villus cell number
    N_c = 30.0 # crypt cell number
    R_t0 = 33.0 # lumen radius 
    V_lumen_t0 = 4/3*pi*R_t0^3

    RM = 28
    Rm = 22
    R0 = Rm + (1 - swelling)*(RM - Rm)

    println("New model with parameters:")
    println("Swelling = ", swelling)
    println("Myosin crypt 1 = ", dg1, "; Myosin crypt 2 = ", dg2)
    V_lumen = 4/3*pi*R0^3

    v_v = v_v + (V_lumen_t0 - V_lumen)/N_v

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

    K = 10

    tv1 = t_v(Rv,Rc1,tc1)
    tv2 = t_v(Rv,Rc2,tc2)
    hc1 = h_c(Rc1,tc1)
    hc2 = h_c(Rc2,tc2)
    hv = h_v(v_v,Rv,tv1,tv2)

    Rc1_new = Rc1
    Rc2_new = Rc2
    tc1_new = tc1 
    tc2_new = tc2
    hc1_new = hc1 
    hc2_new = hc2 
    hv_new = hv
    Rv_new = Rv 
    tv1_new = tv1 
    tv2_new = tv2  

    t = 1
    energies = zeros(T)
    #energies[t] = E
    flag_accept = true
    error_counter = 0.0

    Rc1_r = 0.0
    Rc2_r = 0.0
    tc1_r = 0.0
    tc2_r = 0.0
    Rv_r = 0.0
    hc1_r = 0.0
    hc2_r = 0.0
    counter = 0.0

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
        hc1_new = h_c(Rc1_new,tc1_new)
        hc2_new = h_c(Rc2_new,tc2_new)
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
        hv_new = h_v(v_v,Rv_new,tv1_new,tv2_new)
        E_f = Energy_tot(v_v,ga1,gb1,ga2,gb2,Rc1_new,Rc2_new,tc1_new,tc2_new,Rv_new,tv1_new,tv2_new) + K*(Organoid_volume(v_v,Rc1_new,Rc2_new,tc1_new,tc2_new,Rv_new,tv1_new,tv2_new) - V_lumen)^2

        if flag_accept && ((E_f < E_i) || rand() < exp(-(E_f - E_i)/temp))
            Rc1 = Rc1_new
            Rc2 = Rc2_new
            Rv = Rv_new
            tc1 = tc1_new
            tc2 = tc2_new
            energies[t] = E_f
        else 
            Rc1_new = Rc1
            Rc2_new = Rc2
            Rv_new = Rv
            tc1_new = tc1 
            tc2_new = tc2
            energies[t] = E_i
        end

        if t > T/2
            Rc1_r += Rc1
            Rc2_r += Rc2
            tc1_r += tc1
            tc2_r += tc2
            Rv_r += Rv
            hc1_r += hc1
            hc2_r += hc2
            counter += 1.0
        end

        t = t + 1
    end

    #computes average values
    Rc1 = Rc1_r/counter
    Rc2 = Rc2_r/counter
    tc1 = tc1_r/counter
    tc2 = tc2_r/counter
    Rv = Rv_r/counter
    hc1 = hc1_r/counter
    hc2 = hc2_r/counter
    tv1 = t_v(Rv,Rc1,tc1)
    tv2 = t_v(Rv,Rc2,tc2)
    hv = h_v(v_v,Rv,tv1,tv2)

    neck1 = 2*Rc1*sin(tc1) + hc1
    length1 = Rc1*(1 + cos(tc1))
    neck2 = 2*Rc2*sin(tc2) + hc2
    length2 = Rc2*(1 + cos(tc2))

    BI1 = 5 - neck1/length1
    BI2 = 5 - neck2/length2
    
    villus_color = RGB(217/255,132/255,129/255)
    c1_color = dg_grad(dg1)
    c2_color = dg_grad(dg2)
    sw_color = swell_grad(swelling)
    
    lwv = 1*hv
    lwc1 = 1*hc1
    lwc2 = 1*hc2
    tp = LinRange(tv1,pi-tv2,100)
    fig_sketch = plot(aspect_ratio=:equal, xlim=[-2*R_t0, 2*R_t0], ylim=[-2*R_t0, 2*R_t0])
    
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
    plot!(fig_sketch,legend=false,grid=false)#,showaxis = false)

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


    sw_range = range(0., 1.0, length=10)
    color_array = [swell_grad(1-y) for y in sw_range]
    custom_cmap = cgrad(color_array)
    x_values = [y for y in sw_range]
    v1 = 4*pi/3*Rm^3/1e4
    v2 = 4*pi/3*RM^3/1e4
    y_values = range(v1, v2, length=10) #4.0./3.0.*pi.*(Rm^3.0.*sw_range .+ (1 .- sw_range).*(RM^3 - Rm^3))
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
    savefig(fig_sketch, filename)
    println("Number of errors = ", error_counter)
    println("Volume error = ",(Organoid_volume(v_v,Rc1,Rc2,tc1,tc2,Rv,tv1,tv2) - V_lumen)/V_lumen," %")
    println("Lumen radius = ", Rv-hv/2)
    return BI1, BI2, Rc1, Rc2, tc1, tc2, hc1, hc2, Rv
    
end

"""
    main(dg10, dg20, cpla2)

Run a swelling/myosin time course for the two crypts. `dg10` and `dg20` are
initial myosin asymmetries; `cpla2=true` selects the control condition and
`cpla2=false` selects cPLA2 inhibition.
Returns the two branching-index time series.
"""
function main(dg10,dg20,cpla2,bigfolder="CRL")
    mkpath(bigfolder)
    myo1 = @sprintf("%.2f", dg10)
    myo2 = @sprintf("%.2f", dg20)
    if cpla2
        foldername = "./"*bigfolder*"/CRL_"*Dates.format(now(), "yy-mm-dd_HH:MM")*"_"*myo1*"_"*myo2
    else
        foldername = "./"*bigfolder*"/cPLA2i_"*Dates.format(now(), "yy-mm-dd_HH:MM")*"_"*myo1*"_"*myo2
    end
    mkdir(foldername)
    R0 = 33
    Rc1 = R0/1.3
    Rc2 = R0/1.3
    tc1 = pi/3.
    tc2 = pi/3.
    Rv = R0/1.3
    dg11 = 0.8-dg10
    dg1 = dg10
    dg21 = 0.8 - dg20
    dg2 = dg20
    w0 = 8.0 #8.0
    wmin = 6.0
    alpha = 2.5
    v_c = 700

    Nstep = 100
    T = 2000000
    sw_v = range(0.3, 1.0, length=Nstep)

    sw_v = cat(sw_v,ones(Nstep),dims=1)

    swelling = sw_v[1]

    filename = "frame_"*"00000.pdf"
    filename = joinpath(foldername,filename)
    BI1, BI2, Rc1, Rc2, tc1, tc2, hc1, hc2, Rv = minimizer(T,dg1,dg2,swelling,Rc1,Rc2,Rv,tc1,tc2,filename)

    BI1_v = zeros(Nstep)
    BI2_v = BI1_v.*0.0 
    tau = 10
    noise_g = 0.02
    if cpla2
        flag = 1.0
    else
        flag = 0.0
    end
    for k in 1:Nstep
        kstr = lpad(k,5,'0')
        filename = "frame_"*kstr*".pdf"
        filename = joinpath(foldername,filename)
        sw = sw_v[k]
        w2 = sqrt(v_c/hc2)
        w1 = sqrt(v_c/hc1)
        dg1_target = dg10 + flag*dg11*max(tanh(alpha*(w0 - w1)/(w0 - wmin)),0)
        dg2_target = dg20 + flag*dg21*max(tanh(alpha*(w0 - w2)/(w0 - wmin)),0)
        dg1 = dg1 + (dg1_target - dg1)/tau + noise_g*randn()
        dg2 = dg2 + (dg2_target - dg2)/tau + noise_g*randn()
        BI1, BI2, Rc1, Rc2, tc1, tc2, hc1, hc2, Rv = minimizer(T,dg1,dg2,sw,Rc1,Rc2,Rv,tc1,tc2,filename)
        BI1_v[k] = BI1
        BI2_v[k] = BI2
    end
    BIfig = plot(BI1_v,BI2_v)

    display(BIfig)
    writedlm(foldername*"/BI1.txt",BI1_v)
    writedlm(foldername*"/BI2.txt",BI2_v)

    return BI1_v, BI2_v
end


cd(@__DIR__)
dg01_v = [0.5]
dg02_v = [0.5]
cpla2 = true
bigfolder = "CRL"

for dg10 in dg01_v
    for dg20 in dg02_v
        main(dg10, dg20, cpla2, bigfolder)
    end
end



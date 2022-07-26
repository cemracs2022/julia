ENV["GKSwstype"]="100" #src

# ## GPU Computing

# https://github.com/JuliaGPU

using Plots, BenchmarkTools, FFTW, LinearAlgebra

# ### Advection equation for a rotation in two dimensional domain
#md #
#md # ```math
#md #  \\frac{d f}{dt} +  (y \\frac{d f}{dx} - x \\frac{d f}{dy}) = 0
#md # ```
#md #
#md # ``x \in [-π, π], y \in [-π, π] `` and  `` t \in [0, 200π] ``

#md # ---

#md # # Composite type for mesh definition

struct Mesh
    
    nx   :: Int64
    ny   :: Int64
    x    :: Vector{Float64}
    y    :: Vector{Float64}
    kx   :: Vector{Float64}
    ky   :: Vector{Float64}
    
    function Mesh( xmin, xmax, nx, ymin, ymax, ny)
        ## periodic boundary condition, we remove the end point.
        x = LinRange(xmin, xmax, nx+1)[1:end-1]
        y = LinRange(ymin, ymax, ny+1)[1:end-1]
        kx  = 2π ./ (xmax-xmin) .* [0:nx÷2-1;nx÷2-nx:-1]
        ky  = 2π ./ (ymax-ymin) .* [0:ny÷2-1;ny÷2-ny:-1]
        new( nx, ny, x, y, kx, ky)
    end
end

#-
#md # ---

#md # # Exact computation of solution

function exact!(f, time, mesh :: Mesh; shift=1.0)
   
    for (i, x) in enumerate(mesh.x), (j, y) in enumerate(mesh.y)

        xn = cos(time)*x - sin(time)*y - shift
        yn = sin(time)*x + cos(time)*y - shift

        f[i,j] = exp(-(xn^2+yn^2)/0.1)

    end

end

#-
#md # ---

"""
    exact( time, mesh; shift=1.0)

Computes the solution of the rotation problem

"""
function exact( time, mesh :: Mesh; shift=1.0)
   
    f = zeros(Float64, (mesh.nx, mesh.ny))
    exact!(f, time, mesh, shift = shift)
    return f

end

#-
#md # --

@doc exact

#-
#md # ---

#md # # Create animation to show what we compute

using Plots

function animation( tf, nt)
    
    mesh = Mesh( -π, π, 64, -π, π, 64)
    dt = tf / nt
    t = 0
    f = zeros(Float64, (mesh.nx, mesh.ny))

    anim = @animate for n=1:nt
       
       exact!(f, t, mesh)
       t += dt
       p = contour(mesh.x, mesh.y, f, axis=[], framestyle=:none)
       plot!(p[1]; clims=(0.,1.), aspect_ratio=:equal, colorbar=false, show=false)
       plot!(√2 .* cos.(-π:0.1:π+0.1), 
             √2 .* sin.(-π:0.1:π+0.1), label="", show=false)
       xlims!(-π,π)
       ylims!(-π,π)
        
    end

    anim
    
end
#-

#md # ---

anim = animation( 2π, 100)
#md gif(anim, "rotation2d.gif", fps = 20);
#md nothing # hide

#md # ![](rotation2d.gif)

#md # ---
#-

function rotation_on_cpu( mesh :: Mesh, nt :: Int64, tf :: Float64) 
    
    dt = tf / nt
    
    f   = zeros(ComplexF64,(mesh.nx,mesh.ny))
    exact!( f, 0.0, mesh )
    
    exky = exp.( 1im*tan(dt/2) .* mesh.x  .* mesh.ky')
    ekxy = exp.(-1im*sin(dt)   .* mesh.y' .* mesh.kx )

    p_x, pinv_x = plan_fft!(f,  [1]), plan_ifft!(f, [1])
    p_y, pinv_y = plan_fft!(f,  [2]), plan_ifft!(f, [2])  
        
    for n = 1:nt
        p_y * f
        f .*= exky 
        pinv_y * f
            
        p_x * f
        f .*= ekxy 
        pinv_x * f
            
        p_y * f
        f .*= exky 
        pinv_y * f
    end

    real(f)
    
end

#md # ---
#-

#md # Run the simulation and test error.

mesh = Mesh( -π, π, 1024, -π, π, 1024)

nt, tf = 100, 20.

rotation_on_cpu(mesh, 1, 0.1) # trigger building

etime = @time norm( rotation_on_cpu(mesh, nt, tf) .- exact( tf, mesh))

println(etime)
#-

#md # ---

#md # # Test if GPU packages are installed

using Suppressor #hide
@suppress begin #hide
using CUDA
end #hide

GPU_ENABLED = CUDA.functional()

if GPU_ENABLED

    using CUDA.CUFFT
    
    println(CUDA.name(CuDevice(0)))

end
#-

#md # **JuliaGPU** GPU Computing in Julia
#md # 
#md # https://juliagpu.org/
#md # 

#md # ---
    
if GPU_ENABLED

    function rotation_on_gpu( mesh :: Mesh, nt :: Int64, tf :: Float64)
        
        dt  = tf / nt
        f   = CUDA.zeros(ComplexF64,(mesh.nx, mesh.ny))
        exact!( f, 0.0, mesh)
        
        p_x, pinv_x = plan_fft!(f,  [1]), plan_ifft!(f, [1])
        p_y, pinv_y = plan_fft!(f,  [2]), plan_ifft!(f, [2])  
        
        exky = cu(exp.( 1im*tan(dt/2) .* mesh.x  .* mesh.ky'))
        ekxy = cu(exp.(-1im*sin(dt)   .* mesh.y' .* mesh.kx ))
        
        for n = 1:nt
            p_y * f
            f .*= exky 
            pinv_y * f
            
            p_x * f
            f .*= ekxy 
            pinv_x * f
            
            p_y * f
            f .*= exky 
            pinv_y * f
        end
        
        real(collect(f)) # Transfer f from GPU to CPU
        
    end

end
#-

#md # ---

if GPU_ENABLED

    nt, tf = 100, 20.

    rotation_on_gpu(mesh, 1, 0.1)

    etime = @time norm( rotation_on_gpu(mesh, nt, tf) .- exact( tf, mesh))

    println(etime)

end

#md # ---

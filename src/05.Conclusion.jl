ENV["GKSwstype"]="100" #src

# ### Optimizing Julia code is often done at the expense of transparency

using Random, LinearAlgebra, BenchmarkTools

A = rand(1024, 256); B = rand(256, 1024); C = rand(1024, 1024)

function test1(A, B, C)
    C = C - A * B
    return C
end

@btime test1(A, B, C); #C, A and B are matrices. 
#md nothing # hide

# --

function test2!(C, A, B)
    C .-= A * B
end

@btime test2!(C, A, B); #C, A and B are matrices. 
#md nothing # hide

# ---

function test_opt!(C, A, B)
    BLAS.gemm!('N','N', -1., A, B, 1., C)
    return C
end
@btime test_opt!(C, A, B) # avoids taking two unnecessary copies of the matrix C.
#md nothing # hide

# --

C = rand(1024, 1024)
test2!(C, A, B)
all(test1(A, B, C) .== C)

# --

C = rand(1024, 1024)
test_opt!(C, A, B)
all(test1(A, B, C) .== C)

#md # ---

# ### Derivative computation with FFT

using FFTW

xmin, xmax, nx = 0, 4π, 1024
ymin, ymax, ny = 0, 4π, 1024

x = LinRange(xmin, xmax, nx+1)[1:end-1]
y = LinRange(ymin, ymax, ny+1)[1:end-1]

ky  = 2π ./ (ymax-ymin) .* [0:ny÷2-1;ny÷2-ny:-1]
exky = exp.( 1im .* ky' .* x)

function du_dy( u, exky )
    ifft(exky .* fft(u, 2), 2)
end

u = sin.(x) .* cos.(y') # f is a 2d array created by broadcasting

@btime du_dy(u, exky)
#md nothing # hide

#md # ---

# ### Memory alignement, and inplace computation.

u  = zeros(ComplexF64, (nx,ny)) 
uᵗ = zeros(ComplexF64, reverse(size(u)))
ûᵗ = zeros(ComplexF64, reverse(size(u)))

u .= sin.(x) .* cos.(y')

fft_plan = plan_fft(uᵗ, 1, flags=FFTW.PATIENT)

function du_dy!( u, uᵗ, ûᵗ, exky )
    transpose!(uᵗ,u)
    mul!(ûᵗ,  fft_plan, uᵗ)
    ûᵗ .= ûᵗ .* exky
    ldiv!(uᵗ, fft_plan, ûᵗ)
    transpose!(u, uᵗ)
end

@btime du_dy!(u, uᵗ, ûᵗ, exky )
#md nothing # hide

#md # ---

#md # # Why use Julia language!
#md #
#md # - **You develop in the same language in which you optimize.**
#md # - Packaging system is very efficient (3173 registered packages)
#md # - PyPi (198,360 projects) R (14993 packages)
#md # - It is very easy to create a package (easier than R and Python)
#md # - It is very easy to use GPU device.
#md # - Nice interface for Linear Algebra and Differential Equations
#md # - Easy access to BLAS and LAPACK
#md # - Julia talks to all major Languages - mostly without overhead!

#md # ---

#md # # What's bad
#md #
#md # - It is still hard to build shared library or executable from Julia code.
#md # - Compilation times can be unbearable.
#md # - Plotting takes time (20 seconds for the first plot)
#md # - OpenMP is better than the Julia multithreading library but it is progressing.
#md # - For parallelization, The Julia community seems to prefer the distributed processing approach. 
#md # - Does not work well with vectorized code, you need to do a lot of inplace computation to avoid memory allocations and use explicit views to avoid copy.
#md # - Julia website proclaims that it is faster than Fortran but this is not true. But it is very close  and it is progressing.
#md 
#md # [What's Bad About Julia by Jeff Bezanson](https://www.youtube.com/watch?v=TPuJsgyu87U)

#md # ---

#md # ## Python-Julia benchmarks by Thierry Dumont
#md #
#md # https://github.com/Thierry-Dumont/BenchmarksPythonJuliaAndCo/wiki
#md 
#md # # Julia is a language made for Science.
#md #
#md #  * JuliaDiffEq – Differential equation solving and analysis.
#md #  * JuliaDiff – Differentiation tools.
#md #  * JuliaGeometry – Computational Geometry.
#md #  * JuliaGraphs – Graph Theory and Implementation.
#md #  * JuliaIntervals - Rigorous numerics with interval arithmetic & applications.
#md #  * JuliaMath – Mathematics made easy in Julia.
#md #  * JuliaOpt – Optimization.
#md #  * JuliaPolyhedra – Polyhedral computation.
#md #  * JuliaSparse – Sparse matrix solvers.
#md #  * JuliaStats – Statistics and Machine Learning.
#md #  * JuliaPlots - powerful convenience for visualization.
#md #  * JuliaGPU - GPU Computing for Julia.
#md #  * FluxML - The Elegant Machine Learning Stack.
#md #

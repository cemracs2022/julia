ENV["GKSwstype"]="100" #src

#md # # Who am I ?
#md #
#md #  - My name is *Pierre Navaro*
#md #
#md #  - **Fortran 77 + PVM** : during my PhD 1998-2002 (Université du Havre)
#md #
#md #  - **Fortran 90-2003 + OpenMP-MPI** : Engineer in Strasbourg (2003-2015) at IRMA
#md #
#md #  - **Numpy + Cython, R + Rcpp** : Engineer in Rennes (2015-now) at IRMAR
#md #
#md #  - **Julia v1.0** since July 2018
#md #
#md #  ## Instructions to open the notebooks
#md #
#md #  https://github.com/cemracs2022/julia
#md #

#md # ---

#md #
#md # # Why Julia?
#md #
#md # - Started in 2009 and first version was released in 2012.
#md # - High-level languages like Python and R let one explore and experiment rapidly, but can run slow.
#md # - Low-level languages like Fortran/C++ tend to take longer to develop, but run fast.
#md # - This is sometimes called the "two language problem" and is something the Julia developers set out to eliminate.
#md # - Julia's promise is to provide a "best of both worlds" experience for programmers who need to develop novel algorithms and bring them into production environments with minimal effort.
#md #

#md # **Julia: A Fresh Approach to Numerical Computing**
#md # 
#md # *Jeff Bezanson, Alan Edelman, Stefan Karpinski, Viral B. Shah*
#md # 
#md # SIAM Rev., 59(1), 65–98. (34 pages) 2012

#md # ---
#
#md # # First example
#
#md # Implement your own numerical methods to solve
#
#md # $$
#md # y'(t) = 1 - y(t) 
#md # $$
#md # $$
#md # t \in [0,5]
#md # $$
#md # $$
#md # y(0) = 0.
#md # $$

#md # ---

#md # ## Explicit Euler

euler(f, t, y, h) = t + h, y + h * f(t, y)

#md # ## Runge-Kutta 2nd order

rk2(f, t, y, h) = begin
    ỹ = y + h / 2 * f(t, y)
    t + h, y + h * f(t + h / 2, ỹ)
end

#md # ---

#md # ## Runge-Kutta 4th order

function rk4(f, t, y, dt)

    y₁ = dt * f(t, y)
    y₂ = dt * f(t + dt / 2, y + y₁ / 2)
    y₃ = dt * f(t + dt / 2, y + y₂ / 2)
    y₄ = dt * f(t + dt, y + y₃)

    t + dt, y + (y₁ + 2y₂ + 2y₃ + y₄) / 6

end

#md # ---

#md # ## Solve function

function solve(f, method, t₀, y₀, h, nsteps)

    t = zeros(typeof(t₀), nsteps)
    y = zeros(typeof(y₀), nsteps)

    t[1] = t₀
    y[1] = y₀

    for i = 2:nsteps
        t[i], y[i] = method(f, t[i-1], y[i-1], h)
    end

    t, y

end

#md # ---

#md # ## Plot solutions

using Plots

nsteps, tfinal = 10, 5.0
t₀, x₀ = 0.0, 0.0
dt = tfinal / (nsteps - 1)
f(t, x) = 1 - x

t, y_euler = solve(f, euler, t₀, x₀, dt, nsteps)

t, y_rk2 = solve(f, rk2, t₀, x₀, dt, nsteps)

t, y_rk4 = solve(f, rk4, t₀, x₀, dt, nsteps)

#md # ---

plot(t, y_euler; marker = :o, label = "Euler")
plot!(t, y_rk2; marker = :d, label = "RK2")
plot!(t, y_rk4; marker = :p, label = "RK4")
plot!(t -> 1 - exp(-t); line = 3, label = "true solution", legend = :right)
savefig("solve1.png") #hide
# ![solve1](solve1.png)

#md # ---

using Measurements

t₀ = 0.0
x₀ = 0.0 ± 0.2

t, y_rk4 = solve(f, rk4, t₀, x₀, dt, nsteps)

plot(t, y_rk4; marker = :circle, label = "RK4", legend = :right)
savefig("solve2.png") #hide
# ![solve2](solve2.png)

#md # ---

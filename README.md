# Introduction to Julia language

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cemracs2022/julia/HEAD)

CEMRACS 2022 July, 28th Marseille France.

Link to [slides](https://cemracs2022.github.io/julia).

To install Julia go to http://julialang.org/downloads/.

- On MacOS you can use [brew](https://formulae.brew.sh/formula/julia)
- On Linux use [Command line installer of the Julia Language](https://github.com/abelsiqueira/jill)

To open the notebooks and run them locally:

```bash
git clone https://github.com/cemracs2022/julia
cd julia
julia --project
```

```julia
julia> using Pkg
julia> Pkg.instantiate()
julia> include("generate_nb.jl")
julia> using IJulia
julia> notebook(dir=joinpath(pwd(),"notebooks"))
[ Info: running ...
```

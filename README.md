# Introduction to Julia language

CEMRACS 2022 July, 28th Marseille France.

Link to [slides](https://cemracs2022.github.io/julia).

To open the notebooks run them locally:

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

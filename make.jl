ENV["GKSwstype"]="100"

using  Literate
using  Plots
import Remark

files =  filter( f -> startswith(f, "0"), readdir("src")) |> collect

slides_path = joinpath("src")
mkpath(slides_path)
run(pipeline(`cat src/$files`; stdout=joinpath(slides_path, "index.jl" )))
s = Remark.slideshow(@__DIR__, title="Introduction to Julia language")

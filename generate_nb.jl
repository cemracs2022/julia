using  Literate
using  Plots

files =  filter( f -> startswith(f, "0"), readdir("src")) |> collect

for file in files
    Literate.notebook("src/$file", "notebooks",  execute=false)
end

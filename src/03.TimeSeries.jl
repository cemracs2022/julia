# # Time Series 

struct TimeSeries{T,N}

   nt  :: Int
   nv  :: Int
   t   :: Vector{T}
   u   :: Vector{Array{T, 1}}

   function TimeSeries{T,N}( nt :: Int) where {T,N}
 
       t  = zeros(T, nt)
       u  = [zeros(T, N) for i in 1:nt]
       nv = N

       new( nt, nv, t, u)

   end
    
end
#nb # -

#md # ---

# ## Overload `Base.length` function

#md # --

import Base:length

length(ts :: TimeSeries) = ts.nt

nt, nv = 100, 2
ts = TimeSeries{Float64, nv}(nt);

@show length(ts) == nt

# Generate data

ts.t[1] = 0.0
ts.u[1] = [0.0, 1.0]

dt = 0.01
for i in 2:nt

   ts.t[i] = ts.t[i-1] + dt
   ts.u[i][1] = sin(ts.t[i])
   ts.u[i][2] = cos(ts.t[i])

end

#md # ---

using Plots

plot(ts.t, vcat(ts.u'...))
#md savefig("plot1.svg"); nothing #hide
#md # ![](plot1.svg)

#md # ---

plot(ts.t, [getindex.(ts.u, i) for i in 1:nv])
#md savefig("plot2.svg"); nothing #hide
#md # ![](plot2.svg)

#md # ---

# ## Overload the `[]` operator
#
# we want `ts[i]` equal to `ts.u[:][i]` values

#md # --

import Base: getindex

#md # --

getindex( ts :: TimeSeries, i ) = getindex.(ts.u, i)

#md # ---

plot(ts[1], ts[2])
#md savefig("plot3.svg"); nothing #hide
#md # ![](plot3.svg)

#md # ---

# ## Overload the `+` operator to add noise

import Base:+

#md # --

function +(ts :: TimeSeries, ϵ ) 

    for n in 1:ts.nt, d in 1:ts.nv
       ts.u[n][d] += ϵ[n,d]
    end
    return ts

end
#nb # -

#md # ---

ts = ts + 0.1*randn((nt,nv));

#md # --

scatter(ts.t, [ts[1],ts[2]])
#md savefig("plot4.svg"); nothing #hide
#md # ![](plot4.svg)

# ---

# # Linear regression with obvious operation

using LinearAlgebra

X = hcat(ones(nt), ts.t, ts[1])
y = ts[2]

@show β = inv(X'X) * X'y

#md # ---

# # Version with QR factorisation

@show β = X \ y

# The `\` operator is the short-hand for

Q, R = qr(X)

@show β = inv(factorize(R))Q'y

#md # ---

# # Version with singular values decomposition

U, S, V = svd(X)

@show β = V * diagm(1 ./ S) * U' * y


#md # --

@show β = pinv(X, atol=1e-6) * y

#md # ---

# ## With GLM.jl
using GLM

fitted = lm(X, y)

# ---

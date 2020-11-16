# Nadia Lucas
# November 12, 2020
# PPHA 44320
# Computational PSET, Problem 3
# 
# running on Julia version 1.5.2
################################################################################
# uncomment to install packages used
# import Pkg; Pkg.add("Plots")
# import Pkg; Pkg.add("Random")
# import Pkg; Pkg.add("Distributions")

using Plots, Random, Distributions

# change for which directory to save figures in
figpath = "/Users/nadialucas/Dropbox/Second year/PPHA 44320/computational_pset/"

# part a
P_max = 80
N = 81
prices = 0:1:80
D = 3000000
X = 100000
delta = 1.0/1.05
profits = [X*i - D for i in prices]

# part b
Random.seed!(1234) # for consistent replication
d = Normal(0, 4)
cdf(d, 1)

# construct state transition matrix
cdf_list = [cdf(d, i+0.5) - cdf(d,i-0.5) for i in 0:P_max]
T = [ y==1 ? sum(cdf_list[abs(x-y)+1:N]) : y==N ? sum(cdf_list[abs(x-y)+1:N]) : cdf_list[abs(x-y)+1] for x in 1:N, y in 1:N]

# part c
global V = [0.0 for i = 1:N]
distance = 10.0
while distance > 1e-8
# iterate to get V
    V_old = V
    V_tomorrow = T*V
    V = [ max(profits[i], delta * V_tomorrow[i]) for i in 1:N]
    C = [ profits[i] > delta*V_tomorrow[i] ? 1 : 0 for i in 1:N]
    distance = norm(V-V_old)
end
# NOTE: C gives optimal action, when this flips to 1 is the trigger price
# That happens when the price corresponds to $41

# part d: plotting
x = 0:80
y = V
plot(x, y)
p = plot(x, y, 
    xlabel = "Price", 
    ylabel = "Value function", 
    linecolor = :dodgerblue, 
    fontfamily = "Courier New, monospace", 
    legend = false
)
png(p, string(figpath, "p3_value_function.png"))

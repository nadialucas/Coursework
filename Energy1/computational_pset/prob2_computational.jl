# Nadia Lucas
# November 12, 2020
# PPHA 44320
# Computational PSET, Problem 2
# 
# running on Julia version 1.5.2
################################################################################
# uncomment to install packages used
# import Pkg; Pkg.add("SparseArrays")
# import Pkg; Pkg.add("LinearAlgebra")
# import Pkg; Pkg.add("Plots")

using SparseArrays, LinearAlgebra, Plots

# change for which directory to save figures in
figpath = "/Users/nadialucas/Dropbox/Second year/PPHA 44320/computational_pset/"

# part a (the problem setup here is all the same as in 1)
S_tot = 1000
cost = 0
r = 0.05
delta = 1.0/1.05
N = 501
steps = Array(0:2:S_tot)
A = Array(0:2:S_tot)
nA = length(A)
sqrt_actions = Array(0: sqrt(S_tot)/500: sqrt(S_tot))
actions = [i^2 for i in sqrt_actions]
u1(y) = 2*(sqrt(y))
u2(y) = 5*y - .05*(y^2)
u1_prime(y) = 1.0/(sqrt(y))
u2_prime(y) = 5.0 - (0.1*y)
action_utility1 = zeros(nA)
action_utility2 = zeros(nA)
for i = 1:nA
    action_utility1[i] = u1(actions[i])
    action_utility2[i] = u2(actions[i])
end
numrows = N
numcols = nA
U1 = [ actions[y]<=steps[x] ? action_utility1[y] : -Inf for x in 1:numrows, y in 1:numcols]
U2 = [ actions[y]<=steps[x] ? action_utility2[y] : -Inf for x in 1:numrows, y in 1:numcols]

# part b
# initialize next periods state given each action (to be used in constructing T)
next_state = [ actions[y] <= steps[x] ? steps[x] - actions[y] : 0 for x in 1:numrows, y in 1:numcols]
T = fill(0.0, (N, N*nA))
for i = 1:N
    for j = 1:nA
        # linear interpolation
        lo = searchsortedlast(steps, next_state[i, j])
        hi = searchsortedfirst(steps, next_state[i, j])
        # and fill in T using next_state
        if lo == hi
            T[i, ((j-1)*nA + lo)] = 1
        else
            prob_lo = 1-(next_state[i, j] - steps[lo])/2
            prob_hi = 1-prob_lo
            T[i, ((j-1)*nA + lo)] = prob_lo
            T[i, ((j-1)*nA + hi)] = prob_hi
        end
    end
end
T = sparse(T)

# part c
# cycle through each utility function
for utility in 1:2
    if utility == 1
        U = U1
    else
        U = U2
    end
    # initialize values and corersponding optimal actions
    global V = [0.0 for i = 1:N]
    global C = [0.0 for i = 1:N]
    distance = 10.0
    while distance > 1e-8
    # given V, calculate Vnext (same as problem 1)
        V_old = V
        Vnext = [0.0 for i = 1:N, j =  1:nA]
        for i = 1:nA
            block_begin = nA*(i-1)+1
            block_end = nA*i
            T_next = T[:, block_begin: block_end]
            vals = T_next * V
            Vnext[:,i] = vals
        end
        # using Vnext, update V
        (V, C) = findmax((U + delta*Vnext), dims = 2)
        distance = norm(V-V_old)
    end

    # solving for T_opt using C (optimal actions)
    T_opt = zeros(N, N)
    for i in 1:N
        extract = actions[C[i][2]]
        state = steps[C[i][1]]
        state_i = C[i][1]
        next = state - extract
        # linear interpolation
        lo = searchsortedlast(steps, next)
        hi = searchsortedfirst(steps, next)
        # and fill in T_opt 
        if lo == hi
            T_opt[state_i, lo] = 1
        else
            prob_lo = 1-(next - steps[lo])/2
            prob_hi = 1-prob_lo
            T_opt[state_i, lo] = prob_lo
            T_opt[state_i, hi] = prob_hi
        end
    end

    # part d
    stock = [convert(Float64, S_tot)]
    for t in 1:80
        today_stock = stock[t]
        tomorrows = T_opt*steps
        # linear interpolation
        lo = searchsortedlast(steps, today_stock)
        hi = searchsortedfirst(steps, today_stock)
        # fill in the sequence of stocks
        if lo == hi
            tomorrow_stock = tomorrows[lo]
        else
            prob_lo = 1-(today_stock - steps[lo])/2
            prob_hi = 1-prob_lo
            tomorrow_stock = prob_lo * tomorrows[lo] + prob_hi * tomorrows[hi]
        end
        append!(stock, tomorrow_stock)
    end
    # using the sequence of stocks fill in optimal extraction path
    extraction = []
    for t in 1:80
        extract = stock[t] - stock[t+1]
        append!(extraction, extract)
    end

    # part e
    x = 1:80
    y = extraction
    p = plot(x, y, 
        xlabel = "Time", 
        ylabel = "Extraction", 
        linecolor = :dodgerblue, 
        fontfamily = "Courier New, monospace", 
        title = string("Extraction with interpolation: utility ",string(utility)),
        legend = false
    )
    png(p, string(figpath, "p2_extraction_utility", string(utility), ".png"))
    # calculate prices using marginal utilities
    if utility == 1
        prices = [u1_prime(i) for i in extraction]
    else
        prices = [u2_prime(i) for i in extraction]
    end
    price_plot = plot(x, prices, 
        xlabel = "Time", 
        ylabel = "Price", 
        linecolor = :dodgerblue, 
        fontfamily = "Courier New, monospace", 
        legend = false, 
        title = string("Prices with interpolation: utility ", string(utility)),
    )
    png(price_plot, string(figpath, "p2_prices_utility", string(utility), ".png"))
end
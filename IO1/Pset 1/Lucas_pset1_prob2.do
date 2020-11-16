* Nadia Lucas
* Industrial Organization 1, Problem Set 1, Question 2
* October 22, 2020

clear all
set more off
cd "/Users/nadialucas/Dropbox/Second year/BUSN 33921/Psets/Pset 1"

* create the dataset
qui set obs 5000
gen id = mod(_n-1, 100) + 1
gen time = floor((_n-1)/100) + 1

xtset id time
gen epsilon_it = rnormal(0,.1)

reshape wide epsilon_it, i(id) j(time)
gen gamma_i = rnormal(0,.5)
reshape long epsilon_it, i(id) j(time)

xtset id time
gen omega_it = 0
* initializing omega_i0 = 0
replace omega_it = epsilon_it if time == 1
replace omega_it = 0.8*omega_it[_n-1] + epsilon_it if time!=1


gen a_it = gamma_i +omega_it
gen k_it = rnormal(0,.1)
gen w_it = rnormal(0,.5)

* from part a derivation 
gen l_it = (1/.3)*(a_it + log(0.7) - w_it) + k_it

* part b
gen y_it = a_it + 0.7 * l_it + 0.3 * k_it

label var k_it "Logged capital inputs"
label var l_it "Logged labor inputs"
label var w_it "Exogenous wage rate"
label var y_it "Logged output"

* part c
reg y_it l_it k_it, robust
eststo ols

esttab `ols' using ols.tex, title("Production function estimation") replace ///
	b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
	mtitles("OLS") compress label nogaps
eststo clear

* part d
xtreg y_it l_it k_it, fe vce(cluster id)
eststo fereg
xtreg y_it l_it k_it, re vce(cluster id)
eststo rereg

esttab `fereg' `rereg' using fixed_random.tex, title("Production function estimation using Fixed and Random Effects") replace ///
	b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
	mtitles("Fixed Effects" "Random Effects") compress label nogaps
eststo clear

* part e
xtivreg y_it k_it (l_it = w_it), fe vce(bootstrap)
eststo ivreg1
 
esttab `ivreg1' using iv_fixedeffects.tex, title("Production function estimation using IV") replace ///
	b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
	mtitles("IV") compress label nogaps
eststo clear

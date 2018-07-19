import scipy.stats as st
import json
import numpy as np

# compute ANOVA on the observables


observables_names = ['d', 'v_G', 'v_diff']
intensities = ['0', '1', '2', '3']

couples =  [('0', '1'), ('1', '2'), ('2', '3')]

with open('data/observables/mean_obs.json') as json_data:
    observables = json.load(json_data)

for i in intensities:
    for o in observables_names:
        to_np = np.array(observables[i][o])
        no_nan = to_np[~np.isnan(to_np)]
        observables[i][o] = no_nan

for o in observables_names:
    print('==================\nANOVA for {}'.format(o))
    for c in couples:
        i1, i2 = c[0], c[1]
        print('-------------------\nComparing intensities {} and {}:'.format(i1, i2))
        print(st.f_oneway(observables[i1][o], observables[i2][o]))
        
        # # manual ANOVA
        # f_i = np.array([np.mean(observables[i][o]) for i in intensities])
        # sigma_i = np.array([np.std(observables[i][o]) for i in intensities])
        # n_i = np.array([len(observables[i][o]) for i in intensities])
        # N = sum(n_i)

        # f_bar = sum(f_i * n_i) / N

        # d_i = f_i - np.ones(len(f_i)) * f_bar

        # SST = sum(n_i * d_i** 2)
        # SSE = sum(n_i * sigma_i**2)

        # n_bar = len(f_i)
        # k = n_bar - 1 # v1: 1st deg of freedom
        # Nk = N - n_bar # v1: 2nd deg of freedom

        # MST = SST / k
        # MSE = SSE / Nk

        # F_d = MST / MSE

    # print(k, Nk, F_d)



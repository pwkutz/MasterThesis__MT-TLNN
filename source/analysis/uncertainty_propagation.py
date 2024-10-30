import numpy as np

# UP in single-fidelity GP is done via analytical formulas 
# UP in non-linear multi-fidelity GP, like in case of a NARGP, is done via numerical MC integration (=MC sampling)
# UP in DGP is done using... creating GP surrogates for each level, by approximating them via KL div
# Felix's proposal: use additional dimension in GP, in order to preserve the uncertainties...

# in proper Bayesian approach: during the propagation, we 

def nargp_proper_propagation(X_hf_test, lower_fidelity_surrogate, higher_fidelity_surrogate):
    '''
    source: https://royalsocietypublishing.org/doi/full/10.1098/rspa.2016.0751
    github (implementation is in the example): https://github.com/paraklas/NARGP

    In MuDaFuGP implementation there is no uncertainty propagation through the multi-fidelity levels, only mean prediction is given to the next level (during test)
    '''
    ### proper propagation through the fidelity level
    Nts = X_hf_test.shape[0]
    nsamples = 1000
    mu1, C1 = lower_fidelity_surrogate.predict(X_hf_test, full_cov=True)
    Z = np.random.multivariate_normal(mu1.flatten(),C1,nsamples)

    # push samples through f_2
    tmp_m = np.zeros((nsamples,Nts))
    tmp_v = np.zeros((nsamples,Nts))
    for j in range(0,nsamples):
        mu, v = higher_fidelity_surrogate.predict(np.hstack((X_hf_test, Z[j,:][:,None])))
        tmp_m[j,:] = mu.flatten()
        tmp_v[j,:] = v.flatten()

    # get posterior mean and variance
    mean_proper = np.mean(tmp_m, axis = 0)[:,None]
    var_proper = np.mean(tmp_v, axis = 0)[:,None] + np.var(tmp_m, axis = 0)[:,None]
    
    return mean_proper, var_proper
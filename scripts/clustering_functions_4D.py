import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# Perform k--means clustering using the mean of ln p per walker. This method is performed to optimise
# the efficiency of the MCMC. Walkers that are probing log posterior parameter space that are not as ideal 
# are removed.
def ln_p_clustering(ln_p_lastiters, nwalkers, minClusters, maxClusters):
    # Get the mean of the log posterior per walker.
    ln_p_mean_tmp = np.mean(np.array(ln_p_lastiters).reshape(-1, nwalkers), axis=0)
    ln_p_mean = ln_p_mean_tmp.reshape(-1,1)
    # Use ln_p_mean to perform k means clustering. Gap statistic is used to determine the number of
    # clusters. The number of clusters may only be 1 or 2.
    k, gapdf = optimalK(ln_p_mean, nrefs = 3, minClusters = minClusters, maxClusters = maxClusters)
    n_clusters = int(k)
    kmeans = KMeans(n_clusters = n_clusters)
    y_km = kmeans.fit(ln_p_mean)

    return ln_p_mean, kmeans

# NOTE: This function, performing k means using the Gap statistic (Tibshirani, Walther, Hastie), was taken from the internet. Please consult: 
# https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad
def optimalK(data, nrefs=3, minClusters=5, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(minClusters, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            #plt.show(randomReference[0:], randomReference[1:])
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    return (gaps.argmax() + minClusters, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

def best_walkers(state):
    ln_p_params = list(zip(state.log_prob, state.coords))
    ln_p_params_sorted = sorted(ln_p_params, key=lambda x: x[0], reverse = True)
    ln_p_params_best = [ln_p_params_sorted[i][0:3] for i in range(0,8)]
    ln_p_params_best = list(zip(*ln_p_params_best))
    state.log_prob = np.array(ln_p_params_best[0])
    state.coords = np.array(ln_p_params_best[1])
    return state

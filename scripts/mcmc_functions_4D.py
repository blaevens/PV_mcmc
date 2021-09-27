# Import Python functions
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import random, sys, os, glob
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import basic_chain, ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import emcee
from multiprocessing import Pool
from collections import Counter
from operator import itemgetter
from IPython.display import clear_output, Image
import warnings
warnings.simplefilter("ignore")

# Import custom functions
from clustering_functions_4D import ln_p_clustering, optimalK, best_walkers
from plot_functions_4D import burnin_chains, sample_chains, autocorrelation
from data_processing_4D import overwrite_files


# Assign guess values for theta, phi and Np. These are subsequently used in get_starting_values_ball below.
def guess_values():
    #params_pred = input('input predicted values for theta, phi and Np = ').split()
    #a, b, c = params_pred.split(',')
    
    theta_guess, phi_guess, Np_guess, sigma_guess = [int(x) for x in input("input predicted values (separated by a space) for theta, phi, Np and sigma = : ").split()]
    
    if (theta_guess < 0):
        print('a negative tilt angle is not possible. Exiting. Please try again'); sys.exit()
    if (theta_guess > 90):
        print('an angle larger than 90 degrees is not possible for the tilt angle. Exiting. Please try again.'); sys.exit()
    if (phi_guess < 0):
        phi_guess += 360
        print('Warning: a negative angle is possible; this is the same as phi += 360. Adding 360 degrees.')
    if (phi_guess > 360):
        phi_guess -= 360
        print('Warning: an angle larger than 360 degrees is possible; this is the same as phi -= 360. Subtracting 360 degrees.')
    if (Np_guess > 100):
        print('Warning: you are running an MCMC simulation for a very large system size. Are you sure?')
    if (sigma_guess < 0):
        print('a negative sigma is not possible. Exiting. Please try again'); sys.exit()
    
    return theta_guess, phi_guess, Np_guess, sigma_guess

# Assign MCMC starting values for nwalkers in 3D parameter space (theta, phi and Np).
def get_starting_values_ball(guess_values, ndim, nwalkers, variation):
    values = [guess_values + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    
    return values

# Obtain PVlib energy day series for a chosen set of PV system characteristics using the Sandia module and SAPM inverter databases.
# Here, a SunPower module and AE Solar inverter were chosen. These may be substituted with other choices.
def get_energy_day(lat, lon, theta, phi, Np, year, month, day, naive_times, sandia_modules, sapm_inverters, timezone):

    module = sandia_modules['SunPower_SPR_230_WHT__2007__E__']
    inverter = sapm_inverters['AE_Solar_Energy__AE5_0__208V_']
    system = PVSystem(surface_tilt = theta, surface_azimuth = phi, module_parameters = module, modules_per_string = int(Np), strings_per_inverter = 1, inverter_parameters = inverter)
    location = Location(lat, lon, name='Amsterdam', altitude=10, tz=timezone)
    mc = ModelChain(system, location, orientation_strategy=None)
    mc.run_model(location.get_clearsky(naive_times))

    return [0 if np.isnan(e) else int(e) for e in mc.ac]

# Define the log likelihood function to be evaluated by Emcee. 
def lnlike(params, dict_observed, times, hm, d0, lon, lat, sandia_modules, sapm_inverters, timezone):
    
    theta, phi, Np, sigma = params
    Np = int(Np)
    E_mod = get_energy_day(lat, lon, theta, phi, Np, d0.year, d0.month, d0.day, times, sandia_modules, sapm_inverters, timezone)
    dict_model = dict(list(zip(hm, E_mod)))
    obsmod = [(k, dict_observed[k], dict_model[k]) for k in sorted(dict_observed)]

    # Could define the value of N*log*2pi to speed up code.
    return (((-1*np.sum([((val[1]-val[2])**2) for val in obsmod])/(2*(sigma**2))))-(len(obsmod)*np.log(sigma)))
    #-(len(obsmod)*np.log(np.sqrt(2*3.14)))

# Define the prior for the MCMC. A uniform prior is defined which allows for all orientations and tilts and number of panels ranging 
# between 1 and 100. Future improvement will include allowing for the prior to be constrained.
def lnprior(params):
    
    theta, phi, Np, sigma = params
    if 0.0 < theta < 90.0 and 0.0 < phi < 360.0 and 1 < Np < 100 and 0 < sigma < 1000:
        return 0.0
    return -np.inf

# Define the log probability as the sum of lnprior and lnlike. If phi is smaller than 0 or larger than 360, then phi is re-calculated.
def lnprob(params, dict_observed, times, hm, d0, lon, lat, sandia_modules, sapm_inverters, timezone):
    
    theta, phi_tmp, Np, sigma = params

    if (params[1]>360.0):
        params[1]-=360.0
    elif (params[1]<0.0):
        params[1]+=360.0
    else:
        params[1]=params[1]
    
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + lnlike(params, dict_observed, times, hm, d0, lon, lat, sandia_modules, sapm_inverters, timezone)
    
# This function is employed in the sampling function (below) to check whether the chains have converged. The lines of code here come from 
# the Emcee manual.
def check_convergence(tau, old_tau, iteration):
    converged = np.all(tau * 50 < iteration)
    print('50*tau/iteration = ', tau * 50/iteration)
    print('(old tau - tau)/tau = ', np.abs(old_tau - tau) / tau)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    return converged
   
# This function performs MCMC sampling up until the burn in phase is deemed completed. This is determined by clustering the walker chains
# and checking how close to each other the chains are in parameter space. For more information, look at the individual comments in the function.
def burnin(system_meta, system_measurements, date, mcmc_config, pvlib_databases, clustering_params, paths):
    # Assign variable values from tuples.
    (sid, lon, lat, timezone) = system_meta
    (dict_measurement, times, hm) = system_measurements
    (nwalkers, ndim, nproc, iterations, values) = mcmc_config
    (sandia_modules, sapm_inverters) = pvlib_databases
    (minClusters, maxClusters) = clustering_params
    (chains_path, plots_path) = paths
    
    nwalkers_initial = nwalkers
    year = int(date[0:4]); month = int(date[4:6]); day = int(date[6:8])
    d0 = datetime.datetime(year, month, day)
    
    # Check whether the MCMC chain files already exist.
    overwrite_files(chains_path, sid, date, nwalkers_initial, iterations)
    

    
    index = 0; index_burnin = -1; niters = 50; storage = True; ln_p_lastiters = []; index_var = 0
    
    # Define MCMC chain file which will be used by the Emcee backend function.
    filename = chains_path + 'chain_id%s_%s_w%s_i%s.h5' %(str(sid), date, str(nwalkers_initial), str(iterations))
    backend = emcee.backends.HDFBackend(filename); backend.reset(nwalkers, ndim)
    
    with Pool(processes=nproc) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=([dict_measurement, times, hm, d0, lon, lat, sandia_modules, sapm_inverters, timezone]), pool=pool, backend=backend, moves=emcee.moves.WalkMove())
        # The while loop keeps running until the end of the burn in period has been reached.
        # The end is determined on the basis of statistics applied to the remaining number of clustered walkers.
        while (index == 0):
            index_burnin += 1
            # The MCMC sampling is performed for 50 iterations afte
            for sample in sampler.sample(initial_state = values, iterations = niters, progress=True, store=storage):
                ln_p_lastiters.extend(sample.log_prob)
                # Ignore all calculations in this loop unless the number of iterations (sampler.iteration) equals niters.
                if sampler.iteration % niters:
                    continue
                clear_output(wait=True)
                # All code within this if statement is concerned with the plotting of the MCMC chains. 
                # This is not performed on the first run, since no chains have been generated yet.
                if (index_burnin > 0):
                    #files = sorted(glob.glob(chains_path + '*i10000*h5'))
                    files = sorted(glob.glob(chains_path + 'chain_id%s_%s_w%s_i%s*.h5' %(str(sid), date, str(nwalkers_initial), str(iterations))))
                    samples, log_samples = get_last_sample(files)
                    # Create plot for the first set of chains and keep appending plot afterwards.
                    if (index_burnin == 1):
                        fig, axes = plt.subplots(6, figsize=(15, 11), sharex=True)
                    starter_val = index_burnin*niters
                    burnin_chains(fig, axes, starter_val, starter_val + niters, samples, log_samples, plots_path, \
                                  sid, nwalkers_initial, iterations, date)

#                    display(Image(filename = plots_path + 'mcmc_id%s_%s_w%s_i%s.png' %(str(sid), date, str(nwalkers_initial), str(iterations))))
                
                # Perform k means clustering. See functions ln_p_clustering and optimalK for more info. 
                ln_p_mean, kmeans = ln_p_clustering(ln_p_lastiters, nwalkers, minClusters, maxClusters)
                occurrences = Counter(kmeans.labels_)                
                walker_group = list(zip([i for i in range(len(kmeans.labels_))], kmeans.labels_))
                # Calculate the variance (spread) of the mean log posterior per walker.
                ln_p_var = np.var(ln_p_mean)
                
                

                # Check whether the variance is small or large. 
                print(ln_p_var)
                if (ln_p_var > 5):
                    index=0
                    # Check whether the number of clustered groups is larger than 1.
                    if (len(occurrences)>1):
                        # Determine which walkers belong to which group.
                        min_key, min_count = min(occurrences.items(), key=itemgetter(1))
                        min_val = kmeans.cluster_centers_[min_key][0]
                        if (abs(np.min(kmeans.cluster_centers_)-min_val)<1e-5):
                            id_walkers = [num[0] for num in walker_group if num[1] != min_key]
                        else:
                            id_walkers = [num[0] for num in walker_group]
                    # If there is only one group of walkers, then all walkers are retained for the next 50 iterations.
                    else:
                        id_walkers = [num[0] for num in walker_group]
                
                # If the variance is smaller than 5, this could indicate that the chains are starting to converge towards the same point.
                # To test whether this is the case, a counter (index_var) is initialised. If the algorithm ends up five successive times
                # in this else statement, the MCMC burn in is considered finished. 
                else:
                    index=0
                    id_walkers = [num[0] for num in walker_group]
                    index_var += 1
                    print('Index var equals ', index_var)
                    # Set a check index to the burn in index. The difference between the indices must be five if else statement was performed
                    # 5 successive times.
                    if index_var == 1:
                        check_first = index_burnin
                    if index_var == 6:
                        check_last = index_burnin
                        if ((check_last - check_first) == 5):
                            index = 1
                        else:
                            index_var = 0
                
                # Define a new filename for the new backend. Please see explanation below as to why this is necessary.
                filename = chains_path + 'chain_id%s_%s_w%s_i%s_' %(str(sid), date, str(nwalkers_initial), str(iterations)) + "{:02}".format(index_burnin) + '.h5'                              
                new_backend = emcee.backends.HDFBackend(filename)
                state = sampler.get_last_sample()
                
                # Account for larger or smaller angles for orientation and subtract or add 360 degrees.
                for i in range(len(state.coords)):
                    if state.coords[i][1]>360.0:
                        state.coords[i][1] = state.coords[i][1]-360.0
                    if state.coords[i][1]<0.0:
                        state.coords[i][1] = state.coords[i][1]+360.0
                
                # Change Emcee's 'state' to allow for the clustering approach above. Only want to retain the relevant walkers.
                state.log_prob = np.array([i for j, i in enumerate(state.log_prob) if j in id_walkers])
                state.coords = np.array([i for j, i in enumerate(state.coords) if j in id_walkers])
                print('Best walker value is ', np.min(state.log_prob))
                state.random_state = None
                # Determine the new number of walkers, after clustering.
                nwalkers = len(state.log_prob)
                # Reset the backend with the new number of walkers.
                backend.reset(nwalkers, ndim)
                ndim = 4
                print('new number of walkers is ', nwalkers)
                # Define the sampler with the new backend. This is necessary because it is not possible to append to the same h5 file, when the 
                # number of chains decreases.
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=([dict_measurement, times, hm, d0, lon, lat, sandia_modules, sapm_inverters, timezone]), backend = new_backend, pool = pool, moves = emcee.moves.WalkMove())
                values = state; storage = True; ln_p_lastiters = []
                
            
            # Check whether the burn in phase is completed. Else continue.
            if (index == 1):
                print('Burn in phase completed')
                break 
            else:
                print('Burn in phase is still happening')
                os.system("clear")

    clear_output(wait=True)
    print('MCMC sampling can now commence')
    
    # Having exited the while loop, the best 8 walkers are obtained with the function best_walkers. User can choose to keep more walkers 
    # if desired, though bear in mind that this number also depends on the number of walkers used to start with. Here we used 40, so this number 
    # should also be increased.
    nwalkers = 8
    state = best_walkers(state)    
    return state



# This function is performed after the burnin function (above). 
def sampling(system_meta, system_measurements, date, mcmc_config, pvlib_databases, clustering_params, paths):
    # Assign variable values from tuples.
    (sid, lon, lat, timezone) = system_meta
    (dict_measurement, times, hm) = system_measurements
    (nwalkers, ndim, nproc, iterations, values) = mcmc_config
    (sandia_modules, sapm_inverters) = pvlib_databases
    (minClusters, maxClusters) = clustering_params
    (chains_path, plots_path) = paths
    
    nwalkers_initial = nwalkers
    year = int(date[0:4]); month = int(date[4:6]); day = int(date[6:8])
    d0 = datetime.datetime(year, month, day)
    filename = chains_path + 'chain_id%s_%s_w%s_i%s_sample.h5' %(str(sid), date, str(nwalkers_initial), str(iterations))
    filename_plot = plots_path + 'autocorrelation_id%s_%s_w%s_i%s.png' %(str(sid), date, str(nwalkers_initial), str(iterations))
    new_backend = emcee.backends.HDFBackend(filename)
    max_n = iterations
    storage = True
    autocorr = np.empty(max_n)
    index = 0
    niters = 50
    # Necessary to initialise old tau at infinity (see Emcee documentation)
    old_tau = np.inf
    
    # Similar to burnin function.
    with Pool(processes=nproc) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=([dict_measurement, times, hm, d0, lon, lat, sandia_modules, sapm_inverters, timezone]), pool = pool, backend = new_backend, moves = emcee.moves.WalkMove())
        for sample in sampler.sample(initial_state = values, iterations = max_n, progress = True, store = storage):
            if sampler.iteration % niters:
                continue
            
            # Compute the autocorrelation time so far. See Emcee documentation for this snippet. 
            clear_output(wait = True)
            tau = sampler.get_autocorr_time(tol = 0)
            autocorr[index] = np.mean(tau)
            index += 1
            
            # Check convergence
            converged = check_convergence(tau, old_tau, sampler.iteration)
            
            # Plot the auto correlation time
            autocorrelation(index, autocorr, filename_plot)
            
            # Check if convergence has occurred
            if converged:
                break
            old_tau = tau
            
            sample_chains(filename, plots_path, sid, nwalkers, iterations, date)
#            display(Image(filename = plots_path + 'mcmc_id%s_w%s_i%s_%s.png' %(str(sid), str(nwalkers), str(iterations), date)))
            
#    display(Image(filename = plots_path + 'mcmc_id%s_w%s_i%s_%s.png' %(str(sid), str(nwalkers), str(iterations), date)))        
    
    print("Mean acceptance fraction:{0:.3f}".format(np.mean(sampler.acceptance_fraction)))
            

# Obtain the last MCMC chain sample. The output of this function is used as input for the burnin_chains function.    
def get_last_sample(files):
    all_samples = []; all_log_samples = []
    for file in files[-1:]:
        reader = emcee.backends.HDFBackend(file)
        total_samples = reader.get_chain()
        length = total_samples.shape[0]
        log_prob_samples = reader.get_log_prob()
        for sample in total_samples:
            all_samples.append(sample)
        for logprob in log_prob_samples:
            all_log_samples.append(logprob)

    samples = np.asarray(all_samples)
    log_samples = np.array(all_log_samples)
    
    return samples, log_samples    


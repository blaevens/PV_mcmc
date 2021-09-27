import numpy as np
import pandas as pd
import pvlib
import emcee
import os
import corner
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.ticker as plticker
import matplotlib.dates as mdates

from data_processing_4D import retrieve_measurements, retrieve_meta, get_naive_times
import mcmc_functions_4D



# Plot the auto correlation time every 50 iterations.
def autocorrelation(index, autocorr, filename_plot):    
    n = 50 * np.arange(1, index + 1)
    y = autocorr[:index]
    plt.cla(); plt.clf();
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.plot(n, n / 50.0, "--k")
    ax.plot(n, y)
    ax.set_xlim(0, n.max())
    ax.set_ylim(0, y.max() + 0.1 * (y.max() - y.min()))
    ax.set_xlabel("number of steps", fontsize = 14)
    ax.set_ylabel(r"mean $\hat{\tau}$", fontsize = 14)
    fig.savefig(filename_plot, facecolor='w', transparent = False)
    plt.close(); plt.cla(); plt.close()
    
# Plot the sample chains (in the sampling phase, which is post burn in).     
def sample_chains(filename, path_output, N_id, walker, iterations, date):
    # get the samples for orientation, tilt and number of panels as well as the log samples.
    reader = emcee.backends.HDFBackend(filename)
    fig, axes = plt.subplots(5, figsize=(15, 11), sharex=True)
    samples = reader.get_chain()
    log_prob_samples = reader.get_log_prob()
    labels = [r'$\theta_{pred}~(^{\circ})$', r'$\phi_{pred}~(^{\circ})$', r'$N_{p,pred}$', r'$\sigma_{pred}$']
    for i in range(4):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i], rotation=0, fontsize=14)
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    ax = axes[4]
    ax.plot(log_prob_samples[:, :], "k", alpha=0.3)    
    ax.set_ylabel(r'$ln~p(\beta|D)$', rotation=0, fontsize=14)
    ax.yaxis.labelpad = 45
        
    axes[-1].set_xlabel(r'$step~number$', fontsize=14);
    axes[0].set_title(r'Samples for $\theta_{pred}$, $\phi_{pred}$, $N_{p,pred}$, $\sigma_{pred}$ and $ln(p(\beta|D))$', fontsize = 18)
    fig.savefig(path_output + 'mcmc_id%s_w%s_i%s_%s.png' %(str(N_id), str(walker), str(iterations), date), facecolor='w', transparent = False)

# Plot the sample chains for the burn in phase. 
def burnin_chains(fig, axes, starter_val, end_val, samples, log_prob_samples, path_output, N_id, walker, iterations, date):
    labels = [r'$\theta_{pred}~(^{\circ})$', r'$\phi_{pred}~(^{\circ})$', r'$N_{p,pred}$', r'$\sigma_{pred}$']
    for i in range(4):
        ax = axes[i]
        ax.plot([j for j in range(starter_val, end_val)], samples[:, :, i], color="k", alpha=0.3)
        ax.set_ylabel(labels[i], rotation=0, fontsize=14)
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    # fourth and fifth subplot are the same, with the difference that the fifth subplot is zoomed in on the range of -1000, 0. 
    ax = axes[4]
    ax.plot([j for j in range(starter_val, end_val)], log_prob_samples[:, :], "k", alpha=0.3)    
    ax.set_ylabel(r'$ln~p(\beta|D)$', rotation=0, fontsize=14)
    ax.yaxis.labelpad = 45
    
    ax = axes[5]
    ax.plot([j for j in range(starter_val, end_val)], log_prob_samples[:, :], "k", alpha=0.3)    
    ax.set_ylim(-1000, 0)
    ax.set_ylabel(r'$ln~p(\beta|D)$', rotation=0, fontsize=14)
    ax.yaxis.labelpad = 45
    
    axes[-1].set_xlabel(r'$step~number$', fontsize=14);
    axes[0].set_title(r'Burn-in samples for $\theta_{pred}$, $\phi_{pred}$, $N_{p,pred}$, $\sigma_{pred}$ and $ln(p(\beta|D))$', fontsize = 18)
    fig.savefig(path_output + 'mcmc_id%s_%s_w%s_i%s.png' %(str(N_id), date, str(walker), str(iterations)), facecolor='w', transparent = False)

# Plot the pdfs of orientation, tilt and system size.
def plot_pdfs(flat_samples, sid, date, nwalkers, iterations, path_plot, best_params):

    # Cosmetic operation: to plot Np correctly, we add two extra datapoints to our file. 
    # This is because the corner.corner module does not allow just one bin to be plotted.
    # Adding these two points is not strictly correct, but won't matter because only 2 points amongst many.
    flat_samples = np.vstack([flat_samples, flat_samples[len(flat_samples)-1]])
    flat_samples[len(flat_samples)-1][2] = int(round(flat_samples[:, 2].max() + 1))
    flat_samples = np.vstack([flat_samples, flat_samples[len(flat_samples)-1]])
    flat_samples[len(flat_samples)-1][2] = int(round(flat_samples[:, 2].min() - 1))

    labels = [r'$\theta_{pred}~(^{\circ})$', r'$\phi_{pred}~(^{\circ})$', r'$N_{p, pred}$', r'$\sigma_{pred}~(^{\circ})$']
    bins_npanels = int(flat_samples[:, 2].max()-flat_samples[:, 2].min())
    if bins_npanels==1:
        bins_npanels+=1
    fig2 = corner.corner(flat_samples, labels=labels, bins=[20, 20, bins_npanels, 20], label_kwargs=dict(fontsize=14));
    axes = np.array(fig2.axes).reshape((4, 4))
    

    for i in range(4):
        if (i != 2):
            ax = axes[i, i]
            ax.axvline(best_params[i], color="g")

    

    fig2.savefig(path_plot + 'pdfs_id%s_%s_w%s_i%s.png' %(str(sid), date, str(nwalkers), str(iterations)), facecolor='w', transparent = False, dpi=1000)
    plt.close()

# Get the chain samples (post burn in onwards until convergence). These are used as input for plotting of pdfs (see above).
def retrieve_samples(sid, date, nwalkers, iterations, chains_path):
    file_chain = chains_path + 'chain_id%s_%s_w%s_i%s_sample.h5' %(str(sid), date, str(nwalkers), str(iterations))
    if os.path.exists(file_chain):
        reader = emcee.backends.HDFBackend(file_chain)
    else:
        print('The file does not exist. Please check what has gone wrong.')

    while True:
        try:
            tau = reader.get_autocorr_time()
            # Define extra burn in segment in function of the auto correlation time and remove later. 
            burnin = int(2 * np.max(tau))
            # Thin the samples later.
            thin = int(0.5 * np.min(tau))
            break
        except emcee.autocorr.AutocorrError:
            print('There is an auto-correlation function error. Please check what\'s going on.')
            break
        
    total_samples = reader.get_chain()
    length = total_samples.shape[0]
    
    if length==20000:
        print('The sample is very long: 20 000 iterations were necessary implying sampling took too long. Please re-run the sample.')
    
    flat_samples = reader.get_chain(discard = burnin, thin = thin, flat = True)
    log_prob_samples = reader.get_log_prob(discard = burnin, thin = thin, flat = True)
    best_log_prob = np.max(log_prob_samples)
    
    return flat_samples


# Retrieve the 16th, 50th and 84th percentile of the pdfs for the three different parameters.
def retrieve_best_params(flat_samples, ndim, rounding):
    # User can specify the number of significant digits with 'rounding'.
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        # Obtain the uncertainties by calculating the difference between 16th and 50th or 84th and 50th percentiles.
        q = np.diff(mcmc)
        if (i==0):
            p1 = round(mcmc[1], rounding)
            p1_min = round(q[0], rounding)
            p1_max = round(q[1], rounding)
        if (i==1):
            p2 = round(mcmc[1], rounding)
            p2_min = round(q[0], rounding)
            p2_max = round(q[1], rounding)
        if (i==2):
            p3 = int(mcmc[1])
            p3_min = int(q[0])
            p3_max = int(q[1])
        if (i==3):
            p4 = round(mcmc[1], rounding)
            p4_min = round(q[0], rounding)
            p4_max = round(q[1], rounding)
    
    df_params = pd.DataFrame(data={'value':[p1, p2, p3, p4], '-sigma': [p1_min, p2_min, p3_min, p4_min], '+sigma' : [p1_max, p2_max, p3_max, p4_max]})
    df_params.index = ['theta', 'phi', 'Np', 'sigma']
    
    return df_params, [p1, p2, p3, p4]        


# This function plots the 'observed profile' with the initial parameters as determined in the notebook (green) and compares it to 
# the profile, generated with the preferred parameters after performing MCMC (orange). The user will note that this is somewhat 
# contrived given that the observed data is also 'modelled'.

# This function needs to be improved upon.
def observed_modelled_profiles(sid, date, time_resolution, data_path, df_params, option):
    
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    lat, lon = retrieve_meta(sid, data_path)
    dict_measurement = retrieve_measurements(sid, date, data_path, option)
        
    year = int(date[0:4]); month = int(date[4:6]); day= int(date[6:8]);
    
    theta = df_params.loc['theta']['value']
    phi = df_params.loc['phi']['value']
    Np = df_params.loc['Np']['value']
    
    times, ymd, hm, timezone = get_naive_times(date, str(time_resolution))
    energy = mcmc_functions_4D.get_energy_day(lat, lon, theta, phi, Np, year, month, day, times, sandia_modules, sapm_inverters, timezone)


    start_profile = hm.index(list(dict_measurement.keys())[0])
    end_profile = hm.index(list(dict_measurement.keys())[len(dict_measurement)-1])
    


    xx = [str(year) + str(month) + str(day) + h for h in hm]
    xx = pd.to_datetime(xx, format="%Y%m%d%H:%M")
    
    
    yy = list(dict_measurement.keys())
    yy = [str(year) + str(month) + str(day) + h for h in yy]
    yy = pd.to_datetime(yy, format="%Y%m%d%H:%M")
    

    fig, ax = plt.subplots(1, figsize=(12, 6), sharex=True)
    ax.scatter(yy, list(dict_measurement.values()), lw=2, color='green')
    ax.scatter(xx[start_profile:end_profile], energy[start_profile:end_profile], lw=2, color='orange')
    loc = plticker.MultipleLocator(base=1/12) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%S'))
    plt.xticks(rotation='horizontal',horizontalalignment="center", fontsize = 12)
    plt.yticks(fontsize = 12)
    ax.set_xlabel('Time (h)', fontsize = 13)
    ax.set_ylabel('Power (W)', fontsize = 13)
    #plt.show()






    

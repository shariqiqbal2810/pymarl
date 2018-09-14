""" Plots the results from an experiment stored by a FileStorageObserver. """

import json
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import torch
import numpy as np
import math
import pymongo
from copy import deepcopy
from time import sleep
import sys

#named_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


def load_experiments(keys, ids, directory="results", test=False):
    assert isinstance(keys, list) and len(keys) > 0, "The loaded keys must be a list of (a list of) strings."
    plot_2d = isinstance(keys[0], list)
    t_env_str = "T env" if not test else "T_env_test"

    # Load all experiments
    data = []
    for l in range(len(ids)):
        print("Load ID", ids[l])
        while True:
            try:
                data.append(json.load(open('%s/%u/info.json' % (directory, ids[l]))))
                break
            except json.decoder.JSONDecodeError:
                print("JSON decode error in ID %u. Repeat." % ids[l])
                sleep(1)

    # Find maximum time
    max_time = 0
    for l in range(len(data)):
        if t_env_str in data[l]:
            if len(data[l][t_env_str]) > max_time:
                max_time = len(data[l][t_env_str])
                time = np.array(data[l][t_env_str])
        else:
            print("No environment steps found in ID %u" % ids[l])
            if len(data[l]["Episode reward"]) > max_time:
                max_time = len(data[l]["Episode reward"])
                time = np.array([i + 1 for i in range(max_time)])

    # Fill result Tensor
    if plot_2d:
        res = np.ones((len(ids), len(keys), len(keys[0]), max_time)) * float("nan")
    else:
        res = np.ones((len(ids), len(keys), 1, max_time)) * float("nan")
    for l in range(len(ids)):
        for i in range(len(keys)):
            for j in range(len(keys[i]) if plot_2d else 1):
                if plot_2d:
                    res[l, i, j, :min(len(data[l][keys[i][j]]), max_time)] = \
                        np.array(data[l][keys[i][j]])[:min(len(data[l][keys[i][j]]), max_time)]
                else:
                    res[l, i, 0, :min(len(data[l][keys[i]]), max_time)] = \
                        np.array(data[l][keys[i]])[:min(len(data[l][keys[i]]), max_time)]

    # Return result Tensor
    return res, time, data


def draw_experiments(gca, time, res_key, colors=['blue'], plot_individuals=None, pm_std=False, use_sem=False):
    if not isinstance(colors, list):
        colors = [colors]
    res_key = res_key.squeeze()
    if len(res_key.shape) < 2:
        res_key = np.expand_dims(res_key, axis=0)
    #
    if res_key.shape[0] > 1:
        m = np.nanmean(res_key, axis=0)
    else:
        m = res_key.squeeze(0)
    #
    if res_key.shape[0] > 1:
        # Compute standard deviation/s
        if pm_std:
            sm = np.zeros(m.shape)
            sp = np.zeros(m.shape)
            for t in range(res_key.shape[1]):
                g = np.zeros(res_key.shape[0], dtype=bool)
                ng = np.zeros(res_key.shape[0], dtype=bool)
                for i in range(res_key.shape[0]):
                    g[i] = res_key[i, t] - m[t] > 0
                    ng[i] = res_key[i, t] - m[t] <= 0
                if np.sum(g) > 0:
                    sp[t] = np.nanmean((res_key[g, t] - m[t]) ** 2 / (np.nansum(g) if use_sem else 1)) ** 0.5
                if np.sum(ng) > 0:
                    sm[t] = np.nanmean((res_key[ng, t] - m[t]) ** 2 / (np.nansum(ng) if use_sem else 1)) ** 0.5
            gca.fill_between(x=time, y1=m - sm, y2=m + sp, alpha=0.2, linewidth=0, facecolor=colors[0])
        else:
            s = np.nanstd(res_key, axis=0) / (res_key.shape[0] ** 0.5 if use_sem else 1)
            gca.fill_between(x=time, y1=m - s, y2=m + s, alpha=0.2, linewidth=0, facecolor=colors[0])
        if plot_individuals is not None and plot_individuals != '' and plot_individuals != ' ':
            for l in range(res_key.shape[0]):
                gca.plot(time, res_key[l, :], linestyle=plot_individuals, color=colors[(l+1) % len(colors)])
    if len(time) == len(m):
        gca.plot(time, m, color=colors[0])
    else:
        print(len(time), len(m))


def plot_batch(ids, directory="results", plot_2d=False, title=None, plot_individuals=True, colors=['blue'], test=False):
    if plot_2d:
        keys = [["Episode reward", "Episode length"], ["Policy loss", "Critic loss"], ["Policy entropy", "Critic mean"]]
    else:
        #keys = ["Episode reward", "Episode length"]
        keys = ["Win_rate_test", "episode_reward_test", "episode_length_test"]

    res, time, data = load_experiments(keys, ids, directory=directory, test=test)

    if plot_2d:
        if "NoisyNet Regularization" in data[0]:
            keys[2][1] = "NoisyNet Regularization"
        res, time, data = load_experiments(keys, ids, directory)

    print("POSSIBLE STATISTICS:")
    for key in data[0]:
        print(key)

    # Plot
    if plot_2d:
        fig, ax = plt.subplots(len(keys), len(keys[0]))
    else:
        fig, ax = plt.subplots(len(keys))
    for i in range(len(keys)):
        for j in range(len(keys[i]) if plot_2d else 1):
            if plot_2d:
                gca = ax[i][j]
            else:
                gca = ax[i]
            draw_experiments(gca, time, res[:, i, j, :], colors=colors, plot_individuals=plot_individuals)
            gca.set_xlabel("Environmental Steps")
            gca.set_xlim(time.min(), time.max())
            if plot_2d:
                gca.set_ylabel(keys[i][j])
            else:
                gca.set_ylabel(keys[i])

    # Set global title
    gca = ax[0][0] if plot_2d else ax[0]
    if 'title' is not None:
        gca.set_title(title)

    # Show figure
    plt.show()


def plot_comparison(id_lists, title=None, legend=None, directory="results", max_time=None, plot_individuals=False, test=False):
    #keys = ["Episode reward", "Episode length"]
    keys = ["Win_rate_test", "episode_reward_test", "episode_length_test"]
    legend_pos = ['lower right']
    #colors = ["blue", "green", "red", "cyan", "magenta"]
    colors = ["red", "blue", "green", "red", "cyan", "magenta"]
    find_max_time = max_time is None
    # Make figure
    fig, ax = plt.subplots(len(keys))
    if legend is not None:
        for k in range(len(keys)):
            gca = ax[k]
            for i in range(len(id_lists)):
                gca.fill_between(x=np.array([-2, -1]), y1=np.array([float('nan'), float('nan')]),
                                 y2=np.array([float('nan'), float('nan')]), alpha=0.2, linewidth=0, facecolor=colors[i])
                #gca.plot(np.array([-2, -1]), np.array([0, 0]), color=colors[i], linestyle='-')
            gca.legend(legend, loc=legend_pos[k % len(legend_pos)])

    # Plot horizontal helper lines
    if True:
        horizons = [0.6, 0.8, 0.9, 0.95, 0.975]
        for h in range(len(horizons)):
            ax[0].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')

    # Plot experiments
    if find_max_time:
        max_time = 0
    for i in range(len(id_lists)):
        res, time, data = load_experiments(keys, id_lists[i], directory=directory, test=test)
        if find_max_time and time.max() > max_time:
            max_time = time.max()
        for k in range(len(keys)):
            draw_experiments(ax[k], time, res[:, k, 0, :], colors=[colors[i]], plot_individuals=plot_individuals)
    # Decorate plots
    for k in range(len(keys)):
        gca = ax[k]
        gca.set_xlim(0, max_time)
        gca.set_ylabel(keys[k])
        if k is len(keys) - 1:
            gca.set_xlabel("Environmental Steps")
    if title is not None:
        ax[0].set_title(title)
    # Show figure
    plt.show()
    return ax


def load_db(keys, name, test=False, max_time=None, fill_in=True, longest_runs=0, min_time=0, bin_size=200,
            refactored=False):
    min_time_steps = 10
    if refactored:
        time_str = "test_episode_T" if test else "episode_T"
    else:
        time_str = "T env test" if test else "T env"
    print("Looking for ", name, keys[0], time_str, test)
    # Open a mongo DB client
    db_url = "mongodb://pymarlOwner:EMC7Jp98c8rE7FxxN7g82DT5spGsVr9A@gandalf.cs.ox.ac.uk:27017/pymarl"
    db_name = "pymarl"
    loaded = False
    while not loaded:
        try:
            client = pymongo.MongoClient(db_url, 27017, ssl=True, serverSelectionTimeoutMS=1000)
            client.server_info()
            loaded = True
        except pymongo.errors.ServerSelectionTimeoutError:
            print("No connection to MongoDB server! Press Ctrl+C to stop tying.")
    # Find maximum time
    max_time_found = 0
    min_time_found = 0
    len_time = 0
    time_list = []
    experiment_list = []
    if not isinstance(name, list):
        name = [name]
    look_for_name = [({'$regex': '^' + name[n] + '.*'} if fill_in else name[n]) for n in range(len(name))]
    for this_name in look_for_name:
        for experiment in client[db_name]['runs'].find({'config.name': this_name}):
            # Non-refactored plots have one time-line
            if time_str in experiment['info'] \
                    and len(experiment['info'][time_str]) >= min_time_steps\
                    and experiment['info'][time_str][-1] > min_time:
                experiment_list.append(experiment)
                m = max(experiment['info'][time_str])
                time_list.append(m)
                if m > max_time_found:
                    max_time_found = m
                if m < min_time_found:
                    min_time_found = m
                m = len(set(experiment['info'][time_str]))
                if m > len_time:
                    len_time = m
        print("Found %u experiments with name '%s', max(time)=%u" % (len(experiment_list), this_name, max_time_found))
    if len(experiment_list) == 0:
        return None, None

    # Determine the maximum time and number of longest_runs we can extract
    if max_time is None:
        max_time = max_time_found
    longest_runs = min(longest_runs, len(experiment_list)) if longest_runs > 0 else len(experiment_list)
    time_arr = np.array(time_list)
    time_idx = np.argsort(time_arr)
    time_arr = time_arr[time_idx]
    longest_runs = max(longest_runs, np.sum(time_arr >= max_time))
    max_time = min(max_time, time_list[time_idx[len(time_idx)-longest_runs]])
    print("Times: ", time_arr)

    # Rearrange the experiment_list
    new_list = []
    for i in range(longest_runs):
        new_list.append(experiment_list[time_idx[len(time_idx)-i-1]])
    experiment_list = new_list
    num_experiments = len(experiment_list)

    # Create time-line
    time_bins = min(bin_size, len_time // 2)
    time = np.linspace(0, max_time, num=time_bins)
    # Create result Tensor
    res = np.ones((num_experiments, len(keys), time_bins)) * float("nan")
    num = np.zeros((num_experiments, len(keys), time_bins), dtype=int)
    # Collect result Tensor
    for e, experiment in enumerate(experiment_list):
        for k in range(len(keys)):
            if keys[k] in experiment['info']:
                eti = experiment['info'][time_str]
                val = experiment['info'][keys[k]]
                if isinstance(val[0], float):
                    for t in range(min(len(val), len(eti))):
                        if eti[t] <= max_time and eti[t] <= max_time_found:
                            i = int(eti[t] / (max_time+1) * time_bins)
                            if num[e, k, i] == 0:
                                res[e, k, i] = val[t]
                            else:
                                res[e, k, i] += val[t]
                            num[e, k, i] += 1
                else:
                    print("No float list found in experiment '%s', key '%s'" % (experiment['config']['name'], keys[k]))
    # Normalize each observed data point
    for e in range(num_experiments):
        for k in range(len(keys)):
            for i in range(time_bins):
                if num[e, k, i] > 0:
                    res[e, k, i] /= num[e, k, i]
    # Return result and time
    return res, time


def load_refactored_db(keys, name, test=False, max_time=None, fill_in=True, min_time=0, bin_size=200):
    if len(keys) == 0:
        return None, None
    min_time_steps = 10
    # Open a mongo DB client
    db_url = "mongodb://pymarlOwner:EMC7Jp98c8rE7FxxN7g82DT5spGsVr9A@gandalf.cs.ox.ac.uk:27017/pymarl"
    db_name = "pymarl"
    loaded = False
    while not loaded:
        try:
            client = pymongo.MongoClient(db_url, 27017, ssl=True, serverSelectionTimeoutMS=1000)
            client.server_info()
            loaded = True
        except pymongo.errors.ServerSelectionTimeoutError:
            print("No connection to MongoDB server! Press Ctrl+C to stop tying.")
    # Find maximum time
    max_time_found = [0.0 for _ in range(len(keys))]
    min_time_found = [float('inf') for _ in range(len(keys))]
    len_time_found = [float('inf') for _ in range(len(keys))]
    len_time = 0
    time_list = []
    experiment_list = []
    if not isinstance(name, list):
        name = [name]
    look_for_name = [({'$regex': '^' + name[n] + '.*'} if fill_in else name[n]) for n in range(len(name))]
    for this_name in look_for_name:
        for experiment in client[db_name]['runs'].find({'config.name': this_name}):
            # Filter out invalid experiments
            if all([keys[k] + "_T" in experiment['info'] and len(experiment['info'][keys[k] + "_T"]) >= min_time_steps
                    and experiment['info'][keys[k] + "_T"][-1] >= min_time for k in range(len(keys))]):
                # Add experiment
                experiment_list.append(experiment)
                # Update time statistics for each key
                for k in range(len(keys)):
                    time_str = keys[k] + "_T"
                    m = max(experiment['info'][time_str])
                    max_time_found[k] = max(max_time_found[k], m)
                    min_time_found[k] = min(min_time_found[k], m)
                    len_time_found[k] = min(len_time_found[k], len(experiment['info'][time_str]))
        print("Found %u experiments with name '%s', %g < max(time) < %g" % (len(experiment_list), this_name,
                                                                            min(min_time_found), max(max_time_found)))
        for k in range(len(keys)):
            print(keys[k], sorted([experiment['info'][keys[k] + "_T"][-1] for experiment in experiment_list]))
    if len(experiment_list) == 0:
        return None, None

    for i in range(len(min_time_found)):
        if min_time_found[i] == float('inf'):
            min_time_found[i] = 0

    # Create time-lines for each key
    max_time = float('inf') if max_time is None else max_time
    times = []
    for k in range(len(keys)):
        time_bins = min(bin_size, len_time_found[k] // 2)
        times.append(np.linspace(0.0, min(max_time, min_time_found[k]), num=time_bins))

    # Create result lists
    res = [np.ones((len(experiment_list), times[k].size), dtype=float) * float("nan") for k in range(len(keys))]
    num = [np.zeros((len(experiment_list), times[k].size), dtype=int) for k in range(len(keys))]
    # Collect result Tensor
    for e, experiment in enumerate(experiment_list):
        for k in range(len(keys)):
            time_str = keys[k] + "_T"
            #max_time = min_time_found[k]
            if keys[k] in experiment['info'] and time_str in experiment['info']:
                eti = experiment['info'][time_str]
                val = experiment['info'][keys[k]]
                if isinstance(val[0], float):
                    for t in range(min(len(val), len(eti))):
                        if eti[t] <= times[k][-1] and eti[t] <= max_time_found[k]:
                            i = int(eti[t] / (times[k][-1] + 1) * times[k].size)
                            if num[k][e, i] == 0:
                                res[k][e, i] = val[t]
                            else:
                                res[k][e, i] += val[t]
                            num[k][e, i] += 1
                else:
                    print("No float list found in experiment '%s', key '%s'" % (experiment['config']['name'], keys[k]))
    # Normalize each observed data point
    for e in range(len(experiment_list)):
        for k in range(len(keys)):
            for i in range(num[k].shape[1]):
                if num[0][e, i] > 0:
                    res[k][e, i] /= num[k][e, i]
    # Return result and time
    return res, times


def plot_db_name(name, title=None, plot_individuals=":"):
    # Define things
    keys = ["Episode reward", "Episode length"]
    colors = ["blue", "green", "red", "cyan", "magenta"]
    # Retrieve experiments results from data base
    res, time = load_db(keys=keys, name=name)
    res = np.expand_dims(res, axis=2)
    # Plot results
    fig, ax = plt.subplots(len(keys))
    for i in range(len(keys)):
        gca = ax[i]
        draw_experiments(gca, time, res[:, i, 0, :], colors=colors, plot_individuals=plot_individuals)
        gca.set_xlabel("Environmental Steps")
        gca.set_xlim(time.min(), time.max())
        gca.set_ylabel(keys[i])
    # Set global title
    gca = ax[0]
    if 'title' is not None:
        gca.set_title(title)
    # Show figure
    plt.show()


def plot_db_compare(names, title=None, legend=None, keys=None, max_time=None, plot_individuals=None, pm_std=False,
                    use_sem=False, test=False, colors=None, ax=None, fill_in=False, longest_runs=0, min_time=0,
                    legend_pos=None, legend_plot=None, bin_size=200, refactored=False):
    # Definitions
    if keys is None:
        keys = ["Episode reward", "Episode length"]
    else:
        keys = deepcopy(keys)
    if len(keys) == 0:
        return
    if test:
        for i in range(len(keys)):
            if refactored:
                keys[i] = "test_" + keys[i]
            else:
                keys[i] = keys[i] + " test"
    if legend_pos is None:
        legend_pos = ['lower right', 'upper right']
    if legend_plot is None:
        legend_plot = [True, False]
    if colors is None:
        colors = ["blue", "green", "red", "black", "magenta", "cyan"]
    # Retrieve experiments results from data base
    res = []
    time = []
    max_time_found = [0 for _ in range(len(keys))]
    for i in range(len(names)):
        if refactored:
            res_i, time_i = load_refactored_db(keys=keys, name=names[i], test=test, max_time=max_time, fill_in=fill_in,
                                               bin_size=bin_size, min_time=min_time)
        else:
            res_i, time_i = load_db(keys=keys, name=names[i], test=test, max_time=max_time, fill_in=fill_in,
                                    longest_runs=longest_runs, bin_size=bin_size, min_time=min_time)
        if res_i is not None and time_i is not None:
            res.append(res_i)
            time.append(time_i)
            for k in range(len(keys)):
                max_time_found[k] = max(max_time_found[k], time_i[k][-1])

    # Make the figure
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(len(keys))
        show_plot = True
    else:
        ax = np.reshape(ax, np.size(ax))

    # Make legend
    if legend is not None:
        legend = legend.copy()
        for k in range(len(keys)):
            if len(keys) > 1:
                gca = ax[k]
            else:
                gca = ax
            if isinstance(gca, np.ndarray):
                gca = gca[0]
            for i in range(min(len(names), len(legend))):
                gca.fill_between(x=np.array([-2, -1]), y1=np.array([float('nan'), float('nan')]),
                                 y2=np.array([float('nan'), float('nan')]), alpha=0.2, linewidth=0,
                                 facecolor=colors[i % len(colors)])
                if k == 0:
                    if i < len(res):
                        legend[i] += " [%u]" % (res[i][0].shape[0])
            if legend_plot[k % len(legend_plot)]:
                gca.legend(legend, loc=legend_pos[k % len(legend_pos)])
    # Plot results
    for k in range(len(keys)):
        if max_time_found[k] > 0:
            gca = ax[k]
            for n in range(len(names)):
                if len(time) > n:
                    draw_experiments(gca, time[n][k], res[n][k][:, :], colors=[colors[n % len(colors)]],
                                     plot_individuals=plot_individuals, pm_std=pm_std, use_sem=use_sem)
            gca.set_xlabel("Environmental Steps")
            gca.set_xlim(0, max_time_found[k])
            gca.set_ylabel(keys[k] + (" (SEM)" if use_sem else " (STD)"))
    # Set global title
    gca = ax[0]
    if title is not None:
        gca.set_title(title)
    # Show figure
    if show_plot:
        plt.show()


def print_db_experiments(starting_with):
    # Open a mongo DB client
    db_url = "mongodb://pymarlOwner:EMC7Jp98c8rE7FxxN7g82DT5spGsVr9A@gandalf.cs.ox.ac.uk:27017/pymarl"
    db_name = "pymarl"
    try:
        client = pymongo.MongoClient(db_url, 27017, ssl=True, serverSelectionTimeoutMS=5)
        client.server_info()
    except pymongo.errors.ServerSelectionTimeoutError:
        print("No connection to MongoDB server!")
        return None, None
    # Scan the data base and collect all experiments starting_with a sting
    experiment_list = []
    experiment_nums = []
    for experiment in client[db_name]['runs'].find({'config.name': {'$regex': starting_with + '.*'}}):
        name = experiment['config']['name']
        if name not in experiment_list:
            experiment_list.append(name)
            experiment_nums.append(1)
        else:
            experiment_nums[experiment_list.index(name)] += 1
    # Print all found names
    for i in range(len(experiment_list)):
        print(experiment_list[i], "(%ux)" % experiment_nums[i])


def plot_db_comparison_and_test(names, title=None, legend=None, keys=None, max_time=None, plot_individuals=None,
                                pm_std=False, use_sem=False, colors=None, ax=None):
    fig, ax = fig, ax = plt.subplots(2, 2)
    for i in range(2):
        plot_db_compare(names, title, legend, keys, max_time, plot_individuals, pm_std, use_sem,
                        test=(i == 1), colors=colors, ax=ax[i])
    plt.show()

def plot_last_tests(ids=[0], max_time=20):
    (epi, format) = torch.load("results/last_test_run.torch")
    max_time = min(epi.shape[1], max_time)
    fig, ax = plt.subplots(len(ids), max_time)
    plt.axis('off')
    #plt.tick_params(
    #    axis='x',  # changes apply to the x-axis
    #    which='both',  # both major and minor ticks are affected
    #    bottom=False,  # ticks along the bottom edge are off
    #    top=False,  # ticks along the top edge are off
    #    labelbottom=False)  # labels along the bottom edge are off
    for i in range(len(ids)):
        for t in range(max_time):
            s = epi[ids[i], t, :].cpu().numpy().reshape(4, 4, 2)
            if np.any(s != s):
                break
            ax[i][t].imshow(-2*s[:, :, 0] - s[:, :, 1], interpolation='none', cmap=plt.get_cmap('gray'))
    for i in range(len(ids)):
        for t in range(max_time):
            ax[i][t].axes.get_xaxis().set_visible(False)
            ax[i][t].axes.get_yaxis().set_visible(False)
    plt.show()


def plot_db_final_performances(list_of_name_lists, x_axis, time_slice, title=None, legend=None, use_sem=False,
                               keys=None, ax=None, colors=None, x_label=None):
    if keys is None:
        keys = ["Episode reward"]
    if colors is None:
        colors = ['black', 'blue', 'cyan', 'green', 'orange', 'red', 'magenta']
    legend_pos = ['lower right', 'upper right']
    legend_plot = [True, False]
    # Load all experiments in list_of_name_lists
    res_list = []
    time_list = []
    for i in range(len(list_of_name_lists)):
        res_list.append([])
        time_list.append([])
        for j in range(len(list_of_name_lists[i])):
            res, time = load_db(keys, list_of_name_lists[i][j])
            res_list[i].append(res)
            time_list[i].append(time)
    # For each list, select the results in the given slice of all experiments in it
    m_performances = []
    s_performances = []
    for i in range(len(res_list)):
        m_performances.append(np.ones((len(res_list[i]), len(keys))) * float('nan'))
        s_performances.append(np.ones((len(res_list[i]), len(keys))) * float('nan'))
        for j in range(len(res_list[i])):
            # Select time-slice in current experiment
            is_slice = np.logical_and(time_list[i][j] >= time_slice[0], time_list[i][j] <= time_slice[1])
            # Average over all time-steps
            x = res_list[i][j][:, :, is_slice].mean(axis=2)
            # Compute mean and std for each key
            for k in range(len(keys)):
                m_performances[i][j, k] = np.nanmean(x[:, k])
                s_performances[i][j, k] = np.nanstd(x[:, k])
            # If needed, convert std into sem
            if use_sem:
                s_performances[i] /= x.shape[0] ** 0.5

    # Make the figure
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(len(keys))
        show_plot = True
    # Make legend
    if legend is not None:
        for k in range(len(keys)):
            if len(keys) > 1:
                gca = ax[k]
            else:
                gca = ax
            for i in range(len(legend)):
                gca.fill_between(x=np.array([-2, -1]), y1=np.array([float('nan'), float('nan')]),
                                 y2=np.array([float('nan'), float('nan')]), alpha=0.2, linewidth=0,
                                 facecolor=colors[i])
                if k == 0:
                    if i < len(res_list):
                        n = np.array([res_list[i][j].shape[0] for j in range(len(res_list[i]))])
                        legend[i] += (" [%u" % np.nanmin(n)) + "]" if np.nanmin(n) == np.nanmax(n) else "+]"
            if legend_plot[k % len(legend_plot)]:
                gca.legend(legend, loc=legend_pos[k % len(legend_pos)])
    # Plot each key
    for k in range(len(keys)):
        if isinstance(ax, list):
            gca = ax[k]
        else:
            gca = ax
        for i in range(len(m_performances)):
            print(x_axis.shape, m_performances[i][:, k].shape)
            gca.fill_between(x_axis, m_performances[i][:, k] - s_performances[i][:, k],
                             m_performances[i][:, k] + s_performances[i][:, k],
                             alpha=0.2, linewidth=0, facecolor=colors[i % len(colors)])
            gca.plot(x_axis, m_performances[i][:, k], color=colors[i % len(colors)])
        # add labels
        gca.set_ylabel(keys[k] + " (SEM)" if use_sem else " (STD)")
        if x_label is not None:
            gca.set_xlabel(x_label)
        if k == 0:
            if title is not None:
                gca.set_title(title)

    if show_plot:
        plt.show()


def plot_cumulative_rewards(list_of_names, title=None, legend=None, max_time=None, ax=None, colors=None,
                            test=True, use_sem=True, boxplot=True):
    if test:
        keys = ["Episode reward test"]
    else:
        keys = ["Episode reward"]
    # Prepare results
    mean_reward = np.zeros(len(list_of_names))
    std_reward = np.zeros(len(list_of_names))
    # Load all experiments
    res_list, time_list = [], []
    for i in range(len(list_of_names)):
        res, time = load_db(keys, list_of_names[i], max_time=max_time, bin_size=40, test=test)
        res_list.append(res)
        time_list.append(time)
        if res is not None:
            resi = np.squeeze(res, axis=1)
            resi = np.nanmean(resi, axis=1)
        else:
            resi = np.zeros(1)
        mean_reward[i] = np.mean(resi)
        std_reward[i] = np.std(resi) / (np.sqrt(len(resi)) if use_sem else 1)
    fig, ax = plt.subplots(1, 1)
    gca = ax
    if boxplot:
        # Do a box plot of the average reward
        ind = np.arange(0, len(list_of_names))
        bars = gca.bar(ind, mean_reward)
        gca.set_xticks(np.arange(0.5, len(list_of_names) + 0.5))
        gca.set_xticklabels(['p=%g' % legend[i] for i in range(len(legend))])
        gca.set_ylabel('Average reward in the first %u steps %s' % (max_time, '(SEM)' if use_sem else '(STD)'))
        gca.set_title(title)
        for i in range(len(bars)):
            bars[i].set_facecolor(colors[i % len(colors)])
            gca.plot(np.array([i + 0.4, i + 0.4]),
                     np.array([mean_reward[i] - std_reward[i], mean_reward[i] + std_reward[i]]),
                     color='black', linewidth=6)
    else:
        # Do an error-bar plot of the average reward
        gca.fill_between(legend, mean_reward - std_reward, mean_reward + std_reward,
                         alpha=0.2, linewidth=0, facecolor='blue')
        gca.plot(legend, mean_reward, color='blue')
        #gca.errorbar(legend, mean_reward, yerr=std_reward)
        gca.set_ylabel('Average reward in the first %u steps %s' % (max_time, '(SEM)' if use_sem else '(STD)'))
        gca.set_xlabel(title)
        gca.set_xlim(legend[0] - 0.05, legend[-1] + 0.05)
        gca.legend(['ICQL (0 iter) [16]'], loc='lower right')
    plt.show()

def print_all_names(starting_with):
    # Open a mongo DB client
    db_url = "mongodb://pymarlOwner:EMC7Jp98c8rE7FxxN7g82DT5spGsVr9A@gandalf.cs.ox.ac.uk:27017/pymarl"
    db_name = "pymarl"
    try:
        client = pymongo.MongoClient(db_url, 27017, ssl=True, serverSelectionTimeoutMS=5)
        client.server_info()
    except pymongo.errors.ServerSelectionTimeoutError:
        print("No connection to MongoDB server!")
        return None, None
    # Scan the data base and collect all experiments starting_with a sting
    experiment_list = []
    experiment_nums = []
    for experiment in client[db_name]['runs'].find({'config.name': {'$regex': starting_with + '.*'}}):
        name = experiment['config']['name']
        if name not in experiment_list:
            experiment_list.append(name)
            experiment_nums.append(1)
        else:
            experiment_nums[experiment_list.index(name)] += 1
    # Print all found names
    for i in range(len(experiment_list)):
        print(experiment_list[i], "(%ux)" % experiment_nums[i])


def plot_key_vs_key(names, keya, keyb, test=False, fill_in=True, colors=None, ax=None, title=None, legend=None,
                    later_than=0, the_last=0):
    the_last = int(the_last)
    if colors is None:
        colors = ['blue', 'red', 'green']
    res = []
    time = []
    for i in range(len(names)):
        res_i, time_i = load_db(keys=[keya, keyb], name=names[i], test=test, fill_in=fill_in)
        res.append(res_i[:, :, :])
        time.append(time_i)

    show = False
    if ax is None:
        fig, ax = plt.subplots(1)
        show = True

    for i in range(len(res)):
        x = res[i][:, 0, :]
        y = res[i][:, 1, :]
        if the_last == 0:
            for t in range(x.shape[1]):
                print(time[i][t], np.nanmax(time[i]))
                ax.scatter(x[:, t], y[:, t], c=colors[i % len(colors)], alpha=min(time[i][t]/np.nanmax(time[i]), 1.0), linewidths=0)
        else:
            x = np.mean(x[:, -the_last:-1], axis=1)
            y = np.mean(y[:, -the_last:-1], axis=1)
            ax.scatter(x, y)

    ax.set_xlabel(keya)
    ax.set_ylabel(keyb)
    if legend is not None:
        ax.legend(legend, loc='lower left')
    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()


cols = ['blue', 'green', 'red', 'black', 'magenta', 'cyan']
#print_db_experiments("wen_")
#plot_batch(ids=[9, 10, 11], plot_2d=False, plot_individuals=True, colors=cols)
#plot_batch(ids=[i+1 for i in range(5)], plot_2d=False, plot_individuals=True, colors=cols) # COMA on ppc4x4
# show_timelines([i+1 for i in range(5)])
#plot_comparison([[xxx], [yyy]], "Title of the comparison", legend=["xxx", "yyy"])
#plot_db_name(name='wen_4x4ppc_4agents_coma', title='COMA on 4x4 new predator-prey with 4 agents')
#plot_db_name(name='wen_4x4ppc_4agents_coma_noexplore', title='COMA on 4x4 new predator-prey with 4 agents')
#plot_db_compare(['wen_4x4ppc_4agents_coma_noexplore', 'wen_4x4ppc_4agents_coma', 'wen_4x4ppc_4agents_coma_explore100k',
#                 'wen_4x4ppc_4agents_coma_explore200k', 'wen_4x4ppc_4agents_coma_explore300k'],
#                title='COMA on 4x4 new 4 agent predator prey',
#                legend=['no exploration', '20k exploration',  '100k exploration', '200k exploration', '300k exploration'],
#                plot_individuals=None, pm_std=False, use_sem=True)
#plot_db_compare(['wen_4x4ppc_4agents_coma_explore100k', 'wen_4x4ppc_4agents_iql_noexplore',
#                 'wen_4x4ppc_4agents_iql_explore20k_to0.05', 'wen_4x4ppc_4agents_iql_explore100k',
#                 'wen_4x4ppc_4agents_iql_explore300k', 'wen_4x4ppc_4agents_iql_explore300k_to0.1'],
#                legend=['COMA (100k)', 'iQL (0k)', 'iQL(20k to 0.05)', 'iQL (100k)', 'iQL (300k)', 'iQL (300k to 0.1)'],
#                test=False, colors=['blue', 'magenta', 'red', 'green', 'orange', 'black', 'cyan'],
#                title='New 4 agent predator prey (4x4)', plot_individuals=None, pm_std=True, use_sem=True)
#plot_db_compare(['wen_7x7ppc_4agents_iql_default', 'wen_7x7ppc_4agents_coma_default'], legend=['iQL', 'COMA'],
#                title='New 7x7 4 predators prey', pm_std=True, use_sem=False, plot_individuals=':')
#plot_db_compare(['wen_7x7ppc_4agents_coma_4prey', 'wen_7x7ppc_4agents_iql_4prey_noexplore',
#                 'wen_7x7ppc_4agents_iql_4prey_longer', 'wen_7x7ppc_4agents_iql_4prey_explore1M'],
#                legend=['COMA (100k)', 'iQL (noexplore)', 'iQL (20k exp)', 'iQL (1M to 0.05)'],
#                title='New 7x7 4 predators 4 prey', pm_std=True, use_sem=True, plot_individuals=None, test=False)
#plot_db_compare(['wen_7x7ppc_4agents_iql_4prey_longer', 'wen_7x7ppc_4agents_icql_4prey_default'],
#                legend=['IQL', 'ICQL (10iter)'], test=True,
#                title='New 4x4 4 predators 1 prey', pm_std=True, use_sem=False, plot_individuals=':')
#plot_db_comparison_and_test(['wen_4x4ppc_4agents_iql_explore100k', 'wen_4x4ppc_4agents_icql_0iterations_explore100k_v2',
#                    'wen_4x4ppc_4agents_icql_10iterations_explore100k',
#                    'wen_4x4ppc_4agents_icql_10iterations_explore100k_v4'],
#                    legend=['IQL', 'ICQL (0)', 'ICQL (<10)', 'ICQL(<10,v4)'],
#                    title='New (4x4) 4 predators 1 prey, exploration 100k',
#                    pm_std=True, use_sem=True, plot_individuals='',
#                    colors=['blue', 'green', 'red',  'magenta',  'black', 'cyan'])
#plot_db_comparison_and_test(['wen_7x7ppc_4agents_coma_4prey', 'wen_7x7ppc_4agents_iql_4prey_longer',
#                             'wen_7x7ppc_4agents_4prey_icql_10iterations_explore100k'],
#                    legend=['COMA', 'IQL', 'ICQL (<10)'], title='New (7x7) 4 predators 4 prey, exploration 100k',
#                    pm_std=True, use_sem=True, plot_individuals='',
#plot_db_compare(['wen_4x4pp_4agents_1prey_iql_explore100k', 'wen_4x4pp_4agents_1prey_qmix_explore100k',
#                 'wen_4x4pp_4agents_1prey_vdn_explore100k'],
#                legend=['IQL', 'QMIX', 'VDN'],
#                title='New (4x4) 4 predators 1 prey', pm_std=True, use_sem=True, plot_individuals=':',
#                colors=['blue', 'green', 'red', 'magenta', 'black', 'cyan'])
#plot_db_compare(['wen_starcraft2_5m_iql_test', 'wen_starcraft2_5m_iql_baseline', 'wen_starcraft2_5m_iql_replicate'],
#                title='IQL on SC2 5m vs 5m', legend=['test', 'baseline', 'replicate'],
#                plot_individuals='--', pm_std=True, use_sem=True, test=False, max_time=3000000,
#                keys=["Episode reward", "Win rate"])
#                #keys=["Restarts", "Target q mean", "Battles won", "Win rate", "Episode length", "Episode reward"])
#plot_db_compare(['wen_7x7ppc_4predators_4prey_coma_default', 'wen_7x7ppc_4predators_4prey_iql_default'],
#                legend=['COMA', 'IQL'], title='New (7x7) capture with 4 predators 4 prey, exploration 100k',
#                pm_std=True, use_sem=True, plot_individuals='', test=False)
#plot_db_compare(['wen_4x4pp_4predators_1prey_iql_default', 'wen_4x4pp_4predators_1prey_iql_update',
#                 #'wen_4x4pp_4predators_1prey_pseudoiqn_default', 'wen_4x4pp_4predators_1prey_vdn_default',
#                 'wen_4x4pp_4predators_1prey_coma_default', 'wen_4x4pp_4predators_1prey_coma_mod',
#                 'wen_4x4pp_4predators_1prey_coma_fix', 'wen_4x4pp_4predators_1prey_coma_fix2'],
#                #legend=['true IQL', 'IQL in QMIX', 'VDN in QMIX', 'COMA', 'COMA2'],
#                legend=['IQL', 'IQL (updated)', 'COMA (broke)', 'COMA (broke)', 'COMA (fixed)', 'COMA (fixed)'],
#                title='New (4x4) capture with 4 predators 1 prey',
#                pm_std=False, use_sem=True, plot_individuals='', test=False,
#                colors=['magenta', 'red', 'green', 'black', 'blue', 'cyan'], keys=["Episode reward", "Episode length"])
#plot_db_compare(['wen_starcraft2_5m_iql_available_actions'],
#                legend=['IQL'], title='Starcraft 2 5m vs. 5m',
#                pm_std=False, use_sem=True, plot_individuals=':', test=False,)
#                #keys=["Restarts", "Target q mean", "Battles won", "Win rate", "Episode length", "Episode reward"])

plot_please = -1
if len(sys.argv) > 1:
    plot_please = int(sys.argv[1])

if plot_please == 0:
    plot_db_compare(['wen_6x6pp_4predators_1prey_coma_default', 'wen_6x6pp_4predators_1prey_coma_almostzero',
                     'wen_6x6pp_4predators_2prey_coma_default', 'wen_6x6pp_4predators_4prey_coma_default',
                     'wen_6x6pp_4predators_4prey_coma_almostzero', 'wen_6x6pp_4predators_4prey_coma_zero'],
                    legend=['1 prey', '1 prey (az)', '2 prey', '4 prey', '4 prey (az)', '4 prey (z)'],
                    title='COMA on 6x6 toroidal 4 predators (5x5 vision)',
                    pm_std=False, use_sem=True, plot_individuals=':', test=False, max_time=5E6)
if plot_please == 1:
    plot_db_compare(['wen_7x7pp_4predators_4prey_coma_default', 'wen_7x7pp_4predators_2prey_coma_default',
                     'wen_7x7pp_4predators_1prey_coma_default', 'wen_7x7pp_4predators_4prey_coma_almostzero',
                     'wen_7x7pp_4predators_4prey_coma_zero'],
                    legend=['4 prey', '2 prey', '1 prey', '4 prey (az)', '4 prey (z)'],
                    title='COMA on 7x7 toroidal 4 predators (5x5 vision)',
                    colors=['green', 'red', 'magenta', 'blue', 'black', 'cyan'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, max_time=5E6)

if plot_please == 2:
    plot_db_compare(['wen_4x4pp_4predators_1prey_coma_zero', 'wen_4x4pp_3predators_1prey_coma_zero',
                     'wen_4x4pp_2predators_1prey_coma_zero', 'wen_4x4pp_3predators_1prey_coma_rewone',
                     'wen_4x4pp_3predators_1prey_coma_onlycapture',],
                    legend=['4 agents', '3 agents', '2 agents', '3 agents, rew=1', '3 agents, only capture'],
                    title='COMA on 4x4 bounded (3x3 vision) 1 prey, no reward shaping',
                    pm_std=False, use_sem=True, plot_individuals=':', test=False, max_time=5E5)

if plot_please == 3:
    plot_last_tests([0, 3, 5, 6], max_time=13)

if plot_please == 4:
    plot_db_compare(['wen_4x4pp_2predators_1prey_coma_zero', 'wen_4x4pp_2predators_1prey_coma_realnew',
                     'wen_4x4pp_2predators_1prey_coma_new', 'wen_3x3pp_2predators_1prey_coma_new',],
                    legend=['4x4 (old)', '4x4 (new)', '3x3 (new)', '3x3 (new)'],
                    title='COMA 2 agents (3x3 vision) 1 prey',
                    pm_std=False, use_sem=True, plot_individuals=':', test=False, max_time=5E5,
                    colors=['green', 'blue', 'red', 'magenta', 'black', 'cyan'],)

if plot_please == 5:
    plot_db_compare(['wen_matrix_game_coma_default', 'wen_matrix_game_iql_default',
                     'wen_matrix_game_coma_debug', 'wen_matrix_game_iql_debug',
                     'wen_matrix_game_coma_original'],
                    legend=['COMA (obs5)', 'IQL(obs5)', 'COMA(obs1)', 'IQL(obs1)', 'COMA(epi_lim)'],
                    title='Matrix Game',
                    pm_std=False, use_sem=True, plot_individuals=':', test=False, max_time=1E5,
                    colors=['blue', 'red', 'green', 'magenta', 'black', 'cyan'],)

if plot_please == 6:
    plot_db_compare(['wen_matrix_game_coma_explore0k', 'wen_matrix_game_coma_explore1k',
                     'wen_matrix_game_coma_debug', 'wen_matrix_game_coma_explore50k'],
                    legend=['no exploration', '1k exploration', '10k exploration', '50k exploration'],
                    title='COMA on Matrix Game', keys=['Episode reward'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, max_time=1E5,
                    colors=['blue', 'red', 'green', 'magenta', 'black', 'cyan'],)

if plot_please == 7:
    plot_db_compare(['wen_matrix_game_coma_debug', 'wen_matrix_game_xxx_default'],
                    legend=['COMA', 'XXX'],
                    title='Matrix Game', keys=['Episode reward'],
                    pm_std=False, use_sem=True, plot_individuals=':', test=False, max_time=5E5,
                    colors=['blue', 'red', 'green', 'magenta', 'black', 'cyan'],)

if plot_please == 8:
    plot_db_compare(['wen_matrix_game_xxx_explore0k', 'wen_matrix_game_xxx_explore1k',
                     'wen_matrix_game_xxx_explore10k', 'wen_matrix_game_xxx_explore50k'],
                    legend=['no exploration', 'explore 1k', 'explore 10k', 'explore 50k'],
                    title='XXX on Matrix Game with p=0.9', keys=['Episode reward'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, max_time=1E5,
                    colors=['blue', 'red', 'green', 'magenta', 'black', 'cyan'])

if plot_please == 9:
    # old MACKRL/COMA on 3x3 pred-prey with reward shaping
    plot_db_compare(['CBEAR23', 'CBEAR33', 'XXXBEAR23', 'XXXBEAR33'],
                    legend=['COMA (2 agents)', 'COMA (3 agents)', 'MACKRL (2 agents)', 'MACKRL (3 agents)'],
                    title='3x3 Predator Prey', keys=['Episode reward'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, #max_time=5E6,
                    colors=['red', 'magenta', 'blue', 'green', 'black', 'cyan'], )

if plot_please == 10:
    # old MACKRL/COMA on 4x4 pred-prey with reward shaping
    plot_db_compare(['CBEAR24', 'CBEAR34', 'XXXBEAR24', 'XXXBEAR34',],
                    legend=['COMA (2 agents)', 'COMA (3 agents)', 'MACKRL (2 agents)', 'MACKRL (3 agents)'],
                    title='4x4 Predator Prey', keys=['Episode reward'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, #max_time=5E5,
                    colors=['red', 'magenta', 'blue', 'green', 'black', 'cyan'], )

if plot_please == 11:
    plot_db_compare(['wen_pp4x4_2predators_1prey_coma_cs_default',
                     'wen_pp4x4_2predators_1prey_coma_cs_otherdefault',
                     'wen_pp4x4_2predators_1prey_coma_cs_otherdefault_explore',
                     'wen_pp4x4_2predators_1prey_coma_cs_otherdefault_newexplore'],
                    legend=['reward shaping', 'new default', '0.5 < eps < 0.01', '0.99 < eps < 0.01'],
                    title='COMA 2 agents (3x3vision) 1 prey (4x4)',
                    pm_std=False, use_sem=True, plot_individuals=':', test=False, max_time=5E5,
                    colors=['green', 'blue', 'red', 'black', 'magenta', 'cyan'],)

if plot_please == 12:
    # old MACKRL on the Matrix Game for various p
    plot_db_compare(['wen_matrix_game_xxx_p0', 'wen_matrix_game_xxx_p2',
                     'wen_matrix_game_xxx_p4', 'wen_matrix_game_xxx_p5',
                     'wen_matrix_game_xxx_p6', 'wen_matrix_game_xxx_p8',
                     'wen_matrix_game_xxx_p10'],
                    legend=['p=0.0', 'p=0.2', 'p=0.4', 'p=0.5', 'p=0.6', 'p=0.8', 'p=1.0'],
                    title='MACKRL on Matrix Game', keys=['Episode reward'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, max_time=1E5,
                    colors=['black', 'blue', 'cyan', 'green', 'orange', 'red', 'magenta'])

if plot_please == 13:
    # old MACKRL/COMA on 3x3 pred-prey with no reward shaping
    plot_db_compare(['COMDELIN23', 'COMDELIN33', 'XXX2_23', 'XXX2_33'],
                    legend=['COMA (2 agents)', 'COMA (3 agents)', 'MACKRL (2 agents)', 'MACKRL (3 agents)'],
                    title='3x3 Predator Prey', keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=True, max_time=1E6,
                    colors=['red', 'magenta', 'blue', 'green', 'black', 'cyan'], )

if plot_please == 14:
    # old MACKRL/COMA on 4x4 pred-prey with no reward shaping
    plot_db_compare(['COMDELIN24', 'COMDELIN34', 'XXX2_24', 'XXX2_34',],
                    legend=['COMA (2 agents)', 'COMA (3 agents)', 'MACKRL (2 agents)', 'MACKRL (3 agents)'],
                    title='4x4 Predator Prey', keys=['Episode reward'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=True, max_time=1E6,
                    colors=['red', 'magenta', 'blue', 'green', 'black', 'cyan'], )

if plot_please == 15:
    # old MACKRL/COMA on 6x6 pred-prey with no reward shaping
    plot_db_compare(['CBEAR26', 'CBEAR36', 'XXXBEAR26', 'XXXBEAR36',],
                    legend=['COMA (2 agents)', 'COMA (3 agents)', 'MACKRL (2 agents)', 'MACKRL (3 agents)'],
                    title='6x6 Predator Prey (5x5 vision)', keys=['Episode reward'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, #max_time=5E5,
                    colors=['red', 'magenta', 'blue', 'green', 'black', 'cyan'], )

if plot_please == 16:
    # IQL on the Matrix Game for various p
    plot_db_compare(['wen_matrix_game_iql_p0', 'wen_matrix_game_iql_p2',
                     'wen_matrix_game_iql_p4', 'wen_matrix_game_iql_p5', 'wen_matrix_game_iql_p6',
                     'wen_matrix_game_iql_p8', 'wen_matrix_game_iql_p10',],
                    legend=['p=0.0', 'p=0.2', 'p=0.4', 'p=0.5', 'p=0.6', 'p=0.8', 'p=1.0'],
                    title='IQL on Matrix Game', keys=['Episode reward'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, max_time=1E5,
                    colors=['black', 'blue', 'cyan', 'green', 'orange', 'red', 'magenta'])

if plot_please == 17:
    # COMA on the Matrix Game for various p
    plot_db_compare(['wen_matrix_game_coma_p0', 'wen_matrix_game_coma_p2', 'wen_matrix_game_coma_p4',
                     'wen_matrix_game_coma_p5', 'wen_matrix_game_coma_p6',
                     'wen_matrix_game_coma_p8', 'wen_matrix_game_coma_p10'],
                    legend=['p=0.0', 'p=0.2', 'p=0.4', 'p=0.5', 'p=0.6', 'p=0.8', 'p=1.0'],
                    title='COMA on Matrix Game', keys=['Episode reward'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, max_time=1E5,
                    colors=['black', 'blue', 'cyan', 'green', 'orange', 'red', 'magenta'])

if plot_please == 18:
    plot_db_final_performances([['wen_matrix_game_xxx_p0', 'wen_matrix_game_xxx_p2',
                                 'wen_matrix_game_xxx_p4', 'wen_matrix_game_xxx_p5',
                                 'wen_matrix_game_xxx_p6', 'wen_matrix_game_xxx_p8',
                                 'wen_matrix_game_xxx_p10'],
                                ['wen_matrix_game_coma_p0', 'wen_matrix_game_coma_p2', 'wen_matrix_game_coma_p4',
                                 'wen_matrix_game_coma_p5', 'wen_matrix_game_coma_p6',
                                 'wen_matrix_game_coma_p8', 'wen_matrix_game_coma_p10'],
                                ['wen_matrix_game_iql_p0', 'wen_matrix_game_iql_p2', 'wen_matrix_game_iql_p4',
                                 'wen_matrix_game_iql_p5', 'wen_matrix_game_iql_p6', 'wen_matrix_game_iql_p8',
                                 'wen_matrix_game_iql_p10']],
                               legend=['old MACKRL', 'COMA', 'IQL'], colors=['blue', 'red', 'green'],
                               x_axis=np.array([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]), time_slice=[8E4, 10E4], use_sem=True,
                               title='Matrix Game Performance during 80k < t < 100k',
                               x_label='probability of common observation')


if plot_please == 19:
    # new MACKRL/COMA on 3x3 pred-prey with no reward shaping
    plot_db_compare(['CBEAR23', 'CBEAR33',  'XXXON_23', 'XXXON_33', ],
                    legend=['COMA (2 agents)', 'COMA (3 agents)', 'MACKRL (2 agents)', 'MACKRL (3 agents)'],
                    title='3x3 Predator Prey', keys=['Episode reward'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=1E6,
                    colors=['red', 'magenta', 'blue', 'green', 'black', 'cyan'], )


if plot_please == 20:
    # new MACKRL/COMA on 4x4 pred-prey with no reward shaping
    plot_db_compare(['CBEAR24', 'CBEAR34', 'XXXON_24', 'XXXON_34', ],
                    legend=['COMA (2 agents)', 'COMA (3 agents)', 'MACKRL (2 agents)', 'MACKRL (3 agents)'],
                    title='4x4 Predator Prey', keys=['Episode reward'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=1E6,
                    colors=['red', 'magenta', 'blue', 'green', 'black', 'cyan'], )

if plot_please == 21:
    # new MACKRL/COMA on 6x6 pred-prey with no reward shaping
    plot_db_compare(['CBEAR26', 'CBEAR36', 'XXXON_26', 'XXXON_36', 'COMDO_46_2', 'COMDO_56',
                     'wen_pp6x6_walls_coma_default'],
                    legend=['COMA (2 agents)', 'COMA (3 agents)', 'MACKRL (2 agents)', 'MACKRL (3 agents)',
                            'COMA (4 agents)', 'COMA (5 agents)', 'COMA (2 agents, walls)'],
                    title='6x6 Predator Prey', keys=['Episode reward'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=1E6,
                    colors=['red', 'magenta', 'blue', 'green', 'black', 'cyan', 'orange'], )

if plot_please == 22:
    # new MACKRL/COMA on Starcraft 2 with no reward shaping
    plot_db_compare(['COMAJAKOB', 'XXXJAKOB'],
                    legend=['COMA', 'MACKRL'],
                    title='Starcraft 3m3m', keys=['Episode reward', 'Win rate', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, #max_time=4E6,
                    colors=['red', 'blue', 'green', 'black', 'cyan'], )

if plot_please == 23:
    # new COMA on Starcraft 2 with no reward shaping
    plot_db_compare(['coma_jakob_sc2_3m_ngc', 'coma_jakob_sc2_5m_ngc', 'coma_jakob_sc2_2d3z_ngc',
                     'coma_jakob_sc2_2d3z_ngc_80lim', 'COMSC23'],
                    legend=['3m vs 3m', '5m vs 5m', '2d vs 3z', '2d vs 3z (80lim)', '2d vs 3z (80lim)'],
                    title='COMA on Starcraft2', keys=['Win rate', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, #max_time=2.5E5,
                    colors=['red', 'blue', 'green', 'black', 'cyan'], )

if plot_please == 24:
    # new MACKRL on Starcraft 2 with no reward shaping
    plot_db_compare(['XXXSC3', 'XXXSC5', 'XXXSC23'],
                    legend=['3m vs 3m', '5m vs 5m', '2d vs 3z'],
                    title='MACKRL on Starcraft2', keys=['Win rate', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, #max_time=4E6,
                    colors=['red', 'blue', 'green', 'black', 'cyan'], )

if plot_please == 25:
    # Feature view: MACKRL vs. COMA on Starcraft 2
    plot_db_compare(['XXXSC3', 'coma_jakob_sc2_3m_ngc'],
                    legend=['MACKRL', 'COMA'],
                    title='3m vs 3m Starcraft2', keys=['Win rate', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, #max_time=4E6,
                    colors=['red', 'blue', 'green', 'black', 'cyan'], )

if plot_please == 26:
    # 6x6 Predator prey
    plot_db_compare(['wen_pp6x6_walls_xxx_cluster', 'wen_pp6x6_walls_xxx_unknown', 'wen_pp6x6_walls_coma_cluster'],
                    legend=['MACKRL', 'MACKRL (unknown)', 'COMA' ],
                    title='6x6 predator prey', keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=True, max_time=1E6,
                    colors=['red', 'blue', 'green', 'black', 'cyan'], )

if plot_please == 27:
    # First Stag hunt
    plot_db_compare(['wen_staghunt4x4_3agents_xxx_default', 'wen_staghunt4x4_3agents_coma_default',
                     'wen_staghunt4x4_3agents_xxx_correct', 'wen_staghunt4x4_3agents_coma_correct',
                     'wen_staghunt6x6_3agents_coma_default',],
                    legend=['MACKRL (4x4, 2a)', 'COMA (4x4, 2a)', 'MACKRL (4x4, 3a)', 'COMA (4x4, 3a)', 'COMA (6x6, 2a)'],
                    title='The mighty Stag Hunt', keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=True, max_time=9.5E5,
                    colors=['blue', 'green', 'red', 'magenta', 'black', 'cyan'], )

if plot_please == 28:
    # 4x4 Collision Pred-prey
    plot_db_compare(['wen_pp4x4_3agents_xxx_collision01', 'wen_pp4x4_3agents_coma_collision01',
                     'wen_pp4x4_3agents_xxx_collision05', 'wen_pp4x4_3agents_coma_collision05',
                     'wen_pp4x4_3agents_xxx_collision1',  'wen_pp4x4_3agents_coma_collision1',],
                    legend=['MACKRL (rew -0.1)', 'COMA (rew -0.1)',
                            'MACKRL (rew -0.5)', 'COMA (rew -0.5)',
                            'MACKRL (rew -1.0)', 'COMA (rew -1.0)', ],
                    title='3 Predators 1 Prey (4x4 with collisions)', keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=True, max_time=9.5E5,
                    colors=['blue', 'black',  'red', 'magenta', 'green', 'cyan'], )

#plot_please = 29
if plot_please == 29:
    print('Second Stag hunt with 2 bunnies, p_rest=0.2 for both animals, intersection_global_view=False and new logit_bias!')
    plot_db_compare(['wen_staghunt4x4_3agents_coma_2bunnies', 'wen_staghunt4x4_3agents_xxx_2bunnies',
                     'wen_staghunt6x6_3agents_coma_2bunnies', 'wen_staghunt6x6_3agents_xxx_2bunnies'],
                    legend=['COMA (4x4)', 'MACKRL (4x4)', 'COMA (6x6)', 'MACKRL (6x6)'],
                    title='The mighty Stag Hunt with 3 agents and 2 bunnies', keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals=':', test=False, fill_in=True, max_time=9.5E5,
                    colors=['red', 'magenta', 'green', 'blue', 'black', 'cyan'], )

#plot_please = 30
if plot_please == 30:
    print('Third Stag hunt with only 1 bunny, but only reward=10 for the stag!')
    plot_db_compare(['wen_staghunt4x4_3agents_coma_evilreward', 'wen_staghunt4x4_3agents_xxx_evilreward',
                     'wen_staghunt6x6_3agents_coma_evilreward', 'wen_staghunt6x6_3agents_xxx_evilreward'],
                    legend=['COMA (4x4)', 'MACKRL (4x4)', 'COMA (6x6)', 'MACKRL (6x6)'],
                    title='The mighty Stag Hunt with 3 agents and evil reward', keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals=':', test=False, fill_in=True, max_time=9.5E5,
                    colors=['red', 'magenta', 'green', 'blue', 'black', 'cyan'], )

if plot_please == 31:
    # Starcraft 2 with logit_bias=1.5
    plot_db_compare(['coma_jakob_sc2_3m_ngc', 'XN3:1.5', 'coma_jakob_sc2_5m_ngc', 'XN5:1.5',
                     'coma_jakob_sc2_2d3z_ngc', 'XNN23:1.5',  ],
                    legend=['3m vs 3m COMA', '3m vs 3m MACKRL', '5m vs 5m COMA', '5m vs 5m MACKRL',
                            '2d vs 3z COMA', '2d vs 3z MACKRL'],
                    title='MACKRL on Starcraft2 (logit_bias=1.5)', keys=['Win rate', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=4E5,
                    colors=['red', 'magenta', 'blue', 'cyan', 'green', 'black'], )

if plot_please == 32:
    # Starcraft 2 with logit_bias=-1.5
    plot_db_compare(['coma_jakob_sc2_3m_ngc', 'XN3:-1.5', 'coma_jakob_sc2_5m_ngc', 'XN5:-1.5',
                     'coma_jakob_sc2_2d3z_ngc', 'XNN23:-1.5',  ],
                    legend=['3m vs 3m COMA', '3m vs 3m MACKRL', '5m vs 5m COMA', '5m vs 5m MACKRL',
                            '2d vs 3z COMA', '2d vs 3z MACKRL'],
                    title='MACKRL on Starcraft2 (logit_bias=-1.5)', keys=['Win rate', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=4E5,
                    colors=['red', 'magenta', 'blue', 'cyan', 'green', 'black'], )

if plot_please == 33:
    # Starcraft 2 with logit_bias=0.01
    plot_db_compare(['coma_jakob_sc2_3m_ngc', 'XN3:0.01', 'coma_jakob_sc2_5m_ngc', 'XN5:0.01',
                     'coma_jakob_sc2_2d3z_ngc', 'XNN23:0.01'],
                    legend=['3m vs 3m COMA', '3m vs 3m MACKRL', '5m vs 5m COMA', '5m vs 5m MACKRL',
                            '2d vs 3z COMA', '2d vs 3z MACKRL'],
                    title='MACKRL on Starcraft2 (logit_bias=0.01)', keys=['Win rate', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=4E5,
                    colors=['red', 'magenta', 'blue', 'cyan', 'green', 'black'], )

if plot_please == 34:
    # First good looking plot
    plot_db_compare(['coma_jakob_sc2_3m_ngc', 'XN3:1.5', 'XN3:-1.5'],
                    legend=['COMA', 'MACKRL (bias=1.5)', 'MACKRL (bias=-1.5)'],
                    title='[34] Starcraft2: 3m vs. 3m', keys=['Win rate', 'Level2 delegation rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=3.5E5,
                    colors=['blue', 'magenta', 'red', 'cyan', 'green', 'black'], longest_runs=6)

if plot_please == 35:
    # Starcraft 2 with logit_bias=1.5
    plot_db_compare(['coma_jakob_sc2_5m_ngc', 'XN5:1.5', 'XN5:-1.5'],
                    legend=['COMA', 'MACKRL (bias=1.5)', 'MACKRL (bias=-1.5)'],
                    title='[35] Starcraft2: 5m vs. 5m', keys=['Win rate', 'Level2 delegation rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=3.5E5,
                    colors=['blue', 'magenta', 'red', 'cyan', 'green', 'black'], longest_runs=4)

#plot_please = 36
if plot_please == 36:
    # Last shot on Starcraft 2 3m vs 3m
    plot_db_compare(['coma_jakob_sc2_3m_ngc',
                     'xxx_jakob_sc2_3m__lastshot_m1_16_ngc2', 'xxx_jakob_sc2_3m__lastshot_1_16_ngc2',
                     'xxx_jakob_sc2_3m__lastshot_m1_8_ngc2',  'xxx_jakob_sc2_3m__lastshot_1_8_ngc2'],
                    legend=['COMA', 'MACKRL (m1_16)', 'MACKRL (1_16)', 'MACKRL (m1_8)', 'MACKRL (1_8)'],
                    title='[36] Starcraft2: 3m vs. 3m (tabz)', keys=['Win rate', 'Level2 delegation rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=3.5E5,
                    colors=['blue', 'magenta', 'red', 'cyan', 'green', 'black'], longest_runs=0)

if plot_please == 37:
    # Last shot on Starcraft 2 5m vs 5m
    plot_db_compare(['coma_jakob_sc2_5m_ngc',
                     'xxx_jakob_sc2_5m__lastshot_m1_16_ngc2', 'xxx_jakob_sc2_5m__lastshot_1_16_ngc2',
                     'xxx_jakob_sc2_5m__lastshot_m1_8_ngc2',  'xxx_jakob_sc2_5m__lastshot_1_8_ngc2'],
                    legend=['COMA', 'MACKRL (m1_16)', 'MACKRL (1_16)', 'MACKRL (m1_8)', 'MACKRL (1_8)'],
                    title='[37] Starcraft2: 5m vs. 5m (tabz)', keys=['Win rate', 'Level2 delegation rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=3.5E5,
                    colors=['blue', 'magenta', 'red', 'cyan', 'green', 'black'], longest_runs=0)

#plot_please = 38
if plot_please == 38:
    # Last shot on Starcraft 2 3m vs 3m
    plot_db_compare(['coma_jakob_sc2_3m_ngc', 'XLS3_8', 'XLS3_16'],
                    legend=['COMA', 'MACKRL (8)', 'MACKRL (16)'],
                    title='[38] Starcraft2: 3m vs. 3m (cs)', keys=['Win rate', 'Level2 delegation rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=5E5,
                    colors=['blue', 'magenta', 'red', 'cyan', 'green', 'black'], longest_runs=0)

if plot_please == 39:
    # Last shot on Starcraft 2 5m vs 5m
    plot_db_compare(['coma_jakob_sc2_5m_ngc', 'XLS5_8', 'XLS5_16'],
                    legend=['COMA', 'MACKRL (8)', 'MACKRL (16)'],
                    title='[39] Starcraft2: 5m vs. 5m (cs)', keys=['Win rate', 'Level2 delegation rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=5E5,
                    colors=['blue', 'magenta', 'red', 'cyan', 'green', 'black'], longest_runs=15)

#plot_please = 40
if plot_please == 40:
    # Last shot on Starcraft 2 3m vs 3m
    plot_db_compare(['coma_jakob_sc2_3m_ngc', 'XXXLS3_8', 'XXXLS3_16', 'COC3_16'],
                    legend=['COMA (old)', 'MACKRL (8)', 'MACKRL (16)', 'COMA (16)'],
                    title='[40] Starcraft2: 3m vs. 3m (cs2)', keys=['Win rate', 'Level2 delegation rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=5E5,
                    colors=['blue', 'magenta', 'red', 'cyan', 'green', 'black'], longest_runs=0)

if plot_please == 41:
    # Last shot on Starcraft 2 5m vs 5m
    plot_db_compare(['coma_jakob_sc2_5m_ngc', 'XXXLS5_8', 'XXXLS5_16', 'COC5_16'],
                    legend=['COMA (old)', 'MACKRL (8)', 'MACKRL (16)', 'COMA (16)'],
                    title='[41] Starcraft2: 5m vs. 5m (cs2)', keys=['Win rate', 'Level2 delegation rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=5E5,
                    colors=['blue', 'magenta', 'red', 'cyan', 'green', 'black'], longest_runs=0)

# ------------- Plot rearrangement ----------------------------

#plot_please = 42
if plot_please == 42:
    # Last shot on Starcraft 2 3m vs 3m, frameskip 16
    plot_db_compare(['COC3_16', 'XXXLS3_16', 'XLS3_16'],
                    legend=['COMA', 'MACKRL-XXX', 'MACKRL-X'],
                    title='[42] Starcraft2: 3m vs. 3m (frameskip 16)',
                    keys=['Win rate', 'Level2 delegation rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=5E5,
                    colors=['blue', 'red', 'magenta', 'cyan', 'green', 'black'], longest_runs=0)

#plot_please = 43
if plot_please == 43:
    # Last shot on Starcraft 2 3m vs 3m, frameskip 8
    plot_db_compare(['coma_jakob_sc2_3m_ngc', 'XXXLS3_8', 'XLS3_8'],
                    legend=['COMA', 'MACKRL-XXX', 'MACKRL-X'],
                    title='[43] Starcraft2: 3m vs. 3m (frameskip 8)',
                    keys=['Win rate', 'Level2 delegation rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=5E5,
                    colors=['blue', 'red', 'magenta', 'cyan', 'green', 'black'], longest_runs=0)

#plot_please = 44
if plot_please == 44:
    # Last shot on Starcraft 2 5m vs 5m, frameskip 16
    plot_db_compare(['COC5_16', 'XXXLS5_16', 'XLS5_16'],
                    legend=['COMA', 'MACKRL-XXX', 'MACKRL-X'],
                    title='[44] Starcraft2: 5m vs. 5m (frameskip 16)',
                    keys=['Win rate', 'Level2 delegation rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=5E5,
                    colors=['blue', 'red', 'magenta', 'cyan', 'green', 'black'], longest_runs=0)

#plot_please = 45
if plot_please == 45:
    # Last shot on Starcraft 2 5m vs 5m, frameskip 8
    plot_db_compare(['coma_jakob_sc2_5m_ngc', 'XXXLS5_8', 'XLS5_8'],
                    legend=['COMA', 'MACKRL-XXX', 'MACKRL-X'],
                    title='[45] Starcraft2: 5m vs. 5m (frameskip 8)',
                    keys=['Win rate', 'Level2 delegation rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=5E5,
                    colors=['blue', 'red', 'magenta', 'cyan', 'green', 'black'], longest_runs=0)

#plot_please = 46
if plot_please == 46:
    xlim = 15E5
    # Starcraft results combined
    #fig, ax = plt.subplots(1, 4)
    #axes = [ax[0], ax[1], ax[2], ax[3]]
    fig, ax = plt.subplots(2, 2)
    axes = [ax[0][0], ax[0][1], ax[1][0], ax[1][1]]
    plot_db_compare(['coma_jakob_sc2_3m_ngc', 'XXXLS3_8', 'XLS3_8'],
                    legend=['COMA', 'MACKRL'],
                    title='Starcraft2: 3m vs. 3m (frameskip 8)',
                    keys=['Win rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=xlim,
                    colors=['blue', 'red', 'magenta', 'cyan', 'green', 'black'], longest_runs=0, ax=axes[0])
    plot_db_compare(['COC3_16', 'XXXLS3_16', 'XLS3_16'],
                    legend=['COMA', 'MACKRL'],
                    title='Starcraft2: 3m vs. 3m (frameskip 16)',
                    keys=['Win rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=xlim,
                    colors=['blue', 'red', 'magenta', 'cyan', 'green', 'black'], longest_runs=0, ax=axes[1])
    plot_db_compare(['coma_jakob_sc2_5m_ngc', 'XXXLS5_8', 'XLS5_8'],
                    legend=['COMA', 'MACKRL'],
                    title='Starcraft2: 5m vs. 5m (frameskip 8)',
                    keys=['Win rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=xlim,
                    colors=['blue', 'red', 'magenta', 'cyan', 'green', 'black'], longest_runs=0, ax=axes[2])
    plot_db_compare(['COC5_16', 'XXXLS5_16', 'XLS5_16'],
                    legend=['COMA', 'MACKRL'],
                    title='Starcraft2: 5m vs. 5m (frameskip 16)',
                    keys=['Win rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=xlim,
                    colors=['blue', 'red', 'magenta', 'cyan', 'green', 'black'], longest_runs=0, ax=axes[3])
    plt.show()

#plot_please = 47
if plot_please == 47:
    # Last shot on Starcraft 2 5m vs 5m
    xlim = 5E6
    #key = 'Win rate'
    key = 'Episode reward'
    #key = 'Level2 delegation rate'
    leg = ['COMA', 'MACKRL (-1)', 'MACKRL (+1)']
    test = True
    fig, ax = plt.subplots(2, 2)
    plot_db_compare(['coma_jakob_sc2_3m_ngc',
                     'xxx_jakob_sc2_3m__lastshot_m1_8_ngc3','xxx_jakob_sc2_3m__lastshot_1_8_ngc3', ],
                    legend=leg, title='Starcraft2: 3m vs. 3m (frameskip 8)', keys=[key],
                    pm_std=False, use_sem=True, plot_individuals='', test=test, fill_in=True, max_time=xlim,
                    colors=['blue', 'magenta', 'red', 'cyan', 'green', 'black'], longest_runs=0, ax=ax[0][0])
    plot_db_compare(['COC3_16', 'xxx_jakob_sc2_3m__lastshot_m1_16_ngc3','xxx_jakob_sc2_3m__lastshot_1_16_ngc3'],
                    legend=leg, title='Starcraft2: 3m vs. 3m (frameskip 16)', keys=[key],
                    pm_std=False, use_sem=True, plot_individuals='', test=test, fill_in=True, max_time=xlim,
                    colors=['blue', 'magenta', 'red', 'cyan', 'green', 'black'], longest_runs=0, ax=ax[0][1])
    plot_db_compare(['coma_jakob_sc2_5m_ngc',
                     'xxx_jakob_sc2_5m__lastshot_m1_8_ngc3','xxx_jakob_sc2_5m__lastshot_1_8_ngc3', ],
                    legend=leg, title='Starcraft2: 5m vs. 5m (frameskip 8)', keys=[key],
                    pm_std=False, use_sem=True, plot_individuals='', test=test, fill_in=True, max_time=xlim,
                    colors=['blue', 'magenta', 'red', 'cyan', 'green', 'black'], longest_runs=0, ax=ax[1][0])
    plot_db_compare(['COC5_16', 'xxx_jakob_sc2_5m__lastshot_m1_16_ngc3','xxx_jakob_sc2_5m__lastshot_1_16_ngc3', ],
                    legend=leg, title='Starcraft2: 5m vs. 5m (frameskip 16)', keys=[key],
                    pm_std=False, use_sem=True, plot_individuals='', test=test, fill_in=True, max_time=xlim,
                    colors=['blue', 'magenta', 'red', 'cyan', 'green', 'black'], longest_runs=0, ax=ax[1][1])
    plt.show()

#plot_please = 48
if plot_please == 48:
    keys = ['Episode reward', 'Episode length']#'Level2 delegation rate']
    plot_db_compare(['CBOX2', 'XBOX2:-1', 'XBOX2:0', 'XBOX2:1'],
                    legend=['COMA', 'MACKRL -1', 'MACKRL 0', 'MACKRL +1'],
                    title='Box Pushing', keys=keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, #max_time=xlim,
                    colors=['blue', 'magenta', 'red', 'cyan', 'green', 'black'], longest_runs=0)

#plot_please = 49
if plot_please == 49:
    fig, ax = plt.subplots(2, 2)
    later_than=0
    keys = ['Win rate test', 'Level2 delegation rate test']
    plot_key_vs_key(['xxx_jakob_sc2_3m__lastshot_m1_8_ngc3', 'xxx_jakob_sc2_3m__lastshot_1_8_ngc3'],
                    keys[0], keys[1], ax=ax[0, 0], later_than=later_than,
                    title='Starcraft2: 3m vs. 3m (frameskip 8)', legend=['-1', '+1'])
    plot_key_vs_key(['xxx_jakob_sc2_3m__lastshot_m1_16_ngc3', 'xxx_jakob_sc2_3m__lastshot_1_16_ngc3'],
                    keys[0], keys[1], ax=ax[0, 1], later_than=later_than,
                    title='Starcraft2: 3m vs. 3m (frameskip 8)', legend=['-1', '+1'])
    plot_key_vs_key(['xxx_jakob_sc2_5m__lastshot_m1_8_ngc3', 'xxx_jakob_sc2_5m__lastshot_1_8_ngc3'],
                    keys[0], keys[1], ax=ax[1, 0], later_than=later_than,
                    title='Starcraft2: 5m vs. 5m (frameskip 8)', legend=['-1', '+1'])
    plot_key_vs_key(['xxx_jakob_sc2_5m__lastshot_m1_16_ngc3', 'xxx_jakob_sc2_5m__lastshot_1_16_ngc3'],
                    keys[0], keys[1], ax=ax[1, 1], later_than=later_than,
                    title='Starcraft2: 5m vs. 5m (frameskip 8)', legend=['-1', '+1'])
    plt.show()

#plot_please = 50
if plot_please == 50:
    # Final plot
    xlim = 5E5
    #key = 'Win rate'
    key = 'Episode reward'
    #key = 'Level2 delegation rate'
    leg = ['MACKRL', 'COMA' ]
    test = True
    fig, ax = plt.subplots(1, 2)
    plot_db_compare(['xxx_jakob_sc2_5m__lastshot_1_8_ngc3', 'coma_jakob_sc2_5m_ngc',],
                    legend=leg, title='StarCraft II: 5m vs. 5m (skip rate 8)', keys=[key],
                    pm_std=False, use_sem=True, plot_individuals='', test=test, fill_in=True, max_time=xlim,
                    colors=['blue', 'red', 'cyan', 'green', 'black'], longest_runs=0, ax=ax[0])
    plot_db_compare(['xxx_jakob_sc2_5m__lastshot_1_16_ngc3', 'COC5_16',],
                    legend=leg, title='StarCraft II: 5m vs. 5m (skip rate 16)', keys=[key],
                    pm_std=False, use_sem=True, plot_individuals='', test=test, fill_in=True, max_time=xlim,
                    colors=['blue', 'red', 'magenta',  'cyan', 'green', 'black'], longest_runs=0, ax=ax[1])
    plt.show()

#plot_please = 51
if plot_please == 51:
    print('Fourth Stag hunt with reward=10 for the stag')
    plot_db_compare(['wen_staghunt6x6_3agents_coma_evilreward',
                     'wen_staghunt6x6_3agents_xxx_evilreward_faststag',
                     'wen_staghunt6x6_3agents_coma_evilreward_equallyfast',
                     'wen_staghunt6x6_3agents_xxx_evilreward_equallyfast',
                     'wen_staghunt6x6_3agents_coma_evilreward_fastbunny',
                     'wen_staghunt6x6_3agents_xxx_evilreward_fastbunny'],
                    legend=['COMA: p(s)=0.1, p(h)=0.5', 'MACKRL: p(s)=0.1, p(h)=0.5',
                            'COMA: p(s)=0.2, p(h)=0.2', 'MACKRL: p(s)=0.2, p(h)=0.2',
                            'COMA: p(s)=0.5, p(h)=0.1', 'MACKRL: p(s)=0.5, p(h)=0.1',],
                    title='The mighty 6x6 Stag Hunt with 3 agents and evil reward', keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False, max_time=9.5E5,
                    colors=['red', 'magenta', 'blue', 'cyan', 'black', 'green'], longest_runs=0,)

#plot_please = 52
if plot_please == 52:
    print('Fifth Stag hunt with reward=8 for the stag')
    plot_db_compare(['wen_staghunt6x6_3agents_coma_evilreward_fastbunny',
                     'wen_staghunt6x6_3agents_xxx_evilreward_fastbunny',
                     'wen_staghunt6x6_3agents_coma_stagrew8_fastbunny',
                     'wen_staghunt6x6_3agents_xxx_stagrew8_fastbunny'],
                    legend=['COMA: rew(s)=10', 'MACKRL: rew(s)=10', 'COMA: rew(s)=8', 'MACKRL: rew(s)=8'],
                    title='The mighty 6x6 Stag Hunt with 3 agents and p(s)=0.5 p(h)=0.1',
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False, max_time=9.5E5,
                    colors=['red', 'magenta', 'blue', 'green', 'cyan', 'black'], longest_runs=0,)

#plot_please = 53
if plot_please == 53:
    print('Sixth Stag hunt with equal observation/intersection spaces for everyone')
    plot_db_compare(['wen_staghunt6x6_3agents_xxx_allobs',
                     'wen_staghunt6x6_3agents_coma_allobs',
                     'wen_staghunt6x6_3agents_xxx_noidobs',
                     'wen_staghunt6x6_3agents_coma_noidobs',
                     'wen_staghunt6x6_3agents_xxx_evilreward_equallyfast',
                     'wen_staghunt6x6_3agents_coma_evilreward_equallyfast'],
                    legend=['MACKRL (see id)', 'COMA (see id)',
                            'MACKRL (no id)', 'COMA (no id)',
                            'MACKRL (unfair)', 'COMA (unfair)'],
                    title='The mighty 6x6 Stag Hunt with 3 agents and p(s)=0.2 p(h)=0.2',
                    keys=['Episode reward', 'Episode length', 'Level2 delegation rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False, max_time=9.5E5,
                    colors=['blue', 'green', 'magenta', 'red', 'cyan', 'black'], longest_runs=0,
                    legend_pos=['', 'upper right', ''], legend_plot=[False, True, False])

#plot_please = 54
if plot_please == 54:
    print('Excerpt from the sixth stag hunt with equal observation/intersection spaces for everyone')
    plot_db_compare(['wen_staghunt6x6_3agents_xxx_allobs', 'wen_staghunt6x6_3agents_coma_allobs',
                     #'wen_staghunt6x6_3agents_xxx_noidobs', 'wen_staghunt6x6_3agents_coma_noidobs',
                     'wen_staghunt6x6_3agents_xxx_evilreward_equallyfast',
                     'wen_staghunt6x6_3agents_coma_evilreward_equallyfast'],
                    legend=['MACKRL (see id)', 'COMA (see id)',
                            'MACKRL (old)', 'COMA (old)'],
                    title='The mighty 6x6 Stag Hunt with 3 agents',
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False, max_time=9.5E5,
                    colors=['blue', 'green', 'magenta', 'red', 'cyan', 'black'], longest_runs=0,)

#plot_please = 55
if plot_please == 55:
    print('False-positive stag hunt to see why it differs so much after the last merge.')
    plot_db_compare(['wen_staghunt6x6_3agents_xxx_allobs', 'wen_staghunt6x6_3agents_coma_allobs',
                     'wen_staghunt6x6_3agents_xxx_nowalls_noids', 'wen_staghunt6x6_3agents_coma_nowalls_noids',
                     'wen_staghunt6x6_3agents_coma_evilreward_equallyfast',
                     'wen_staghunt6x6_3agents_xxx_nounknown_nowalls_noids'
                    ],
                    legend=['MACKRL (see walls, see id)', 'COMA (see walls, see id)',
                            'MACKRL (no walls, no id)', 'COMA (no walls, no id)',
                            'COMA (old)', 'MACKRL (no walls, no id, no unknown)'],
                    title='The mighty 6x6 Stag Hunt with 3 agents',
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False, max_time=9.5E5,
                    colors=['blue', 'green', 'magenta', 'red', 'black', 'cyan'], longest_runs=0,)

#plot_please = 56
if plot_please == 56:
    print('Stag hunt that mimics predator-prey (all observations).')
    plot_db_compare(['wen_staghunt6x6_3agents_xxx_allobs', 'wen_staghunt6x6_3agents_coma_allobs',
                     'wen_staghunt6x6_3agents_xxx_predprey', 'wen_staghunt6x6_3agents_coma_predprey',
                     #'CBEAR36'
                    ],
                    legend=['MACKRL (1 bunny)', 'COMA (1 bunny)',
                            'MACKRL (no bunny)', 'COMA (no bunny)',
                            #'COMA (old pred-prey)'
                            ],
                    title='The mighty 6x6 Stag Hunt with 3 agents (all observations)',
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=True, max_time=9.5E5,
                    colors=['blue', 'green', 'magenta', 'red', 'black', 'cyan'], longest_runs=0,
                    legend_pos=['', 'upper right'], legend_plot=[False, True])

#plot_please = 57
if plot_please == 57:
    print('Stag hunt with one-hot features.')
    plot_db_compare(['wen_staghunt6x6_3agents_xxx_onehot_no_walls_noids',
                     'wen_staghunt6x6_3agents_coma_onehot_no_walls_noids',
                     'wen_staghunt6x6_3agents_xxx_onehot_default',
                     'wen_staghunt6x6_3agents_coma_onehot_default',
                    ],
                    legend=['MACKRL (no walls, no ids)', 'COMA (no walls, no ids)',
                            'MACKRL (see walls, see ids)', 'COMA (see walls, see ids)'],
                    title='The mighty 6x6 Stag Hunt with 3 agents (one-hot features)',
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False, max_time=9.5E5,
                    colors=['blue', 'green', 'magenta', 'red', 'black', 'cyan'], longest_runs=0,
                    legend_pos=['', 'upper right'], legend_plot=[False, True])

#plot_please = 58
if plot_please == 58:
    print('False-positive stag hunt to see why it differs so much after the last merge.')
    plot_db_compare(['wen_staghunt6x6_3agents_xxx_nowalls_noids',
                     'wen_staghunt6x6_3agents_coma_nowalls_noids',
                     'wen_staghunt6x6_3agents_xxx_evilreward_equallyfast',
                     'wen_staghunt6x6_3agents_coma_evilreward_equallyfast',
                    ],
                    legend=['MACKRL (no walls, no id)', 'COMA (no walls, no id)',
                            'MACKRL (old)', 'COMA (old)'],
                    title='The mighty 6x6 Stag Hunt with 3 agents',
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False, max_time=9.5E5,
                    colors=['blue', 'green', 'magenta', 'red', 'black', 'cyan'], longest_runs=0,)

#plot_please = 59
if plot_please == 59:
    print('Stag hunt that mimics predator prey')
    plot_db_compare(['wen_staghunt6x6_3agents_xxx_onehot_2s0h_seewall_seeid',
                     'wen_staghunt6x6_3agents_coma_onehot_2s0h_seewall_seeid',
                     'wen_staghunt6x6_3agents_xxx_onehot_2s0h_nowall_noid',
                     'wen_staghunt6x6_3agents_coma_onehot_2s0h_nowall_noid',
                    ],
                    legend=['MACKRL (see walls, see id)', 'COMA (see walls, see id)',
                            'MACKRL (no walls, no id)', 'COMA (no walls, no id)'],
                    title='The mighty 6x6 Stag Hunt with 3 agents, no hares, but 2 stags',
                    keys=['Episode reward', 'Level2 delegation rate', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False, max_time=9.5E5,
                    colors=['blue', 'green', 'magenta', 'red', 'black', 'cyan'], longest_runs=0,)

#plot_please = 60
if plot_please == 60:
    print('Starcraft 2 plot with MACKRL-V')
    plot_db_compare(['xxx_jakob_sc2_3m__lastshot_1_8_ngc3', 'XVSC2_3_8', 'coma_jakob_sc2_3m_ngc',
                     #'XVSC2_5_8', 'coma_jakob_sc2_5m_ngc'
                     ],
                    legend=['MACKRL-COMA (3m vs 3m)', 'MACKRL-V (3m vs 3m)', 'COMA (3m vs 3m)',
                            #'MACKRL-V (5m vs 5m)', 'COMA (5m vs 5m)'
                            ],
                    title='Starcraft II (skip frame 8)',
                    keys=['Win rate', 'Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=2.25E6,
                    colors=['blue', 'cyan', 'red', 'magenta', 'red', 'black'], longest_runs=0,
                    legend_pos=['', '', 'upper right'], legend_plot=[False, False, True])

#plot_please = 61
if plot_please == 61:
    print('Starcraft 2 plot with various hyper parameters of MACKRL-COMA ')
    plot_db_compare(['xxx_jakob_sc2_3m__lastshot_1_8_ngc3', 'XBS8', 'XLRA', 'XLRC',
                     'coma_jakob_sc2_3m_ngc',
                     ],
                    legend=['MACKRL-COMA (NIPS)', 'MACKRL-COMA (smaller batch)', 'MACKRL-COMA (larger agent LR)',
                            'MACKRL-COMA (larger critic LR)', 'COMA (NIPS)' ],
                    title='Starcraft II (3m vs. 3m, skip frame 8)',
                    keys=['Win rate', 'Level2 delegation rate', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=2.3E6,
                    colors=['blue', 'magenta', 'green', 'cyan', 'red', 'orange'], longest_runs=4,
                    legend_pos=['lower right'], legend_plot=[True, False, False])

#plot_please = 62
if plot_please == 62:
    print('First Starcraft 1 plot (defunct)')
    plot_db_compare(['LUCKYPUNCH4'],
                    legend=['COMA'],
                    title='Starcraft 1 (5m vs. 5m)',
                    keys=['Win rate', 'Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=True, max_time=1E6,
                    colors=['blue', 'magenta', 'green', 'cyan', 'red', 'orange'], longest_runs=0,
                    legend_pos=['lower right'], legend_plot=[False, True, False])

#plot_please = 63
if plot_please == 63:
    print('New money-shot plot of Starcraft 2')
    plot_db_compare(['XBS8',
                     #'XLRA',
                     'coma_jakob_sc2_3m_ngc',],
                    legend=['MACKRL', 'COMA'],
                    title='Starcraft II (3m vs. 3m, skip frame 8)',
                    keys=['Win rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=2.3E6,
                    colors=['blue', 'red'], longest_runs=0, bin_size=200,
                    legend_pos=['lower right'], legend_plot=[True, False])

#plot_please = 64
if plot_please == 64:
    print('Redone money-shot plot of Starcraft 2')
    plot_db_compare(['MONEYSHOT',
                     'CMONEYSHOT',],
                    legend=['MACKRL', 'COMA'],
                    title='Starcraft II (3m vs. 3m, skip frame 8)',
                    keys=['Win rate', 'Level2 delegation rate'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=1.3E6,
                    colors=['blue', 'red'], longest_runs=15, bin_size=200,
                    legend_pos=['lower right'], legend_plot=[True, False])

#plot_please = 65
if plot_please == 65:
    print('Starcraft 2 3m vs 3m')
    plot_db_compare(['XXXFO2_3', 'COMAFO3_3'],
                    legend=['MACKRL', 'COMA'],
                    title='Starcraft II (3m vs. 3m)',
                    keys=['Win rate', 'Episode reward', 'Level2 delegation rate', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=2.4E6,
                    colors=['blue', 'red'], longest_runs=15, bin_size=200,
                    legend_pos=['lower right'], legend_plot=[True, False, False, False])

#plot_please = 66
if plot_please == 66:
    print('Starcraft 2 5m vs 5m')
    plot_db_compare(['XXXFO2_5', 'COMAFO2_5'],
                    legend=['MACKRL', 'COMA'],
                    title='Starcraft II (5m vs. 5m)',
                    keys=['Win rate', 'Episode reward', 'Level2 delegation rate', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=True, fill_in=True, max_time=1.55E6,
                    colors=['blue', 'red'], longest_runs=15, bin_size=200,
                    legend_pos=['lower right'], legend_plot=[True, False, False, False])

#plot_please = 67
if plot_please == 67:
    print('Short test to see the delegation rates in stag hunt')
    fig,ax = plt.subplots(2, 3)
    for i in range(2):
        plot_db_compare(['wen_staghunt_xxx_test'],
                    legend=['MACKRL'],
                    title='Stag hunt test',
                    keys=['Episode reward', 'Level2 delegation rate', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals=':', test=i==0, fill_in=False, #max_time=1.55E6,
                    colors=['blue', 'red'], longest_runs=0, bin_size=200,
                    legend_pos=['lower right'], legend_plot=[False, True, False], ax=ax[i, :])
    plt.show()

#plot_please = 68
if plot_please == 68:
    print('Test in 4x4 pred-prey to see whether or not IQL still works')
    fig,ax = plt.subplots(2, 2)
    for i in range(2):
        plot_db_compare(['wen_pp6x6_iql_default', 'wen_pp4x4_iql_explore100k',
                         'wen_pp4x4_iql_targetupdate10', 'wen_pp4x4_iql_tu_explore50k', 'wen_pp4x4_iql_tu_explore100k'],
                    legend=['IQL (default)', 'IQL (explore 100k)',
                            'IQL (up x10, exp 20k)', 'IQL (up x10, exp 50k', 'IQL (up x10, exp 100k)'],
                    title='4 predators (3x3 obs) hunting 1 prey in 4x4 env' if i==0 else None,
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, #max_time=1.55E6,
                    colors=['magenta', 'red', 'green', 'blue', 'black'], longest_runs=0, bin_size=100,
                    legend_pos=['lower right'], legend_plot=[i==0, False], ax=ax[i, :])
    plt.show()

#plot_please = 69
if plot_please == 69:
    print('Another test in 6x6 pred-prey to see whether or not IQL still works')
    fig,ax = plt.subplots(2, 2)
    for i in range(2):
        plot_db_compare(['wen_pp6x6_iql_explore50k', 'wen_pp6x6_iql_explore100k',
                         'wen_pp6x6_2_prey_iql_explore50k', 'wen_pp6x6_2_prey_iql_explore100k',
                         'wen_pp6x6_2_prey_coma_realexplore50k', 'wen_pp6x6_2_prey_coma_explore50k'],
                    legend=['IQL (1 prey, exp 50k)', 'IQL (1 prey, exp 100k)',
                            'IQL (2 prey, exp 50k)', 'IQL (2 prey, exp 100k)',
                            'COMA (2 prey, exp 50k)', 'COMA (2 prey, exp 100k)'],
                    title='4 predators (5x5 obs) hunting in 6x6 env' if i==0 else None,
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, #max_time=1.55E6,
                    colors=['magenta', 'red', 'cyan', 'green', 'blue', 'black'], longest_runs=0, bin_size=100,
                    legend_pos=['lower left'], legend_plot=[False, i==0, False], ax=ax[i, :])
    plt.show()

#plot_please = 70
if plot_please == 70:
    print('Starcraft2 3m vs 3m and 5m vs 5m')
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate', 'Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'test': True, 'fill_in': True, 'bin_size': 100}
    colors = ['blue', 'green', 'red']
    fig, ax = plt.subplots(4, 2)
    plot_db_compare(['XMON3', 'XXMONBIAS3', 'COMAFO3_3'], legend=['MACKRL', 'MACKRL (bias)', 'COMA'],
                    title='Starcraft II (3m vs. 3m)', keys=keys,
                    colors=colors, longest_runs=0, ax=[ax[i][0] for i in range(4)], max_time=1.1E6,
                    legend_pos=['lower right'], legend_plot=[True, False, False, False], **kwargs)
    plot_db_compare(['XMON5', 'XXMON5BIAS', 'COMAFO2_5'], legend=['MACKRL', 'MACKRL (bias)', 'COMA'],
                    title='Starcraft II (5m vs. 5m)', keys=keys,
                    colors=colors, longest_runs=8, ax=[ax[i][1] for i in range(4)], max_time=1.1E6,
                    legend_pos=['lower right'], legend_plot=[True, False, False, False], **kwargs)
    plt.show()

#plot_please = 71
if plot_please == 71:
    print('First test of IQL in the ICQL implementation on 6x6 pred-prey')
    fig,ax = plt.subplots(2, 2)
    for i in range(2):
        plot_db_compare(['wen_pp6x6_2_prey_iql_is_default', 'wen_pp6x6_2_prey_iql_cs_default',
                         'wen_pp6x6_2_prey_iql_cs_in10', 'wen_pp6x6_2_prey_iql_cs_hidden_in10',
                         'wen_pp6x6_2_prey_iql_cs_hidden_lr10', 'wen_pp6x6_2_prey_iql_cs_hidden_lr5_ni2'],
                    legend=['IQL (decentral explore)', 'IQL (central explore)',
                            'IQL (central, n_i=10)', 'IQL (central, hidden, n_i=10)',
                            'IQL (central, hidden, lr_i x10)', 'IQL (central, hidden, lr_i x5, n_i=2)'],
                    title='4 predators (5x5 obs) hunting 2 prey in 6x6 env' if i==0 else None,
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, #max_time=1.55E6,
                    colors=['red', 'green', 'cyan', 'blue', 'black', 'magenta'], longest_runs=0, bin_size=100,
                    legend_pos=['lower left'], legend_plot=[False, i==0, False], ax=ax[i, :])
    plt.show()

#plot_please = 72
if plot_please == 72:
    print('Second test of IQL in the ICQL implementation on 6x6 pred-prey')
    fig,ax = plt.subplots(2, 2)
    for i in range(2):
        plot_db_compare(['wen_pp6x6_2prey_iql_original_default_double_080618',
                         'wen_pp6x6_2prey_iql_is_default_double_080618',
                         'wen_pp6x6_2prey_iql_cs_double_080618',
                         'wen_pp6x6_2prey_iql_cs_lrc10_double_080618',
                         'wen_pp6x6_2_prey_coma_realexplore50k'],
                    legend=['IQL (original)', 'IQL (new code)', 'IQL (central explore)', 'IQL (cen_exp, lr_c x10)',
                            'COMA'],
                    title='4 predators (5x5 obs) hunting 2 prey in 6x6 env' if i==0 else None,
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, #max_time=1.55E6,
                    colors=['red', 'green', 'blue', 'black', 'cyan', 'magenta'], longest_runs=0, bin_size=100,
                    legend_pos=['lower left'], legend_plot=[False, i==0, False], ax=ax[i, :])
    plt.show()

#plot_please = 73
if plot_please == 73:
    print('Third test of IQL in the ICQL implementation on 4x4 pred-prey')
    fig,ax = plt.subplots(2, 2)
    for i in range(2):
        plot_db_compare(['wen_pp4x4_1prey_iql_is_default_110618',
                         'wen_pp4x4_1prey_iql_cs_default_110618',
                         'wen_pp4x4_1prey_iql_cs_nohidden_110618',
                         'wen_pp4x4_1prey_cql_cs_nohidden_110618',
                         'wen_pp4x4_1prey_cql_real_nohidden_110618',
                         'wen_pp4x4_1prey_cql_regression_nohidden_120618'],
                    legend=['IQL (IQL explore)', 'IQL (ICQL explore)',
                            'IQL (ICQL, no hidden)', 'IQL (ICQL, no hidden)',
                            'CQL (IQL learn, no hid.)', 'CQL (regression, no hid.)'],
                    title='4 predators (3x3 obs) hunting 1 prey in 4x4 env' if i==0 else None,
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, #max_time=1.55E6,
                    colors=['red', 'magenta', 'cyan', 'blue', 'black', 'green'], longest_runs=0, bin_size=100,
                    legend_pos=['lower left'], legend_plot=[False, i==0, False], ax=ax[i, :])
    plt.show()

#plot_please = 74
if plot_please == 74:
    print('Third test of IQL in the ICQL implementation on 4x4 pred-prey')
    fig,ax = plt.subplots(2, 2)
    for i in range(2):
        plot_db_compare(['wen_pp4x4_1prey_iql_is_default_110618',
                         #'wen_pp4x4_1prey_cql_regression_nohidden_120618',
                         'wen_pp4x4_1prey_coma_120618',
                         'wen_pp4x4_cql_iql_0iter_140618',
                         'wen_pp4x4_cql_iql_1iter_140618',
                         'wen_pp4x4_cql_iql_2iter_140618',
                         'wen_pp4x4_cql_iql_4iter_140618',
                         'wen_pp4x4_cql_iql_20iter_140618'],
                    legend=['IQL', 'COMA', 'CQL (iql, 0 iter)', 'CQL (iql, 1 iter)',
                            'CQL (iql, 2 iter)', 'CQL (iql, 4 iter)', 'CQL (iql, 20 iter)'],
                    title='4 predators (3x3 obs) hunting 1 prey in 4x4 env' if i==0 else None,
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, #max_time=1.55E6,
                    colors=['red', 'magenta', 'green', 'cyan', 'blue', 'black', 'orange'], longest_runs=0, bin_size=100,
                    legend_pos=['upper right'], legend_plot=[False, i==0, False], ax=ax[i, :])
    plt.show()

#plot_please = 75
if plot_please == 75:
    print('IQL  anc CQL in the ICQL implementation, for fully observable 4x4 pred-prey')
    fig, ax = plt.subplots(2, 3)
    colors = ['red', 'green', 'blue', 'magenta', 'black', 'cyan',  'orange']
    for i in range(2):
        plot_db_compare(['wen_pp4x4_fully_iql_defaultr_140618',
                         'wen_pp4x4_fully_cql_1iter_140618',
                         'wen_pp4x4_fully_icql_iql_1iter_140618',
                         'wen_pp4x4_fully_icql_reg_1iter_140618',
                         'wen_pp4x4_fully_coma_150618'],
                    legend=['IQL', 'CQL (iql, 1 iter)', 'ICQL (iql, 1 iter)', 'ICQL (reg, 1 iter)', 'COMA'],
                    title='4 predators (fully observable) hunting 1 prey in 4x4 env' if i==0 else None,
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, #max_time=1.55E6,
                    colors=colors, longest_runs=0, bin_size=100,
                    legend_pos=['upper right'], legend_plot=[False, i==0, False], ax=ax[i, 0:2])
    plot_db_compare(['wen_pp4x4_fully_iql_defaultr_140618',
                     'wen_pp4x4_fully_cql_1iter_140618',
                     'wen_pp4x4_fully_icql_iql_1iter_140618',
                     'wen_pp4x4_fully_icql_reg_1iter_140618',
                     'wen_pp4x4_fully_coma_150618'],
                    legend=['IQL', 'CQL (iql, 1 iter)', 'ICQL (iql, 1 iter)', 'ICQL (reg, 1 iter)', 'COMA'],
                    title='4 predators (fully observable) hunting 1 prey in 4x4 env' if i == 0 else None,
                    keys=['Individual loss', 'Central loss'],
                    pm_std=False, use_sem=True, plot_individuals='', test=(i == 0), fill_in=False,  # max_time=1.55E6,
                    colors=colors, longest_runs=0, bin_size=100,
                    legend_pos=['upper right'], legend_plot=[False, i == 0, False], ax=ax[:, 2])
    plt.show()

#plot_please = 76
if plot_please == 76:
    print('IQL  anc CQL in the ICQL implementation, for fully observable 6x6 pred-2prey')
    fig,ax = plt.subplots(2, 2)
    for i in range(2):
        plot_db_compare(['wen_pp6x6_2prey_fully_iql_default_140618',
                         'wen_pp6x6_2prey_fully_cql_iql_1iter_140618',
                         'wen_pp6x6_2prey_fully_icql_iql_1iter_140618',
                         'wen_pp6x6_2prey_fully_icql_reg_1iter_140618',
                         'wen_pp6x6_2prey_fully_coma_140618'],
                    legend=['IQL', 'CQL (iql, 1 iter)', 'ICQL (iql, 1 iter)', 'ICQL (reg, 1 iter)', 'COMA'],
                    title='4 predators (fully observable) hunting 2 prey in 6x6 env' if i==0 else None,
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, #max_time=1.55E6,
                    colors=['red', 'green', 'blue', 'black', 'cyan', 'magenta', 'orange'], longest_runs=0, bin_size=100,
                    legend_pos=['lower left'], legend_plot=[False, i==0, False], ax=ax[i, :])
    plt.show()

#plot_please = 77
if plot_please == 77:
    print('ICQL regressions for fully observable 4x4 pred-prey')
    fig, ax = plt.subplots(2, 3)
    colors = ['red', 'magenta', 'green', 'blue', 'cyan', 'orange', 'black']
    experiments = ['wen_pp4x4_fully_iql_defaultr_140618',
                   'wen_pp4x4_fully_icql_iql_1iter_140618',
                   'wen_pp4x4_fully_icql_reg_1iter_140618',
                   'wen_pp4x4_fully_icql_reg_nonstat_1iter_140618',
                   'wen_pp4x4_fully_icql_reg_nonstat_restrict_1iter_140618',
                   'wen_pp4x4_fully_icql_reg_restrict_1iter_140618',
                   'wen_pp4x4_fully_icql_reg_nonstat_4updates_1iter_150618']
    for i in range(2):
        plot_db_compare(experiments,
                    legend=['IQL', 'ICQL (iql)', 'ICQL (reg, stat, non-rest)', 'ICQL (reg, non-stat, non-rest)',
                            'ICQL (reg, non-stat, rest)', 'ICQL (reg, stat, rest)', 'ICQL (reg, n-s, n-r, ni=4)',],
                    title='4 predators (fully observable) hunting 1 prey in 4x4 env' if i==0 else None,
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, #max_time=1.55E6,
                    colors=colors, longest_runs=0, bin_size=100,
                    legend_pos=['upper right'], legend_plot=[False, i==0, False], ax=ax[i, 0:2])
    plot_db_compare(experiments, keys=['Individual loss', 'Central loss'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  # max_time=1.55E6,
                    colors=colors, longest_runs=0, bin_size=100,
                    legend_pos=['upper right'], legend_plot=[False, False], ax=ax[:, 2])
    plt.show()

#plot_please = 78
if plot_please == 78:
    print('ICQL with IQL for fully observable 4x4 pred-prey')
    fig, ax = plt.subplots(2, 3)
    colors = ['red', 'green', 'blue', 'magenta', 'cyan', 'orange', 'black']
    experiments = ['wen_pp4x4_fully_iql_defaultr_140618',
                   'wen_pp4x4_fully_icql_iql_1iter_140618',
                   'wen_pp4x4_fully_icql_iql_hidden_140618']
    for i in range(2):
        plot_db_compare(experiments,
                    legend=['IQL', 'ICQL (iql, state)', 'ICQL (iql, state+hidden)'],
                    title='4 predators (fully observable) hunting 1 prey in 4x4 env' if i==0 else None,
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, #max_time=1.55E6,
                    colors=colors, longest_runs=0, bin_size=100,
                    legend_pos=['upper right'], legend_plot=[False, i==0, False], ax=ax[i, 0:2])
    plot_db_compare(experiments, keys=['Individual loss', 'Central loss'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  # max_time=1.55E6,
                    colors=colors, longest_runs=0, bin_size=100,
                    legend_pos=['upper right'], legend_plot=[False, False], ax=ax[:, 2])
    plt.show()

#plot_please = 79
if plot_please == 79:
    print('ICQL with IQL for partially observable 4x4 pred-prey')
    fig, ax = plt.subplots(2, 5)
    colors = ['red', 'magenta', 'green', 'blue', 'orange', 'cyan', 'black' ]
    experiments = ['wen_pp4x4_iql_default_180618',
                   'wen_pp4x4_iql_icql_1iter_190618',
                   #'wen_pp4x4_icql_iql_1iter_180618',
                   'wen_pp4x4_icql_iql_1iter_190618',
                   'wen_pp4x4_icql_iql_1iter_lossact_190618',
                   #'wen_pp4x4_icql_iql_1iter_hidden_180618',
                   #'wen_pp4x4_icql_iql_1iter_hidden_190618',
                   #'wen_pp4x4_icql_iql_1iter_onlyhidden_180618'
                   ]
    legend = ['IQL (cql, state)', 'IQL (icql, state)', 'ICQL (iql, state)', 'ICQL (iql, new)',
              #'ICQL (iql, state+hidden)', 'ICQL (iql, state+hidden)', 'ICQL (iql, hidden)'
             ]
    for i in range(2):
        plot_db_compare(experiments,
                    title='4 predators (3x3 obs) hunting 1 prey in 4x4 env' if i==0 else None,
                    keys=['Episode reward', 'Episode length'],
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, #max_time=1.55E6,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[i, 0:2])
    plot_db_compare(experiments, keys=['Individual loss', 'Central loss'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  # max_time=1.55E6,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[:, 2],
                    legend_pos=['upper right'], legend_plot=[False, True], legend=legend)
    plot_db_compare(experiments, keys=['Target individual q mean', 'Target central q mean',
                                       'Target average q diff', 'Qvalues entropy'],
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  # max_time=1.55E6,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[:, 3:5])
    #plot_db_compare(experiments, keys=['Target average q diff', 'Qvalues entropy'],
    #                pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  # max_time=1.55E6,
    #                colors=colors, longest_runs=0, bin_size=100, ax=ax[:, 4])
    plt.show()

#plot_please = 80
if plot_please == 80:
    print('IQL with ICQL for partially observable 4x4 pred-prey with reduced simultaneous envs')
    colors = ['red', 'magenta', 'green', 'cyan', 'orange', 'blue', 'black']
    experiments = ['wen_pp4x4_iql_cql_200618', 'wen_pp4x4_iql_icql_200618',
                   'wen_pp4x4_cql_iql_200618', 'wen_pp4x4_cql_iql_10iter_200618', 'wen_pp4x4_cql_iql_mix0.5_200618',
                   'wen_pp4x4_icql_iql_200618', 'wen_pp4x4_icql_iql_mix0.5_200618']
    legend = ['IQL (cql)', 'IQL (icql)', 'CQL (iql)', 'CQL (10 iter)', 'CQL (iql, 0.5)', 'ICQL (iql)', 'ICQL (iql, 0.5)']
    max_time = 9E5
    test_keys = ['Episode reward']
    rest_keys = ['Target individual q mean', 'Target central q mean']
    rest_legend = [True, False]
    # test_keys = ['Episode reward', 'Episode length']
    # rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
    #              'Central loss', 'Target central q mean', 'Qvalues entropy']
    # rest_legend = [False, False, True, False, False, False]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2, wide)
    for i in range(2):
        plot_db_compare(experiments,
                    title='4 predators (3x3 obs) hunting 1 prey in 4x4 env' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=7, bin_size=100, ax=ax[i, 0:len(test_keys)])
    plot_db_compare(experiments, keys=rest_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                    colors=colors, longest_runs=7, bin_size=100, ax=ax[:, len(test_keys):wide],
                    legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 81
if plot_please == 81:
    print('Starcraft2 3m vs 3m and 5m vs 5m')
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate', 'Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'test': True, 'fill_in': True, 'bin_size': 100}
    colors = ['blue', 'green', 'red']
    fig, ax = plt.subplots(4, 2)
    plot_db_compare(['FLOK6_3m', 'XXMONBIAS3', 'CMONEYSHOT'], legend=['FLOUNDRL', 'MACKRL (bias)', 'COMA'],
                     title='Starcraft II (3m vs. 3m)', keys=keys,
                     colors=colors, longest_runs=10, ax=[ax[i][0] for i in range(4)], max_time=None, #1.1E6,
                     legend_pos=['lower right'], legend_plot=[True, False, False, False], **kwargs)
    plot_db_compare(['FLOK6_5m', 'XXMON5BIAS', 'CJAK2'], legend=['FLOUNDRL', 'MACKRL (bias)', 'COMA'],
                     title='Starcraft II (5m vs. 5m)', keys=keys,
                     colors=colors, longest_runs=10, ax=[ax[i][1] for i in range(4)], max_time=None, #2.1E6,
                     legend_pos=['lower right'], legend_plot=[True, False, False, False], **kwargs)
    plt.show()

#plot_please = 82
if plot_please == 82:
    print('IQL with ICQL for partially observable 4x4 pred-prey as in 80, but with 200k exploration steps')
    colors = ['red', 'green', 'cyan', 'blue', 'black', 'magenta', 'orange']
    experiments = ['wen_pp4x4_iql_icql_explore200k_210618',
                   'wen_pp4x4_cql_iql_explore200k_210618', 'wen_pp4x4_cql_iql_mix0.5_explore200k_210618',
                   'wen_pp4x4_icql_iql_explore200k_210618', 'wen_pp4x4_icql_iql_mix0.5_explore200k_210618',
                   'wen_pp4x4_cql_iql_mix0.5_explore200k_v2_210618', 'wen_pp4x4_icql_iql_mix0.5_explore200k_v2_210618']
    legend = ['IQL (icql)', 'CQL (iql, 0.0)', 'CQL (iql, 0.5, bug)', 'ICQL (iql)', 'ICQL (iql, 0.5, bug)',
              'CQL (iql, 0.5)', 'ICQL (iql, 0.5)']
    max_time = None # 9E5
    if True:
        test_keys = ['Episode reward']
        rest_keys = ['Target individual q mean', 'Target central q mean']
        rest_legend = [True, False]
    else:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, True, False, False, False]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2, wide)
    for i in range(2):
        plot_db_compare(experiments,
                    title='4 predators (3x3 obs) hunting 1 prey in 4x4 env' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[i, 0:len(test_keys)])
    plot_db_compare(experiments, keys=rest_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                    legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 83
if plot_please == 83:
    print('IQL with ICQL for partially observable 4x4 pred-prey as in 80, but with 200k exploration steps')
    colors = ['red', 'green',  'cyan', 'blue', 'black', 'magenta', 'orange']
    experiments = ['wen_pp4x4_iql_icql_nobuffer_220618',
                   'wen_pp4x4_cql_iql_nobuffer_220618',
                   'wen_pp4x4_cql_iql_nobuffer_mix0.5_220618',
                   'wen_pp4x4_icql_iql_nobuffer_220618',
                   'wen_pp4x4_icql_iql_nobuffer_mix0.5_220618',
                   'wen_pp4x4_icql_reg_nobuffer_nonstat_220618',
                   'wen_pp4x4_icql_reg_nobuffer_nonstat_mix0.5_220618']
    legend = ['IQL (icql)', 'CQL (iql, 0.0)', 'CQL (iql, 0.5)', 'ICQL (iql, 0.0)', 'ICQL (iql, 0.5)',
              'ICQL (reg, 0.0)', 'ICQL (reg, 0.5)']
    max_time = None # 9E5
    if False:
        test_keys = ['Episode reward']
        rest_keys = ['Target individual q mean', 'Target central q mean']
        rest_legend = [True, False]
    else:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2, wide)
    for i in range(2):
        plot_db_compare(experiments,
                    title='4 predators (3x3 obs) hunting 1 prey in 4x4 env' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[i, 0:len(test_keys)])
    plot_db_compare(experiments, keys=rest_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                    legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 84
if plot_please == 84:
    print('Rerun of COMA experiment for exploration')
    colors = ['blue', 'red', 'green',  'cyan', 'black', 'magenta', 'orange']
    if False:
        experiments = ['wen_4p1p4x4_coma_noexplore_250618',
                       'wen_4p1p4x4_coma_explore100k_250618',
                       'wen_4p1p4x4_coma_explore200k_250618',]
        n_agents = 4
    elif False:
        experiments = ['wen_4p1p4x4_coma_noexplore_v2_250618',
                       'wen_4p1p4x4_coma_explore100k_v2_250618',
                       'wen_4p1p4x4_coma_explore200k_v2_250618', ]
        n_agents = 4
    elif True:
        experiments = ['wen_2p1p4x4_coma_noexplore_long_250618',
                       'wen_2p1p4x4_coma_explore100k_long_250618',
                       'wen_2p1p4x4_coma_explore200k_long_250618', ]
        n_agents = 2
    else:
        experiments = ['wen_3p1p4x4_coma_noexplore_250618',
                       'wen_3p1p4x4_coma_explore100k_250618',
                       'wen_3p1p4x4_coma_explore200k_250618', ]
        n_agents = 3
    legend = ['no exploration', '100k exploration', '200k exploration']
    max_time = None # 9E5
    if True:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = []
        rest_legend = [True, False]
    else:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2, wide)
    for i in range(2):
        plot_db_compare(experiments,
                    title=('%u COMA predators (3x3 obs) hunting 1 prey in 4x4 env' % n_agents) if i==0 else None,
                    keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[i, 0:len(test_keys)],
                    legend=legend, legend_pos=['lower right'], legend_plot=[i==0, False])
    if len(rest_keys) > 0:
        plot_db_compare(experiments, keys=rest_keys,
                        pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                        colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                        legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 85
if plot_please == 85:
    print('VDN with ICQL for partially observable 4x4 2pred-1prey')
    colors = ['red', 'magenta', 'cyan', 'green', 'blue', 'black', 'orange']
    experiments = ['wen_2p1p4x4_iql_icql_250618',
                   'wen_2p1p4x4_vdn_icql_250618',
                   'wen_2p1p4x4_icql_iql_250618',
                   'wen_2p1p4x4_icql_iql_mix0.5_250618',
                   'wen_2p1p4x4_icql_vdn_250618',
                   'wen_2p1p4x4_icql_vdn_mix0.5_250618']
    legend = ['IQL (icql)', 'VDN (icql)', 'ICQL (iql, 0.0)', 'ICQL (iql, 0.5)', 'ICQL (vdn, 0.0)', 'ICQL (vdn, 0.5)']
    max_time = None # 9E5
    if False:
        test_keys = ['Episode reward']
        rest_keys = ['Target individual q mean', 'Target central q mean']
        rest_legend = [True, False]
    else:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2, wide)
    for i in range(2):
        plot_db_compare(experiments,
                    title='2 predators (3x3 obs) hunting 1 prey in 4x4 env' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[i, 0:len(test_keys)])
    plot_db_compare(experiments, keys=rest_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=10000, ax=ax[:, len(test_keys):wide],
                    legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 86
if plot_please == 86:
    print('VDN with ICQL for partially observable 4x4 2pred-1prey')
    colors = ['red', 'magenta', 'cyan', 'green', 'blue', 'black', 'orange']
    experiments = ['wen_2p1p4x4_iql_icql_gam9_270618',
                   'wen_2p1p4x4_vdn_icql_gam9_270618',
                   'wen_2p1p4x4_icql_iql_gam9_250618',
                   'wen_2p1p4x4_icql_iql_gam9_mix5_270618',
                   'wen_2p1p4x4_icql_vdn_gam9_250618',
                   'wen_2p1p4x4_icql_vdn_gam9_mix5_270618']
    legend = ['IQL (icql 0.0, g0.9)', 'VDN (icql 0.0, g0.9)', 'ICQL (iql 0.0, g0.9)', 'ICQL (iql 0.5, g0.9)',
              'ICQL (vdn 0.0, g0.9)', 'ICQL (vdn 0.5, g0.9)']
    max_time = None # 9E5
    if False:
        test_keys = ['Episode reward']
        rest_keys = ['Target individual q mean', 'Target central q mean']
        rest_legend = [True, False]
    else:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2, wide)
    for i in range(2):
        plot_db_compare(experiments,
                    title='2 predators (3x3 obs) hunting 1 prey in 4x4 env' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[i, 0:len(test_keys)])
    plot_db_compare(experiments, keys=rest_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                    legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 87
if plot_please == 87:
    print('COMA on the new predator prey tasks')
    colors = ['red', 'cyan', 'magenta', 'blue', 'green', 'black',  'orange']
    experiments = ['wen_4p1p6x6_bounded_coma_explore100k_long_280618',
                   'wen_4p1p6x6_toroidal_coma_explore100k_long_280618',
                   'wen_4p1p4x4_bounded_coma_explore100k_long_290618',
                   'wen_4p1p4x4_toroidal_coma_explore100k_long_290618',
                   'wen_4p1p4x4_toroidal_almostcap_coma_explore100k_long_290618',
                   'wen_4p1p4x4_toroidal_nowalls_coma_explore100k_long_290618']
    legend = ['6x6 (bounded)', '6x6 (toroidal)', '4x4 (bounded)', '4x4 (toroidal)',
              '4x4 (tor., almost_cap)', '4x4 (tor., nowalls)']
    max_time = None # 9E5
    if True:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = []
        rest_legend = [True, False]
    else:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2, wide)
    for i in range(2):
        plot_db_compare(experiments,
                    title='4 COMA agents on pred-prey-new' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[i, 0:len(test_keys)],
                    legend_pos=['lower right'], legend_plot=[i==0, False], legend=legend)
    if len(rest_keys) > 0:
        plot_db_compare(experiments, keys=rest_keys,
                        pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                        colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                        legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 88
if plot_please == 88:
    print('ICQL for partially observable 4x4 3pred-1prey')
    colors = ['magenta', 'green', 'red', 'blue', 'black', 'cyan', 'orange']
    experiments = ['wen_3p1p4x4_bounded_iql_icql_030718',
                   'wen_3p1p4x4_bounded_icql_iql_030718',
                   'wen_3p1p4x4_bounded_iql_icql_hidden_030718',
                   'wen_3p1p4x4_bounded_icql_iql_hidden_030718']
    legend = ['IQL (icql 0.0)', 'ICQL (iql 0.0)', 'IQL (icql 0.0, hid)', 'ICQL (iql 0.0, hid)', ]
    max_time = None # 9E5
    if False:
        test_keys = ['Episode reward']
        rest_keys = ['Target individual q mean', 'Target central q mean']
        rest_legend = [True, False]
    else:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2, wide)
    for i in range(2):
        plot_db_compare(experiments,
                    title='3 predators (3x3 obs) hunting 1 prey in 4x4 env' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[i, 0:len(test_keys)])
    plot_db_compare(experiments, keys=rest_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                    legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 89
if plot_please == 89:
    print('ICQL for partially observable non-truncated 4x4 3pred-1prey')
    colors = ['magenta', 'green', 'red', 'blue', 'black', 'cyan', 'orange']
    experiments = ['wen_3p1p4x4_bounded_iql_icql_notrunc_030718',
                   'wen_3p1p4x4_bounded_icql_iql_notrunc_030718',
                   'wen_3p1p4x4_bounded_iql_icql_notrunc_hidden_030718',
                   'wen_3p1p4x4_bounded_icql_iql_notrunc_hidden_030718']
    legend = ['IQL (icql 0.0)', 'ICQL (iql 0.0)', 'IQL (icql 0.0, hid)', 'ICQL (iql 0.0, hid)', ]
    max_time = None # 9E5
    if False:
        test_keys = ['Episode reward']
        rest_keys = ['Target individual q mean', 'Target central q mean']
        rest_legend = [True, False]
    else:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2, wide)
    if wide == 1:
        ax = np.expand_dims(ax, axis=1)
    for i in range(2):
        plot_db_compare(experiments,
                    title='3 predators (3x3 obs) hunting 1 prey in 4x4 env (not tuncated)' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[i, 0:len(test_keys)])
    if len(rest_keys) > 0:
        plot_db_compare(experiments, keys=rest_keys,
                        pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                        colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                        legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 90
if plot_please == 90:
    print('ICQL for partially observable non-truncated 4x4 3pred-1prey')
    colors = ['magenta', 'green', 'red', 'blue', 'black', 'cyan', 'orange']
    experiments = ['wen_3p1p4x4_bounded_iql_icql_notrunc_debug_030718',
                   'wen_3p1p4x4_bounded_icql_iql_notrunc_debug_030718',
                   'wen_3p1p4x4_bounded_iql_icql_notrunc_hidden_debug_030718',
                   'wen_3p1p4x4_bounded_icql_iql_notrunc_hidden_debug_030718']
    legend = ['IQL (icql 0.0)', 'ICQL (iql 0.0)', 'IQL (icql 0.0, hid)', 'ICQL (iql 0.0, hid)', ]
    max_time = None # 9E5
    if False:
        test_keys = ['Episode reward']
        rest_keys = ['Target individual q mean', 'Target central q mean']
        rest_legend = [True, False]
    else:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2, wide)
    if wide == 1:
        ax = np.expand_dims(ax, axis=1)
    for i in range(2):
        plot_db_compare(experiments,
                    title='3 predators (3x3 obs) hunting 1 prey in 4x4 env (not tuncated)' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[i, 0:len(test_keys)])
    if len(rest_keys) > 0:
        plot_db_compare(experiments, keys=rest_keys,
                        pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                        colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                        legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 91
if plot_please == 91:
    print('ICQL (iql uses available actions) for partially observable 4x4 3pred-1prey')
    colors = ['magenta', 'green', 'blue', 'black', 'cyan', 'orange']
    experiments = ['wen_3p1p4x4_bounded_iql_icql_010818',
                   'wen_3p1p4x4_bounded_icql_iql_010818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.5_010818']
    legend = ['IQL (icql 0.0)', 'ICQL (iql 0.0)', 'ICQL (iql 0.5)' ]
    max_time = None # 9E5
    if False:
        test_keys = ['Episode reward']
        rest_keys = ['Target individual q mean', 'Target central q mean']
        rest_legend = [True, False]
    else:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2, wide)
    if wide == 1:
        ax = np.expand_dims(ax, axis=1)
    for i in range(2):
        plot_db_compare(experiments,
                    title='3 predators (3x3 obs) hunting 1 prey in 4x4 env' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[i, 0:len(test_keys)])
    if len(rest_keys) > 0:
        plot_db_compare(experiments, keys=rest_keys,
                        pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                        colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                        legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 92
if plot_please == 92:
    print('ICQL (iql uses available actions in learner and controller) for partially observable 4x4 3pred-1prey')
    colors = ['red', 'green', 'blue', 'black', 'magenta', 'cyan', 'orange']
    experiments = [#'wen_3p1p4x4_bounded_iql_icql_fix3_010818',
                   'wen_3p1p4x4_bounded_iql_only_fix3_020818',
                   'wen_3p1p4x4_bounded_icql_iql_fix3_010818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.5_fix3_010818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.5_fix3_10iter_010818',
                   #'wen_3p1p4x4_bounded_coma_only_020818',
                   'wen_3p1p4x4_bounded_coma_fix1_020818',
                    'wen_3p1p4x4_bounded_icql_iql_mix0.5_hidden_nostate_030818']
    legend = ['IQL (iql)', 'ICQL (iql 0.0, 0iter)', 'ICQL (iql 0.5, 0iter)', 'ICQL (iql 0.5, 10iter)',
              #'COMA (coma)',
              'COMA (coma, lr)', 'ICQL (iql 0.5, nostate)']
    max_time = None  # 1.3E6
    if True:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = []
        rest_legend = [True, False]
        test_legend = [False, True]
    else:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
        test_legend = [False]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2, wide)
    if wide == 1:
        ax = np.expand_dims(ax, axis=1)
    for i in range(2):
        plot_db_compare(experiments,
                    title='3 predators (3x3 obs) hunting 1 prey in 4x4 env' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[i, 0:len(test_keys)],
                    legend_plot=(test_legend if i==1 else [False]), legend=legend)
    if len(rest_keys) > 0:
        plot_db_compare(experiments, keys=rest_keys,
                        pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                        colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                        legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 93
if plot_please == 93:
    print('ICQL for partially observable 6x6 4pred-1prey')
    colors = ['red', 'green', 'blue', 'black', 'cyan', 'orange']
    experiments = ['wen_4p1p6x6_bounded_iql_only_020818',
                   'wen_4p1p6x6_bounded_cql_iql_mix0.5_020818',
                   'wen_4p1p6x6_bounded_icql_iql_mix0.5_020818']
    legend = ['IQL (iql)', 'CQL (iql 0.5)', 'ICQL (iql 0.5)']
    max_time = None  # 1E6
    if True:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = []
        rest_legend = [True, False]
        test_legend = [False, True]
    else:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
        test_legend = [False]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2, wide)
    if wide == 1:
        ax = np.expand_dims(ax, axis=1)
    for i in range(2):
        plot_db_compare(experiments,
                    title='4 predators (5x5 obs) hunting 1 prey in 6x6 env' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=100, ax=ax[i, 0:len(test_keys)],
                    legend_plot=test_legend if i==1 else [False], legend=legend)
    if len(rest_keys) > 0:
        plot_db_compare(experiments, keys=rest_keys,
                        pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                        colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                        legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 94
if plot_please == 94:
    print('ICQL (iql uses available actions in learner and controller) for partially observable 4x4 3pred-1prey')
    colors = ['red', 'blue', 'orange',
              'magenta', 'black', 'green', 'cyan'
              ]
    experiments = ['wen_3p1p4x4_bounded_iql_only_fix3_020818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.5_fix3_010818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.5_hidden_020818',
                   'wen_3p1p4x4_bounded_vdn_only_020818',
                   'wen_3p1p4x4_bounded_icql_vdn_mix0.5_020818',
                   'wen_3p1p4x4_bounded_icql_vdn_mix0.5_hidden_020818'
                   ]
    legend = ['IQL (iql)', 'ICQL (iql 0.5)', 'ICQL (iql 0.5, hidden)',
              'VDN (vdn)', 'ICQL (vdn 0.5)', 'ICQL (vdn 0.5, hidden)'
              ]
    max_time = None  # 3.5E5
    if True:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = []
        rest_legend = [True, False]
        test_legend = [False, True]
    else:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
        test_legend = [False]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2, wide)
    if wide == 1:
        ax = np.expand_dims(ax, axis=1)
    for i in range(2):
        plot_db_compare(experiments,
                    title='3 predators (3x3 obs) hunting 1 prey in 4x4 env' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=40, ax=ax[i, 0:len(test_keys)],
                    legend_plot=(test_legend if i==1 else [False]), legend=legend)
    if len(rest_keys) > 0:
        plot_db_compare(experiments, keys=rest_keys,
                        pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                        colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                        legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 95
if plot_please == 95:
    print('Parameter search for fixed ICQL mixture ratio (4x4 3pred-1prey)')
    colors = ['magenta', 'red', 'darkorange', 'y', 'lime', 'green', 'c', 'deepskyblue', 'blue', 'gray', 'black']
    experiments = ['wen_3p1p4x4_bounded_icql_iql_fix3_010818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.01_060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.05_.*060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.1_060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.2_060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.4_v2_060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.5_(fix3_01|v2_06)0818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.6_.*060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.7_.*060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.8_060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.9_090818',
                   'wen_3p1p4x4_bounded_iql_only_fix3_020818']
    legend = ['p_iql = 0.0', 'p_iql = 0.01', 'p_iql = 0.05', 'p_iql = 0.1', 'p_iql = 0.2', 'p_iql = 0.4',
              'p_iql = 0.5', 'p_iql = 0.6', 'p_iql = 0.7', 'p_iql = 0.8', 'p_iql = 0.9', 'p_iql = 1.0']
    max_time = 1E6
    if True:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = []
        rest_legend = [True, False]
        test_legend = [False, True]
    else:
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
        test_legend = [False]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2, wide)
    if wide == 1:
        ax = np.expand_dims(ax, axis=1)
    for i in range(2):
        plot_db_compare(experiments,
                    title='ICQL on 3 pred (3x3 obs), 1 prey in 4x4 env' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=True, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=40, ax=ax[i, 0:len(test_keys)],
                    legend_plot=(test_legend if i==0 else [False]), legend=legend)
    if len(rest_keys) > 0:
        plot_db_compare(experiments, keys=rest_keys,
                        pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                        colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                        legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 96
if plot_please == 96:
    print('ICQL with all independent learners for partially observable 4x4 3pred-1prey')
    colors = ['red', 'blue', 'orange', 'magenta', 'green', 'gray', 'black', 'cyan']
    experiments = ['wen_3p1p4x4_bounded_iql_only_fix3_020818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.5_fix3_010818',
                   'wen_3p1p4x4_bounded_icql_vdn_mix0.5_020818',
                   'wen_3p1p4x4_bounded_icql_reg_vanilla_060818',
                   'wen_3p1p4x4_bounded_icql_reg_seen_060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.33_lr_100818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.5_equallr_100818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.5_equallr_fastupdate_100818'
                   ]
    legend = ['IQL (iql 1.0)', 'ICQL (iql 0.5)', 'ICQL (vdn 0.5)', 'ICQL (reg 0.5)', 'ICQL (reg 0.5, seen)',
              'ICQL (iql 0.33, lr 1e-4)', 'ICQL (iql 0.5, eq. lr)', 'ICQL (iql 0.5, eq. lr, update/10)']
    max_time = 1E6
    if True:
        plot_train = True
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = []
        rest_legend = [True, False]
        test_legend = [False, True]
    else:
        plot_train = True
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
        test_legend = [False]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2 if plot_train else 1, wide)
    if wide == 1:
        ax = np.expand_dims(ax, axis=1)
    for i in range(2 if plot_train else 1):
        my_ax = ax[i, 0:len(test_keys)] if plot_train else ax[0:len(test_keys)]
        plot_db_compare(experiments,
                    title='3 predators (3x3 obs) hunting 1 prey in 4x4 env' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=40, ax=my_ax,
                    legend_plot=(test_legend if i==(1 if plot_train else 0) else [False]), legend=legend)
    if len(rest_keys) > 0:
        plot_db_compare(experiments, keys=rest_keys,
                        pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                        colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                        legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 97
if plot_please == 97:
    experiments = ['wen_3p1p4x4_bounded_icql_iql_fix3_010818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.01_060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.05_.*060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.1_060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.2_060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.4_v2_060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.5_(fix3_01|v2_06)0818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.6_.*060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.7_.*060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.8_060818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.9_090818',
                   'wen_3p1p4x4_bounded_iql_only_fix3_020818']
    labels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    colors = ['magenta', 'red', 'darkorange', 'y', 'lime', 'green', 'c', 'deepskyblue', 'blue', 'gray', 'black']
    max_time = 4E5
    plot_cumulative_rewards(list_of_names=experiments, title='IQL vs. ICQL mixture probability p',
                            legend=labels, max_time=max_time, colors=colors, test=True, use_sem=True, boxplot=False)

#plot_please = 98
if plot_please == 98:
    print('ICQL on Starcraft2 5m vs 5m')
    keys = ['Win rate', 'Episode reward', 'Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100, 'max_time': 3E6}
    colors = ['gray', 'black', 'red', 'blue', 'green', 'lime', 'cyan', 'darkorange', 'red', 'magenta']
    horizons = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare([#'wen_sc2_5m_icql_iql_mix1.0_090818',
                         #'wen_sc2_5m_icql_iql_mix1.0_explore50k_100818',
                         #'wen_sc2_5m_icql_iql_mix0.5_090818',
                         #'wen_sc2_5m_icql_iql_mix0.5_explore50k_100818',
                         #'wen_sc2_5m_icql_iql_mix1.0_130818',
                         #'wen_sc2_5m_icql_iql_mix0.5_130818',
                         #'wen_sc2_5m_icql_iql_mix1.0_explore200k_130818',
                         'WEN_ELBA_CV0001_5m',
                         'WEN_ELBA_CV00001_5m',
                         #'BUCKWEED_PO__exp001_5m',
                         #'BW_CENTRALV_PO_5m',
                         'wen_sc2_5m_icql_iql_mix1.0_140818',
                         'wen_sc2_5m_icql_iql_mix0.5_140818',
                         'wen_sc2_5m_icql_iql_mix0.5_lrq1Em4_200818',
                         'wen_sc2_5m_icql_iql_mix0.5_lrq1Em4_ncgu4_200818',
                         ],
                        legend=[#'IQL (1.0, exp 750k)', 'ICQL (0.5, exp 750k)',
                                #'IQL (1.0, exp 50k)', 'ICQL (0.5, exp 50k)',
                                #'IQL (1.0, exp 200k)',
                                #'COMA (exp 50k)',
                                'Central-V (lr_a=1E-3)', 'Central-V (lr_a=1E-4)',
                                'IQL(1.0, lr_a=1E-3)', 'ICQL(0.5, lr_a=1E-3)',
                                'ICQL(0.5, lr_a=1E-4)', 'ICQL(0.5, lr_a=1E-4, n=4)',
                                ],
                        title='Starcraft II (5m vs. 5m) [98]', keys=keys, test=t==1,
                        colors=colors, longest_runs=4, ax=[ax[i, t] for i in range(len(ax))],
                        legend_pos=['lower right'], legend_plot=[False, False, t==1], **kwargs)
    plt.show()

#plot_please = 99
if plot_please == 99:
    print('ICQL with all different hyperparameters for partially observable 4x4 3pred-1prey')
    colors = ['red', 'magenta', 'cyan', 'blue', 'gray', 'black', 'orange', 'green']
    experiments = ['wen_3p1p4x4_bounded_iql_only_fix3_020818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.5_fix3_010818',
                   'wen_3p1p4x4_bounded_icql_iql_mix1.0_equallr_fastupdate_130818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.5_equallr_fastupdate_100818',
                   'wen_3p1p4x4_bounded_icql_iql_mix1.0_equallr_fastupdate_shrinklinearexplore_130818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.5_equallr_fastupdate_shrinklinearexplore_130818'
                   ]
    legend = ['IQL (iql 1.0, old)', 'ICQL (iql 0.5, old)',
              'IQL (iql 1.0, new, eps>0.1)', 'ICQL (iql 0.5, new, eps>0.1)',
              'IQL (iql 1.0, new eps>0.01)', 'ICQL (iql 0.5, new, eps>0.01)']
    max_time = 1E6
    if True:
        plot_train = True
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = []
        rest_legend = [True, False]
        test_legend = [False, True]
    else:
        plot_train = True
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
        test_legend = [False]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2 if plot_train else 1, wide)
    if wide == 1:
        ax = np.expand_dims(ax, axis=1)
    for i in range(2 if plot_train else 1):
        my_ax = ax[i, 0:len(test_keys)] if plot_train else ax[0:len(test_keys)]
        plot_db_compare(experiments,
                    title='3 predators (3x3 obs) hunting 1 prey in 4x4 env' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=40, ax=my_ax,
                    legend_plot=(test_legend if i==(1 if plot_train else 0) else [False]), legend=legend)
    if len(rest_keys) > 0:
        plot_db_compare(experiments, keys=rest_keys,
                        pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                        colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                        legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 100
if plot_please == 100:
    print('ICQL with new hyperparameters for partially observable 6x6 4pred-1prey')
    colors = ['red', 'green', 'magenta', 'blue', 'cyan', 'gray', 'black', 'orange']
    experiments = ['wen_4p1p6x6_bounded_iql_only_020818',
                   'wen_4p1p6x6_bounded_icql_iql_mix0.5_020818',
                   'wen_4p1p6x6_bounded_icql_iql_mix1.0_130818',
                   'wen_4p1p6x6_bounded_icql_iql_mix0.5_130818',]
    legend = ['IQL (iql 1.0, old)', 'ICQL (iql 0.5, old)', 'IQL (iql 1.0, new)', 'ICQL (iql 0.5, new)']
    max_time = 15E5
    if True:
        plot_train = True
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = []
        rest_legend = [True, False]
        test_legend = [False, True]
    else:
        plot_train = True
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
        test_legend = [False]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2 if plot_train else 1, wide)
    if wide == 1:
        ax = np.expand_dims(ax, axis=1)
    for i in range(2 if plot_train else 1):
        my_ax = ax[i, 0:len(test_keys)] if plot_train else ax[0:len(test_keys)]
        plot_db_compare(experiments,
                    title='4 predators (5x5 obs) hunting 1 prey in 6x6 env' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=40, ax=my_ax,
                    legend_plot=(test_legend if i==(1 if plot_train else 0) else [False]), legend=legend)
    if len(rest_keys) > 0:
        plot_db_compare(experiments, keys=rest_keys,
                        pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                        colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                        legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 101
if plot_please == 101:
    print('FLOUNDRL on Starcraft2 3m vs 3m')
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate'] #, 'Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100, 'max_time': 3E6}
    colors = ['red', 'magenta', 'green', 'c', 'blue', 'black', 'darkorange']
    horizons = [0.8, 0.85, 0.9, 0.95]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(['ELBA_CV_3m',
                         'ELBA_COMA_3m',
                         'ELBA_FL001_3m',
                         'ELBA_FL0005_3m',
                         'ELBA_FL_3m',
                         'WEN_ELBA_FL00005_3m',
                         'WEN_ELBA_FL00001_3m',
                         ],
                        legend=['Central V (lr_a=0.0005)', 'COMA (lr_a=0.0005)', 'FLOUNDRL (lr_a=0.01)',
                                'FLOUNDRL (lr_a=0.005)', 'FLOUNDRL (lr_a=0.0015)',
                                'FLOUNDRL (lr_a=0.0005)', 'FLOUNDRL (lr_a=0.0001)'
                                ],
                        title='Starcraft II (3m vs. 3m)', keys=keys, test=t==1,
                        colors=colors, longest_runs=3, ax=[ax[i, t] for i in range(len(ax))],
                        legend_pos=['lower right'], legend_plot=[False, t==1, False, False, False], **kwargs)
    plt.show()

#plot_please = 102
if plot_please == 102:
    print('FLOUNDRL on Starcraft2 5m vs 5m')
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate'] #, 'Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100, 'max_time': 2E6}
    colors = ['red', 'magenta', 'green', 'c', 'blue', 'black', 'darkorange', 'y']
    horizons = [0.8, 0.85, 0.9, 0.95]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(['ELBA_CV_5m',
                         'ELBA_COMA_5m',
                         'ELBA_FL001_5m',
                         'ELBA_FL0005_5m',
                         'WEN_ELBA_FL0001_5m',
                         'WEN_ELBA_FL00005_5m',
                         'WEN_ELBA_FL00001_5m',
                         'WEN_ELBA_FL000001_5m',
                         ],
                        legend=['Central V', 'COMA',
                                'FLOUNDRL (lr_a=0.01)', 'FLOUNDRL (lr_a=0.005)',
                                'FLOUNDRL (lr_a=0.001)', 'FLOUNDRL (lr_a=0.0005)',
                                'FLOUNDRL (lr_a=0.0001)', 'FLOUNDRL (lr_a=0.00001)'],
                        title='Starcraft II (5m vs. 5m)', keys=keys, test=t==1,
                        colors=colors, longest_runs=0, ax=[ax[i, t] for i in range(len(ax))],
                        legend_pos=['lower right'], legend_plot=[False, t==1, False, False, False], **kwargs)
    plt.show()

#plot_please = 103
if plot_please == 103:
    print('FLOUNDRL with comparable parameters (lr=1E-3) on Starcraft2 5m vs 5m')
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate'] #, 'Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100, 'max_time': 2E6}
    colors = ['red', 'green', 'blue', 'black', 'darkorange', 'y', 'c', 'magenta']
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        plot_db_compare(['WEN_ELBA_CV0001_5m',
                         'WEN_ELBA_COMA0001_5m',
                         'WEN_ELBA_FL0001_5m',
                         ],
                        legend=['Central V', 'COMA', 'FLOUNDRL'],
                        title='Starcraft II (5m vs. 5m)', keys=keys, test=t==1,
                        colors=colors, longest_runs=6, ax=[ax[i, t] for i in range(len(ax))],
                        legend_pos=['lower right'], legend_plot=[False, t==1, False, False, False], **kwargs)
    plt.show()

#plot_please = 104
if plot_please == 104:
    print('FLOUNDRL vs. CENTRAL-V (varying agent learning rates) on Starcraft2 5m vs 5m')
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate'] #, 'Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100, 'max_time': 3E6}
    colors = ['red', 'magenta', 'blue', 'green', 'c', 'black',  'y', 'magenta']
    horizons = [0.9, 0.95]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(['WEN_ELBA_CV0001_5m',
                         'ELBA_CV_5m',
                         'WEN_ELBA_CV00001_5m',
                         'WEN_ELBA_FL0001_5m',
                         'WEN_ELBA_FL00005_5m',
                         'WEN_ELBA_FL00001_5m',
                         ],
                        legend=['Central-V (lr_a=1E-3)', 'Central-V (lr_a=5E-4)', 'Central-V (lr_a=1E-4)',
                                'FLOUNDRL (lr_a=1E-3)', 'FLOUNDRL (lr_a=5E-4)', 'FLOUNDRL (lr_a=1E-4)'],
                        title='Starcraft II (5m vs. 5m)', keys=keys, test=t==1,
                        colors=colors, longest_runs=4, ax=[ax[i, t] for i in range(len(ax))],
                        legend_pos=['lower right'], legend_plot=[False, t==1, False, False, False], **kwargs)
    plt.show()

#plot_please = 105
if plot_please == 105:
    print('ICQL with intrinsic reward on 4x4 3 predator 1 prey environments.')
    colors = ['black', 'gray', 'orange', 'red',  'magenta', 'blue', 'c', 'green',]
    experiments = ['wen_3p1p4x4_bounded_icql_iql_mix1.0_equallr_fastupdate_shrinklinearexplore_130818',
                   'wen_3p1p4x4_bounded_icql_iql_mix0.5_equallr_fastupdate_shrinklinearexplore_130818',
                   'wen_3p1p4x4_icql_iql_mix0.5_irew1.0_brew0.0_200818',
                   'wen_3p1p4x4_icql_iql_mix0.5_irew1.0_brew0.2_200818',
                   'wen_3p1p4x4_icql_iql_mix0.5_irew1.0_brew0.5_200818',
                   'wen_3p1p4x4_icql_iql_mix0.5_irew0.1_brew0.0_200818',
                   'wen_3p1p4x4_icql_iql_mix0.5_irew0.1_brew0.2_200818',
                   'wen_3p1p4x4_icql_iql_mix0.5_irew0.01_brew0.0_200818',]
    legend = ['IQL (iql 1.0, no ir)', 'ICQL (iql 0.5, no ir)',
              'ICQL (iql 0.5, ir=1.0, br=0.0)', 'ICQL (iql 0.5, ir=1.0, br=0.2)', 'ICQL (iql 0.5, ir=1.0, br=0.5)',
              'ICQL (iql 0.5, ir=0.1, br=0.0)', 'ICQL (iql 0.5, ir=0.1, br=0.2)', 'ICQL (iql 0.5, ir=0.01, br=0.0)',]
    horizons = [-2, 0, 2, 4, 6, 8, 8.5, 9, 9.25, 9.5]
    max_time = 3E5
    if True:
        plot_train = True
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = []
        rest_legend = [True, False]
        test_legend = [False, True]
    else:
        plot_train = True
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
        test_legend = [False]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2 if plot_train else 1, wide)
    if wide == 1:
        ax = np.expand_dims(ax, axis=1)
    for i in range(2 if plot_train else 1):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for j in range(len(test_keys)):
                if test_keys[i] == 'Episode reward':
                    ax[j, i].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        my_ax = ax[i, 0:len(test_keys)] if plot_train else ax[0:len(test_keys)]
        plot_db_compare(experiments,
                    title='3 predators (3x3 obs) hunting 1 prey in 4x4 env [105]' if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=6, bin_size=40, ax=my_ax,
                    legend_plot=(test_legend if i==(1 if plot_train else 0) else [False]), legend=legend)
    if len(rest_keys) > 0:
        plot_db_compare(experiments, keys=rest_keys,
                        pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                        colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                        legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 106
if plot_please == 106:
    print('ICQL with intrinsic reward on 6x6 4 predator 1 prey environments.')
    title = '4 predators (5x5 obs) hunting 1 prey in 6x6 env [106]'
    colors = ['black', 'gray', 'red',  'magenta', 'blue', 'c', 'green', 'orange', ]
    experiments = ['wen_4p1p6x6_bounded_icql_iql_mix1.0_130818',
                   'wen_4p1p6x6_bounded_icql_iql_mix0.5_130818',
                   'wen_4p1p6x6_icql_iql_mix0.5_irew0.1_brew0.0_210818',
                   'wen_4p1p6x6_icql_iql_mix0.5_irew0.1_brew0.5_210818',
                   'wen_4p1p6x6_icql_iql_mix0.5_irew1.0_brew0.0_210818',
                   'wen_4p1p6x6_icql_iql_mix0.5_irew1.0_brew0.5_210818']
    legend = ['IQL (iql 1.0, no ir)', 'ICQL (iql 0.5, no ir)',
              'ICQL (iql 0.5, ir=0.1, br=0.0)', 'ICQL (iql 0.5, ir=0.1, br=0.5)',
              'ICQL (iql 0.5, ir=1.0, br=0.0)', 'ICQL (iql 0.5, ir=1.0, br=0.5)',]
    horizons = [-2, 0, 2, 4, 6, 8, 8.5, 9, 9.25, 9.5]
    max_time = 1E6
    if True:
        plot_train = True
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = []
        rest_legend = [True, False]
        test_legend = [False, True]
    else:
        plot_train = True
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff',
                     'Central loss', 'Target central q mean', 'Qvalues entropy']
        rest_legend = [False, False, False, False, False, True]
        test_legend = [False]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2 if plot_train else 1, wide)
    if wide == 1:
        ax = np.expand_dims(ax, axis=1)
    for i in range(2 if plot_train else 1):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for j in range(len(test_keys)):
                if test_keys[i] == 'Episode reward':
                    ax[j, i].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        my_ax = ax[i, 0:len(test_keys)] if plot_train else ax[0:len(test_keys)]
        plot_db_compare(experiments,
                    title=title if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals=':', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=40, ax=my_ax,
                    legend_plot=(test_legend if i==(1 if plot_train else 0) else [False]), legend=legend)
    if len(rest_keys) > 0:
        plot_db_compare(experiments, keys=rest_keys,
                        pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                        colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                        legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 107
if plot_please == 107:
    print('ICQL with intrinsic reward on 4 predator hunting 1 stag and 1 hare in bounded 4x4 environment.')
    title = '4 agents (5x5 obs) hunting 1 stag and 1 hare in 6x6 env [107]'
    colors = ['red', 'green', 'c', 'blue', 'magenta', 'black', 'gray', 'orange']
    experiments = ['wen_staghunt_6x6_icql_iql_mix0.0_irew0.0_brew0.0_220818',   # woma
                   'wen_staghunt_6x6_icql_iql_mix0.5_irew0.0_brew0.0_220818',   # woma
                   'wen_staghunt_6x6_icql_iql_mix0.5_irew1.0_brew0.0_220818',   # savitar
                   'wen_staghunt_6x6_icql_iql_mix0.5_irew1.0_brew0.5_220818',   # gollum
                   'wen_staghunt_6x6_icql_iql_mix0.5_irew1.0_brew1.0_220818',   # woma
                   'wen_staghunt_6x6_icql_iql_mix0.5_irew10.0_brew0.0_220818',  # gollum
                   'wen_staghunt_6x6_icql_iql_mix0.5_irew10.0_brew0.5_220818',  # gollum
                   'wen_staghunt_6x6_icql_iql_mix0.5_iter10_220818']            # savitar
    legend = ['IQL (iql 1.0, no ir)', 'ICQL (iql 0.5, no ir)',
              'ICQL (iql 0.5, ir=1.0, br=0.0)', 'ICQL (iql 0.5, ir=1.0, br=0.5)', 'ICQL (iql 0.5, ir=1.0, br=1.0)',
              'ICQL (iql 0.5, ir=10.0, br=0.0)', 'ICQL (iql 0.5, ir=10.0, br=0.5)', 'ICQL (iql 0.5, 10 max-iter)']
    horizons = [3, 3.5, 4, 4.5, 5, 8, 8.5, 9, 9.5]
    max_time = 1E6
    if False:
        plot_train = True
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = []
        rest_legend = [True, False]
        test_legend = [False, True]
    else:
        plot_train = True
        test_keys = ['Episode reward', 'Episode length']
        rest_keys = ['Individual loss', 'Target individual q mean', 'Target average q diff', # 'Intrinsic reward',
                     'Central loss', 'Target central q mean', 'Qvalues entropy', #'Intrinsic reward all'
                     ]
        rest_legend = [False, False, False, False, False, True]
        test_legend = [False]
    wide = len(test_keys) + len(rest_keys) // 2
    fig, ax = plt.subplots(2 if plot_train else 1, wide)
    if wide == 1:
        ax = np.expand_dims(ax, axis=1)
    for i in range(2 if plot_train else 1):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for j in range(len(test_keys)):
                if test_keys[i] == 'Episode reward':
                    ax[j, i].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        my_ax = ax[i, 0:len(test_keys)] if plot_train else ax[0:len(test_keys)]
        plot_db_compare(experiments,
                    title=title if i==0 else None, keys=test_keys,
                    pm_std=False, use_sem=True, plot_individuals='', test=(i==0), fill_in=False, max_time=max_time,
                    colors=colors, longest_runs=0, bin_size=40, ax=my_ax,
                    legend_plot=(test_legend if i==(1 if plot_train else 0) else [False]), legend=legend)
    if len(rest_keys) > 0:
        plot_db_compare(experiments, keys=rest_keys,
                        pm_std=False, use_sem=True, plot_individuals='', test=False, fill_in=False,  max_time=max_time,
                        colors=colors, longest_runs=0, bin_size=100, ax=ax[:, len(test_keys):wide],
                        legend_pos=['lower right'], legend_plot=rest_legend, legend=legend)
    plt.show()

#plot_please = 108
if plot_please == 108:
    print('FLOUNDRL vs. CENTRALV on Starcraft2 3m vs 3m')
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate'] #, 'Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100, 'max_time': 3E6}
    colors = ['red',  'blue', 'magenta', 'green', 'c', 'black', 'darkorange']
    horizons = [0.8, 0.85, 0.9, 0.95]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(['WEN_ELBA_CV00001_3m_220818',
                         'WEN_ELBA_FL00001_3m_220818',],
                        legend=['Central V', 'FLOUNDRL'],
                        title='Starcraft II (3m vs. 3m) lr_a=1E-4', keys=keys, test=t==1,
                        colors=colors, longest_runs=8, ax=[ax[i, t] for i in range(len(ax))],
                        legend_pos=['lower right'], legend_plot=[False, t==1, False, False, False], **kwargs)
    plt.show()

#plot_please = 109
if plot_please == 109:
    print('2PAIRS vs. FLOUNDRL vs. CENTRAL-V on Starcraft2 5m vs 5m')
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate'] #, 'Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100, 'max_time': 1E6}
    colors = ['red', 'blue', 'green', 'c', 'black',  'y', 'magenta']
    horizons = [0.8, 0.9, 0.95, 0.975]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(['WEN_ELBA_CV00001_5m',
                         'WEN_ELBA_FL00001_5m',
                         'WEN_2PAIRS_FL00001_5m'],
                        legend=['Central-V', 'FLOUNDRL (1 pair)', 'FLOUNDRL (2 pairs)'],
                        title='Starcraft II (5m vs. 5m), lr_a=1E-4', keys=keys, test=t==1,
                        colors=colors, longest_runs=0, ax=[ax[i, t] for i in range(len(ax))],
                        legend_pos=['lower right'], legend_plot=[False, t==1, False, False, False], **kwargs)
    plt.show()

#plot_please = 110
if plot_please == 110:
    print('CENTRAL-V on variations of Starcraft2 5m vs 5m')
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate']  # ''Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100, 'max_time': 2.5E6}
    colors = ['green', 'red', 'magenta', 'blue', 'black',  'y', 'c', ]
    horizons = [0.6, 0.8, 0.9, 0.95, 0.975]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(['WEN_2PAIRS_CV00001_5m_16STEP_240818',
                         'WEN_ELBA_CV00001_5m',
                         'WEN_2PAIRS_CV00001_5m_DC0_DV10_240818',
                         'WEN_2PAIRS_CV00001_5m_DC0_DV0_240818',
                         'WEN_2PAIRS_FL00001_5m_DC0_DV10_240818'],
                        legend=['hit x1, kill=10, step=16', 'hit x1, kill=10, step=8',
                                'hit x0, kill=10, step=8', 'hit x0, kill=0, step=8',
                                'FLOUNDRL (h0, k10, s8)'],
                        title='Central-V on Starcraft II (5m vs. 5m)', keys=keys, test=t==1,
                        colors=colors, longest_runs=0, ax=[ax[i, t] for i in range(len(ax))],
                        legend_pos=['lower right'], legend_plot=[False, t==1, False, False, False], **kwargs)
        for i in range(2):
            y_min, y_max = ax[i, t].get_ylim()
            ax[i, t].set_ylim(y_min - (y_max - y_min) / 50.0, y_max)
    plt.show()

#plot_please = 111
if plot_please == 111:
    print('CENTRAL-V vs. FLOUNDRL in no-hit-reward variation of Starcraft2 5m vs 5m')
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate']  # ''Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100}
    max_time = None
    min_time = int(9E5)
    colors = ['red', 'blue', 'green', 'black', 'y', 'c', ]
    horizons = [0.6, 0.8, 0.9, 0.95, 0.975]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(['WEN_2PAIRS_CV00001_5m_DC0_DV10_280818',
                         #'WEN_2PAIRS_FL00001_5m_DC0_DV10_280818',  # contained deadly bug
                         'WEN_2PAIRS_FL00001_5m_DC0_DV10_300818',
                         #'FGFLOUNDERL_5m'  # @cs
                         #'FFFLOUNDERL_LOGITm1_5m',
                         #'FFFLOUNDERL_LOGIT1_5m'
                         ],
                        legend=['CENTRAL-V', 'FLOUNDRL'],  # , 'FLOUNDRL (logit -1)', 'FLOUNDRL (logit +1)'],
                        title='Starcraft II (5m vs. 5m) without hit-reward', keys=keys, test=t==1, min_time=min_time,
                        colors=colors, longest_runs=0, ax=[ax[i, t] for i in range(len(ax))], max_time=max_time,
                        legend_pos=['lower right'], legend_plot=[False, t==1, False, False, False], **kwargs)
        for i in range(2):
            y_min, y_max = ax[i, t].get_ylim()
            ax[i, t].set_ylim(y_min - (y_max - y_min) / 50.0, y_max)
    plt.show()

#plot_please = 112
if plot_please == 112:
    print('CENTRAL-V vs. FLOUNDRL in no-hit-reward variation of Starcraft2 3m vs 3m')
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate']  # ''Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100}
    max_time = None
    min_time = 0  # int(5E5)
    colors = ['red', 'blue',  'y', 'c', ]
    horizons = [0.6, 0.8, 0.9, 0.95, 0.975]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(['(WEN_2PAIRS_CV00001_3m_DC0_DV10_280818|FFCENTRALV_3m)',
                         #'(WEN_2PAIRS_FL00001_3m_DC0_DV10_280818|FFFLOUNDERL_3m)'  # contained deadly bug
                         'WEN_2PAIRS_FL00001_3m_DC0_DV10_300818'
                         ],
                        legend=['CENTRAL-V', 'FLOUNDRL'],
                        title='Starcraft II (3m vs. 3m) without hit-reward', keys=keys, test=t==1, max_time=max_time,
                        colors=colors, longest_runs=0, ax=[ax[i, t] for i in range(len(ax))], min_time=min_time,
                        legend_pos=['lower right'], legend_plot=[False, t==1, False, False, False], **kwargs)
        for i in range(2):
            y_min, y_max = ax[i, t].get_ylim()
            ax[i, t].set_ylim(y_min - (y_max - y_min) / 50.0, y_max)
    plt.show()

#plot_please = 113
if plot_please == 113:
    ax = plot_comparison(id_lists=[[i for i in range(16)], [i for i in range(16, 30)]], max_time=2E6,
                         directory='../results/results', plot_individuals=None, test=True,
                         title='StarCraft 2 5m vs. 5m', legend=['Central-V', 'Floundrl'])

#plot_please = 114
if plot_please == 114:
    print('Christian\'s StarCraft experiments overview')
    levels = ['StarCraft 2 (3s vs. 4z)', 'StarCraft 2 (2s3z vs. 2s3z)',
              'StarCraft 2 (3m vs. 3m, no hit reward)', 'StarCraft 2 (5m vs. 5m, no hit reward)']
    #subplots = [['FFCENTRALV_3svs4z__run__29', 'FFFLOUNDERL_3svs4z'], ['FFCENTRALV_2s3z', 'FFFLOUNDERL_2s3z'],
    #            ['FFCENTRALV_3m', 'FFFLOUNDERL_3m', 'FFFLOUNDERL_LOGITm1_3m', 'FFFLOUNDERL_LOGIT1_3m'],
    #            ['FFCENTRALV_5m', 'FFFLOUNDERL_5m', 'FFFLOUNDERL_LOGITm1_5m', 'FFFLOUNDERL_LOGIT1_5m']]
    subplots = [['FFCENTRALV_3svs4z__run__29', 'FGFLOUNDERL_3svs4z'],
                ['FFCENTRALV_2s3z', 'FGFLOUNDERL_2s3z'],
                ['FFCENTRALV_3m', 'FGFLOUNDERL_3m', 'FGFLOUNDERL_LOGITm1_3m'],
                ['FFCENTRALV_5m', 'FGFLOUNDERL_5m', 'FGFLOUNDERL_LOGITm1_5m']]
    legend = ['Central-V', 'FloundRL (no logit)', 'FloundRL (logit -1)']
    max_times = [2E6, 2E6, 2E6, 2E6]
    min_times = [1E6, 0, 1E6, 5E5]
    longest_runs = [0, 0, 0, 0]
    #key = 'Episode reward'
    key = 'Win rate'
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': ':', 'fill_in': True, 'bin_size': 100}
    colors = ['red', 'blue',  'green', 'black', 'y', 'c', ]
    horizons = [0.6, 0.8, 0.9, 0.95, 0.975]
    fig, ax = plt.subplots(1, len(subplots))
    for p in range(len(subplots)):
        #for t in range(ax.shape[0]):
            t = 1
            gca = ax[p]

            # Main plot
            plot_db_compare(subplots[p], legend=legend, title=levels[p], keys=[key], test=t==1, min_time=min_times[p],
                            colors=colors, longest_runs=longest_runs[p], ax=gca, max_time=max_times[p],
                            legend_pos=['upper left'], legend_plot=[True, False, False, False], **kwargs)

            # Plot horizontal helper lines
            if key == 'Win rate':
                for h in range(len(horizons)):
                    gca.plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')

            # Shift the limits a bit
            y_min, y_max = gca.get_ylim()
            gca.set_ylim(y_min - (y_max - y_min) / 50.0, y_max)
    plt.show()

#plot_please = 115
if plot_please == 115:
    print('CENTRAL-V vs. FLOUNDRL in Starcraft2 2s3z vs 2s3z')
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate']  # ''Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100}
    max_time = None
    min_time = int(2E6)
    colors = ['red', 'magenta', 'blue', 'c', ]
    horizons = [0.6, 0.8, 0.9, 0.95, 0.975]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(['(FFCENTRALV_2s3z__run__28|FFCENTRALV_2s3z__run__31)', 'FFCENTRALV_REC_2s3z',
                         'FFFLOUNDERL_2s3z', 'FGFLOUNDERL_2s3z'],
                        legend=['Central-V (feed-forw.)', 'Central-V (recurrent)',
                                'FloundRL (feed-forw.)', 'FloundRL (recurrent)'],
                        title='Starcraft II (2s3z vs. 2s3z)', keys=keys, test=t==1, max_time=max_time,
                        colors=colors, longest_runs=30, ax=[ax[i, t] for i in range(len(ax))], min_time=min_time,
                        legend_pos=['lower right'], legend_plot=[False, t==1, False, False, False], **kwargs)
        for i in range(2):
            y_min, y_max = ax[i, t].get_ylim()
            ax[i, t].set_ylim(y_min - (y_max - y_min) / 50.0, y_max)
    plt.show()

#plot_please = 116
if plot_please == 116:
    print('CENTRAL-V vs. FLOUNDRL in Starcraft2 2s3z vs 2s3z')
    keys = ['Win rate', 'Episode reward']  #, 'Level2 delegation rate']  # ''Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100}
    max_time = 1E6
    min_time = int(1E5)
    colors = ['red', 'green', 'blue', 'c', ]
    horizons = [0.6, 0.8, 0.9, 0.95, 0.975]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(['(FFCENTRALV_2s3z__run__28|FFCENTRALV_2s3z__run__31)',
                         'FFCENTRALV_2s3z_4HYPER', 'WEN_FFCENTRALV_2s3z_5HYPER'],
                        legend=['Central-V (4 upd., lr=1E-4)', 'Central-V (60 upd., lr=5E-4)',
                                'Central-V (critic upd., lr=5E-4)'] if t==1 else
                               ['FFCENTRALV_2s3z', 'FFCENTRALV_2s3z_4HYPER', 'WEN_FFCENTRALV_2s3z_5HYPER'],
                        title='Starcraft II (2s3z vs. 2s3z)', keys=keys, test=t==1, max_time=max_time,
                        colors=colors, longest_runs=30, ax=[ax[i, t] for i in range(len(ax))], min_time=min_time,
                        legend_pos=['lower right'], legend_plot=[False, True, False, False, False], **kwargs)
        for i in range(2):
            y_min, y_max = ax[i, t].get_ylim()
            ax[i, t].set_ylim(y_min - (y_max - y_min) / 50.0, y_max)
    plt.show()

#plot_please = 117
if plot_please == 117:
    print('CENTRAL-V vs. FLOUNDRL in Starcraft2 2s3z vs 2s3z (Hawaii2)')
    names = ['WEN_FFCENTRALV_2s3z_5HYPER',
             #'HAWAII_CENTRALV_2s3z', 'HAWAII_FLOUNDERL_2s3z',  # bad env parameter
             'HAWAII2_CENTRALV_2s3z', 'HAWAII2_FLOUNDERL_2s3z'
             ]
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate']  # ''Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100}
    max_time = 10E6
    min_time = 4E6
    colors = ['red', 'green', 'blue', 'c', ]
    horizons = [0.6, 0.8, 0.9, 0.95, 0.975]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(names,
                        legend=['Central-V (Firefly)', 'Central-V (Hawaii2)',
                                'FloundRL (Hawaii2)'] if t==1 else names,
                        title='Starcraft II (2s3z vs. 2s3z)', keys=keys, test=t==1, max_time=max_time,
                        colors=colors, longest_runs=30, ax=[ax[i, t] for i in range(len(ax))], min_time=min_time,
                        legend_pos=['lower right'], legend_plot=[False, True, False, False, False], **kwargs)
        for i in range(2):
            y_min, y_max = ax[i, t].get_ylim()
            ax[i, t].set_ylim(y_min - (y_max - y_min) / 50.0, y_max)
    plt.show()

#plot_please = 118
if plot_please == 118:
    print('CENTRAL-V vs. FLOUNDRL in Starcraft2 3m vs 3m  (Hawaii2)')
    names = ['HAWAII2_CENTRALV_3m', 'HAWAII2_FLOUNDERL_3m', 'HAWAII2_CENTRALV_no_hit_3m', 'HAWAII2_FLOUNDERL_no_hit_3m']
    # 'HAWAII_CENTRALV_3m', 'HAWAII_FLOUNDERL_3m', 'HAWAII_CENTRALV_no_hit_3m', 'HAWAII_FLOUNDERL_no_hit_3m']
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate']  # ''Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100}
    max_time = 5E6
    min_time = int(0)
    colors = ['red', 'blue', 'green', 'c', ]
    horizons = [0.6, 0.8, 0.9, 0.95, 0.975]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(names,
                        legend=['Central-V (Hawaii, rew.)', 'FloundRL (Hawaii, rew.)',
                                'Central-V (Hawaii, no rew.)', 'FloundRL (Hawaii, no rew.)'
                                ] if t==1 else names,
                        title='Starcraft II (3m vs. 3m)', keys=keys, test=t==1, max_time=max_time,
                        colors=colors, longest_runs=30, ax=[ax[i, t] for i in range(len(ax))], min_time=min_time,
                        legend_pos=['upper right'], legend_plot=[False, False, True, False, False], **kwargs)
        for i in range(2):
            y_min, y_max = ax[i, t].get_ylim()
            ax[i, t].set_ylim(y_min - (y_max - y_min) / 50.0, y_max)
    plt.show()

#plot_please = 119
if plot_please == 119:
    print('CENTRAL-V vs. FLOUNDRL in Starcraft2 5m vs 5m (Hawaii2)')
    names = ['HAWAII2_CENTRALV_5m', 'HAWAII2_FLOUNDERL_5m']
             #'HAWAII_CENTRALV_no_hit_5m', 'HAWAII_FLOUNDERL_no_hit_5m',
             #'WEN_HAWAII_CENTRALV_5m', 'WEN_HAWAII_CENTRALV_moveonly_5m', 'WEN_HAWAII_CENTRALV_actiononly_5m',]
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate']  # ''Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100}
    max_time = 1E6
    min_time = int(0)
    colors = ['red', 'green', 'blue', 'orange', 'magenta', 'c', 'black']
    horizons = [0.6, 0.8, 0.9, 0.95, 0.975]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(names,
                        legend=['Central-V (Hawaii, rew.)', 'FloundRL (Hawaii, rew.)',
                                'Central-V (Hawaii, no rew.)', 'FloundRL (Hawaii, no rew.)',
                                'Central-V (Hawaii, m.5, last)', 'Central-V (Hawaii, m.5, no last)',
                                'Central-V (Hawaii, m.2, last)'] if t==1 else names,
                        title='Starcraft II (5m vs. 5m)', keys=keys, test=t==1, max_time=max_time,
                        colors=colors, longest_runs=0, ax=[ax[i, t] for i in range(len(ax))], min_time=min_time,
                        legend_pos=['lower right'], legend_plot=[False, False, True, False, False], **kwargs)
        for i in range(2):
            y_min, y_max = ax[i, t].get_ylim()
            ax[i, t].set_ylim(y_min - (y_max - y_min) / 50.0, y_max)
    plt.show()

#plot_please = 120
if plot_please == 120:
    print('CENTRAL-V vs. FLOUNDRL in Starcraft2 3s vs 4z  (Hawaii)')
    names = ['HAWAII_CENTRALV_3svs4z', 'HAWAII_FLOUNDERL_3svs4z']
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate']  # ''Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100}
    max_time = None  # 2E6
    min_time = int(0)
    colors = ['red', 'green', 'blue', 'black', ]
    horizons = [0.6, 0.8, 0.9, 0.95, 0.975]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(names,
                        legend=['Central-V (Hawaii)', 'FloundRL (Hawaii)'] if t==1 else names,
                        title='Starcraft II (3s vs. 4z)', keys=keys, test=t==1, max_time=max_time,
                        colors=colors, longest_runs=0, ax=[ax[i, t] for i in range(len(ax))], min_time=min_time,
                        legend_pos=['lower right'], legend_plot=[False, False, True, False, False], **kwargs)
        for i in range(2):
            y_min, y_max = ax[i, t].get_ylim()
            ax[i, t].set_ylim(y_min - (y_max - y_min) / 50.0, y_max)
    plt.show()

#plot_please = 121
if plot_please == 121:
    print('CENTRAL-V vs. FLOUNDRL in Starcraft2 5m vs 5m (Hawaii)')
    names = ['WEN_HAWAII_CENTRALV_5m', 'WEN_HAWAII_CENTRALV_moveonly_5m', 'WEN_HAWAII_CENTRALV_actiononly_5m',
             'WEN_HAWAII_FLOUNDERL_oldenv_5m']
    keys = ['Win rate', 'Episode reward', 'Level2 delegation rate']  # ''Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100}
    max_time = 1E6
    min_time = int(0)
    colors = ['magenta', 'c' , 'black', 'blue', 'red', 'green', 'orange']
    horizons = [0.6, 0.8, 0.9, 0.95, 0.975]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(names,
                        legend=['Central-V (Hawaii, m.5, last)', 'Central-V (Hawaii, m.5, no last)',
                                'Central-V (Hawaii, m.2, last)', 'FloundRL (Hawaii, m.5, last)'] if t==1 else names,
                        title='Starcraft II (5m vs. 5m)', keys=keys, test=t==1, max_time=max_time,
                        colors=colors, longest_runs=0, ax=[ax[i, t] for i in range(len(ax))], min_time=min_time,
                        legend_pos=['lower right'], legend_plot=[False, False, True, False, False], **kwargs)
        for i in range(2):
            y_min, y_max = ax[i, t].get_ylim()
            ax[i, t].set_ylim(y_min - (y_max - y_min) / 50.0, y_max)
    plt.show()

#plot_please = 122
if plot_please == 122:
    print('Final plot for AAAI of 2s3z')
    test = True
    names = [#'HAWAII_FLOUNDERL_2s3z',  'HAWAII_CENTRALV_2s3z' # broken critic
             'HAWAII2_FLOUNDERL_2s3z', 'HAWAII2_CENTRALV_2s3z', # right critic
             #'WEN_FFCENTRALV_2s3z_5HYPER'
             ]
    keys = ['Win rate']  # , 'Episode reward', 'Level2 delegation rate']  # ''Episode length']
    #keys = ['Episode reward']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': True, 'bin_size': 100}
    max_time = 5E6
    min_time = 4E6  # 5E5
    colors = ['blue', 'red', 'green', 'c', ]
    horizons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #, 0.95] #, 0.975]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(1):
        gca = ax[1]
        # Main plot
        plot_db_compare(names, legend=['MACKRL', 'Central-V'], #, 'Firefly baseline'],
                        title='StarCraft II: 2s3z vs. 2s3z', keys=keys, test=test, max_time=max_time,
                        colors=colors, longest_runs=0, ax=gca, min_time=min_time,
                        legend_pos=['lower right'], legend_plot=[True, False, False, False], **kwargs)
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    gca.plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        gca.set_ylim(0.0, 1.0)
    print('Final plot for AAAI of 3m')
    names = ['HAWAII2_FLOUNDERL_3m', 'HAWAII2_CENTRALV_3m']
    keys = ['Win rate']  # , 'Episode reward', 'Level2 delegation rate']  # ''Episode length']
    max_time = 3E6
    min_time = 0
    colors = ['blue', 'red', 'green', 'c', ]
    for t in range(1):
        gca = ax[0]
        # Main plot
        plot_db_compare(names, legend=['MACKRL', 'Central-V'],
                        title='StarCraft II: 3m vs. 3m', keys=keys, test=test, max_time=max_time,
                        colors=colors, longest_runs=0, ax=gca, min_time=min_time,
                        legend_pos=['lower right'], legend_plot=[True, False, False, False], **kwargs)
        # Plot horizontal h
        # elper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    gca.plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        gca.set_ylim(0.6, 1.0)
    plt.show()

#plot_please = 123
if plot_please == 123:
    print("Laura's stuff")
    names = ['LAURA_T07C01_PP_CV_4A_6x6_IN_REG1_L20_SIG1_CE',
             'LAURA_T07C01_PP_CV_4A_6x6_IN_REG1_L20_SIG100_CE',
             'LAURA_T07C01_PP_CV_4A_6x6_NN_NOLA_DE']
    keys = ['Episode reward', 'Episode length']
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': ':', 'fill_in': True, 'bin_size': 100}
    max_time = 1E6
    min_time = int(0)
    colors = ['magenta', 'c' , 'black', 'blue', 'red', 'green', 'orange']
    horizons = [0.6, 0.8, 0.9, 0.95, 0.975]
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(ax.shape[1]):
        # Plot horizontal helper lines
        for h in range(len(horizons)):
            for i in range(len(keys)):
                if keys[i] == 'Win rate':
                    ax[i, t].plot(np.array([0, 1E100]), horizons[h] * np.ones(2), linestyle=':', color='black')
        # Main plot
        plot_db_compare(names, legend=names,
                        title='4 Pred 1 Prey 6x6', keys=keys, test=t==1, max_time=max_time,
                        colors=colors, longest_runs=0, ax=[ax[i, t] for i in range(len(ax))], min_time=min_time,
                        legend_pos=['lower right'], legend_plot=[False, t==1, True, False, False], **kwargs)
        for i in range(2):
            y_min, y_max = ax[i, t].get_ylim()
            ax[i, t].set_ylim(y_min - (y_max - y_min) / 50.0, y_max)
    plt.show()

#plot_please = 124
if plot_please == 124:
    plot_key_vs_key(['HAWAII2_FLOUNDERL_2s3z'], 'Win rate', 'Level2 delegation rate', test=True, fill_in=True,
                    colors=None, ax=None, title=None, legend=None, later_than=0, the_last=2)

#plot_please = 125
if plot_please == 125:
    print("Refactored ICQL 6x6 predator-prey experiment (comaprison with IQL)")
    names = ['wen_pp6x6_riql_100918', 'wen_pp6x6_ricql_0.0_100918',
             #'wen_pp6x6_ricql_0.2_100918', 'wen_pp6x6_ricql_0.5_100918',
             'wen_pp6x6_ricql_0.5_detach_100918',
             'wen_pp6x6_ricql_0.9_100918', 'wen_pp6x6_ricql_1.0_100918',
             ]
    legend = ['IQL (refactor)', 'ICQL (0.0)', #'ICQL (0.2)',
              'ICQL (0.5)', 'ICQL (0.9)', 'ICQL (1.0)', #'ICQL (0.5 fix)'
              ]
    keys = ['return_mean', 'ep_length_mean']
    #single_keys = ['loss', 'td_error_abs', 'q_taken_mean', 'grad_norm']
    single_keys = []
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': False, 'bin_size': 100}
    max_time = None  # 1E6
    min_time = int(0)
    colors = ['red', 'green', 'blue', 'black', 'orange', 'c', 'magenta',]
    reward_horizons = [-5, -4, -3, -2, -1, -0.5, 0.0]
    ep_length_horizons = [15, 20, 25, 30, 40, 50]
    fig, ax = plt.subplots(2, int(len(keys) + math.ceil(len(single_keys) / 2.0)))
    # Plot keys and their test
    for t in range(len(keys)):
        # Main plot
        plot_db_compare(names, legend=legend, keys=keys, refactored=True,
                        title='4 Pred(3x3) 1 Prey in 6x6 Env.', test=t==1, max_time=max_time,
                        colors=colors, longest_runs=0, ax=[ax[t, i] for i in range(len(keys))], min_time=min_time,
                        legend_pos=['upper right'], legend_plot=[False, t==1, True, False, False], **kwargs)
        # Plot horizontal helper lines
        for i in range(len(keys)):
            if keys[i] == 'return_mean':
                for h in range(len(reward_horizons)):
                    ax[t, i].plot(np.array([0, 1E100]), reward_horizons[h] * np.ones(2), linestyle=':', color='black')
            if keys[i] == 'ep_length_mean':
                for h in range(len(ep_length_horizons)):
                    ax[t, i].plot(np.array([0, 1E100]), ep_length_horizons[h] * np.ones(2), linestyle=':',
                                  color='black')
        #for i in range(2):
        #    y_min, y_max = ax[i, t].get_ylim()
        #    ax[i, t].set_ylim(y_min - (y_max - y_min) / 1.0, y_max)
    # Plot single keys
    width = math.ceil(len(single_keys) / 2.0)
    sax = [ax[0, len(keys) + i] for i in range(width)]
    sax.extend([ax[1, len(keys) + i] for i in range(min(width, len(single_keys) - width))])
    plot_db_compare(names, legend=legend, keys=single_keys, refactored=True,
                    title='4 Pred(3x3) 1 Prey in 6x6 Env.', test=False, max_time=max_time,
                    colors=colors, longest_runs=0, ax=sax, min_time=min_time,
                    legend_pos=['upper right'], legend_plot=[False, False, False, False], **kwargs)
    plt.show()

#plot_please = 126
if plot_please == 126:
    print("Refactored ICQL 10x10 predator-prey experiment (comaprison with IQL)")
    #names = ['wen_pp10x10_riql_long_100918', 'wen_pp10x10_ricql_0.0_long_100918', 'wen_pp10x10_ricql_0.5_long_100918']
    names = ['wen_pp10x10_riql_110918', 'wen_pp10x10_ricql_0.0_110918', 'wen_pp10x10_ricql_0.5_110918']  # separate_critic_update
    legend = ['IQL (refactor)', 'ICQL (0.0)', 'ICQL (0.5)']
    keys = ['return_mean', 'ep_length_mean']
    #single_keys = ['loss', 'td_error_abs', 'q_taken_mean', 'grad_norm']
    single_keys = []
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': False, 'bin_size': 100}
    max_time = None  # 1E6
    min_time = int(0)
    colors = ['red', 'green', 'blue', 'magenta',  'orange', 'black', 'c']
    reward_horizons = [-5, -4, -3.5, -3, -2.5, -2]
    ep_length_horizons = [15, 20, 25, 30, 40, 50]
    fig, ax = plt.subplots(2, int(len(keys) + math.ceil(len(single_keys) / 2.0)))
    # Plot keys and their test
    for t in range(len(keys)):
        # Main plot
        plot_db_compare(names, legend=legend, keys=keys, refactored=True,
                        title='4 Pred(3x3) 1 Prey in 10x10 Env.', test=t==1, max_time=max_time,
                        colors=colors, longest_runs=0, ax=[ax[t, i] for i in range(len(keys))], min_time=min_time,
                        legend_pos=['upper right'], legend_plot=[False, t==1, True, False, False], **kwargs)
        # Plot horizontal helper lines
        for i in range(len(keys)):
            if keys[i] == 'return_mean':
                for h in range(len(reward_horizons)):
                    ax[t, i].plot(np.array([0, 1E100]), reward_horizons[h] * np.ones(2), linestyle=':', color='black')
            if keys[i] == 'ep_length_mean':
                for h in range(len(ep_length_horizons)):
                    ax[t, i].plot(np.array([0, 1E100]), ep_length_horizons[h] * np.ones(2), linestyle=':',
                                  color='black')
        #for i in range(2):
        #    y_min, y_max = ax[i, t].get_ylim()
        #    ax[i, t].set_ylim(y_min - (y_max - y_min) / 1.0, y_max)
    # Plot single keys
    width = math.ceil(len(single_keys) / 2.0)
    sax = [ax[0, len(keys) + i] for i in range(width)]
    sax.extend([ax[1, len(keys) + i] for i in range(min(width, len(single_keys) - width))])
    plot_db_compare(names, legend=legend, keys=single_keys, refactored=True,
                    title='4 Pred(3x3) 1 Prey in 6x6 Env.', test=False, max_time=max_time,
                    colors=colors, longest_runs=0, ax=sax, min_time=min_time,
                    legend_pos=['upper right'], legend_plot=[False, False, False, False], **kwargs)
    plt.show()

#plot_please = 127
if plot_please == 127:
    print("Refactored 10x10 staghunt experiment (IQL, QMIX, COMA)")
    names = ['wen_staghunt10x10_refactor_iql_110918', 'wen_staghunt10x10_refactor_qmix_110918',
             'wen_staghunt10x10_refactor_coma_110918']
    legend = ['IQL', 'QMIX', 'COMA']
    keys = ['return_mean', 'ep_length_mean']
    #single_keys = ['loss', 'td_error_abs', 'q_taken_mean', 'grad_norm']
    single_keys = []
    kwargs = {'pm_std': False, 'use_sem': False, 'plot_individuals': '', 'fill_in': False, 'bin_size': 100}
    max_time = None
    min_time = 0  # int(3E6)
    colors = ['red', 'green', 'blue', 'magenta', 'c', 'black', 'orange', ]
    reward_horizons = []  # [-5, -4, -3.5, -3, -2.5, -2]
    ep_length_horizons = []  # [15, 20, 25, 30, 40, 50]
    fig, ax = plt.subplots(2, int(len(keys) + math.ceil(len(single_keys) / 2.0)))
    # Plot keys and their test
    for t in range(len(keys)):
        # Main plot
        plot_db_compare(names, legend=legend, keys=keys, refactored=True,
                        title='4 agents(5x5) 1 Stag 1 Hare in 10x10 Env.' if t==0 else None,
                        test=t==1, max_time=max_time, min_time=min_time,
                        colors=colors, longest_runs=0, ax=[ax[t, i] for i in range(len(keys))],
                        legend_pos=['upper right'], legend_plot=[False, t==1, True, False, False], **kwargs)
        # Plot horizontal helper lines
        for i in range(len(keys)):
            if keys[i] == 'return_mean':
                for h in range(len(reward_horizons)):
                    ax[t, i].plot(np.array([0, 1E100]), reward_horizons[h] * np.ones(2), linestyle=':', color='black')
            if keys[i] == 'ep_length_mean':
                for h in range(len(ep_length_horizons)):
                    ax[t, i].plot(np.array([0, 1E100]), ep_length_horizons[h] * np.ones(2), linestyle=':',
                                  color='black')
        #for i in range(2):
        #    y_min, y_max = ax[i, t].get_ylim()
        #    ax[i, t].set_ylim(y_min - (y_max - y_min) / 1.0, y_max)
    # Plot single keys
    width = math.ceil(len(single_keys) / 2.0)
    sax = [ax[0, len(keys) + i] for i in range(width)]
    sax.extend([ax[1, len(keys) + i] for i in range(min(width, len(single_keys) - width))])
    plot_db_compare(names, legend=legend, keys=single_keys, refactored=True,
                    test=False, max_time=max_time,
                    colors=colors, longest_runs=0, ax=sax, min_time=min_time,
                    legend_pos=['upper right'], legend_plot=[False, False, False, False], **kwargs)
    plt.show()

#plot_please = 128
if plot_please == 128:
    print("Refactored 20x20 staghunt experiment (IQL, QMIX, CENTRAL-V, COMA)")
    names = ['wen_staghunt_20x20_refaqctor_iql_120918', 'wen_staghunt_20x20_refaqctor_qmix_120918',
             'wen_staghunt_20x20_refaqctor_coma_120918', 'wen_staghunt_20x20_refaqctor_centralV_120918']
    legend = ['IQL', 'QMIX', 'COMA (fix)', 'CENTRAL-V']
    keys = ['return_mean', 'ep_length_mean']
    #single_keys = ['loss', 'td_error_abs', 'q_taken_mean', 'grad_norm']
    single_keys = []
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': False, 'bin_size': 100}
    max_time = None  # 1E6
    min_time = int(0E6)
    colors = ['red', 'green', 'blue', 'black', 'magenta',  'orange', 'c']
    reward_horizons = []  # [-5, -4, -3.5, -3, -2.5, -2]
    ep_length_horizons = []  # [15, 20, 25, 30, 40, 50]
    fig, ax = plt.subplots(2, int(len(keys) + math.ceil(len(single_keys) / 2.0)))
    # Plot keys and their test
    for t in range(len(keys)):
        # Main plot
        plot_db_compare(names, legend=legend, keys=keys, refactored=True,
                        title='4 agents(5x5) 1 Stag 1 Hare in 20x20 Env.' if t==0 else None,
                        test=t==1, max_time=max_time, min_time=min_time,
                        colors=colors, longest_runs=0, ax=[ax[t, i] for i in range(len(keys))],
                        legend_pos=['upper right'], legend_plot=[False, t==1, True, False, False], **kwargs)
        # Plot horizontal helper lines
        for i in range(len(keys)):
            if keys[i] == 'return_mean':
                for h in range(len(reward_horizons)):
                    ax[t, i].plot(np.array([0, 1E100]), reward_horizons[h] * np.ones(2), linestyle=':', color='black')
            if keys[i] == 'ep_length_mean':
                for h in range(len(ep_length_horizons)):
                    ax[t, i].plot(np.array([0, 1E100]), ep_length_horizons[h] * np.ones(2), linestyle=':',
                                  color='black')
        #for i in range(2):
        #    y_min, y_max = ax[i, t].get_ylim()
        #    ax[i, t].set_ylim(y_min - (y_max - y_min) / 1.0, y_max)
    # Plot single keys
    width = math.ceil(len(single_keys) / 2.0)
    sax = [ax[0, len(keys) + i] for i in range(width)]
    sax.extend([ax[1, len(keys) + i] for i in range(min(width, len(single_keys) - width))])
    plot_db_compare(names, legend=legend, keys=single_keys, refactored=True,
                    test=False, max_time=max_time,
                    colors=colors, longest_runs=0, ax=sax, min_time=min_time,
                    legend_pos=['upper right'], legend_plot=[False, False, False, False], **kwargs)
    plt.show()

plot_please = 129
if plot_please == 129:
    print("Refactored 20x20 staghunt experiment with more reward (IQL, QMIX, COMA)")
    names = ['wen_staghunt_20x20_refactor_iql_reward_120918',
             #'wen_staghunt_20x20_refactor_coma_reward_120918',
             #'wen_staghunt_20x20_refactor_coma_nstep1_reward_120918',
             #'wen_refactor_coma_stag_hunt_20x20_reward_csparams_130918',
             #'wen_staghunt_20x20_refactor_icql_reward_130918',
             'wen_refactor_vdn_stag_hunt_20x20_reward_130918',
             'wen_staghunt_20x20_refactor_qmix_reward_120918',
             'wen_refactor_qmix_stag_hunt_20x20_reward_skip_130918',
             'wen_refactor_qmix_stag_hunt_20x20_reward_init_130918',
             'wen_refactor_qmix_stag_hunt_20x20_reward_initskip_130918']
    legend = ['IQL', #'COMA (0-step)', 'COMA (1-step)', 'COMA(params)', 'ICQL(0.5)',
              'VDN', 'QMIX', 'QMIX (skip)', 'QMIX (init)', 'QMIX (new skip)']
    keys = ['return_mean', 'ep_length_mean']
    #single_keys = ['loss', 'td_error_abs', 'q_taken_mean', 'grad_norm']
    single_keys = []
    kwargs = {'pm_std': False, 'use_sem': True, 'plot_individuals': '', 'fill_in': False, 'bin_size': 100}
    max_time = 1E6
    min_time = int(0E6)
    #colors = ['red', 'cyan', 'c', 'lightblue', 'black', 'magenta', 'green', 'y', 'orange', 'blue']
    colors = ['red', 'magenta', 'green', 'blue', 'orange', 'black', 'y',  'cyan', 'c', 'lightblue',]
    reward_horizons = [-10, -5, 0, 5, 10]
    ep_length_horizons = [50, 60, 70, 80, 90, 100]
    fig, ax = plt.subplots(2, int(len(keys) + math.ceil(len(single_keys) / 2.0)))
    # Plot keys and their test
    for t in range(len(keys)):
        # Main plot
        plot_db_compare(names, legend=legend, keys=keys, refactored=True,
                        title='4 agents(5x5) 1 Stag 1 Hare in 20x20 Env. (high reward)' if t==0 else None,
                        test=t==1, max_time=max_time, min_time=min_time,
                        colors=colors, longest_runs=0, ax=[ax[t, i] for i in range(len(keys))],
                        legend_pos=['upper right'], legend_plot=[False, t==1, True, False, False], **kwargs)
        # Plot horizontal helper lines
        for i in range(len(keys)):
            if keys[i] == 'return_mean':
                for h in range(len(reward_horizons)):
                    ax[t, i].plot(np.array([0, 1E100]), reward_horizons[h] * np.ones(2), linestyle=':', color='black')
            if keys[i] == 'ep_length_mean':
                for h in range(len(ep_length_horizons)):
                    ax[t, i].plot(np.array([0, 1E100]), ep_length_horizons[h] * np.ones(2), linestyle=':',
                                  color='black')
        #for i in range(2):
        #    y_min, y_max = ax[i, t].get_ylim()
        #    ax[i, t].set_ylim(y_min - (y_max - y_min) / 1.0, y_max)
    # Plot single keys
    width = math.ceil(len(single_keys) / 2.0)
    sax = [ax[0, len(keys) + i] for i in range(width)]
    sax.extend([ax[1, len(keys) + i] for i in range(min(width, len(single_keys) - width))])
    plot_db_compare(names, legend=legend, keys=single_keys, refactored=True,
                    test=False, max_time=max_time,
                    colors=colors, longest_runs=0, ax=sax, min_time=min_time,
                    legend_pos=['upper right'], legend_plot=[False, False, False, False], **kwargs)
    plt.show()

#plot_please = 130
if plot_please == 130:
    print("Refactored 10x10 staghunt experiment without time punishment.")
    names = ['wen_staghunt_10x10_refactor_iql_nopain_130918',
             'wen_staghunt_10x10_refactor_qmix_nopain_130918',
             'wen_refactor_icql_stag_hunt_10x10_nopain_mix0.0_130918',
             'wen_staghunt_10x10_refactor_icql_nopain_130918',
             'wen_refactor_iql_stag_hunt_10x10_nopain_smallobs_130918',
             'wen_refactor_qmix_stag_hunt_10x10_nopain_smallobs_130918',
             'wen_refactor_icql_stag_hunt_10x10_nopain_smallobs_mix0.0_130918',
             'wen_refactor_icql_stag_hunt_10x10_nopain_smallobs_130918']
    legend = ['IQL (5x5)', 'QMIX (5x5)', 'ICQL(0.0, 5x5)', 'ICQL(0.5, 5x5)',
              'IQL (3x3)', 'QMIX (3x3)', 'ICQL(0.0, 3x3)', 'ICQL(0.5, 3x3)']
    keys = ['return_mean', 'ep_length_mean']
    #single_keys = ['loss', 'td_error_abs', 'q_taken_mean', 'grad_norm']
    single_keys = []
    kwargs = {'pm_std': False, 'use_sem': False, 'plot_individuals': '', 'fill_in': False, 'bin_size': 100}
    max_time = None
    min_time = 0  # int(3E6)
    colors = ['red', 'green', 'gray', 'black', 'magenta', 'orange', 'c', 'blue']
    reward_horizons = [5, 10]
    ep_length_horizons = []  # [15, 20, 25, 30, 40, 50]
    fig, ax = plt.subplots(2, int(len(keys) + math.ceil(len(single_keys) / 2.0)))
    # Plot keys and their test
    for t in range(len(keys)):
        # Main plot
        plot_db_compare(names, legend=legend, keys=keys, refactored=True,
                        title='4 agents 1 Stag 1 Hare in 10x10 Env. (no punishment)' if t==0 else None,
                        test=t==1, max_time=max_time, min_time=min_time,
                        colors=colors, longest_runs=0, ax=[ax[t, i] for i in range(len(keys))],
                        legend_pos=['upper right'], legend_plot=[False, t==1, True, False, False], **kwargs)
        # Plot horizontal helper lines
        for i in range(len(keys)):
            if keys[i] == 'return_mean':
                for h in range(len(reward_horizons)):
                    ax[t, i].plot(np.array([0, 1E100]), reward_horizons[h] * np.ones(2), linestyle=':', color='black')
            if keys[i] == 'ep_length_mean':
                for h in range(len(ep_length_horizons)):
                    ax[t, i].plot(np.array([0, 1E100]), ep_length_horizons[h] * np.ones(2), linestyle=':',
                                  color='black')
        #for i in range(2):
        #    y_min, y_max = ax[i, t].get_ylim()
        #    ax[i, t].set_ylim(y_min - (y_max - y_min) / 1.0, y_max)
    # Plot single keys
    width = math.ceil(len(single_keys) / 2.0)
    sax = [ax[0, len(keys) + i] for i in range(width)]
    sax.extend([ax[1, len(keys) + i] for i in range(min(width, len(single_keys) - width))])
    plot_db_compare(names, legend=legend, keys=single_keys, refactored=True,
                    test=False, max_time=max_time,
                    colors=colors, longest_runs=0, ax=sax, min_time=min_time,
                    legend_pos=['upper right'], legend_plot=[False, False, False, False], **kwargs)
    plt.show()

#plot_please = 131
if plot_please == 131:
    print("Test of COMA on 2s3z.")
    names = ['wen_refactor_coma_sc2_2s3z_test_130918']
    legend = ['COMA']
    #keys = ['return_mean', 'ep_length_mean']
    keys = ['battle_won_mean', 'return_mean']
    kwargs = {'pm_std': False, 'use_sem': False, 'plot_individuals': '', 'fill_in': False, 'bin_size': 100}
    max_time = None
    min_time = 0  # int(3E6)
    colors = ['blue', 'red', 'green', 'gray', 'black', 'magenta', 'orange', 'c', 'blue']
    fig, ax = plt.subplots(len(keys), 2)
    for t in range(2):
        plot_db_compare(names, legend=legend, keys=keys, refactored=True,
                        test=t==1, max_time=max_time, title="StarCraft II (2s3z)",
                        colors=colors, longest_runs=0, min_time=min_time, ax=ax[:, t],
                        legend_pos=['upper right'], legend_plot=[False, False, False, False], **kwargs)
    plt.show()
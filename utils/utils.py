import random
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split


def check_cuda():
    # check for available GPU and distribute work
    if torch.cuda.device_count() > 1:
        torch.cuda.empty_cache()

        gpu_sel = 1
        gpu_av = [torch.cuda.is_available()
                for ii in range(torch.cuda.device_count())]
        print("Detected {} GPUs. The load will be shared.".format(
            torch.cuda.device_count()))
        for gpu in range(len(gpu_av)):
            if True in gpu_av:
                if gpu_av[gpu_sel]:
                    device = torch.device("cuda:"+str(gpu))
                    # torch.cuda.set_per_process_memory_fraction(0.9, device=device)
                    print("Selected GPUs: {}" .format("cuda:"+str(gpu)))
                else:
                    device = torch.device("cuda:"+str(gpu_av.index(True)))
            else:
                device = torch.device("cpu")
                print("No GPU detected. Running on CPU.")
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

                print("Single GPU detected. Setting up the simulation there.")
                device = torch.device("cuda:0")
                # torch.cuda.set_per_process_memory_fraction(0.9, device=device)
            else:
                device = torch.device("cpu")
                print("No GPU detected. Running on CPU.")
    return device


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile! :(
            return None
        return directory_path



def load_and_extract(params, file_name, taxels=None, letter_written=None):

    max_time = (350*10)  # samples time dt [ms]
    time_bin_size = int(params['time_bin_size'])  # ms
    time = range(0, max_time, time_bin_size)

    time_step = time_bin_size*0.001
    data_steps = len(time)

    data_dict = pd.read_pickle(file_name)

    # Extract data
    data = []
    labels = []
    bins = 1000  # ms conversion
    nchan = len(data_dict['events'][1])  # number of channels per sensor
    # loop over all trials
    for i, sample in enumerate(data_dict['events']):
        events_array = np.zeros(
            [nchan, round((max_time/time_bin_size)+0.5), 2])
        # loop over sensors (taxel)
        for taxel in range(len(sample)):
            # loop over On and Off channels
            for event_type in range(len(sample[taxel])):
                if sample[taxel][event_type]:
                    indx = bins*(np.array(sample[taxel][event_type]))
                    indx = np.array((indx/time_bin_size).round(), dtype=int)
                    events_array[taxel, indx, event_type] = 1
        if taxels != None:
            events_array = np.reshape(np.transpose(events_array, (1, 0, 2))[
                                      :, taxels, :], (events_array.shape[1], -1))
            selected_chans = 2*len(taxels)
        else:
            events_array = np.reshape(np.transpose(
                events_array, (1, 0, 2)), (events_array.shape[1], -1))
            selected_chans = 2*nchan
        data.append(events_array)
        labels.append(letter_written.index(data_dict['letter'][i]))

    # return data,labels
    data = np.array(data)
    labels = np.array(labels)

    # # one-hot encoding
    # from sklearn.preprocessing import OneHotEncoder
    # enc = OneHotEncoder()
    # labels = enc.fit_transform(labels.reshape(-1,1))

    # print(labels)
    data = torch.as_tensor(data, dtype=torch.float) # torch.tensor() always copies the data
    labels = torch.as_tensor(labels, dtype=torch.long)

    # create 70/20/10 train/test/validation split
    x_train, y_train, x_test, y_test, x_validation, y_validation = train_test_validation_split(data, labels, split=[70, 20, 10], seed=None)

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)
    ds_validation = TensorDataset(x_validation, y_validation)

    return ds_train, ds_test, ds_validation, labels, selected_chans, data_steps



def train_test_validation_split(data, labels, split=[70, 20, 10], seed=None):
    """
    Creates a train-test-validation split using the sklearn train_test_split() twice.
    Function accepts lists, arrays, and tensor.
    Default split: [70, 20, 10]

    data.shape: [trials, time, sensor]
    label.shape: [trials] 
    split: [train, test, validation]
    """
    # do some sanity checks first
    if len(split) != 3:
        raise ValueError(
            f"Split dimensions are wrong. Expected 3 , but got {len(split)}. Please provide split in the form [train size, test size, validation size].")
    if min(split) == 0.0:
        raise ValueError(
            "Found entry 0.0. If you want to use only perfrom a two-folded split, use the sklearn train_test_split function only please.")
    if sum(split) > 99.0:
        split = np.array(split)/100
    if sum(split) < 0.99 or sum(split) > 1.01:
        raise ValueError("Please use a split summing up to 1, or 100%.")

    # create train and (test + validation) split
    x_train, x_test_validation, y_train, y_test_validation = train_test_split(
        data, labels, test_size=split[0], shuffle=True, stratify=labels, random_state=seed)
    # create test and validation split
    ratio = split[1]/sum(split[1:])
    x_test, x_validation, y_test, y_validation = train_test_split(
        x_test_validation, y_test_validation, test_size=ratio, shuffle=True, stratify=y_test_validation, random_state=seed)

    return x_train, y_train, x_test, y_test, x_validation, y_validation


def ConfusionMatrix(dataset, save, layers=None, labels=None):

    g = torch.Generator()
    g.manual_seed(seed)
    generator = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    accs = []
    trues = []
    preds = []
    for x_local, y_local in generator:
        x_local, y_local = x_local.to(
            device, non_blocking=True), y_local.to(device, non_blocking=True)
        if layers == None:
            if use_trainable_out and use_trainable_tc:
                layers = [w1, w2, v1, alpha1, beta1, alpha2,
                          beta2, out_scale, out_offset]
            elif use_trainable_out:
                layers = [w1, w2, v1, out_scale, out_offset]
            elif use_trainable_tc:
                layers = [w1, w2, v1, alpha1, beta1, alpha2, beta2]
            else:
                layers = [w1, w2, v1]
            spks_out, _, _ = run_snn(x_local, layers)
        else:
            spks_out, _, _ = run_snn(x_local, layers)
        # with output spikes
        if use_trainable_out:
            m = spks_out
        else:
            m = torch.sum(spks_out, 1)  # sum over time
        _, am = torch.max(m, 1)     # argmax over output units
        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)
        trues.extend(y_local.detach().cpu().numpy())
        preds.extend(am.detach().cpu().numpy())

    print("Accuracy from Confusion Matrix: {:.2f}% +- {:.2f}%".format(np.mean(accs)
                                                                             * 100, np.std(accs)*100))

    cm = confusion_matrix(trues, preds, normalize='true')
    cm_df = pd.DataFrame(cm, index=[ii for ii in labels], columns=[
                         jj for jj in labels])
    plt.figure("cm", figsize=(12, 9))
    sn.heatmap(cm_df,
               annot=True,
               fmt='.1g',
               cbar=False,
               square=False,
               cmap="YlGnBu")
    plt.xlabel('\nPredicted')
    plt.ylabel('True\n')
    plt.xticks(rotation=0)
    if save:
        path_to_save_fig = f'{path_for_plots}/generation_{generation+1}_individual_{best_individual+1}'
        if use_trainable_tc:
            path_to_save_fig = f'{path_to_save_fig}_train_tc'
        if use_trainable_out:
            path_to_save_fig = f'{path_to_save_fig}_train_out'
        if use_dropout:
            path_to_save_fig = f'{path_to_save_fig}_dropout'
        path_to_save_fig = f'{path_to_save_fig}_cm.png'
        plt.savefig(path_to_save_fig, dpi=300)
        plt.close()
    else:
        plt.show()

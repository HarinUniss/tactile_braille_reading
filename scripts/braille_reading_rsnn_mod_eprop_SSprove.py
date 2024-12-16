import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec  # can be used for nice subplot layout
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
from ipdb import set_trace as st

torch.cuda.empty_cache()  # Svuota la cache della memoria CUDA
torch.set_default_dtype(torch.float64)

def psave(topsave1,topsave2=None,topsave3=None,topsave4=None,topsave5=None):
    print(topsave1,topsave2,topsave3,topsave4,topsave5)
    with open('output.txt', 'a') as f:
        print(topsave1,topsave2,topsave3,topsave4,topsave5, file=f)

dtype = torch.float

letters = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# set variables
use_seed = True
threshold = 2  # possible values are: 1, 2, 5, 10
# set the number of epochs you want to train the network (default = 300)
epochs = 100

global batch_size
batch_size = 4  

global lr
lr = 0.0008
psave("Learning rate: ",lr)
global gamma
gamma = 0.3

global lower_bound
lower_bound = -1.0  # set to None to disable
global no_synapse
no_synapse = True
global use_linear_decay
use_linear_decay = False
global ref_per_timesteps
# refractory period is set in simulation time steps for now; set to None to disable
ref_per_timesteps = 1

# some options for plotting
NB_BATCHES_TO_PLOT = 1
NB_TRIALS_TO_PLOT = 1

# create folder to safe figures later
path = './figures'
isExist = os.path.exists(path)

if not isExist:
    os.makedirs(path)

device = torch.device("cuda:0")  ##### <<<< ATTENZIONE PER PC A SASSARI IMPOSTARE cuda:0 ALTROVE cuda:1
psave(f"Using device: {device}") #### <<<<< psave per poter verificare se stiamo usando la GPU corretta 

# use fixed seed for reproducable results
if use_seed:
    seed = 42  # "Answer to the Ultimate Question of Life, the Universe, and Everything"
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    psave("Seed set to {}".format(seed))
else:
    psave("Shuffle data randomly")


def load_and_extract(params, file_name, taxels=None, letter_written=letters):

    max_time = int(1275)  # ms
    time_bin_size = int(params['time_bin_size'])  # so far from laoded file, but can be set manually here
    global time
    time = range(0, max_time, time_bin_size)

    global time_step
    time_step = time_bin_size*0.001  # ms
    global data_steps
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
    #one_hot_encoded = np.eye(len(np.unique(labels)))[labels]
    #output = np.repeat(one_hot_encoded[:, np.newaxis, :], data_steps+1, axis=1)

    # psave(labels)
    data = torch.tensor(data, dtype=dtype)
    labels = torch.tensor(labels, dtype=torch.long)

    # create 70/20/10 train/test/validation split
    # first create 70/30 train/(test + validation)
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.30, shuffle=True, stratify=labels)
    # split test and validation 2/1
    x_test, x_validation, y_test, y_validation = train_test_split(
        x_test, y_test, test_size=0.33, shuffle=True, stratify=y_test)

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)
    ds_validation = TensorDataset(x_validation, y_validation)

    return ds_train, ds_test, ds_validation, labels, selected_chans, data_steps

def grads_batch(x, yo, yt, gamma, thr, v, z, w_in, w_rec, w_out):
    if w_in.grad is None:
        w_in.grad = torch.zeros_like(w_in)
    if w_rec.grad is None:
        w_rec.grad = torch.zeros_like(w_rec)
    if w_out.grad is None:
        w_out.grad = torch.zeros_like(w_out)
    # Surrogate derivatives
    h = gamma * torch.max(torch.zeros_like(v), 1 - torch.abs((v - thr) / thr))

    # Crea una variabile di errore vuota con le stesse dimensioni di yo
    err = torch.zeros_like(yo)

    # Eligibility traces convolution
    beta_conv     = torch.tensor([beta_trace_out ** (data_steps - i - 1) for i in range(data_steps)]).float().view(1, 1, -1).to(device)
    beta_rec_conv = torch.tensor([beta_trace ** (data_steps - i - 1) for i in range(data_steps)]).float().view(1, 1, -1).to(device)

    # Convoluzione Input eligibility traces
    trace_in = F.conv1d(x.permute(1, 2, 0), beta_rec_conv.expand(nb_inputs, -1, -1), padding=data_steps, groups=nb_inputs)[:, :, 1:data_steps+1]
    trace_in = trace_in.unsqueeze(1).expand(-1, nb_hidden, -1, -1)
    trace_in = torch.einsum('tbr,brit->brit', h, trace_in)

    # Convoluzione Recurrent eligibility traces
    trace_rec = F.conv1d(z.permute(1, 2, 0), beta_rec_conv.expand(nb_hidden, -1, -1), padding=data_steps, groups=nb_hidden)[:, :, :data_steps]
    trace_rec = trace_rec.unsqueeze(1).expand(-1, nb_hidden, -1, -1)
    trace_rec = torch.einsum('tbr,brit->brit', h, trace_rec)

    # Output eligibility vector
    trace_out = F.conv1d(z.permute(1, 2, 0), beta_conv.expand(nb_hidden, -1, -1), padding=data_steps, groups=nb_hidden)[:, :, 1:data_steps+1]

    # Ottimizzazione convoluzioni batch-wise
    trace_in = F.conv1d(trace_in.reshape(batch_size, nb_inputs * nb_hidden, data_steps),
                        beta_conv.expand(nb_inputs * nb_hidden, -1, -1),
                        padding=data_steps, groups=nb_inputs * nb_hidden)[:, :, 1:data_steps+1]
    trace_in = trace_in.reshape(batch_size, nb_hidden, nb_inputs, data_steps)

    trace_rec = F.conv1d(trace_rec.reshape(batch_size, nb_hidden * nb_hidden, data_steps),
                         beta_conv.expand(nb_hidden * nb_hidden, -1, -1),
                         padding=data_steps, groups=nb_hidden * nb_hidden)[:, :, 1:data_steps+1]
    trace_rec = trace_rec.reshape(batch_size, nb_hidden, nb_hidden, data_steps)

    # Ciclo for per calcolare l'errore 'err'
    for i in range(yo.shape[0]):
        err[i,:,:] = yo[i,:,:] - yt
    err = err.to(dtype)
    # Calcolo dei segnali di apprendimento
    L = torch.einsum('tbo,or->brt', err, w_out)

    # Weight gradient updates
    w_in.grad += torch.sum(L.unsqueeze(2).expand(-1, -1, nb_inputs, -1) * trace_in, dim=(0, 3))
    w_rec.grad += torch.sum(L.unsqueeze(2).expand(-1, -1, nb_hidden, -1) * trace_rec, dim=(0, 3))
    w_out.grad += torch.einsum('tbo,brt->or', err, trace_out)


def run_snn(inputs, layers, trainable, yt = None):

    w1, w2, v1 = layers

    bs = inputs.shape[0]

    h1 = torch.einsum(
        "abc,cd->abd", (inputs.tile((nb_input_copies,)), w1.t()))

    if ref_per_timesteps:
        spk_rec_hidden, mem_rec_hidden = recurrent_layer.compute_activity(
            bs, nb_hidden, h1, v1, nb_steps, lower_bound, ref_counter_hidden)
    else:
        spk_rec_hidden, mem_rec_hidden = recurrent_layer.compute_activity(
            bs, nb_hidden, h1, v1, nb_steps, lower_bound)

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec_hidden, w2.t()))

    if ref_per_timesteps:
        spk_rec_readout, mem_rec_readout, mem_train, n_spike = feedforward_layer.compute_activity(
            bs, nb_outputs, h2, nb_steps, lower_bound, ref_counter_readout)
    else:
        spk_rec_readout, mem_rec_readout, n_spike = feedforward_layer.compute_activity(
            bs, nb_outputs, h2, nb_steps, lower_bound)

    other_recs = [mem_rec_hidden, spk_rec_hidden, mem_rec_readout]
    layers_update = layers

    if(trainable):
        _, am = torch.max(n_spike, 2)
        yo = torch.nn.functional.one_hot(am, num_classes=len(np.unique(labels)))
        #psave(yo.shape, yt.shape)
        grads_batch(inputs.tile((nb_input_copies,)).permute(1,0,2), yo.permute(1,0,2), yt, gamma, 1, mem_rec_hidden.permute(1,0,2), spk_rec_hidden.permute(1,0,2), w1, v1, w2)

    return spk_rec_readout, other_recs, layers_update


def build_and_train(params, ds_train, ds_test, epochs=epochs):

    global nb_input_copies
    # Num of spiking neurons used to encode each channel
    nb_input_copies = params['nb_input_copies']

    # Network parameters
    global nb_inputs
    nb_inputs = nb_channels*nb_input_copies
    global nb_outputs
    nb_outputs = len(np.unique(labels))
    global nb_hidden
    nb_hidden = 450
    global nb_steps
    nb_steps = data_steps

    tau_mem = 0.06 #params['tau_mem']  # ms
    global tau_mem_rec
    tau_mem_rec = 0.06 #params['tau_mem'] #ms
    global tau_trace
    tau_trace = 0.06
    global tau_trace_out
    tau_trace_out = 0.06
    tau_syn = tau_mem/params['tau_ratio']
    psave(["tau_mem: ", tau_mem, "tau_mem recurrent: ", tau_mem_rec, "tau trace out: ", tau_trace_out, "tau trace: ", tau_trace])
    global alpha
    global beta
    global beta_rec
    global beta_trace
    global beta_trace_out
    if no_synapse:
        alpha = 0.0  # here we disable synapse dynamics
    else:
        alpha = float(np.exp(-time_step/tau_syn))

    if use_linear_decay:
        beta = 0.005  # 0.05 < 0.01 says how much to lose
    else:
        beta = float(np.exp(-time_step/tau_mem))  # says how much to keep
        beta_rec = float(np.exp(-time_step/tau_mem_rec))  # says how much to keep
        beta_trace = float(np.exp(-time_step/tau_trace))
        beta_trace_out = float(np.exp(-time_step/tau_trace_out))
    if ref_per_timesteps:
        # initialize as many we have layers with the size of each layer
        global ref_counter_hidden
        ref_counter_hidden = torch.zeros(
            (batch_size, nb_hidden), device=device)
        global ref_counter_readout
        ref_counter_readout = torch.zeros(
            (batch_size, nb_outputs), device=device)

    fwd_weight_scale = params['fwd_weight_scale']
    rec_weight_scale = fwd_weight_scale*params['weight_scale_factor']

    # Spiking network
    layers = []

    # recurrent layer
    w1, v1 = recurrent_layer.create_layer(
        nb_inputs, nb_hidden, fwd_weight_scale, rec_weight_scale)

    # readout layer
    w2 = feedforward_layer.create_layer(
        nb_hidden, nb_outputs, fwd_weight_scale)

    layers.append(w1), layers.append(w2), layers.append(v1)

    # a fixed learning rate is already defined within the train function, that's why here it is omitted
    loss_hist, accs_hist, best_layers = train(
        params=params, dataset=ds_train, layers=layers, lr=lr, nb_epochs=epochs, dataset_test=ds_test)

    # best training and test at best training
    acc_best_train = np.max(accs_hist[0])  # returns max value
    acc_best_train = acc_best_train*100

    idx_best_train = np.argmax(accs_hist[0])  # returns index of max value
    acc_test_at_best_train = accs_hist[1][idx_best_train]*100

    # best test and training at best test
    acc_best_test = np.max(accs_hist[1])
    acc_best_test = acc_best_test*100
    idx_best_test = np.argmax(accs_hist[1])
    acc_train_at_best_test = accs_hist[0][idx_best_test]*100

    # TODO track time constants!!!
    psave("Final results: ")
    psave("Best training accuracy: {:.2f}% and according test accuracy: {:.2f}% at epoch: {}".format(
        acc_best_train, acc_test_at_best_train, idx_best_train+1))
    psave("Best test accuracy: {:.2f}% and according train accuracy: {:.2f}% at epoch: {}".format(
        acc_best_test, acc_train_at_best_test, idx_best_test+1))
    psave("------------------------------------------------------------------------------------\n")

    return loss_hist, accs_hist, best_layers


def train(params, dataset, layers, lr=0.0015, nb_epochs=300, dataset_test=None):

    # The log softmax function across output units
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

    generator = DataLoader(dataset=dataset, batch_size=batch_size, pin_memory=True,
                           shuffle=True, num_workers=2)

    # The optimization loop
    loss_hist = []
    accs_hist = [[], []]
    pbar_training = tqdm(range(nb_epochs), position=1,
                         total=nb_epochs, leave=False)
    for _ in pbar_training:
        # learning rate decreases over epochs
        optimizer = torch.optim.Adamax(layers, lr=lr, betas=(0.9, 0.995))
        # if e > nb_epochs/2:
        #     lr = lr * 0.9
        local_loss = []
        # accs: mean training accuracies for each batch
        accs = []
        pbar_batches = tqdm(generator, position=2,
                            total=len(generator), leave=False)
        for x_local, y_local in pbar_batches:
            x_local, y_local = x_local.to(device), y_local.to(device)
            # reset refractory period counter for each batch
            if ref_per_timesteps:
                # initialize as many we have layers with the size of each layer
                global ref_counter_hidden
                ref_counter_hidden = torch.zeros(
                    (batch_size, nb_hidden), device=device)
                global ref_counter_readout
                ref_counter_readout = torch.zeros(
                    (batch_size, nb_outputs), device=device)
            optimizer.zero_grad()

            one_hot_encoded = torch.nn.functional.one_hot(y_local, num_classes=len(np.unique(labels)))

            spk_rec_readout, recs, layers_update = run_snn(x_local, layers, True, one_hot_encoded)

            _, spk_rec_hidden, _ = recs
            m = torch.sum(spk_rec_readout, 1)  # sum over time

            # cross entropy loss on the active read-out layer
            log_p_y = log_softmax_fn(m)

            # Here we can set up our regularizer loss
            # reg_loss = params['reg_spikes']*torch.mean(torch.sum(spks1,1)) # L1 loss on spikes per neuron (original)
            # L1 loss on total number of spikes (hidden layer 1)
            reg_loss = params['reg_spikes']*torch.mean(torch.sum(spk_rec_hidden, 1))
            # L1 loss on total number of spikes (output layer)
            # reg_loss += params['reg_spikes']*torch.mean(torch.sum(spk_rec_readout, 1))
            # psave("L1: ", reg_loss)
            # reg_loss += params['reg_neurons']*torch.mean(torch.sum(torch.sum(spks1,dim=0),dim=0)**2) # e.g., L2 loss on total number of spikes (original)
            # L2 loss on spikes per neuron (hidden layer 1)
            reg_loss += params['reg_neurons'] * \
                torch.mean(torch.sum(torch.sum(spk_rec_hidden, dim=0), dim=0)**2)
            # L2 loss on spikes per neuron (output layer)
            # reg_loss += params['reg_neurons'] * \
            #     torch.mean(torch.sum(torch.sum(spk_rec_readout, dim=0), dim=0)**2)
            # psave("L1 + L2: ", reg_loss)

            # Here we combine supervised loss and the regularizer
            loss_val = loss_fn(log_p_y, y_local) + reg_loss
            optimizer.step()

            local_loss.append(loss_val.item())

            # compare to labels
            _, am = torch.max(m, 1)  # argmax over output units
            tmp = np.mean((y_local == am).detach().cpu().numpy())
            accs.append(tmp)

        mean_loss = np.mean(local_loss)
        loss_hist.append(mean_loss)

        # mean_accs: mean training accuracy of current epoch (average over all batches)
        mean_accs = np.mean(accs)
        accs_hist[0].append(mean_accs)

        # Calculate test accuracy in each epoch
        if dataset_test is not None:
            test_acc = compute_classification_accuracy(
                dataset=dataset_test, layers=layers_update)
            accs_hist[1].append(test_acc)  # only safe best test

            # save best training
            if mean_accs >= np.max(accs_hist[0]):
                best_acc_layers = []
                for ii in layers_update:
                    best_acc_layers.append(ii.detach().clone())
        else:
            # save best test
            if np.max(test_acc) >= np.max(accs_hist[1]):
                best_acc_layers = []
                for ii in layers_update:
                    best_acc_layers.append(ii.detach().clone())

        pbar_training.set_description("{:.2f}%, {:.2f}%, {:.2f}.".format(
            accs_hist[0][-1]*100, accs_hist[1][-1]*100, loss_hist[-1]))
        psave("Test acc: ", accs_hist[0][-1]*100, "Train acc", accs_hist[1][-1]*100)
    return loss_hist, accs_hist, best_acc_layers


def compute_classification_accuracy(dataset, layers):
    """ Computes classification accuracy on supplied data in batches. """

    generator = DataLoader(dataset=dataset, batch_size=batch_size, pin_memory=True,
                           shuffle=False, num_workers=2)
    accs = []

    for x_local, y_local in generator:
        x_local, y_local = x_local.to(device), y_local.to(device)
        with torch.no_grad():
            spk_rec_readout, _, _ = run_snn(x_local, layers, False)

        # with output spikes
        m = torch.sum(spk_rec_readout, 1)  # sum over time
        _, am = torch.max(m, 1)     # argmax over output units

        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)
    #psave("Test mean accuracy", np.mean(accs))
    return np.mean(accs)


def plot_training_perfromance(path, acc_train, acc_test, loss_train):
    """Visualize the training performance."""
    # calc mean and std
    acc_mean_train, acc_std_train = np.mean(acc_train, axis=0), np.std(acc_train, axis=0)
    acc_mean_test, acc_std_test = np.mean(acc_test, axis=0), np.std(acc_test, axis=0)
    best_trial, best_val_idx = np.where(np.max(acc_test) == acc_test)
    best_trial, best_val_idx = best_trial[0], best_val_idx[0]
    loss_train_mean, loss_train_std = np.mean(loss_train, axis=0), np.std(loss_train, axis=0)

    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(2, 1, 1)
    ax.fill_between(range(1, len(acc_mean_train)+1), 100*(acc_mean_train+acc_std_train), 100*(
        acc_mean_train-acc_std_train), color='cornflowerblue')
    ax.fill_between(range(1, len(acc_mean_test)+1), 100*(
        acc_mean_test+acc_std_test), 100*(acc_mean_test-acc_std_test), color='sandybrown')
    # plot mean and std
    ax.plot(range(1, len(acc_mean_train)+1),
            100*np.array(acc_mean_train), color='blue', linestyle='dashed')
    ax.plot(range(1, len(acc_mean_test)+1), 100 *
            np.array(acc_mean_test), color='orangered', linestyle='dashed')
    # highlight best trial
    ax.plot(range(1, len(acc_train[best_trial])+1), 100*np.array(
        acc_train[best_trial]), color='blue')
    ax.plot(range(1, len(acc_test[best_trial])+1), 100*np.array(
        acc_test[best_trial]), color='orangered')
    # ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim((0, 105))
    ax.set_title("Accuracy")
    ax.legend(["Training std", "Test std", r"$\overline{\mathrm{Training}}$", r"$\overline{\mathrm{Test}}$", "Training @ best test", "Best test"], loc='lower right')

    ax = fig.add_subplot(2, 1, 2)
    ax.fill_between(range(1, len(loss_train_mean)+1), loss_train_mean+loss_train_std, loss_train_mean-loss_train_std, color='cornflowerblue')
    ax.plot(range(1, len(loss_train_mean)+1), loss_train_mean, color='blue', linestyle='dashed')
    ax.plot(range(1, len(loss_train[best_trial])+1), loss_train[best_trial], color='blue')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim((0, None))
    ax.legend(["Training std", r"$\overline{\mathrm{Training}}$", "Training loss @ best test"])
    ax.set_title("Training loss")
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(f"{path}.pdf")
    plt.close(fig)


def plot_confusion_matrix(dataset, layers, labels):
    '''Takes a dataset and the weight matrix to compute the network activity and compare it to the labels.
    Labels are used to write the ticks.'''
    generator = DataLoader(dataset=dataset, batch_size=batch_size, pin_memory=True,
                           shuffle=False, num_workers=2)
    accs = []
    trues = []
    preds = []
    for x_local, y_local in generator:
        x_local, y_local = x_local.to(device), y_local.to(device)
        with torch.no_grad():
            spk_rec_readout, _, _ = run_snn(x_local, layers, False)

        # with output spikes
        m = torch.sum(spk_rec_readout, 1)  # sum over time
        _, am = torch.max(m, 1)     # argmax over output units

        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)
        trues.extend(y_local.detach().cpu().numpy())
        preds.extend(am.detach().cpu().numpy())

    cm = confusion_matrix(trues, preds, normalize='true')
    cm_df = pd.DataFrame(cm, index=[ii for ii in labels], columns=[
                         jj for jj in labels])
    plt.figure(figsize=(12, 9))
    sn.heatmap(cm_df,
               annot=True,
               fmt='.1g',
               cbar=False,
               square=False,
               cmap="YlGnBu")
    plt.xlabel('\nPredicted')
    plt.ylabel('True\n')
    plt.xticks(rotation=0)
    plt.savefig(
            f"./figures/rsnn_1layers_train_tc_thr_{threshold}_cm.pdf")


def get_network_activity(dataset, layers):
    '''Takes a dataset and the weight matrix to compute the network activity.'''

    generator = DataLoader(dataset=dataset, batch_size=batch_size, pin_memory=True,
                           shuffle=False, num_workers=2)
    accs = []
    spk_rec_readout_list = []
    spk_rec_hidden_list = []
    for x_local, y_local in generator:
        x_local, y_local = x_local.to(device), y_local.to(device)
        with torch.no_grad():
            spk_rec_readout, recs, _ = run_snn(x_local, layers, False)

        _, spk_rec_hidden, _ = recs

        # with output spikes
        m = torch.sum(spk_rec_readout, 1)  # sum over time
        _, am = torch.max(m, 1)     # argmax over output units

        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)
        spk_rec_readout_list.append(spk_rec_readout.detach().cpu().numpy())
        spk_rec_hidden_list.append(spk_rec_hidden.detach().cpu().numpy())

    return accs, spk_rec_readout_list, spk_rec_hidden_list


def plot_network_activity(spr_recs, layer_names, figname='./figures'):
    """
    Creates raster plots for all layers using matplotlib.eventplot().
    Input dimension: [timesteps, neurons]
    """
    nb_layers = len(layer_names)
    fig = plt.figure()
    for counter, name in enumerate(layer_names):
        # TODO plot hidden layer activity
        spk_per_layer = spr_recs[counter]
        num_neurons = spk_per_layer.shape[1]
        ax = fig.add_subplot(nb_layers, 1, counter+1)

        spikes_per_neuron = []
        for neuron_idx in range(spk_per_layer.shape[-1]):
            spk_times_per_neuron = np.where(spk_per_layer[:, neuron_idx])[0]
            spk_times_per_neuron = spk_times_per_neuron*0.001*int(params['time_bin_size'])
            spikes_per_neuron.append(spk_times_per_neuron)

        # # TODO possible optimization
        # # Find the indices of spikes (value 1)
        # spike_times, neuron_ids = np.where(spk_per_layer == 1)
        # # Sort by neuron id and then by spike time
        # # sorted_indices = np.lexsort((spike_times, neuron_ids))
        # # # Get the sorted spike times and neuron ids
        # # sorted_spike_times = spike_times[sorted_indices]  # contains the neuron IDs
        # # sorted_neuron_ids = neuron_ids[sorted_indices]  # contains the according spike times
        # # # Group indices by neuron
        # # spikes_per_neuron = {neuron: sorted_spike_times[sorted_neuron_ids == neuron] for neuron in np.unique(sorted_neuron_ids)}

        # # Sort by neuron id and then by spike time
        # sorted_indices = np.lexsort((spike_times, neuron_ids))

        # # Get the sorted spike times and neuron ids
        # sorted_spike_times = spike_times[sorted_indices]
        # sorted_neuron_ids = neuron_ids[sorted_indices]

        # # Get the total number of neurons
        # num_neurons = spk_per_layer.shape[1]

        # # Include empty lists for neurons with no spikes
        # spikes_per_neuron = {neuron: sorted_spike_times[sorted_neuron_ids == neuron].tolist() for neuron in range(num_neurons)}


        # TODO possible colorcode by nb spikes
        psave(len(spikes_per_neuron))
        psave(len(range(num_neurons)))
        ax.eventplot(spikes_per_neuron, orientation="horizontal", lineoffsets=range(num_neurons), linewidth=0.3, colors="k")
        ax.set_ylabel("Neuron ID")
        ax.set_title(f"{name} activity")
    ax.set_xlabel("Time [sec]")
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(f"{figname}.pdf")
    plt.close(fig)


# Load data and parameters
file_dir_data = './data/'
file_type = 'data_braille_letters_th_'
file_thr = str(threshold)
file_name = file_dir_data + file_type + file_thr + '.pkl'

file_dir_params = './parameters/'
param_filename = 'parameters_th' + str(threshold) + '.txt'
file_name_parameters = file_dir_params + param_filename
params = {}
with open(file_name_parameters) as file:
    for line in file:
        (key, value) = line.split()
        if key == 'time_bin_size' or key == 'nb_input_copies':
            params[key] = int(value)
        else:
            params[key] = np.double(value)


def update_refractory_perdiod_counter(spk, counter):
    counter = counter[:spk.shape[0], :spk.shape[1]]
    counter[counter > 0.0] -= 1
    counter[spk > 0.0] = ref_per_timesteps
    return counter


class feedforward_layer:
    '''
    class to initialize and compute spiking feedforward layer
    '''
    def create_layer(nb_inputs, nb_outputs, scale):
        ff_layer = torch.empty((nb_outputs, nb_inputs),
                               device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(ff_layer, mean=0.0, std=scale/np.sqrt(nb_inputs))
        return ff_layer

    def compute_activity(nb_input, nb_neurons, input_activity, nb_steps, lower_bound=None, ref_per_counter=None):
        syn = torch.zeros((nb_input, nb_neurons), device=device, dtype=dtype)
        new_syn = torch.zeros((nb_input, nb_neurons),
                              device=device, dtype=dtype)
        mem       = torch.zeros((nb_input, nb_neurons), device=device, dtype=dtype)
        mem_t     = torch.zeros((nb_input, nb_neurons), device=device, dtype=dtype)
        out       = torch.zeros((nb_input, nb_neurons), device=device, dtype=dtype)
        n_spike   = torch.zeros((nb_input, nb_neurons), device=device, dtype=dtype)

        mem_rec     = []
        spk_rec     = []
        mem_t_rec   = []
        n_spike_tot = []
        # Compute feedforward layer activity
        for t in range(nb_steps):
            mthr = mem-1.0
            out = torch.zeros_like(mthr)
            out[mthr > 0] = 1
            n_spike[out == 1] = n_spike[out == 1] + 1
            rst = out.detach()

            # update the correct counter
            if ref_per_counter is not None:
                update_refractory_perdiod_counter(rst, ref_per_counter)
                # take care of last batch
                mask = ref_per_counter[:syn.shape[0], :syn.shape[1]] == 0.0
                new_syn = alpha * syn
                new_syn[mask] = (alpha*syn[mask] + input_activity[:, t][mask])
            else:
                new_syn = alpha*syn + input_activity[:, t]

            if use_linear_decay:
                # torch.sign returns: 1 if x > 0, -1 if x < 0, and 0 if x == 0
                new_mem = ((mem-torch.sign(mem)*beta) + syn)*(1.0-rst)
            else:
                new_mem   = (beta*mem + syn)*(1.0-rst)
                new_mem_t = (beta*mem_t + syn)
            if lower_bound:
                new_mem[new_mem < lower_bound] = lower_bound
                new_mem_t[new_mem_t < lower_bound] = lower_bound

            mem_rec.append(mem)
            spk_rec.append(out)
            mem_t_rec.append(mem_t)
            n_spike_tot.append(n_spike)

            mem = new_mem
            mem_t = new_mem_t
            syn = new_syn

        # Now we merge the recorded membrane potentials into a single tensor
        mem_rec     = torch.stack(mem_rec, dim=1)
        spk_rec     = torch.stack(spk_rec, dim=1)
        mem_t_rec   = torch.stack(mem_t_rec, dim=1)
        n_spike_tot = torch.stack(n_spike_tot, dim=1)
        return spk_rec, mem_rec, mem_t_rec, n_spike_tot

class recurrent_layer:
    '''
    class to initialize and compute spiking recurrent layer
    '''
    def create_layer(nb_inputs, nb_outputs, fwd_scale, rec_scale):
        ff_layer = torch.empty((nb_outputs, nb_inputs),
                               device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(ff_layer, mean=0.0,
                              std=fwd_scale/np.sqrt(nb_inputs))
        rec_layer = torch.empty((nb_outputs, nb_outputs),
                                device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(rec_layer, mean=0.0,
                              std=rec_scale/np.sqrt(nb_inputs))
        return ff_layer,  rec_layer

    def compute_activity(batch_size, nb_neurons, input_activity, layer, nb_steps, lower_bound=None, ref_per_counter=None):
        syn = torch.zeros((batch_size, nb_neurons), device=device, dtype=dtype)
        new_syn = torch.zeros((batch_size, nb_neurons),
                              device=device, dtype=dtype)
        mem = torch.zeros((batch_size, nb_neurons), device=device, dtype=dtype)
        out = torch.zeros((batch_size, nb_neurons), device=device, dtype=dtype)
        mem_rec = []
        spk_rec = []

        # Compute recurrent layer activity
        for t in range(nb_steps):
            # input activity plus last step output activity
            h1 = input_activity[:, t] + torch.einsum("ab,bc->ac", (out, layer.t()))
            mthr = mem-1.0
            out = torch.zeros_like(mthr)
            out[mthr > 0] = 1
            rst = out.detach()  # We do not want to backprop through the reset

            if ref_per_counter is not None:
                update_refractory_perdiod_counter(rst, ref_per_counter)
                # only update the membrane potential if not in refractory period
                # take care of last batch
                mask = ref_per_counter[:syn.shape[0], :syn.shape[1]] == 0.0
                new_syn = alpha * syn
                new_syn[mask] = (alpha*syn[mask] + h1[mask])
            else:
                new_syn = alpha*syn + h1

            if use_linear_decay:
                new_mem = ((mem-torch.sign(mem)*beta_rec) + syn)*(1.0-rst)
            else:
                new_mem = (beta_rec*mem + syn)*(1.0-rst)

            if lower_bound:
                # clamp membrane potential
                new_mem[new_mem < lower_bound] = lower_bound

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        # Now we merge the recorded membrane potentials into a single tensor
        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        return spk_rec, mem_rec


if __name__ == '__main__':
    with torch.no_grad():
        acc_train_list = []
        acc_test_list = []
        loss_train_list = []
        max_repetitions = 1

        pbar_repetitions = tqdm(range(max_repetitions), position=0, total=max_repetitions, leave=True)
        for repetition in pbar_repetitions:
            pbar_repetitions.set_description(f"{repetition+1}/{max_repetitions}")
            # load data for each repetition indepoently to get different splits
            ds_train, ds_test, ds_validation, labels, nb_channels, data_steps = load_and_extract(
                params, file_name, letter_written=letters)
            if repetition == 0:
                psave("Number of training data %i." % len(ds_train))
                psave("Number of testing data %i." % len(ds_test))
                psave("Number of validation data %i." % len(ds_validation))
                psave("Number of outputs %i." % len(np.unique(labels)))
                psave("Number of timesteps %i." % data_steps)
                if no_synapse:
                    psave(f"No synapse dynamics.")
                if lower_bound:
                    psave(f"Clamp membrane voltage to: {lower_bound}.")
                if use_linear_decay:
                    psave(f"Use linear decay.")
                else:
                    psave(f"Use exponential decay.")
                if ref_per_timesteps:
                    psave(f"Refractory period set to {ref_per_timesteps} simulation timesteps.")
                psave("Input duration %fs" % (data_steps*time_step))
                psave("---------------------------\n")

            # initialize and train network
            loss_hist, acc_hist, best_layers = build_and_train(
                params, ds_train, ds_test, epochs=epochs)

            # get validation results
            val_acc = compute_classification_accuracy(
                dataset=ds_validation, layers=best_layers)

            # safe overall best layer
            if repetition == 0:
                very_best_layer = best_layers
                best_acc = val_acc
            else:
                if val_acc > best_acc:
                    very_best_layer = best_layers
                    best_acc = val_acc

            acc_train_list.append(acc_hist[0])
            acc_test_list.append(acc_hist[1])
            loss_train_list.append(loss_hist)

        acc_train_list = np.array(acc_train_list)
        acc_test_list = np.array(acc_test_list)
        loss_train_list = np.array(loss_train_list)

        psave("*************************")
        psave("* Best: ", best_acc*100)
        psave("*************************")


        # save the best layer
        st()
        torch.save(very_best_layer, './model/best_model_th'+str(threshold)+'.pt')

        # ### Lets plot the training curve and the confusion matrix
        plot_training_perfromance(path=f"./figures/rsnn_1layers_train_tc_thr_{threshold}_acc", acc_train=acc_train_list, acc_test=acc_test_list, loss_train=loss_train_list)
        # plotting the confusion matrix
        plot_confusion_matrix(dataset=ds_test, layers=very_best_layer, labels=letters)

        #####################################
        ### Lets create some raster plots ###
        #####################################

        # plotting the network activity
        accs, spk_rec_readout_array, spk_rec_hidden_array = get_network_activity(ds_test, layers=very_best_layer)
        #
        layer_names = ["Hidden layer", "Readout layer"]
        nb_layers = len(layer_names)

        total_nb_batches = len(accs)

        # select the batches to plot
        if NB_BATCHES_TO_PLOT > total_nb_batches:
            psave(f"WARNING: Not enough batches to plot. Will plot all {total_nb_batches} batches instead of the asked {NB_BATCHES_TO_PLOT}. Lower the number to avoid this warning.")
            batch_selection = range(NB_BATCHES_TO_PLOT)
        elif NB_BATCHES_TO_PLOT == total_nb_batches:
            psave(f"Plotting all {total_nb_batches} batches.")
            batch_selection = range(NB_BATCHES_TO_PLOT)
        else:
            psave(f"Plotting {NB_BATCHES_TO_PLOT} random batches (out of {total_nb_batches}).")
            found_unique = False
            while not found_unique:
                batch_selection = np.random.choice(total_nb_batches, NB_BATCHES_TO_PLOT)
                if len(np.unique(batch_selection)) == NB_BATCHES_TO_PLOT:
                    found_unique = True

        for batch_idx in batch_selection:
            batch_acc = accs[batch_idx]
            spk_rec_readout_batch = spk_rec_readout_array[batch_idx]  # [trials, timesteps, neurons]
            spk_rec_hidden_batch = spk_rec_hidden_array[batch_idx]  # [trials, timesteps, neurons]
            # select random trials to plot
            total_nb_trials = len(spk_rec_readout_batch)
            if NB_TRIALS_TO_PLOT > total_nb_trials:
                psave(f"WARNING: Not enough trials to plot. Will plot all {total_nb_trials} trials instead of the asked {NB_TRIALS_TO_PLOT}. Lower the number to avoid this warning.")
                trial_selection = range(NB_BATCHES_TO_PLOT)
            elif NB_TRIALS_TO_PLOT == total_nb_trials:
                psave(f"Plotting all {total_nb_trials} trials.")
                trial_selection = range(NB_TRIALS_TO_PLOT)
            else:
                psave(f"Plotting {NB_TRIALS_TO_PLOT} random trials (out of {total_nb_trials}).")
                found_unique = False
                while not found_unique:
                    trial_selection = np.random.choice(total_nb_trials, NB_TRIALS_TO_PLOT)
                    if len(np.unique(trial_selection)) == NB_TRIALS_TO_PLOT:
                        found_unique = True

            for trial_idx in trial_selection:
                spr_recs = [spk_rec_hidden_batch[trial_idx], spk_rec_readout_batch[trial_idx]]
                # TODO include more specifics into the figure name
                plot_network_activity(spr_recs, layer_names, figname=f'./figures/network_activity_batch_{batch_idx}_trial_{trial_idx}')


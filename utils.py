import dataclasses

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm.notebook import tqdm

# set plotting syle
def set_style():
    sns.set_context('notebook')
    sns.set_style('darkgrid')
    plt.rc('axes', linewidth=1)
    plt.rc('axes', edgecolor='k')
    plt.rc('figure', dpi=100)

# custom progress bar for long training
class tqdm_callback(tf.keras.callbacks.Callback):
    
    def __init__(self, num_epochs, desc, loss=None, val_loss=None):
        self.num_epochs = num_epochs
        self.desc = desc
        self.metrics = {'loss':loss, 'val_loss':val_loss}
    
    def on_train_begin(self, logs={}):
        self.epoch_bar = tqdm(total=self.num_epochs, desc=self.desc)
    
    def on_train_end(self, logs={}):
        self.epoch_bar.close()
        
    def on_epoch_end(self, epoch, logs={}):
        for name in self.metrics:
            self.metrics[name]  = logs.get(name, self.metrics[name])
        self.epoch_bar.set_postfix(mse=self.metrics['loss'], val_mse=self.metrics['val_loss'], refresh=False)
        self.epoch_bar.update()

# plot Lorenz 1963 trajectory
def plot_l63_traj_truth_obs(
    x_truth, 
    x_raw,
    dt,
    Nt_shift,
    t_plot,
    linewidth,
):
    Nt_plot_t = int(t_plot/dt)
    Nt_plot_r = int(t_plot/dt/Nt_shift)
    palette = sns.color_palette('deep')
    sub_palette = [
        palette[0],
        palette[3],
        palette[2],
    ]
    fig = plt.figure(figsize=(linewidth, linewidth/3))
    for (n, variable) in enumerate('xyz'):
        plt.plot(
            dt*np.arange(Nt_plot_t), 
            x_truth[:Nt_plot_t, n], 
            label='${}$'.format(variable),
            c=sub_palette[n],
        )
        plt.plot(
            dt*Nt_shift*np.arange(Nt_plot_r), 
            x_raw[:Nt_plot_r, n], 
            '.',
            c = sub_palette[n],
        )
    plt.xlabel('time (MTU)')
    plt.ylabel('L63 variables')
    plt.title('True L63 trajectory (lines) and observations (dots)')
    plt.xlim(0, t_plot)
    plt.ylim(-30, 50)
    plt.legend()
    plt.show()

# plot any learning curve
def plot_learning_curve(
    loss,
    val_loss,
    title,
    linewidth,
):
    palette = sns.color_palette('deep')
    sub_palette = [
        palette[0],
        palette[3],
    ]
    fig = plt.figure(figsize=(linewidth, linewidth/3))
    plt.plot(loss, c=sub_palette[0], label='training loss')
    plt.plot(val_loss, c=sub_palette[1], label='validation loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.title(title)
    plt.legend()

# standard L96 model
@dataclasses.dataclass
class Lorenz1996Model:
    Nx: 'number of variables'
    F: 'forcing'
    dt: 'integration time step'
    steps: 'integration scheme steps' = dataclasses.field(init=False)
    weights: 'integration scheme weights' = dataclasses.field(init=False)

    def __post_init__(self):
        self.steps = np.array([0, self.dt / 2, self.dt / 2, self.dt])
        self.weights = np.array([1, 2, 2, 1])
        self.weights = self.weights / self.weights.sum()

    def tendency(self, x):
        xp = np.roll(x, shift=-1, axis=-1)
        xmm = np.roll(x, shift=+2, axis=-1)
        xm = np.roll(x, shift=+1, axis=-1)
        return (xp - xmm)*xm - x + self.F

    def forward(self, x):
        averaged_dx_dt = np.zeros_like(x)
        current_dx_dt = np.zeros_like(x)
        for (w, dt) in zip(self.weights, self.steps):
            current_x = x + current_dx_dt * dt
            current_dx_dt = self.tendency(current_x)
            averaged_dx_dt += w * current_dx_dt
        return x + averaged_dx_dt * self.dt

# plot single Lorenz 1996 trajectory
def plot_l96_traj(
    x,
    model,
    linewidth,
):
    fig = plt.figure(figsize=(linewidth, linewidth/3))
    plt.grid(False)
    im = plt.imshow(
        x.T, 
        aspect = 'auto',
        origin = 'lower',
        interpolation = 'spline36',
        cmap = sns.diverging_palette(240, 60, as_cmap=True),
        extent = [0, model.dt*x.shape[0], 0, model.Nx],
        vmin = -10,
        vmax = 15,
    )
    plt.colorbar(im)
    plt.xlabel('Time (MTU)')
    plt.ylabel('Lorenz 96 variables')
    plt.tick_params(direction='out', left=True, bottom=True)
    plt.show()

# plot comparative Lorenz 1996 trajectories
def plot_l96_compare_traj(
    x_ref,
    x_pred,
    model,
    linewidth,
):
    error = x_pred - x_ref
    fig = plt.figure(figsize=(linewidth, linewidth))
    ax = plt.subplot(311)
    ax.grid(False)
    im = plt.imshow(
        x_ref.T, 
        aspect = 'auto',
        origin = 'lower',
        interpolation = 'spline36',
        cmap = sns.diverging_palette(240, 60, as_cmap=True),
        extent = [0, model.dt*x_pred.shape[0], 0, model.Nx],
        vmin = -10,
        vmax = 15,
    )
    ax.set_title('true model integration')
    plt.colorbar(im)
    ax.set_ylabel('Lorenz 96 variables')
    ax.tick_params(direction='out', left=True, bottom=True)
    ax.set_xticklabels([])
    ax = plt.subplot(312)
    ax.grid(False)
    im = plt.imshow(
        x_pred.T,
        aspect = 'auto',
        origin = 'lower',
        interpolation = 'spline36',
        cmap = sns.diverging_palette(240, 60, as_cmap=True),
        extent = [0, model.dt*x_pred.shape[0], 0, model.Nx],
        vmin = -10,
        vmax = 15,
    )
    ax.set_title('surrogate model integration')
    plt.colorbar(im)
    ax.set_ylabel('Lorenz 96 variables')
    ax.tick_params(direction='out', left=True, bottom=True)
    ax.set_xticklabels([])
    ax = plt.subplot(313)
    ax.grid(False)
    im = ax.imshow(
        error.T, 
        aspect = 'auto',
        origin = 'lower',
        interpolation = 'spline36',
        cmap = sns.diverging_palette(240, 10, as_cmap=True),
        extent = [0, model.dt*error.shape[0], 0, model.Nx],
        vmin = -15,
        vmax = 15,
    )
    ax.set_title('signed error')
    plt.colorbar(im)
    ax.set_xlabel('Time (MTU)')
    ax.set_ylabel('Lorenz 96 variables')
    ax.tick_params(direction='out', left=True, bottom=True)
    plt.show()
    
# plot Lorenz 1996 forecast skill
def plot_l96_forecast_skill(
    fss,
    model,
    p1,
    p2,
    xmax,
    linewidth,
):
    fig = plt.figure(figsize=(linewidth, linewidth/2))
    palette = sns.color_palette('deep')
    palette.pop(1)
    for (c, key) in zip(palette, fss):
        time = (model.dt/model.lyap_time)*np.arange(fss[key].shape[0])
        rmse_m = fss[key].mean(axis=1) / model.model_var
        rmse_p1 = np.percentile(fss[key], p1, axis=1) / model.model_var
        rmse_p2 = np.percentile(fss[key], p2, axis=1) / model.model_var
        plt.plot(time, rmse_m, color=c, label=key)
        plt.fill_between(time, rmse_p1, rmse_p2, color=c, alpha=0.1)
    plt.axhline(np.sqrt(2), c='k', ls='--', label='$\sqrt{2}$')
    plt.xlabel('Time (Lyapunov time)')
    plt.ylabel('normalised RMSE')
    plt.ylim(0, 2)
    plt.xlim(0, xmax)
    plt.title('Forecast skill')
    plt.legend()
    plt.show()
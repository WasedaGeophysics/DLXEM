import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def loss_plot(hist_df, epochs):
    plt.rcParams["font.size"] = 10
    plt.tight_layout()
    epochx = np.arange(1, epochs+1)
    val_mse = hist_df['val_loss'].values
    loss = hist_df['loss'].values
    fig = plt.figure(figsize=(6,4), edgecolor='w')
    ax1 = fig.add_subplot(111)
    ax1.plot(epochx, loss, label='train loss', linewidth=2)
    ax1.plot(epochx, val_mse, label='validation loss', linewidth=2)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('MSE')
    ax1.set_xlim(0, epochx.max())
    ax1.set_ylim(0, loss.max() // 1 + 1)
    ax1.legend()
    ax1.grid()
    ax1.set_title('Loss Transition', fontsize = 12)
    plt.show()
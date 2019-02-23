import matplotlib.pyplot as plt
import os
def plot_loss_vec(loss_vec, plot_name):
    if os.path.isfile(plot_name):
        os.remove(plot_name)
    plt.plot(loss_vec, linestyle='--', marker='o', color='b')
    plt.title('Pre-training discriminator loss')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.grid()
    plt.savefig(plot_name, bbox_inches='tight')
    plt.close()
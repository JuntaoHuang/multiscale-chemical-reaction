import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import sys
sys.path.append('..')
import regression

# set random seed to make results reproducible
rand_seed_num = 0
torch.manual_seed(rand_seed_num)
    
Nx = 1000
data_x = torch.rand(Nx, 1)

c1, c2 = 1.0, 1e2
p1, p2 = 1.0, 2.0
data_y = c1 * (data_x ** p1) + c2 * (data_x ** p2)

reg = regression.Regression(np.array([c1, c2]))

# plot loss as a 1D function of p1 with fixed p2
np1 = 100000
p1_array = np.linspace(0, 3, np1)

p2_value = 2.0 - 1e-2       # fix p2 = 1.99
loss_1d = reg.compute_loss_1d(data_x, data_y, p1_array, p2_value)
plt.semilogy(p1_array, loss_1d, label=r'$\theta_2$' + ' = 1.99')

p2_value = 2.0              # fix p2 = 2
loss_1d = reg.compute_loss_1d(data_x, data_y, p1_array, p2_value)
plt.semilogy(p1_array, loss_1d, label=r'$\theta_2$' + ' = 2')

p2_value = 2.0 + 1e-2       # fix p2 = 2.01
loss_1d = reg.compute_loss_1d(data_x, data_y, p1_array, p2_value)
plt.semilogy(p1_array, loss_1d, label=r'$\theta_2$' + ' = 2.01')

plt.xlim(0, 3)
plt.xlabel(r'$\theta_1$')
plt.ylabel('loss')
plt.legend(loc='lower right')
plt.savefig('plot_loss_1d.eps')

plt.show()
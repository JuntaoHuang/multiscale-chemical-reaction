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

total_epoch = 200000

# without freeze integer
reg = regression.Regression(np.array([c1, c2]))
p1_init, p2_init = reg.get_params()
reg.train(data_x, data_y, is_freeze_integer=False, total_epoch=total_epoch, loss_params_file_name='loss_params_history_freeze_false.npy')

# with freeze integer
reg = regression.Regression(np.array([c1, c2]))
reg.set_params(np.array([p1_init, p2_init]))
reg.train(data_x, data_y, is_freeze_integer=True, total_epoch=total_epoch, loss_params_file_name='loss_params_history_freeze_true.npy')

# read data and plot
[epoch_array_freeze_int_false, 
 loss_history_freeze_int_false, 
 p_history_freeze_int_false] = reg.read_loss_params('loss_params_history_freeze_false.npy')

[epoch_array_freeze_int_true, 
 loss_history_freeze_int_true, 
 p_history_freeze_int_true] = reg.read_loss_params('loss_params_history_freeze_true.npy')

plt.figure()
plt.semilogy(epoch_array_freeze_int_true, loss_history_freeze_int_true, '-', label='freezing')
plt.semilogy(epoch_array_freeze_int_false, loss_history_freeze_int_false, '--', label='no freezing')
plt.xlim(0, total_epoch)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xticks(np.arange(0, total_epoch+1, step=50000))
plt.legend()
plt.savefig('loss_history.eps')

plt.figure()
plt.plot(epoch_array_freeze_int_true, p_history_freeze_int_true[:,0], '-', label=r'$\theta_1$' + ' (freezing)')
plt.plot(epoch_array_freeze_int_true, p_history_freeze_int_true[:,1], '-', label=r'$\theta_2$' + ' (freezing)')
plt.plot(epoch_array_freeze_int_false, p_history_freeze_int_false[:,0], '--', label=r'$\theta_1$' + ' (no freezing)')
plt.plot(epoch_array_freeze_int_false, p_history_freeze_int_false[:,1], '--', label=r'$\theta_2$' + ' (no freezing)')
plt.xlim(0, total_epoch)
plt.xlabel('epoch')
plt.ylabel(r'$\theta_1, \theta_2$')
plt.xticks(np.arange(0, total_epoch+1, step=50000))
plt.legend()
plt.savefig('params_history.eps')

plt.show()
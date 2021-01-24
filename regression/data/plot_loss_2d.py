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

# plot loss as a 2D function of p1 and p2
p1_arr_1d = np.linspace(-1, 5, 10000)
p2_arr_1d = np.linspace(2 - 1e-1, 2 + 1e-1, 100)

p1_arr_2d, p2_arr_2d, loss_2d = reg.compute_loss_2d(data_x, data_y, p1_arr_1d, p2_arr_1d)
with open('loss_landscape_2d.npy', 'wb') as f:
    np.savez(f, p1_arr_2d=p1_arr_2d,
                p2_arr_2d=p2_arr_2d,
                loss_2d=loss_2d)

# with open('loss_landscape_2d.npy', 'rb') as f:
#     data = np.load(f)
#     p1_arr_2d = data['p1_arr_2d']
#     p2_arr_2d = data['p2_arr_2d']
#     loss_2d = data['loss_2d']

# surface plot
fig = plt.figure()
ax = fig.gca(projection='3d')
f = ax.plot_surface(p1_arr_2d, p2_arr_2d, np.log(loss_2d),
                    linewidth=0,
                    antialiased=True, 
                    cmap=plt.cm.coolwarm)
ax.grid(False)
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')
ax.set_zlabel('log(loss)')
plt.yticks([1.9, 2.0, 2.1])
ax.view_init(elev=14, azim=-81)
plt.savefig('plot_params_2d_surface.eps')


# contour plot
fig, ax = plt.subplots()
f = plt.contour(p1_arr_2d, p2_arr_2d, np.log(loss_2d),
                linewidths = 1.0,
                levels = np.linspace(-10, 0, 51),
                cmap="jet")
cbar = fig.colorbar(f, ticks=np.linspace(-10, 0, 6))
cbar.set_label('log(loss)')

# label local minima
xs, ys = 3.8286325931549072, 1.9745292663574219
plt.scatter(xs, ys, c='k')
plt.annotate(
    'local minima',
    xy=(xs, ys), xytext=(3.5, 1.995),
    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

# label global minima
plt.scatter(1, 2, c='k')
plt.annotate(
    'global minima',
    xy=(1, 2), xytext=(2, 2.005),
    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

plt.xlim(0, 5)
plt.ylim(1.96, 2.02)
plt.xlabel(r'$\theta_1$', fontsize=12)
plt.ylabel(r'$\theta_2$', fontsize=12)
plt.savefig('plot_params_2d_contour.eps')

plt.show()
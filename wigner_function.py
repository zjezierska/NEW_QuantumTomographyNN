from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
import matplotlib
import tensorflow as tf


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


d = 5
D = 40
alpha = 5
rho0 = rand_dm_hs(N=d)  # initial state
rho1 = coherent_dm(N=d, alpha=1.0)

full_array = np.full([D, D], 0. + 0.j)
full_array[0:d, 0:d] = rho0.full()
full_state = Qobj(full_array)

xvec = np.linspace(-5, 5, 200)
a = destroy(d)  # annihilation operator
x = a.dag() + a
p = 1j * (a.dag() - a)
H_quartic = p * p / 4 + (x / alpha) * (x / alpha) * (x / alpha) * (x / alpha)
H_harmonic = a.dag() * a

Wigner = wigner(rho0, xvec, xvec)

fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2.5, 1]})
plot1 = axs[0].contourf(xvec, xvec, Wigner, 100, cmap='RdBu_r')
pd = HarmonicOscillatorProbabilityFunction()  # create probabiliy distribution object
pd.update(rho0)  # function to return P(x) from rho - returns P as pd.data and x as pd.xvecs[0]
axs[1].plot(pd.xvecs[0], np.real(pd.data), '-')

frames = 300
tlist = np.linspace(0, 15, frames)
#
# for i in range(3):
#     W = wigner(result.states[i], xvec, xvec)
#     axs[0, i].contourf(xvec, xvec, W, 100, cmap='RdBu_r')
#     pd.update(result.states[i])
#     axs[1, i].plot(pd.xvecs[0], np.real(pd.data), '-')
#     axs[0, 0].set_xticklabels([])
#     if i != 0:
#         axs[0, i].set_xticklabels([])
#         axs[0, i].set_yticklabels([])
#         axs[1, i].set_yticklabels([])
#
# axs[1, 0].set_xlabel(r'$x$')
# axs[1, 0].set_ylabel(r'$P(x)$')
#
# plt.subplots_adjust(wspace=1)


# request that the solver return the expectation value of the photon number state operator
result = mesolve(H_harmonic, rho0, tlist, [], [])
# for i in range(9):
#     W = wigner(result.states[i], xvec, xvec)
#     axs[0].contourf(xvec, xvec, W, 100, cmap='RdBu_r')
#     pd.update(result.states[i])
#     axs[1].plot(pd.xvecs[0], np.real(pd.data), '-')


def animate_plot(i):
    plt.cla()
    W = wigner(result.states[i], xvec, xvec)
    axs[0].contourf(xvec, xvec, W, 100, cmap='RdBu_r')
    pd.update(result.states[i])
    axs[1].plot(pd.xvecs[0], np.real(pd.data), '-')
    axs[1].set_ylim([0, 0.5])
    # axs[0].set_aspect('equal', adjustable='box')
    # axs[1].set_aspect('auto', adjustable='box')
    axs[1].set_ylabel(r"$P(x)$")
    axs[1].set_xlabel(r"$x$")


ani = FuncAnimation(fig, animate_plot, frames=frames, interval=200)

# # Save as gif

ani.save('animation2.5.gif', fps=10)
#fig.savefig('myimage.svg', format='svg', dpi=1200)
plt.show()

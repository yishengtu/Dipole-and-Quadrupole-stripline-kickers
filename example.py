'''
Yisheng Tu, Tanaji Sen, Jean-Francois Ostiguy
'''

import Dipole as gd
import Quadrupole as qq
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors


def dipole_example():
    a = 25  # radius of beampipe
    b = 20  # radius of the kicker plates
    theta_0 = 0.4*np.pi

    # create a dipole object instance
    dipole = gd.dipole(a = a, b = b, theta_0 = theta_0, mode='odd')

    # print out the characteristic impedance
    # dipole.get_characteristic_impedance(thickness=8, kf_even=5, kf_geo=5.5)     # using customized fitting value
    dipole.get_characteristic_impedance(thickness=8)    # using default fitting value

    # setup 2D-scan parameters
    x = np.linspace(-a, a, 50)  # change here to change the resolution!
    y = np.linspace(-a, a, 50)  # change here to change the resolution!
    x, y = np.meshgrid(x, y)
    x0 = np.concatenate(x)
    y0 = np.concatenate(y)

    # calculate the potential at all points on the 2D-scan
    num_of_pt = len(x[0])
    z_list = [dipole.phi_xy(xx, yy, number_of_item=200) for xx, yy in zip(x0, y0)]
    z = [z_list[i:i + num_of_pt] for i in range(0, len(z_list) - 1, num_of_pt)]

    # plot the figure, setup the plot parameters
    fig = plt.figure()
    fig.set_size_inches(12, 10)
    dim = (2, 2)
    ax1 = plt.subplot2grid(dim, (0, 0), colspan=2, rowspan=2)

    # make 2D-pseudocolor plot
    c = ax1.pcolor(x, y, z, cmap='RdBu', norm=colors.Normalize(vmin=-1, vmax=1))

    # adjust colorbar
    cbar = fig.colorbar(c, ax=ax1)
    cbar.ax.set_yticklabels([round(x, 4) for x in cbar.get_ticks()], fontsize = 20)

    # adjust aspect ratio
    ax1.set_aspect('equal')
    plt.show()


def quadrupole_example():
    a = 25  # radius of beampipe
    b = 20  # radius of the kicker plates
    theta_0 = 0.125 * np.pi

    # create a quadrupole object instance
    quadrupole = qq.quadrupole(a=a, b=b, theta_0=theta_0, mode='sum')

    # print out the characteristic impedance
    # quadrupole.get_characteristic_impedance(thickness=8, kf_sum=3, kf_geo=3.5)     # using customized fitting value
    quadrupole.get_characteristic_impedance(thickness=5)    # using default fitting value

    # setup 2D-scan parameters
    x = np.linspace(-a, a, 150)  # change here to change the resolution!
    y = np.linspace(-a, a, 150)  # change here to change the resolution!
    x, y = np.meshgrid(x, y)
    x0 = np.concatenate(x)
    y0 = np.concatenate(y)

    # calculate the potential at all points on the 2D-scan
    num_of_pt = len(x[0])
    z_list = [quadrupole.phi_xy(xx, yy, number_of_item=200) for xx, yy in zip(x0, y0)]
    z = [z_list[i:i + num_of_pt] for i in range(0, len(z_list) - 1, num_of_pt)]

    # plot the figure, setup the plot parameters
    fig = plt.figure()
    fig.set_size_inches(12, 10)
    dim = (2, 2)
    ax1 = plt.subplot2grid(dim, (0, 0), colspan=2, rowspan=2)

    # make 2D-pseudocolor plot
    c = ax1.pcolor(x, y, z, cmap='RdBu', norm=colors.Normalize(vmin=-1, vmax=1))

    # adjust colorbar
    cbar = fig.colorbar(c, ax=ax1)
    cbar.ax.set_yticklabels([round(x, 4) for x in cbar.get_ticks()], fontsize = 20)

    # adjust aspect ratio
    ax1.set_aspect('equal')
    plt.show()


dipole_example()
quadrupole_example()

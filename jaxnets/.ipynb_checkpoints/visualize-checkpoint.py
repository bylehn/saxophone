import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import jax.numpy as np
import numpy as onp
import jaxnets.simulation as simulation
<<<<<<< HEAD
=======
import matplotlib as mpl

mpl.rcParams.update({'font.size': 28})
>>>>>>> main
sns.set_style(style='white')

def format_plot(x, y):
  plt.xlabel(x, fontsize=20)
  plt.ylabel(y, fontsize=20)

def finalize_plot(shape=(1, 1)):
  plt.gcf().set_size_inches(
    shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
    shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
  plt.tight_layout()



def makemovie(N, G, traj, amp, xylims, stride=10):

    # Set style
    sns.set_style(style='white')

    # Define the init function, which sets up the plot
    def init():

        plt.axis('on')
        return plt

    # Define the update function, which is called for each frame
    def update(frame):

        plt.clf()  # Clear the current figure
        R_plt = traj['position'][frame]
        R_0 = traj['position'][0]
        R_plt = R_0 + amp * (R_plt - R_0)

        pos = {i: (R_plt[i, 0], R_plt[i, 1]) for i in range(N)}
        nx.draw(G, pos=pos, with_labels=False, node_size=2, font_size=8, font_color='black', font_weight='bold')
        plt.xlim([0, xylims])
        plt.ylim([0, xylims])

        plt.axis('on')
        return plt

    # Create the animation
<<<<<<< HEAD
    fig, ax = plt.subplots(figsize=(5, 5))
=======
    fig, ax = plt.subplots(figsize=(10, 10))
>>>>>>> main
    ani = FuncAnimation(fig, update, frames=range(0, len(traj['position']), stride), init_func=init, blit=False)
    #ani.save('example.gif', writer='imagemagick')
    # Display the animation
    display(HTML(ani.to_jshtml()))
    plt.show()
    return ani

def makemovie_bondwidth(system, k, traj, amp=1., xylims=9., stride=10):

    # Set style
    sns.set_style(style='white')
    k = np.squeeze(k)
    # Define the init function, which sets up the plot
    def init():

        plt.axis('on')
        return plt

    # Define the update function, which is called for each frame
    def update(frame):

        plt.clf()  # Clear the current figure
        R_plt = traj['position'][frame]
        R_0 = traj['position'][0]
        R_plt = R_0 + amp * (R_plt - R_0)

        pos = {i: (R_plt[i, 0], R_plt[i, 1]) for i in range(system.N)}
<<<<<<< HEAD
        nx.draw_networkx_edges(system.G, pos, width=1*k, alpha=0.6,edge_color='k')
=======
        nx.draw_networkx_edges(system.G, pos, width=2*k, alpha=0.6,edge_color='k')
>>>>>>> main
        plt.xlim([0, xylims])
        plt.ylim([0, xylims])

        plt.axis('on')
        return plt

    # Create the animation
<<<<<<< HEAD
    fig, ax = plt.subplots(figsize=(5, 5))
=======
    fig, ax = plt.subplots(figsize=(10, 10))
>>>>>>> main
    ani = FuncAnimation(fig, update, frames=range(0, len(traj['position']), stride), init_func=init, blit=False)
    ani.save('compressedexample.gif', writer='imagemagick')
    # Display the animation
    display(HTML(ani.to_jshtml()))
    plt.show()
    return ani

def makemovie_bondwidth_labels(system, k, traj, amp=1., xylims=9., stride=10):
    sns.set_style(style='white')
    k = np.squeeze(k)
    
    def init():
        plt.axis('on')
        return plt

    def update(frame):
        plt.clf()  # Clear the current figure
        R_plt = traj['position'][frame]
        R_0 = traj['position'][0]
        R_plt = R_0 + amp * (R_plt - R_0)

        pos = {i: (R_plt[i, 0], R_plt[i, 1]) for i in range(system.N)}
        nx.draw_networkx_edges(system.G, pos, width=1*k, alpha=0.6, edge_color='k')

        # Draw node numbers on the nodes
        nx.draw_networkx_labels(system.G, pos, font_size=8, font_color='r')

        plt.xlim([0, xylims])
        plt.ylim([0, xylims])
        plt.axis('on')
        return plt

<<<<<<< HEAD
    fig, ax = plt.subplots(figsize=(5, 5))
=======
    fig, ax = plt.subplots(figsize=(10, 10))
>>>>>>> main
    ani = FuncAnimation(fig, update, frames=range(0, len(traj['position']), stride), init_func=init, blit=False)
    ani.save('compressedexample.gif', writer='imagemagick')
    display(HTML(ani.to_jshtml()))
    plt.show()
    return ani

def makemovieDOS(system, k, traj,stride=10):

    # Set style
    sns.set_style(style='white')

    # Define the init function, which sets up the plot
    def init():

        plt.axis('on')
        return plt

    # Define the update function, which is called for each frame
    def update(frame):
        plt.clf()  # Clear the current figure
        R_plt = traj['position'][frame]
        C = simulation.create_compatibility(system, R_plt)
        D, V, forbidden_states,_ = simulation.get_forbidden_states(C, k, system)
<<<<<<< HEAD
        plt.hist(onp.sqrt(onp.abs(D)), bins=onp.arange(-0.025, 4.025, 0.05), density=False)
        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$\rho(\omega)$')
        print(forbidden_states)

=======
        plt.hist(onp.sqrt(onp.abs(D)), bins=onp.arange(-0.025, 3.025, 0.05), density=False)
        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$C (\omega)$')
        print(forbidden_states)
        plt.gca().yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
>>>>>>> main
        #plt.ylim(0,5)
        plt.axis('on')
        return plt

    # Create the animation
<<<<<<< HEAD
    fig, ax = plt.subplots(figsize=(5, 5))
=======
    fig, ax = plt.subplots(figsize=(10, 10))
>>>>>>> main
    ani = FuncAnimation(fig, update, frames=range(0, len(traj['position']), stride), init_func=init, blit=False)
    ani.save('compressedDOS0.2.gif', writer='imagemagick')
    # Display the animation
    display(HTML(ani.to_jshtml()))
    plt.show()
    return ani

def quiver_plot(R_init, R_final, E, ms = 30):
    """
    Creates a quiver plot of the displacements of the atoms.

    R_init: initial positions
    R_final: final positions
    E: edge matrix
    """
    R_plt = np.array(R_final)  # Assuming R_final is already defined

    # Plotting atoms
    plt.plot(R_plt[:, 0], R_plt[:, 1], 'o', markersize=ms * 0.5)

    # Plotting bonds
    for bond in E:  # Assuming E is your list of bonds
        point1 = R_plt[bond[0]]
        point2 = R_plt[bond[1]]
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], c='black')  # Bond color

    # Calculate displacement vectors
    displacements = R_final - R_init # Assuming R_initial is defined

    # Create quiver plot for displacements
    plt.quiver(R_init[:, 0], R_init[:, 1], displacements[:, 0], displacements[:, 1],
            color='red', scale=1, scale_units='xy', angles='xy')  # Adjust color and scale as needed

    # Setting plot limits
    plt.xlim([0, np.max(R_plt[:, 0])])
    plt.ylim([0, np.max(R_plt[:, 1])])

    plt.axis('on')

    # Assuming finalize_plot is a function you've defined
    finalize_plot((1, 1))
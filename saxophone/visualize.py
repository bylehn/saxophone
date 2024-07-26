import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import jax.numpy as np
from jax_md import space
import numpy as onp
import saxophone.simulation as simulation
import matplotlib as mpl

mpl.rcParams.update({'font.size': 28})
sns.set_style(style='white')

def format_plot(x, y):
  plt.xlabel(x, fontsize=20)
  plt.ylabel(y, fontsize=20)

def finalize_plot(shape=(1, 1)):
  plt.gcf().set_size_inches(
    shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
    shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
  plt.tight_layout()


def makemovie_evolution(system, traj, amp=1., xylims=9., stride=10):
    """
    a version of makemovie that takes in the evolution trajectory from optimization tasks to show how k and R evolve in the course of optimization
    the amplification works on bond constants too to amplify the changes
    the traj file is output from generate functions if the specified output_evolution is set to True and proper output is stored.
    """
    # Set style
    sns.set_style(style='white')
    

    # Create the animation
    fig_length = 5 #inches
    dots_per_inch = 100
    fig, ax = plt.subplots(figsize=(fig_length, fig_length), dpi = dots_per_inch)

    #ax.set_aspect('equal', adjustable='box')


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
    
        k_0 = np.squeeze(traj['bond_strengths'][0])
        k_plt = np.squeeze(traj['bond_strengths'][frame])
        k_ratio = k_plt / k_0
        k_plt = k_0  * (np.abs(k_ratio)**amp) * onp.sign(k_ratio)
        

        pos = {i: (R_plt[i, 0], R_plt[i, 1]) for i in range(system.N)}
        
        nx.draw_networkx_edges(system.G, pos, width=2*k_plt*system.distances, alpha=0.6,edge_color='k')

        diameter_in_data_units = system.soft_sphere_sigma
        size_in_points = (2*np.exp(1)* diameter_in_data_units * fig_length*10/xylims)**2 # arbitrary factor of 5.43 = 2e (Cesar's contrib); remove the 10/xylims factor when doing it outside of this function ¯\_(ツ)_/¯

        nx.draw_networkx_nodes(system.G, pos, node_size=size_in_points, node_color='k', linewidths = 0.0, alpha=0.25)

        plt.xlim([0, xylims])
        plt.ylim([0, xylims])
        plt.gca().set_aspect('equal')

        return plt


    
    ani = FuncAnimation(fig, update, frames=range(0, len(traj['position']), stride), init_func=init, blit=False)
    ani.save('compressedexample.gif', writer='imagemagick')
    # Display the animation
    display(HTML(ani.to_jshtml()))
    plt.show()
    return ani




def makemovie(system, k, traj, amp=1., xylims=9., stride=10):
    """
    a cleaned up version of makemovie that plots nodes as well to the size of the soft sphere diameter
    """
    # Set style
    sns.set_style(style='white')
    k = np.squeeze(k)

    # Create the animation
    fig_length = 5 #inches
    dots_per_inch = 100
    fig, ax = plt.subplots(figsize=(fig_length, fig_length), dpi = dots_per_inch)

    #ax.set_aspect('equal', adjustable='box')


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
        
        nx.draw_networkx_edges(system.G, pos, width=2*k*system.distances, alpha=0.6,edge_color='k')

        diameter_in_data_units = system.soft_sphere_sigma
        size_in_points = (2*np.exp(1)* diameter_in_data_units * fig_length*10/xylims)**2 # arbitrary factor of 5.43 = 2e (Cesar's contrib); remove the 10/xylims factor when doing it outside of this function ¯\_(ツ)_/¯

        nx.draw_networkx_nodes(system.G, pos, node_size=size_in_points, node_color='k', linewidths = 0.0, alpha=0.25)

        plt.xlim([0, xylims])
        plt.ylim([0, xylims])
        plt.gca().set_aspect('equal')

        return plt


    
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

    fig, ax = plt.subplots(figsize=(10, 10))
    ani = FuncAnimation(fig, update, frames=range(0, len(traj['position']), stride), init_func=init, blit=False)
    ani.save('compressedexample.gif', writer='imagemagick')
    display(HTML(ani.to_jshtml()))
    plt.show()
    return ani

def makemovieDOS(system, k, traj, stride=10):

    # Set style
    sns.set_style(style='white')

    # Define the init function, which sets up the plot
    def init():

        plt.axis('on')
        return plt

    # Define the update function, which is called for each frame
    def update(frame):
        plt.clf()  # Clear the current figure
        R_0 = traj['position'][0]
        R_plt = traj['position'][frame]
        C = simulation.create_compatibility(system, R_plt)
        B= simulation.create_incidence(system, R_plt)
        length_ratios =  np.linalg.norm(space.map_bond(system.displacement)(R_0 [system.E[:, 0], :], R_0 [system.E[:, 1], :]), axis = 1) / np.linalg.norm(space.map_bond(system.displacement)(R_plt[system.E[:, 0], :], R_plt[system.E[:, 1], :]), axis = 1)
        D, V, forbidden_states,_ = simulation.get_forbidden_states_deformed(C, B, length_ratios, k, system)
        plt.hist(onp.sqrt(onp.abs(D)), bins=onp.arange(-0.025, 3.025, 0.05), density=False)
        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$C (\omega)$')
        print(forbidden_states)
        plt.gca().yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        #plt.ylim(0,5)
        plt.axis('on')
        return plt

    # Create the animation
    fig, ax = plt.subplots(figsize=(10, 10))
    ani = FuncAnimation(fig, update, frames=range(0, len(traj['position']), stride), init_func=init, blit=False)
    ani.save('compressedDOS0.2.gif', writer='imagemagick')
    # Display the animation
    display(HTML(ani.to_jshtml()))
    plt.show()
    return ani


def makemovieDOS_evolution(system, traj, stride=1):

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

    
        k_plt = np.squeeze(traj['bond_strengths'][frame])

        
        C = simulation.create_compatibility(system, R_plt)
        D, V, forbidden_states,_ = simulation.get_forbidden_states(C, k_plt, system)
        plt.hist(onp.sqrt(onp.abs(D)), bins=onp.arange(-0.025, 3.025, 0.05), density=False)
        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$C (\omega)$')
        print(forbidden_states)
        plt.gca().yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        #plt.ylim(0,5)
        plt.axis('on')
        return plt

    # Create the animation
    fig, ax = plt.subplots(figsize=(10, 10))
    ani = FuncAnimation(fig, update, frames=range(0, len(traj['position']), stride), init_func=init, blit=False)
    ani.save('compressedDOS0.2.gif', writer='imagemagick')
    # Display the animation
    display(HTML(ani.to_jshtml()))
    plt.show()
    return ani

def quiver_plot(R_init, R_final, k, system, xylims=9., shaft_width = 0.001):
    """
    Creates a quiver plot of the displacements of the atoms.

    R_init: initial positions
    R_final: final positions
    
    """
    sns.set_style(style='white')

    fig_length = 5 #inches
    dots_per_inch = 100
    fig, ax = plt.subplots(figsize=(fig_length, fig_length), dpi = dots_per_inch)
    R_plt = np.array(R_final)  # Assuming R_final is already defined
    k = np.squeeze(k)
    # Plotting atoms


    

    pos = {i: (R_plt[i, 0], R_plt[i, 1]) for i in range(system.N)}
    
    nx.draw_networkx_edges(system.G, pos, width=2*k*system.distances, alpha=0.6,edge_color='k')

    diameter_in_data_units = system.soft_sphere_sigma
    size_in_points = (2*np.exp(1)* diameter_in_data_units * fig_length)**2 # arbitrary factor of 5.43 = 2e (Cesar's contrib); remove the 10/xylims factor when doing it outside of this function

    nx.draw_networkx_nodes(system.G, pos, node_size=size_in_points, node_color='k', linewidths = 0.0, alpha=0.25)


    # Calculate displacement vectors
    displacements = R_final - R_init # Assuming R_initial is defined

    # Create quiver plot for displacements
    plt.quiver(R_init[:, 0], R_init[:, 1], displacements[:, 0], displacements[:, 1],
            color='red', scale=1, scale_units='xy', angles='xy', width = shaft_width)  # Adjust color and scale as needed

    # Setting plot limits
    plt.xlim([0, xylims])
    plt.ylim([0, xylims])
    plt.gca().set_aspect('equal')


    plt.axis('on')

    # Assuming finalize_plot is a function you've defined
    finalize_plot((1, 1))



def show_network(system, R_plt, k, xylims):
    """
    visualizes a given network statically. 
    """
    sns.set_style(style='white')
    k = np.squeeze(k)

    # Create the animation
    fig_length = 5 #inches
    dots_per_inch = 100
    fig, ax = plt.subplots(figsize=(fig_length, fig_length), dpi = dots_per_inch)

    

    pos = {i: (R_plt[i, 0], R_plt[i, 1]) for i in range(system.N)}
    
    nx.draw_networkx_edges(system.G, pos, width=2*k*system.distances, alpha=0.6,edge_color='k')

    diameter_in_data_units = system.soft_sphere_sigma
    size_in_points = (2*np.exp(1)* diameter_in_data_units * fig_length*10/xylims)**2 # arbitrary factor of 5.43 = 2e (Cesar's contrib); remove the 10/xylims factor when doing it outside of this function

    nx.draw_networkx_nodes(system.G, pos, node_size=size_in_points, node_color='k', linewidths = 0.0, alpha=0.25)

    plt.xlim([0, xylims])
    plt.ylim([0, xylims])
    plt.gca().set_aspect('equal')
    return
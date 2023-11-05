"""
Visualization functions compatible with Pygame
"""
import matplotlib.pyplot as plt
import pygame
import io


def lineplot(x,y, xlabel, ylabel, title, *args, **kwargs):
    """
    Create a lineplot and return it as a stream
    Args:
        x:  x values
        y:  y values
        xlabel:
        ylabel:
        title:
        *args:
        **kwargs:

    Returns:

    """
    plot_stream =  io.BytesIO()
    plt.plot(x,y, *args, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(plot_stream, format='png',dpi=300)
    plot_stream.seek(0)
    return plot_stream

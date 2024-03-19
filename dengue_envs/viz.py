"""
Visualization functions compatible with Pygame
"""
import matplotlib.pyplot as plt
import pygame
import io


def lineplot(x, y, xlabel, ylabel, title, identifier, *args, **kwargs):
    """
    Create a lineplot and return it as a stream
    Args:
        x: x values
        y: y values
        xlabel: label for x-axis
        ylabel: label for y-axis
        title: title of the plot
        identifier: unique identifier for the plot
        *args: additional positional arguments for plt.plot()
        **kwargs: additional keyword arguments for plt.plot()

    Returns:
        BytesIO: a stream containing the plot image
    """
    plot_stream = io.BytesIO()
    plt.clf()
    plt.plot(x, y, *args, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(plot_stream, format='png', dpi=300)
    plot_stream.seek(0)
    return plot_stream

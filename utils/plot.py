import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def model_size():
    # Data points for two lines
    x = [0.2, 2.3, 13.6]  # X-axis values
    y1 = [0.52, 0.78, 1.45]  # BIDA
    y2 = [0.18, 0.29, 0.38]  # BA

    # Plotting the first line with a dashed line and square markers
    plt.plot(x, y1, linestyle='--', marker='*', label='Batch IDA*')

    # Plotting the second line with a dotted line and circle markers
    plt.plot(x, y2, linestyle=':', marker='s', label='Batch A*')

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)


    # Set x and y ticks to indicate discrete values
    plt.xticks(x)  # Explicitly set x-axis ticks to match the data points
    plt.yticks(sorted(set(y1 + y2)))  # Y-axis ticks based on the unique y-values

    # Adding labels and legend
    plt.xlabel('size (MB)')
    plt.ylabel('time (s)')
    # plt.title('')
    # plt.legend()

    # Show plot
    plt.savefig("test")
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def model_size():
    # Data points for two lines
    x = [0.2, 2.3, 13.6]  # X-axis values
    
    #STP
    y1 = [0.52, 0.78, 1.45]  # BIDA
    y2 = [0.18, 0.29, 0.38]  # BA
    
    # RC
    # y1 = [3.46, 4.75, 10.23]  # BIDA
    # y2 = [11.16, 16.71, 20.66]  # BA

    # Plotting the first line with a dashed line and square markers
    plt.plot(x, y1, linestyle='--', marker='*', label='Batch IDA*')

    # Plotting the second line with a dotted line and circle markers
    plt.plot(x, y2, linestyle=':', marker='s', label='Batch A*')

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)


    # Set x and y ticks to indicate discrete values
    plt.xticks(x)  # Explicitly set x-axis ticks to match the data points

    # Apply the y-ticks to the plot
    y_ticks = np.arange(0, 2.2, 0.4)
    # y_ticks = np.arange(0, 25, 5)
    plt.yticks(y_ticks)

    # Adding annotations for y1 (Batch IDA*)
    for xi, yi in zip(x, y1):
        if xi>1:
            # plt.text(xi, yi + 0.2, f'{yi}', ha='center', va='bottom', fontsize=12)            # RC
            plt.text(xi, yi + 0.05, f'{yi}', ha='center', va='bottom', fontsize=12)            # STP
        else:
            # plt.text(xi+0.7, yi - 1.5, f'{yi}', ha='center', va='bottom', fontsize=12)        # RC
            plt.text(xi+0.8, yi - 0.05, f'{yi}', ha='center', va='bottom', fontsize=12)            # STP

    # Adding annotations for y2 (Batch A*)
    for xi, yi in zip(x, y2):
        if xi>1:
            # plt.text(xi+0.05, yi + 0.3, f'{yi}', ha='center', va='bottom', fontsize=12)       # RC
            plt.text(xi+0.2, yi + 0.05, f'{yi}', ha='center', va='bottom', fontsize=12)            # STP
        else:
            # plt.text(xi+1, yi - 1, f'{yi}', ha='center', va='bottom', fontsize=12)            # RC
            plt.text(xi+0.7, yi - 0.1, f'{yi}', ha='center', va='bottom', fontsize=12)            # STP

    # Optional: Adjust y-axis limit to accommodate annotations
    plt.ylim(0, 2)
    # plt.ylim(0, 25)
    
    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.7,axis='y')
    
    plt.tight_layout() 
    # Adding labels and legend
    plt.xlabel('Size (MB)')
    plt.ylabel('Time (s)')
    # plt.title('')
    # plt.legend(loc="upper left")

    # Show plot
    plt.savefig('STP.pdf',format="pdf",bbox_inches='tight',dpi=300)

def hardware():
    
    font=25

    # RC
    categories = ['H1', 'H2', 'H3']
    method1_times = [5.87, 10.78, 4.32]  
    method2_times = [2.32, 6.81, 2.01] 

    # STP
    # categories = ['H1', 'H2', 'H3']
    # method1_times = [0.78, 1.01, 0.60]  
    # method2_times = [0.026, 0.045, 0.015]


    # Number of categories
    n_categories = len(categories)
    # Positions of the groups on the x-axis
    index = np.arange(n_categories)
    # Width of each bar
    bar_width = 0.35  

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for Method 1
    bars1 = ax.bar(index - bar_width/2, method1_times, bar_width, label='SingleGPU Batch IDA*', color='#1f77b4')

    # Plot bars for Method 2
    bars2 = ax.bar(index + bar_width/2, method2_times, bar_width, label='AIDA*', color='#ff7f0e')

    # Add labels and title
    ax.set_xlabel('Hardware Resources', fontsize=font,labelpad=15)
    ax.set_ylabel('Time (s)', fontsize=font,labelpad=15)

    # Set the position and labels of the x-ticks
    ax.set_xticks(index)
    ax.set_xticklabels(categories, fontsize=font)

    # Add legend
    ax.legend(fontsize=19)

    # Add grid lines for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Add value labels on top of each bar for Method 1
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height, 
            f'{height}', 
            ha='center', 
            va='bottom',
            fontsize=font
        )

    # Add value labels on top of each bar for Method 2
    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height, 
            f'{height}', 
            ha='center', 
            va='bottom',
            fontsize=font
        )

    # Adjust layout for better fit
    plt.tight_layout()
    # plt.ylim([0,1.1])
    plt.ylim([0,15])
    plt.yticks(fontsize=19)

    # Show plot
    plt.savefig('hardware-RC.pdf',format="pdf",bbox_inches='tight',dpi=300)

def hyper():

    fontsize=19
    
    # hypers for RC
    # hyperparam1 = [50, 100, 200, 300, 400,500]
    # hyperparam2 = ["4k", "6k", "8k", "10k", "12k"]

    # hypers for STP
    hyperparam1 = [20, 30, 50, 100, 200]
    hyperparam2 = ["500", "1k", "2k", "3k", "4k"]

    # Random scores matrix with a missing value (use np.nan to represent missing values)
    multi_rc_scores = np.array([
        [2.71, 2.69, 2.66, 2.70, 2.85, 2.74],    # 4k
        [2.69, 2.52, 2.59, 2.84, 2.72, 2.69],    # 6k
        [2.61, 2.51, 2.65, 2.68, 2.70, 2.71],    # 8k
        [2.58, 2.59, 2.66, 2.66, 2.69, 2.77],    # 10k
        [2.70, 2.62, 2.63, 2.59, 2.63, 2.85]     # 12k
    ])


    single_rc_scores = np.array([
        [3.49, 3.49, 3.55, 3.66, 3.78, 3.84],    # 4k
        [3.48, 3.51, 3.59, 3.67, 3.89, 3.82],    # 6k
        [3.47, 3.46, 3.74, 3.70, 3.74, 3.74],    # 8k
        [3.61, 3.57, 3.71, 3.68, 3.71, 3.78],    # 10k
        [3.60, 3.64, 3.61, 3.62, 3.77, 3.73]     # 12k
    ])


    multi_stp_scores = np.array([
        [0.45, 0.46, 0.50, 0.60, 0.81],    # 500
        [0.43, 0.44, 0.48, 0.58, 0.79],    # 1k
        [0.43, 0.47, 0.47, 0.57, 0.83],    # 2k
        [0.45, 0.45, 0.47, 0.57, 0.80],    # 3k
        [0.45, 0.44, 0.46, 0.56, 0.82]     # 4k
    ])


    single_stp_scores = np.array([
        [0.61, 0.56, 0.63, 0.70, 0.90],    # 500
        [0.63, 0.52, 0.65, 0.70, 0.87],    # 1k
        [0.62, 0.63, 0.66, 0.74, 0.93],    # 2k
        [0.57, 0.59, 0.66, 0.70, 0.91],    # 3k
        [0.58, 0.60, 0.62, 0.74, 0.96]     # 4k
    ])



    # Plot the heatmap with missing values
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(multi_stp_scores, annot=True, annot_kws={"size": fontsize} , fmt=".2f" , xticklabels=hyperparam1, yticklabels=hyperparam2, cmap="viridis", mask=np.isnan(multi_stp_scores))
    plt.xlabel('number of subtrees',fontsize=22,labelpad=15)
    plt.ylabel('batch size',fontsize=22,labelpad=15)
    plt.title('Time Heatmap',fontsize=22,pad=20)
    # plt.show()

    plt.xticks(fontsize=fontsize)  # Set x-axis label size
    plt.yticks(fontsize=fontsize)

    # Set the font size for the legend, if applicable
    colorbar = ax.collections[0].colorbar  # Access the colorbar from the heatmap
    num_ticks = 4  # Set the number of ticks you want
    ticks = np.linspace(0.45, 0.80, num_ticks)

    # Set the ticks on the colorbar
    colorbar.set_ticks(ticks)
    tick_labels = [f'{tick:.2f}' for tick in ticks]
    colorbar.set_ticklabels(tick_labels)
    colorbar.ax.tick_params(labelsize=fontsize)  # Set the font size of the colorbar (side legend)

    plt.savefig('multi-STP.pdf',format="pdf",bbox_inches='tight',dpi=300)


if __name__ == "__main__":
    # hyper()
    hardware()
    # model_size()
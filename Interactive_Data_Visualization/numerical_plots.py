# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

# frequency Distribution Table
def frequency_distribution(df, column, bins=10):
    # Create bins and calculate frequency counts for each bin
    bin_edges = np.histogram_bin_edges(df[column].dropna(), bins=bins)  # Generate bin edges
    bin_edges = np.round(bin_edges, 2)  # Round bin edges to 2 decimal places
    
    frequency, bin_edges = np.histogram(df[column].dropna(), bins=bin_edges)  # Calculate frequency in each bin
    
    # Create a DataFrame for the frequency distribution table
    frequency_table = pd.DataFrame({
        'Bin Range': pd.IntervalIndex.from_breaks(bin_edges),
        'Frequency': frequency
    })
    
    # Calculate the relative frequency as a percentage
    frequency_table['Relative Frequency (%)'] = (frequency_table['Frequency'] / len(df[column].dropna())) * 100
    
    return frequency_table


# 1. Histogram
def plot_histogram(df, column):
    plt.figure(figsize=(10,5))
    
    # change bins as needed
    bins = np.linspace(df[column].min(), df[column].max(), 25)
    
    # Create the histogram
    ax = sns.histplot(df[column], kde=True, bins=bins, color='mediumseagreen', cbar=True, cumulative=False, edgecolor='black')
    ax.grid(False)
    
    # Set the title and labels
    plt.style.use("bmh")
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    
    # Annotate each bar
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:  # Only annotate non-zero bars
            plt.text(patch.get_x() + patch.get_width() / 2, height + 0.5, 
                     int(height), ha='center', va='bottom', fontsize=10, color='black')
            
    # Calculate percentiles
    quant_5, quant_25, quant_50, quant_75, quant_95 = df[column].quantile(0.05), df[column].quantile(0.25), df[column].quantile(0.5), df[column].quantile(0.75), df[column].quantile(0.95)

    # [quantile, opacity, length]
    quants = [[quant_5, 0.6, 0.16], [quant_25, 0.8, 0.26], [quant_50, 1, 0.36],  [quant_75, 0.8, 0.46], [quant_95, 0.6, 0.56]]

    # Plot the lines with a loop
    for i in quants:
        ax.axvline(i[0], alpha = i[1], ymax = i[-1], linestyle = ":", color = "red")

    # display quantile values
    # ax.text(x=quant_5, y =quant_5, s="5th", size = 10, alpha = 0.8, rotation = 55, transform=ax.transAxes)
    # ax.text(x=quant_25, y=quant_25,s= "25th", size = 11, alpha = 0.85, rotation = 55, transform=ax.transAxes)
    # ax.text(x=quant_50, y=  quant_50,s= "50th", size = 12, alpha = 1, rotation = 55, transform=ax.transAxes)
    # ax.text(x=quant_75,y= quant_75, s="75th", size = 11, alpha = 0.85, rotation = 55, transform=ax.transAxes)
    # ax.text(x=quant_95, y=quant_95, s="95th Percentile", size = 10, alpha =.8, rotation = 55, transform=ax.transAxes)

    # highlight - Mean, Median and Mode
    mean = df[column].mean()
    median = df[column].median()
    mode = df[column].mode()[0]

    ax.axvline(mean, linestyle = '--', color = 'green', label = "Mean - Average")
    ax.axvline(median, linestyle = '-.', color = 'blue', label = "Median (50th percentile)")
    ax.axvline(mode, linestyle = '--', color = 'gray', label = "Mode - Most frequent")
    plt.legend(loc = "upper right")        
    return plt.gcf()



# 2. Box Plot
def plot_boxplot(df, column):
    fig, ax = plt.subplots(figsize=(15, 6))
    custom_palette = np.array(["#007bff", "#28a745","#ffc107", "#dc3545", "#6c757d", "#343a40", "#76b7b2"])
    ax = sns.boxplot(x=column, data=df, color=np.random.choice(custom_palette), width=0.3, linecolor="green")
    plt.axvline(df[column].mean(), linestyle = "-.", linewidth = "1", color= "black", label = "Mean = {}".format(df[column].mean().round(1)))
    plt.axvline((np.percentile(df[column],50)),linestyle = ":", linewidth = "1", color = "blue", label = "Median = {}".format((np.percentile(df[column],50))))

    plt.title(f'Box Plot of {column}')
    plt.legend(bbox_to_anchor=(1.5, 1), loc='upper right', fontsize=20)
    return plt.gcf()



# 3. Violin Plot
def plot_violin(df, column):
    fig, ax = plt.subplots(figsize=(15, 3))

    custom_palette = np.array([	"#98df8a", "#ff7f0e", "#ffbb78", "#1f77b4", "#aec7e8"])
    # plot the graph
    ax = sns.violinplot(x=column, data=df, color=np.random.choice(custom_palette))

    # Mean - vertical line display
    plt.axvline(df[column].mean(), linestyle = "-.", linewidth = "1", color= "black", label = "Mean = {}".format(df[column].mean().round(1)))
    # Median - vertical line display
    plt.axvline((np.percentile(df[column],50)),linestyle = ":", linewidth = "1", color = "blue", label = "Median = {}".format((np.percentile(df[column],50))))
    # 25th percentile - vertical line display
    plt.axvline((np.percentile(df[column],25)), color = "green",linewidth = "1", label = "25th percentile = %d"%np.percentile(df[column],25))
    # 75th percentile - vertical line display
    plt.axvline((np.percentile(df[column],75)), color = "orange", linewidth = "1",label = "75th percentile = {}".format((np.percentile(df[column],75))))
    # Min value - vertical line display
    plt.axvline(np.min(df[column]), color = "#1f77b4", linewidth = "1",label = "Min = {}".format(df[column].min()))
    # Max Value - vertical line display
    plt.axvline(np.max(df[column]), color = "#8c564b",linewidth = "1", label = "Min = {}".format(df[column].min()))


    plt.legend(bbox_to_anchor=(1.2, 1), loc='upper right')

    plt.title(f'Violin Plot of {column}')
    return plt.gcf()


# 9. Density Plot
def density_plot(df, column):
    plt.figure(figsize=(10,4))
    custom_palette = np.array(["#4e79a7", "#f28e2b","#e15759", "#76b7b2", "#59a14f", "#edc949"])
    sns.kdeplot(df[column], fill=False, color=np.random.choice(custom_palette))

    plt.axvline(df[column].mean(), linestyle = "-.", linewidth = "1", color= "black", label = "Mean = {}".format(df[column].mean().round(1)))
    plt.axvline((np.percentile(df[column],50)),linestyle = ":", linewidth = "1", color = "blue", label = "Median = {}".format((np.percentile(df[column],50))))

    plt.axvline((np.percentile(df[column],25)), color = "green",linewidth = "1", label = "25th percentile = %d"%np.percentile(df[column],25))
    plt.axvline((np.percentile(df[column],75)), color = "orange", linewidth = "1",label = "75th percentile = {}".format((np.percentile(df[column],75))))
    plt.legend(loc='upper right')
    plt.title(f'Density Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')
    return plt.gcf()

# 14. Quantile-Quantile (Q-Q) Plot
def qq_plot(df, column):
    plt.figure(figsize=(10,4))
    stats.probplot(df[column], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {column}')
    return plt.gcf()

# 5. Rug Plot
def rug_plot(df, column):
    plt.figure(figsize=(10,4))
    custom_palette = np.array(["#4e79a7", "#f28e2b","#e15759", "#76b7b2", "#59a14f", "#edc949"])
    sns.kdeplot(df[column], fill=True, color=np.random.choice(custom_palette))
    sns.rugplot(df[column], height=0.1, color='black')
    plt.title(f'KDE with Rug Plot of {column}')
    return plt.gcf()

# 6. Line Plot
def line_plot(df, column):
    plt.figure(figsize=(10,4))
    custom_palette = np.array(["#4e79a7", "#f28e2b","#e15759", "#76b7b2", "#59a14f", "#edc949"])
    plt.plot(df.index, df[column], color=np.random.choice(custom_palette))
    plt.title(f'Line Plot of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    return plt.gcf()

# 7. Scatter Plot
def scatter_plot(df, column):
    plt.figure(figsize=(10,5))
    custom_palette = np.array(["#4e79a7", "#f28e2b","#e15759", "#76b7b2", "#59a14f", "#edc949"])
    plt.scatter(df.index, df[column], color=np.random.choice(custom_palette))
    plt.title(f'Scatter Plot of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    return plt.gcf()

# 8. ECDF Plot
def ecdf_plot(df, column):
    plt.figure(figsize=(10,4))
    custom_palette = np.array(["#007bff", "#28a745", "#ffc107" ,"#dc3545", "#6c757d ", "#343a40"])
    sns.ecdfplot(df[column], color=np.random.choice(custom_palette))
    plt.title(f'ECDF Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('ECDF')
    return plt.gcf()

# 10. Hexbin Plot (usually for two continuous columns but can modify for single)
def hexbin_plot(df, column):
    plt.figure(figsize=(6,6))
    custom_palette = np.array(['Greens_r',  'RdGy_r', 'YlGnBu_r', "copper_r", "gist_yarg_r", "terrain_r"])
    plt.hexbin(df.index, df[column], gridsize=50, cmap=np.random.choice(custom_palette), mincnt=1)
    plt.colorbar(label='Count')
    plt.title(f'Hexbin Plot of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    return plt.gcf()

# 12. Strip Plot
def strip_plot(df, column):
    plt.figure(figsize=(8,6))
    sns.stripplot(y=column, data=df, color='darkblue', jitter=True)
    plt.title(f'Strip Plot of {column}')
    return plt.gcf()

# 13. Swarm Plot
def swarm_plot(df, column):
    plt.figure(figsize=(15,6))
    custom_palette = np.array(["#007bff", "#28a745", "#ffc107" ,"#dc3545", "#6c757d ", "#343a40"])
    sns.swarmplot(y=column, data=df, color=np.random.choice(custom_palette))
    plt.title(f'Swarm Plot of {column}')
    return plt.gcf()
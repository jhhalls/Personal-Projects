import matplotlib.pyplot as plt
import seaborn as sns

# scatter plot
def scatter_plot(df, x_col, y_col):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(f'Scatter Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=90)
    return plt.gcf()

# line plot
def line_plot(df, x_col, y_col):
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=df, x=x_col, y=y_col)
    plt.title(f'Line Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=90)
    return plt.gcf()

# joint plot
def joint_plot(df, x_col, y_col, kind='scatter'):
    plt.figure(figsize=(6, 4))
    sns.jointplot(data=df, x=x_col, y=y_col, kind=kind,color="tomato")
    plt.title(f'Line Plot: {x_col} vs {y_col} \n Plot type as {kind}', pad = 80)
    plt.xticks(rotation=90)
    return plt.gcf()

# hexbin plot
def hexbin_plot(df, x_col, y_col, gridsize=30, cmap='Blues'):
    plt.figure(figsize=(8, 6))
    plt.hexbin(df[x_col], df[y_col], gridsize=gridsize, cmap=cmap)
    plt.colorbar(label='Counts')
    plt.title(f'Hexbin Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=90)
    return plt.gcf()

# kde plot
def kde_plot(df, x_col, y_col):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df, x=x_col, y=y_col, fill=True)
    plt.title(f'KDE Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=90)
    return plt.gcf()

# 2d Histogram
def hist2d_plot(df, x_col, y_col, bins=30):
    plt.figure(figsize=(8, 6))
    plt.hist2d(df[x_col], df[y_col], bins=bins, cmap='Blues')
    plt.colorbar(label='Counts')
    plt.title(f'2D Histogram: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=90)
    return plt.gcf()

# box plot
def box_plot(df, x_col, y_col):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=x_col, y=y_col)
    plt.title(f'Box Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=90)
    return plt.gcf()

# correlation heatmap
def correlation_heatmap(df, x_col, y_col):
    plt.figure(figsize=(8, 6))
    correlation = df[[x_col, y_col]].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title(f'Correlation between {x_col} and {y_col}')
    plt.xticks(rotation=90)
    return plt.gcf()

# Regression plot
def regplot(df, x_col, y_col):
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x=x_col, y=y_col, scatter_kws={'s': 10}, line_kws={'color': 'red'})
    plt.title(f'Regression Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=90)
    return plt.gcf()

# swarm plot
def swarm_plot(df, x_col, y_col):
    plt.figure(figsize=(8, 6))
    sns.swarmplot(data=df, x=x_col, y=y_col)
    plt.title(f'Swarm Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=90)
    return plt.gcf()

# Strip Plot
def strip_plot(df, x_col, y_col):
    plt.figure(figsize=(8, 6))
    sns.stripplot(data=df, x=x_col, y=y_col)
    plt.title(f'Strip Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=90)
    return plt.gcf()

# Lollipop Plot
def lollipop_plot(df, x_col, y_col):
    plt.figure(figsize=(8, 6))
    plt.stem(df[x_col], df[y_col], basefmt=" ", use_line_collection=True)
    plt.title(f'Lollipop Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=90)
    return plt.gcf()


# Bubble Chart
def bubble_chart(df, x_col, y_col, size_based_on='x'):
    """
    This function generates a bubble chart with bubble sizes dynamically based on either the x or y column,
    and uses different colors for different categories in the y-axis.
    
    Parameters:
    df (DataFrame): The pandas DataFrame containing the data.
    x_col (str): The name of the column to plot on the x-axis.
    y_col (str): The name of the column to plot on the y-axis (categorical values).
    size_based_on (str): Determines which column ('x' or 'y') to base the bubble size on.
    
    Output:
    A bubble chart with dynamically scaled bubble sizes and different colors for each category on the y-axis.
    """
    
    # Choose the column for bubble sizes based on 'size_based_on' argument
    if size_based_on == 'x':
        size = df[x_col]
    elif size_based_on == 'y':
        size = df[y_col]
    else:
        raise ValueError("size_based_on should be either 'x' or 'y'")
    
    # Normalize the selected column values for bubble sizes
    size_scaled = (size - size.min()) / (size.max() - size.min()) * 1000  # Scale to range 0-1000 for bubble sizes
    
    # Create a color palette based on unique categories in y_col
    unique_categories = df[y_col].unique()
    colors = sns.color_palette('Set2', len(unique_categories))  # Color palette for each unique category
    color_mapping = {category: color for category, color in zip(unique_categories, colors)}
    
    # Map colors to y_col categories
    bubble_colors = df[y_col].map(color_mapping)
    
    # Plot the bubble chart
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], s=size_scaled, alpha=0.6, c=bubble_colors, edgecolors='w', linewidth=0.5)
    
    # Set plot title and labels
    plt.title(f'Bubble Chart: {x_col} vs {y_col} \n(Bubble Size based on {x_col if size_based_on=="x" else y_col})', fontsize=14)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    
    # Add a legend for the y_col categories
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=category) 
               for category, color in color_mapping.items()]
    plt.legend(handles=handles, title=y_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Show the plot
    plt.tight_layout()
    return plt.gcf()


# Step Plot
def step_plot(df, x_col, y_col):
    plt.figure(figsize=(8, 6))
    plt.step(df[x_col], df[y_col], where='mid', color='blue')
    plt.title(f'Step Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    return plt.gcf()
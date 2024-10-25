# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pywaffle import Waffle
import squarify

# 1. Bar Chart
def bar_chart(df, column):
    fig = plt.figure(figsize=(10,5))
    ax = sns.countplot(x=column, 
                       data=df, 
                       order=df[column].value_counts(ascending=False).index)
    abs_values = df[column].value_counts(ascending=False).values
    abs_percent = df[column].value_counts(ascending=False, normalize=True).values*100
    labels = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, abs_percent)]
    ax.bar_label(container=ax.containers[0], labels=labels)
    plt.title(f'Bar Chart of {column}')
    plt.tight_layout()
    return plt.gcf()


# 2. Pie Chart
def pie_chart(df, column):
    data = df[column].value_counts()
    fig = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    ax = plt.pie(data, 
                 labels=data.index,
                 autopct='%1.1f%%', 
                 startangle=140, 
                 colors=sns.color_palette('Set2'))
    plt.title(f'Pie Chart of {column}')
    plt.tight_layout()
    return plt.gcf()

# 3. Donut Chart
def donut_chart(df, column):
    data = df[column].value_counts()
    fig = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    ax = plt.pie(data, 
                labels=data.index, 
                autopct='%1.1f%%', 
                startangle=140, 
                wedgeprops={'width':0.3}, 
                colors=sns.color_palette('Set2'))
    plt.title(f'Donut Chart of {column}')
    plt.tight_layout()
    return plt.gcf()


# 4. Pareto Chart
def pareto_chart(df, column):
    data = df[column].value_counts()
    cumulative = data.cumsum()/data.sum()*100
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    ax1.bar(data.index, data, color="C0")
    ax2 = ax1.twinx()
    ax2.plot(data.index, cumulative, color="C1", marker="D", ms=7)
    
    ax1.set_xlabel(column)
    ax1.set_ylabel("Count")
    ax2.set_ylabel("Cumulative Percentage")
    
    plt.title(f'Pareto Chart of {column}')
    return plt.gcf()

# 5. Frequency Table
def frequency_table(df, column):
    freq_table = pd.Series(df[column].value_counts(), name = "Values")
    normalized_values = pd.Series(df[column].value_counts(normalize=True).mul(100).round(2), name="Relative Share(%)")
    final_freq_table = pd.concat([freq_table, normalized_values], axis=1)
    return final_freq_table


# 6. Waffle Chart
def waffle_chart(df, column):
    # Get value counts for the categorical column
    data = df[column].value_counts().to_dict()
    
    # Generate a color palette based on the number of unique categories
    num_categories = len(data)
    colors = sns.color_palette('Set2', num_categories).as_hex()  # Generate hex colors
    
    # Create a waffle chart
    fig = plt.figure(
        FigureClass=Waffle, 
        rows=5, 
        columns=5, 
        values=data, 
        block_arranging_style='snake',  # or 'new-line'
        interval_ratio_x=0.25,  # Adjust spacing between blocks in x direction
        interval_ratio_y=0.25,  # Adjust spacing between blocks in y direction
        colors=colors,  # Dynamically generated colors
        legend={'loc': 'upper left', 'bbox_to_anchor': (1.1, 1)},  # Legend positioning
        figsize=(2,2)  # Figure size
        )
    plt.title(f'Waffle Chart of {column}')
    return plt.gcf()


# 9. Tree Map
def tree_map(df, column):
    data = df[column].value_counts()
    plt.figure(figsize=(10,6))
    squarify.plot(sizes=data.values, label=data.index, color=sns.color_palette('Set1'), alpha=0.8)
    plt.title(f'Tree Map of {column}')
    plt.axis('off')
    return plt.gcf()

# 10. Heatmap (for frequency)
def heatmap(df, column):
    data = df[column].value_counts().to_frame().T
    plt.figure(figsize=(10,1))
    sns.heatmap(data, annot=True, cmap='coolwarm', cbar=True, fmt="0.0f")
    plt.title(f'Heatmap of {column}')
    return plt.gcf()

# 13. Dot Plot
def dot_plot(df, column):
    data = df[column].value_counts()
    plt.figure(figsize=(10,6))
    plt.plot(data.index, data, 'x', color='dodgerblue')
    # plt.annotate(text=data.index, xy=(data.index, data.values), annotation_clip=True, clip_on=True)
    plt.title(f'Dot Plot of {column}')
    return plt.gcf()

# 14. Bubble Chart
def bubble_chart(df, column):
    data = df[column].value_counts()
    plt.figure(figsize=(10,6))
    plt.scatter(data.index, data, s=data.values * 0.5, alpha=0.5, color='dodgerblue')
    plt.title(f'Bubble Chart of {column}')
    return plt.gcf()

def count_plot(df, x_col):
    # Number of unique categories in the column
    num_categories = df[x_col].nunique()
    
    # Dynamically adjust the figure size based on the number of categories
    plt.figure(figsize=(8, min(max(num_categories * 0.8, 6), 15)))  # Adjust height for horizontal plot
    
    # Create the count plot with bars in horizontal orientation
    ax = sns.countplot(data=df, y=x_col, palette="Set2", orient='h')
    
    # Set title and axis labels
    plt.title(f'Horizontal Count Plot of {x_col}', fontsize=14)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(x_col, fontsize=12)
    
    # Add frequency labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_width())}', 
                    (p.get_width(), p.get_y() + p.get_height() / 2.), 
                    ha = 'left', va = 'center', 
                    fontsize=11, color='black', xytext=(5, 0), 
                    textcoords='offset points')
    
    # Add some spacing to avoid bars and labels touching
    plt.tight_layout()
    
    # Show plot
    return plt.gcf()
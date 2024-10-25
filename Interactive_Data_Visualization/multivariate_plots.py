import plotly.express as px

def scatter_matrix_plot(df, color_col=None, dimensions=None, marker_size=5, opacity=0.7):
    """
    Generates an interactive scatter matrix plot with enhanced details.
    
    Parameters:
    df (DataFrame): The pandas DataFrame containing the data.
    color_col (str): The column name to color the points by (categorical or numerical).
    dimensions (list): List of column names to include in the scatter matrix.
    marker_size (int): Size of the markers in the plot (default is 5).
    opacity (float): Opacity of the markers (default is 0.7).
    
    Output:
    An interactive scatter matrix plot with additional details.
    """
    # Scatter matrix with enhanced details
    fig = px.scatter_matrix(
        df, 
        dimensions=dimensions if dimensions else df.columns,  # Select specific columns or all if not specified
        color=color_col,  # Color by a categorical or numerical column if provided
        title="Scatter Matrix Plot",
        labels={col: col.replace('_', ' ').title() for col in df.columns},  # More readable labels
        opacity=opacity,  # Adjust the opacity of the points
        height=800,  # Set a custom height for better visualization
        width=800  # Set custom width
    )

    # Customize the marker size and style
    fig.update_traces(marker=dict(size=marker_size, line=dict(width=1, color='DarkSlateGrey')))

    # Improve the layout for better interaction
    fig.update_layout(
        title='Enhanced Scatter Matrix Plot',
        hovermode='closest',  # Enable closest hover for better interaction
        showlegend=True,  # Show legend when using color_col
        template='plotly_dark',  # Optional: choose a dark theme or use 'plotly_white' for light mode
        font=dict(family="Arial", size=12),  # Font styling
        dragmode='select',  # Allow users to select points by dragging
        height=800,  # Adjust the height of the figure
        width=800,  # Adjust the width of the figure
        margin=dict(l=0, r=0, b=0, t=40)  # Reduce margins for a cleaner look
    )

    # Add better axis titles
    fig.update_xaxes(showgrid=True, zeroline=True)
    fig.update_yaxes(showgrid=True, zeroline=True)

    # Display the plot
    return fig
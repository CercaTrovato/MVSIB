import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib import font_manager

def plot_silhouette_comparison(cluster_scores_with_balance, cluster_scores_without_balance):
    """
    cluster_scores_with_balance: DataFrame with columns ['cluster', 'sample_count', 'silhouette'] for with imbalance module
    cluster_scores_without_balance: DataFrame with columns ['cluster', 'sample_count', 'silhouette'] for without imbalance module
    """

    # Prepare the data
    clusters = cluster_scores_with_balance['cluster'].astype(str).values
    sil_scores_with_balance = cluster_scores_with_balance['silhouette'].values
    sil_scores_without_balance = cluster_scores_without_balance['silhouette'].values
    sample_counts = cluster_scores_with_balance['sample_count'].values

    # Normalize the sample counts for color mapping
    max_count = sample_counts.max()
    norm_counts = sample_counts / max_count  # Normalize to [0, 1]

    # Create a colormap based on sample count
    cmap = plt.cm.Blues
    bar_colors = cmap(norm_counts)

    # Set the background color to light gray with transparency
    plt.rcParams['axes.facecolor'] = (0.9, 0.9, 0.9, 0.4)  # RGBA format for transparency (0.5 is the transparency value)

    # Define the font properties for Times New Roman
    font_prop = font_manager.FontProperties(fname='C:/Windows/Fonts/times.ttf')

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Add gridlines in the background (zorder=-1 ensures it is behind everything)
    ax1.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5, zorder=-1)

    # Plot the bars for Silhouette Scores (With Imbalance Module)
    ax1.bar(clusters, sil_scores_with_balance, width=0.55, color=bar_colors, label='With Imbalance Module',
            edgecolor='black', align='edge', zorder=2)  # Align to the left and bring it to the top layer
    ax1.set_xlabel('Cluster ID', fontproperties=font_prop, fontsize=14)
    ax1.set_ylabel('Silhouette Score', fontproperties=font_prop, fontsize=14)
    ax1.set_ylim(0.0, 0.8)  # Silhouette Score range from -1 to 1
    ax1.set_title('Comparison of Silhouette Scores with and without Class Imbalance Module (RGB-D)', fontproperties=font_prop, fontsize=16)

    # Plot the bars for Silhouette Scores (Without Imbalance Module) in transparent gray color
    ax1.bar(clusters, sil_scores_without_balance, width=0.55, color='gray', alpha=0.8, label='Without Imbalance Module',
            edgecolor='black', align='center', zorder=1)  # Align to the right and place it below the blue bars

    # Add a color bar for sample count normalization
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(sample_counts), vmax=max(sample_counts)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Sample Count (Normalized)', fontproperties=font_prop, fontsize=12)

    # Add red dashed lines for average Silhouette Score for each case
    avg_sil_with_balance = sil_scores_with_balance.mean()
    avg_sil_without_balance = sil_scores_without_balance.mean()

    ax1.axhline(y=avg_sil_with_balance, color='red', linestyle='-',
                label=f'Mean Silhouette (With) {avg_sil_with_balance:.2f}')
    ax1.axhline(y=avg_sil_without_balance, color='red', linestyle='--',
                label=f'Mean Silhouette (Without) {avg_sil_without_balance:.2f}')

    # Add a legend with the font properties applied
    ax1.legend(loc='upper right', fontsize=12, prop=font_prop)

    # Show the plot
    plt.tight_layout()
    plt.show()


# Your real data
cluster_scores_with_balance = pd.DataFrame({
    'cluster': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'sample_count': [117, 113, 136, 166, 145, 115, 138, 145, 86, 81, 80, 65, 62],
    'silhouette': [0.6259, 0.6444, 0.5635, 0.6458, 0.5936, 0.5395, 0.4673, 0.4161, 0.5594, 0.5173, 0.4304, 0.4536,
                   0.4880]
})

cluster_scores_without_balance = pd.DataFrame({
    'cluster': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'sample_count': [117, 113, 136, 166, 145, 115, 138, 145, 86, 81, 80, 65, 62],
    'silhouette': [0.5300, 0.4379, 0.5628, 0.6048, 0.5669, 0.5140, 0.5577, 0.5042, 0.4532, 0.5235, 0.3603, 0.1815,
                   0.3477]
})

# Call the function to plot the graph
plot_silhouette_comparison(cluster_scores_with_balance, cluster_scores_without_balance)

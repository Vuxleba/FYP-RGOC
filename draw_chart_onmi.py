import matplotlib.pyplot as plt
import numpy as np

def create_onmi_delta_bar_chart():
    # 1. Data Setup (K-Means ONMI Scores from Table 6.6)
    datasets = ['facebook_107', 'facebook_348', 'facebook_414', 'facebook_686', 'facebook_698']
    
    # K-Means ONMI Scores (%)
    oracle_scores = [7.18, 20.63, 43.09, 9.68, 42.05]
    empirical_scores = [7.53, 21.15, 37.62, 10.48, 40.65]
    
    # 2. Plot Configuration
    x = np.arange(len(datasets))  # Label locations
    width = 0.35  # Width of the bars
    
    # Setup the figure (dpi=300 for high-quality thesis printing)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Colorblind-friendly professional colors
    color_oracle = '#4C72B0'  # Solid Muted Blue
    color_empirical = '#DD8452'  # Solid Muted Orange
    
    # 3. Draw the Grouped Bars
    rects1 = ax.bar(x - width/2, oracle_scores, width, label='Ground-Truth K (Oracle)', color=color_oracle, edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2, empirical_scores, width, label='Empirical K (Elbow)', color=color_empirical, edgecolor='black', linewidth=0.5)
    
    # 4. Add Labels, Title, and Custom Axes (Small Font Configuration)
    ax.set_ylabel('ONMI Score (%)', fontsize=10, fontweight='bold')
    ax.set_title('K-Means Performance Variance: Ground-Truth vs. Empirical K Estimation', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    
    # Format Y-axis to look clean
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7) 
    ax.set_axisbelow(True) 
    
    # Add a legend inside the plot area
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # 5. Add Delta Annotations
    for i in range(len(datasets)):
        delta = empirical_scores[i] - oracle_scores[i]
        max_y = max(oracle_scores[i], empirical_scores[i])
        
        # Color coding the variance
        if delta < -0.05:
            color = '#D62728' # Red for decrease
            text = f"↓ {abs(delta):.1f}%"
        elif delta > 0.05:
            color = '#2CA02C' # Green for increase
            text = f"↑ {delta:.1f}%"
        else:
            color = 'gray' # Gray for basically no change
            text = f"~ {abs(delta):.1f}%"
            
        # Place the annotation above the tallest bar
        ax.annotate(text,
                    xy=(x[i], max_y),
                    xytext=(0, 6),  
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color=color, fontsize=9, fontweight='bold')

    # 6. Final Layout Adjustments and Save
    plt.tight_layout()
    plt.savefig('kmeans_onmi_variance_chart.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_onmi_delta_bar_chart()
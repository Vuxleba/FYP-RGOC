import matplotlib.pyplot as plt
import numpy as np

def create_delta_bar_chart():
    # 1. Data Setup (Currently using K-Means F1 Scores from Table 6.6)
    datasets = ['facebook_107', 'facebook_348', 'facebook_414', 'facebook_686', 'facebook_698']
    
    # K-Means F1 Scores (%)
    oracle_scores = [23.52, 37.23, 50.93, 10.99, 61.29]
    empirical_scores = [23.08, 41.52, 47.36, 11.48, 58.71]
    
    # 2. Plot Configuration
    x = np.arange(len(datasets))  # Label locations
    width = 0.35  # Width of the bars
    
    # Setup the figure with a high-quality resolution (dpi=300 for thesis printing)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Colors designed to be colorblind-friendly and professional
    color_oracle = '#4C72B0'  # Solid Muted Blue
    color_empirical = '#DD8452'  # Solid Muted Orange
    
    # 3. Draw the Grouped Bars
    rects1 = ax.bar(x - width/2, oracle_scores, width, label='Ground-Truth K (Oracle)', color=color_oracle, edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2, empirical_scores, width, label='Empirical K (Elbow)', color=color_empirical, edgecolor='black', linewidth=0.5)
    
    # 4. Add Labels, Title, and Custom Axes
    ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('FCM Performance Variance: Ground-Truth vs. Empirical K Estimation', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    
    # Format Y-axis to look clean (removing top and right spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7) # Add horizontal grid lines behind bars
    ax.set_axisbelow(True) # Ensure grid is behind the bars
    
    # Add a legend inside the plot area
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # 5. Add Delta Annotations (The magic part)
    for i in range(len(datasets)):
        # Calculate the exact difference
        delta = empirical_scores[i] - oracle_scores[i]
        
        # Find the highest point of the two bars to place the text above them
        max_y = max(oracle_scores[i], empirical_scores[i])
        
        # Determine color and arrow direction based on the variance
        if delta < -0.05: # Significant decrease
            color = '#D62728' # Red
            text = f"↓ {abs(delta):.1f}%"
        elif delta > 0.05: # Significant increase
            color = '#2CA02C' # Green
            text = f"↑ {delta:.1f}%"
        else: # Basically no change
            color = 'gray'
            text = f"~ {abs(delta):.1f}%"
            
        # Place the annotation precisely in the center of the bar group
        ax.annotate(text,
                    xy=(x[i], max_y),
                    xytext=(0, 8),  # 8 points vertical offset above the tallest bar
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color=color, fontsize=11, fontweight='bold')

    # 6. Final Layout Adjustments and Save
    plt.tight_layout()
    plt.savefig('fcm_f1_variance_chart.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_delta_bar_chart()
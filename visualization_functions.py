import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################
######################## Feature Selection #######################
################################################################################

def plot_selector_agreement(majority_selector, feature_names):
    """"Function to plot a heatmap showing which features were selected by which voters in the MajorityVoteSelectorTransformer."""

    feature_names = np.asarray(feature_names)
    
    # Collect masks from each fitted selector and convert to int for boolean values
    data = {}
    for i, selector in enumerate(majority_selector.fitted_selectors_):
        
        selector_name = selector.__class__.__name__
        
        # If it's a SelectFromModel, add the base estimator name
        if hasattr(selector, 'estimator'):
            base_estimator_name = selector.estimator.__class__.__name__
            selector_name = f"{selector_name}({base_estimator_name})"
        
        # Convert boolean mask to int (0/1) for heatmap
        data[selector_name] = selector.get_support().astype(int)
    
    # Create DataFrame and add the kept and total votes columns
    df_votes = pd.DataFrame(data, index=feature_names)
    df_votes['Total Votes'] = df_votes.sum(axis=1).astype(int)
    df_votes['KEPT'] = (df_votes['Total Votes'] >= majority_selector.min_votes).astype(int)
    
    # Plot Heatmap (all columns are now int, so seaborn can handle them)
    plt.figure(figsize=(10, len(feature_names) * 0.4))
    sns.heatmap(df_votes, annot=True, cbar=False, cmap="Blues", linewidths=0.5, fmt='d')
    plt.title("Feature Selection Agreement")
    plt.tight_layout()
    plt.show()


################################################################################
######################## Model comparison #######################
################################################################################
def plot_val_mae_comparison(df_scores):
    sns.set_style("whitegrid")

    # Plot validation MAE as primary metric
    plt.figure(figsize=(8,5))
    sns.barplot(x=df_scores.index, y='val_mae', data=df_scores)
    plt.title("Validation MAE Comparison (lower is better)")
    plt.ylabel("MAE")
    plt.xlabel("Model")
    plt.show()

# Subplot with train vs. val comparison
def plot_train_val_comparison(df_scores):
    # Reset index for plotting
    df_plot = df_scores.reset_index().rename(columns={'index': 'Model'})

    # Melt for easier plotting
    df_melt = df_plot.melt(id_vars='Model', 
                        value_vars=df_plot.columns[1:], 
                        var_name='Metric', 
                        value_name='Score')

    # Add columns for Metric Type and Dataset
    df_melt['Dataset'] = df_melt['Metric'].apply(lambda x: 'Train' if x.startswith('train') else 'Validation')
    df_melt['Metric'] = df_melt['Metric'].apply(lambda x: x.replace('train_', '').replace('val_', '').upper())

    # Set style
    sns.set_style("whitegrid")
    palette = {'Train':'skyblue','Validation':'orange'}

    # Create subplots for each metric
    metrics = df_melt['Metric'].unique()
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics),5), sharey=False)

    for i, metric in enumerate(metrics):
        ax = axes[i] if len(metrics) > 1 else axes
        sns.barplot(x='Model', y='Score', hue='Dataset',
                    data=df_melt[df_melt['Metric']==metric], palette=palette, ax=ax)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xlabel('')
        ax.legend(title='')

    plt.suptitle("Model Performance Comparison (Train vs Validation)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
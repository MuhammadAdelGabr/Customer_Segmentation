from flask import Flask, render_template
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score  # Import here
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.figure_factory as ff

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv("CC GENERAL.csv")
df.drop(['CUST_ID'], axis=1, inplace=True)

# Drop rows with NaN values or negative values
df = df.dropna()  # Remove NaN values
df = df[(df >= 0).all(axis=1)]  # Keep only rows with non-negative values

# Scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# KMeans Clustering
km = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)
df['label'] = km.fit_predict(data_scaled)

# PCA for visualization
pca = PCA(n_components=2)
x_pca = pca.fit_transform(data_scaled)
df['PCA1'] = x_pca[:, 0]
df['PCA2'] = x_pca[:, 1]

# Create initial plots
def create_plots():
    fig1 = px.histogram(df, x='MINIMUM_PAYMENTS', title='Distribution of Minimum Payments')
    fig2 = px.scatter(df, x='PCA1', y='PCA2', color='label', title='Customer Segmentation using PCA')
    fig3 = px.box(df, y='BALANCE', title='Boxplot of Balance')
    return pio.to_html(fig1, full_html=False), pio.to_html(fig2, full_html=False), pio.to_html(fig3, full_html=False)

# Additional plots function
def create_additional_plots():
    # Display full correlation matrix with larger size
    correlation = df.corr()
    fig_corr = px.imshow(correlation, title='Full Correlation Matrix', width=800, height=600)
    
    # T-SNE before standard scaling
    t_sne = TSNE(init='pca', random_state=42)
    results = t_sne.fit_transform(data_scaled)
    t_sne_fig = px.scatter(x=results[:, 0], y=results[:, 1], title='T-SNE before Scaling')

    # T-SNE after standard scaling
    df_scaled = scaler.fit_transform(df)  # Apply standard scaling
    if df_scaled.shape[0] > 1:
        results_scaled = t_sne.fit_transform(df_scaled)
        t_sne_scaled_fig = px.scatter(x=results_scaled[:, 0], y=results_scaled[:, 1], title='T-SNE after Standard Scaling')
    else:
        t_sne_scaled_fig = px.scatter(x=[], y=[], title='T-SNE after Standard Scaling (Not enough data)', color_discrete_sequence=['red'])

    # Dendrogram using Plotly
    if df_scaled.shape[0] > 1:
        Z = linkage(df_scaled, method='complete')
        fig_dendro = ff.create_dendrogram(Z)
        fig_dendro.update_layout(title='Hierarchical Clustering Dendrogram (Complete Linkage)')
        dendro_html = pio.to_html(fig_dendro, full_html=False)
    else:
        dendro_html = pio.to_html(ff.create_dendrogram(np.zeros((1, 1))), full_html=False)  # Empty dendrogram for no data

    return (pio.to_html(fig_corr, full_html=False), 
            pio.to_html(t_sne_fig, full_html=False), 
            pio.to_html(t_sne_scaled_fig, full_html=False), 
            dendro_html)

@app.route('/')
def index():
    plot1, plot2, plot3 = create_plots()  # Create the initial plots
    fig_corr, t_sne_fig, t_sne_scaled_fig, dendro_fig = create_additional_plots()  # Create additional plots

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(data_scaled, df['label'])
    
    return render_template('index.html', plot1=plot1, plot2=plot2, plot3=plot3,
                           fig_corr=fig_corr, t_sne_fig=t_sne_fig,
                           t_sne_log_fig=t_sne_scaled_fig, dendro_fig=dendro_fig,
                           silhouette_score=silhouette_avg)

if __name__ == '__main__':
    app.run(debug=True)

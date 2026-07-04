import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sns.set_style('whitegrid')


def run_elbow_method(rfm, max_k=10, output_path=None):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    inertias = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(rfm_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k + 1), inertias, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()

    return rfm_scaled, scaler


def assign_clusters(rfm, rfm_scaled, n_clusters=4, output_path=None):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = km.fit_predict(rfm_scaled)

    segment_map = build_segment_map(rfm)
    rfm['Segment'] = rfm['Cluster'].map(segment_map)

    if output_path:
        plot_segments(rfm, output_path)

    return rfm, km


def build_segment_map(rfm):
    cluster_summary = rfm.groupby('Cluster')['Recency'].mean()
    sorted_clusters = cluster_summary.sort_values()

    labels = ['Champions', 'Loyal', 'At Risk', 'Lost']
    return {cluster: label for cluster, label in zip(sorted_clusters.index, labels)}


def plot_segments(rfm, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col in zip(axes, ['Recency', 'Frequency', 'Monetary']):
        sns.boxplot(data=rfm, x='Segment', y=col, ax=ax)
        ax.set_title(f'{col} by Segment')
        ax.set_xlabel('')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
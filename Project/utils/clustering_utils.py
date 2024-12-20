from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import plotly.graph_objects as go

def compute_kmeans_clusters(embeddings, n_clusters, init, max_iter, algorithm):
    kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, algorithm=algorithm, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

def compute_dbscan_clusters(embeddings, eps, min_samples, metric):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    cluster_labels = dbscan.fit_predict(embeddings)
    return cluster_labels

def plot_clustered_KMeans_embeddings(embeddings, urls_image, n_clusters, init, max_iter, algorithm):
    n_clusters = min(n_clusters, len(embeddings))
    cluster_labels = compute_kmeans_clusters(embeddings, n_clusters, init, max_iter, algorithm)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        mode='markers',
        marker=dict(size=10, color=cluster_labels, colorscale='Jet', opacity=0.8),
        text=urls_image,
        name='KMeans Clusters'
    ))

    fig.update_layout(
        title="Clustering of Embeddings Using K-Means",
        title_font=dict(size=24, family='Arial, sans-serif', weight='bold', color='black'),
        title_x=0.5,
        title_y=0.99,
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        xaxis=dict(
            tickfont=dict(size=18),
            title_font=dict(size=22)
        ),
        yaxis=dict(
            tickfont=dict(size=18),
            title_font=dict(size=22)
        ),
        template='plotly',
        autosize=True,
        margin=dict(l=20, r=20, b=50, t=50),
        width=900,
        height=900,
    )

    return fig

def plot_clustered_DBScan_embeddings(embeddings, urls_image, eps, min_samples, metric):
    min_samples = min(min_samples, len(embeddings))
    cluster_labels = compute_dbscan_clusters(embeddings, eps, min_samples, metric)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        mode='markers',
        marker=dict(size=10, color=cluster_labels, colorscale='Jet', opacity=0.8),
        text=urls_image,
        name='DBSCAN Clusters'
    ))

    fig.update_layout(
        title="Density-Based Clustering Using DBSCAN",
        title_font=dict(size=24, family='Arial, sans-serif', weight='bold', color='black'),
        title_x=0.5,
        title_y=0.99,
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        xaxis=dict(
            tickfont=dict(size=18),
            title_font=dict(size=22)
        ),
        yaxis=dict(
            tickfont=dict(size=18),
            title_font=dict(size=22)
        ),
        template='plotly',
        autosize=True,
        margin=dict(l=20, r=20, b=50, t=50),
        width=900,
        height=900,
    )
    
    return fig

def plot_clustered_agglomerative_embeddings(embeddings, urls_image, n_clusters, linkage, compute_distances):
    n_clusters = min(n_clusters, len(embeddings))
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, compute_distances=compute_distances)
    cluster_labels = agglomerative.fit_predict(embeddings)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        mode='markers',
        marker=dict(size=10, color=cluster_labels, colorscale='Jet', opacity=0.8),
        text=urls_image,
        name='Agglomerative Clusters'
    ))

    fig.update_layout(
        title="Hierarchical Clustering Using Agglomerative Method",
        title_font=dict(size=24, family='Arial, sans-serif', weight='bold'),
        title_x=0.5,
        title_y=0.99,
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        xaxis=dict(
            tickfont=dict(size=18),
            title_font=dict(size=22)
        ),
        yaxis=dict(
            tickfont=dict(size=18),
            title_font=dict(size=22)
        ),
        template='plotly',
        autosize=True,
        margin=dict(l=20, r=20, b=50, t=50),
        width=900,
        height=900
    )

    return fig
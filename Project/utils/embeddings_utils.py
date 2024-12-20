from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding, Isomap
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import numpy as np
import umap

from sklearn.metrics.pairwise import cosine_similarity

def plot_lle_embeddings(embeddings, image_urls, LLE_n_neighbors, LLE_method, LLE_eigen_solver):
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=LLE_n_neighbors, method=LLE_method, eigen_solver=LLE_eigen_solver)
    lle_result = lle.fit_transform(embeddings)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lle_result[:, 0],
        y=lle_result[:, 1],
        mode='markers',
        marker=dict(size=8, color='red', opacity=0.7),
        text=image_urls,
        name='Embeddings'
    ))

    fig.update_layout(
        title="Locally Linear Embedding (LLE) - Local Structure Preservation",
        title_font=dict(size=24, family='Arial, sans-serif', weight='bold'),
        title_x=0.5,
        title_y=0.95,
        xaxis_title='LLE Dimension 1',
        yaxis_title='LLE Dimension 2',
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
        margin=dict(l=40, r=40, b=80, t=80),
        height=900,
        width=900
    )

    return fig

def plot_mds_embeddings(embeddings, image_urls, MDS_metric, MDS_max_iter):
    mds = MDS(n_components=2, random_state=42, metric=MDS_metric, max_iter=MDS_max_iter)
    mds_result = mds.fit_transform(embeddings)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mds_result[:, 0],
        y=mds_result[:, 1],
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.7),
        text=image_urls,
        name='Embeddings'
    ))

    fig.update_layout(
        title="Multidimensional Scaling (MDS) - Distance Preservation",
        title_font=dict(size=24, family='Arial, sans-serif', weight='bold'),
        title_x=0.5,
        title_y=0.95,
        xaxis_title='MDS Component 1',
        yaxis_title='MDS Component 2',
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
        margin=dict(l=40, r=40, b=80, t=80),
        height=900,
        width=900
    )

    return fig

def plot_isomap_embeddings(embeddings, image_urls, ISO_n_neighbors, ISO_metric, ISO_path_method, ISO_neighbors_algorithm):
    isomap = Isomap(n_components=2, n_neighbors=ISO_n_neighbors, metric=ISO_metric, path_method=ISO_path_method, neighbors_algorithm=ISO_neighbors_algorithm)
    isomap_result = isomap.fit_transform(embeddings)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=isomap_result[:, 0],
        y=isomap_result[:, 1],
        mode='markers',
        marker=dict(size=8, color='orange', opacity=0.7),
        text=image_urls,
        name='Embeddings'
    ))

    fig.update_layout(
        title="Isomap - Geodesic Distance Preservation",
        title_font=dict(size=24, family='Arial, sans-serif', weight='bold'),
        title_x=0.5,
        title_y=0.95,
        xaxis_title='Isomap Component 1',
        yaxis_title='Isomap Component 2',
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
        margin=dict(l=40, r=40, b=80, t=80),
        height=900,
        width=900
    )

    return fig

def plot_PCA_embeddings(embeddings, image_urls, PCA_whiten, PCA_svd_solver):
    if embeddings.shape[1] < 2:
        return {}
    
    pca = PCA(n_components=2, whiten=PCA_whiten, svd_solver=PCA_svd_solver)
    embeddings_pca = pca.fit_transform(embeddings)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=embeddings_pca[:, 0],
        y=embeddings_pca[:, 1],
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.7),
        name='Embeddings',
        text=image_urls,
        hoverinfo='text'
    ))

    fig.update_layout(
        title='PCA - Principal Component Analysis (Variance Maximization)',
        title_font=dict(size=24, family='Arial, sans-serif', weight='bold'),
        title_x=0.5,
        title_y=0.95,
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
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
        margin=dict(l=40, r=40, b=80, t=80),
        height=900,
        width=900
    )

    return fig

def plot_tsne_embeddings(embeddings, image_urls, tSNE_perplexity, tSNE_learning_rate, tSNE_n_iter, tSNE_metric):
    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=2, random_state=42, perplexity=tSNE_perplexity, learning_rate=tSNE_learning_rate, n_iter=tSNE_n_iter, metric=str(tSNE_metric))
    tsne_result = tsne.fit_transform(pca_result)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tsne_result[:, 0],
        y=tsne_result[:, 1],
        mode='markers',
        marker=dict(size=5, color='green', opacity=0.7),
        text=image_urls,
        name='Embeddings'
    ))

    fig.update_layout(
        title="t-SNE - t-Distributed Stochastic Neighbor Embedding (Local Structure)",
        title_font=dict(size=24, family='Arial, sans-serif', weight='bold'),
        title_x=0.5,
        title_y=0.95,
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
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
        margin=dict(l=40, r=40, b=80, t=80),
        height=900,
        width=900
    )

    return fig

def plot_UMAP_embeddings(embeddings, image_urls):
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)

    umap_model = umap.UMAP(n_components=2)
    umap_result = umap_model.fit_transform(embeddings_pca)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=umap_result[:, 0],
        y=umap_result[:, 1],
        mode='markers',
        marker=dict(size=5, color='purple', opacity=0.7),
        text=image_urls,
        name='Embeddings'
    ))

    fig.update_layout(
        title="UMAP - Uniform Manifold Approximation and Projection",
        title_font=dict(size=24, family='Arial, sans-serif', weight='bold'),
        title_x=0.5,
        title_y=0.95,
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
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
        margin=dict(l=40, r=40, b=80, t=80),
        height=900,
        width=900
    )

    return fig

def find_embedding_for_attribute(embeddings, labels, attribute_name, positive_label=1):
    indices_positive = np.where(labels[attribute_name] == positive_label)[0]
    indices_negative = np.where(labels[attribute_name] == -1)[0]
    embeddings_positive = embeddings.iloc[indices_positive]
    embeddings_negative = embeddings.iloc[indices_negative]
    embeddings_positive_mean = embeddings_positive.mean(axis=0)
    embeddings_negative_mean = embeddings_negative.mean(axis=0)
    return embeddings_positive_mean, embeddings_negative_mean

def find_most_similar_embedding(target_embedding, embeddings):
    if not isinstance(target_embedding, np.ndarray):
        target_embedding = np.array(target_embedding)
    target_embedding = target_embedding.reshape(1, -1)
    similarities = cosine_similarity(target_embedding, embeddings)
    most_similar_index = np.argmax(similarities)
    return most_similar_index, similarities[0, most_similar_index]
import pandas as pd
import plotly.graph_objects as go

def plot_corr_matrix_facial_feats(facial_features):
    correlation_matrix = facial_features.corr()

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis',
        colorbar=dict(title='Correlation', ticks='outside', tickvals=[-1, 0, 1]),
        zmin=-1, zmax=1
    ))

    fig.update_layout(
        title='Facial Features Correlation Matrix',
        title_font=dict(size=24, family='Arial, sans-serif', weight='bold'),
        title_x=0.55,
        title_y=0.95,
        xaxis_title="Facial Feature [/]",
        yaxis_title="Facial Feature [/]",
        template='plotly',
        autosize=True,
        margin=dict(l=50, r=50, b=100, t=100),
        xaxis=dict(
            ticks='outside',
            showgrid=True,
            tickfont=dict(size=14),
            title_font=dict(size=22)
        ),
        yaxis=dict(
            ticks='outside',
            tickfont=dict(size=14),
            title_font=dict(size=22)
        ),
        height=1000,
        width=1100
    )

    fig.data[0].colorbar.tickvals = [-1, 0, 1]
    fig.data[0].colorbar.ticktext = ['-1', '0', '1']
    fig.data[0].colorbar.title = "Correlation"
    fig.data[0].colorbar.tickfont = dict(size=20)
    fig.data[0].colorbar.titlefont = dict(size=22)

    return fig

def plot_binary_distribution(facial_features):
    feature_counts = facial_features.apply(pd.Series.value_counts).fillna(0)

    fig = go.Figure()
    for label, label_meaning, color in zip([-1, 1], ['absence', 'presence'], ['lightcoral', 'skyblue']):
        fig.add_trace(go.Bar(
            x=feature_counts.columns,
            y=feature_counts.loc[label],
            name=f'Label {label} ({label_meaning})',
            marker_color=color
        ))

    fig.update_layout(
        barmode='stack',
        title="Binary Distribution of Facial Features",
        title_font=dict(size=24, family='Arial, sans-serif', weight='bold'),
        title_x=0.45,
        title_y=0.95,
        xaxis_title="Facial Feature [/]",
        yaxis_title="Count [/]",
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=14),
            title_font=dict(size=22) 
        ),
        yaxis=dict(
            tickfont=dict(size=18),
            title_font=dict(size=22),
            range=[0, 30000]
        ),
        legend=dict(
            font=dict(size=18)
        ),
        autosize=False,
        width=1100,
        height=1000,
        template="plotly"
    )

    return fig

def plot_embedding_distribution(embeddings, feature="All"):
    from app import facial_features

    if feature != "All":
        condition = (facial_features[[feature]] == 1).any(axis=1)
        filtered_embeddings = embeddings[condition]
    else:
        filtered_embeddings = embeddings

    flattened_filtered_embeddings = filtered_embeddings.values.flatten()

    fig = go.Figure(data=go.Histogram(
        x=flattened_filtered_embeddings,
        nbinsx=300,
        marker_color='royalblue',
        opacity=0.75
    ))

    fig.update_layout(
        title="Embedding Value Distribution",
        title_font=dict(size=24, family='Arial, sans-serif', weight='bold'),
        title_x=0.5,
        title_y=0.95,
        xaxis_title="Embedding Value [/]",
        yaxis_title="Frequency [/]",
        xaxis=dict(
            tickfont=dict(size=18),
            title_font=dict(size=22)
        ),
        yaxis=dict(
            tickfont=dict(size=18),
            title_font=dict(size=22)
        ),
        autosize=False,
        width=1000,
        height=1000,
        template="plotly"
    )

    return fig

def plot_feature_cooccurrence(facial_features):
    cooccurrence_matrix = (facial_features.T @ facial_features).fillna(0)
    fig = go.Figure(data=go.Heatmap(
        z=cooccurrence_matrix.values,
        x=cooccurrence_matrix.columns,
        y=cooccurrence_matrix.columns,
        colorscale='Blues',
        colorbar=dict(
            tickfont=dict(size=18),
            title_font=dict(size=22)
        ),

        zmin=0
    ))

    fig.update_layout(
        title="Feature Co-occurrence Heatmap",
        title_font=dict(size=24, family='Arial, sans-serif', weight='bold'),
        title_x=0.5,
        title_y=0.95,
        xaxis_title="Facial Feature [/]",
        yaxis_title="Facial Feature [/]",
        xaxis=dict(
            tickfont=dict(size=14),
            title_font=dict(size=22)
        ),
        yaxis=dict(
            tickfont=dict(size=14),
            title_font=dict(size=22)
        ),
        autosize=False,
        width=1100,
        height=1000,
        template="plotly"
    )
    
    return fig

def plot_attribute_embedding_correlation(facial_features, embeddings):
    correlations = facial_features.corrwith(embeddings.mean(axis=1))
    
    fig = go.Figure(data=go.Bar(
        x=correlations.index,
        y=correlations.values,
        marker_color='seagreen'
    ))

    fig.update_layout(
        title="Attribute Correlation with Embedding Averages",
        title_font=dict(size=24, family='Arial, sans-serif', weight='bold'),
        title_x=0.5,
        title_y=0.95,
        xaxis_title="Facial Attribute [/]",
        yaxis_title="Correlation [/]",
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=14),
            title_font=dict(size=22)
        ),
        yaxis=dict(
            tickfont=dict(size=18),
            title_font=dict(size=22)
        ),
        autosize=False,
        width=1000,
        height=1000,
        template="plotly"
    )
    
    return fig
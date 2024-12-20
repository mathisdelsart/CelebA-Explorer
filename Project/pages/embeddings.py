from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA

from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dcc, html
import dash

from datetime import datetime
import numpy as np

from utils.embeddings_utils import plot_lle_embeddings, plot_mds_embeddings, \
        plot_isomap_embeddings, find_embedding_for_attribute, \
        find_most_similar_embedding, plot_PCA_embeddings, plot_tsne_embeddings

from utils.clustering_utils import plot_clustered_KMeans_embeddings, \
        plot_clustered_DBScan_embeddings, plot_clustered_agglomerative_embeddings

def create_embeddings_page():
    from app import facial_features

    section_title = html.Div([
        html.H1(
            "Explore the Latent Space Through Embeddings", 
            style={
                'text-align': 'center', 
                'margin-bottom': '20px', 
                'font-size': '48px', 
                'font-weight': 'bold', 
                'color': '#4A90E2',
                'text-transform': 'uppercase',
                'letter-spacing': '3px',
                'text-shadow': '3px 3px 5px rgba(0, 0, 0, 0.2)',
                'font-family': 'Arial, sans-serif',
                'border-bottom': '4px solid #4A90E2',
                'padding-bottom': '10px'
            }
        ),

        html.Div([
            html.H3(
                "Select and weight features to find the best matches in the data's hidden space!", 
                style={
                    'text-align': 'center',
                    'font-size': '30px',
                    'font-weight': 'normal',
                    'color': '#7F8C8D',
                    'font-family': 'Arial, sans-serif',
                    'max-width': '80%',
                    'margin': '0 auto',
                    'line-height': '1.5',
                    'transition': 'all 0.3s ease',
                    'letter-spacing': '1px'
                }
            ),
        ], style={
            'padding': '20px', 
            'background-color': '#f9f9f9', 
            'border-radius': '10px', 
            'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
            'margin-bottom': '20px',
            'text-align': 'center'
        }),
    ])

    feature_weights_store = dcc.Store(
        id='feature-weights-store',
        storage_type='session',
        data={feature: 0 for feature in facial_features}
    )

    run_algo_store = dcc.Store(id='run-algo-store', storage_type='session', data={'run': False})

    embedding_hover_data = dcc.Store(id='embedding-hover-data-store', storage_type='session', data=None)
    facial_hover_data = dcc.Store(id='facial-features-hover-data-store', storage_type='session', data=None)

    filter_mode_store = dcc.Store(id='filter-mode-store', storage_type='session', data='AND')

    reduct_algo_hist_facial = dcc.Store(id='reduction-algo-store-facial', data=None)
    reduct_algo_hist_embed = dcc.Store(id='reduction-algo-store-embed', data=None)

    weights_controls = html.Div([
        html.H3(
            "Assign Weights to Features:", 
            style={
                'margin-bottom': '30px',
                'font-weight': '900',
                'font-size': '32px',
                'text-decoration': 'underline',
                'text-align': 'center',
                'color': '#2c3e50',
                'letter-spacing': '1.5px',
                'font-family': 'Arial, sans-serif',
                'text-shadow': '2px 2px 4px rgba(0, 0, 0, 0.2)'
            }
        ),
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col(
                        dbc.Button(
                            f"Toggle {feature}", 
                            id=f"toggle-button-{feature}",
                            color="primary", 
                            size="sm",
                            style={'margin-bottom': '5px'}
                        ), width=4
                    ),
                    dbc.Col(
                        dcc.Input(
                            id=f"weight-{feature}",
                            type='number',
                            value=0,
                            min=0,
                            step=0.1,
                            style={'width': '100%'}
                        ), width=8
                    ),
                ]) for feature in facial_features
            ])
        ], style={'overflow-y': 'auto', 'max-height': '500px'}),

        html.Div([
            dbc.Button("Enable All", id="enable-all", color="success", size="lg", style={'margin-top': '15px', 'margin-right': '10px'}),
            dbc.Button("Disable All", id="disable-all", color="danger", size="lg", style={'margin-top': '15px', 'margin-right': '10px'}),
            dbc.Button(
                id="filter-mode-switch", 
                color="info", 
                size="lg", 
                style={'margin-top': '15px'},
            ),
            html.Br(),
            html.Div(
                id="celeb-count-box",
                children=[
                    html.P(
                        "Number of Images Present in the Filtered Dataset: 0",
                        id="celeb-count-text",
                        style={
                            'font-size': '18px',
                            'color': '#555',
                            'margin-top': '15px',
                            'padding': '10px',
                            'border': '1px solid #ccc',
                            'border-radius': '5px',
                            'backgroundColor': '#e9ecef',
                            'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)',
                            'text-align': 'center'
                        }
                    )
                ]
            ),
            html.Br(),
            html.Div(
                id="filter-mode-indicator",
                children=[
                    html.Div(
                        children=[
                            html.P(
                                [
                                    "The ", html.Strong("'Enable All'"), " button activates all features by setting their weights to 1. ",
                                    "Similarly, the ", html.Strong("'Disable All'"), " button deactivates all features by setting their weights to 0. ",
                                    "You can also choose between ", html.Strong("AND"), " and ", html.Strong("OR"), " filter modes to control how features are included in dimensionality reduction.\n",
                                    "The ", html.Strong("'AND mode'"), " requires all selected features to be present for an image to be included. ",
                                    "The ", html.Strong("'OR mode'"), " only requires at least one selected feature to be present for an image to be included."
                                ],
                                style={
                                    'font-size': '18px',
                                    'color': '#2c3e50',
                                    'background-color': '#f7f7f7',
                                    'padding': '20px',
                                    'border': '1px solid #ddd',
                                    'border-radius': '8px',
                                    'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                                    'line-height': '1.6',
                                    'font-family': 'Arial, sans-serif',
                                    'margin-top': '20px',
                                    'white-space': 'pre-line',
                                    'text-align': 'left'
                                }
                            )
                        ],
                        style={
                            'border': '1px solid #ccc', 
                            'border-radius': '5px', 
                            'padding': '15px',
                            'backgroundColor': '#f9f9f9',
                            'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.1)'
                        }
                    )
                ]
            )
        ], style={'text-align': 'center', 'margin-top': '20px'}),
    ], style={'padding': '10px', 'border-right': '1px solid #ddd'})

    image_display = html.Div([
        html.H3(
            "Best Match with the weighted features:", 
            style={
                'margin-bottom': '30px',
                'font-weight': '900',
                'font-size': '32px',
                'text-decoration': 'underline',
                'text-align': 'center',
                'color': '#2c3e50',
                'letter-spacing': '1.5px',
                'font-family': 'Arial, sans-serif',
                'text-shadow': '2px 2px 4px rgba(0, 0, 0, 0.2)'
            }
        ),

        html.Div(
            id='selected-image-section',
            children=[
                dcc.Loading(
                    id="loading-image-to-display",
                    type="circle",
                    children=html.Img(id='image-to-display', style={'width': '45%', 'height': '45%'}),
                    style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'height': '200px'}
                )
            ],
            style={'text-align': 'center', 'padding': '10px'}
        )
    ], style={'padding': '10px'})

    title_plots_section = html.Div([
            html.H3(
                "Explore the Power of Dimensionality Reduction and Clustering Visualizations",  
                style={
                    'font-size': '36px',
                    'font-weight': '700',
                    'color': '#2c3e50',
                    'text-align': 'center',
                    'margin-bottom': '20px',
                    'letter-spacing': '1px',
                    'font-family': 'Arial, sans-serif',
                    'text-shadow': '2px 2px 4px rgba(0, 0, 0, 0.1)'
                }
            ),
            html.H3(
                "Dive into the Latent Space: Unlock Insights with Multiple Dimensionality Reduction Techniques on Curated Embeddings and Facial Features",
                style={
                    'font-size': '24px',
                    'font-weight': '400',
                    'color': '#34495e',
                    'text-align': 'center',
                    'line-height': '1.6',
                    'font-family': 'Arial, sans-serif',
                    'margin-bottom': '30px',
                    'letter-spacing': '0.5px',
                    'text-shadow': '1px 1px 3px rgba(0, 0, 0, 0.1)'
                }
            ),

        ], style={
            'padding': '20px', 
            'background-color': '#f9f9f9', 
            'border-radius': '10px', 
            'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
            'margin-bottom': '20px',
            'text-align': 'center'
        })
    
    plots_section = html.Div([
        dbc.Row([
            dbc.Row([
                html.H4(
                    "Set Parameters for Dimensionality Reduction Algorithms", 
                    style={
                        'font-size': '28px',
                        'font-weight': '700',
                        'color': '#2c3e50',
                        'text-align': 'center',
                        'margin-bottom': '25px',
                        'letter-spacing': '1px',
                        'font-family': 'Arial, sans-serif',
                        'text-shadow': '2px 2px 4px rgba(0, 0, 0, 0.1)'
                    }
                ),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("PCA Parameters", className="text-center", style={'color': '#007bff'})),
                        dbc.CardBody([
                            html.Div([
                                html.Label("SVD Solver", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id="pca-svd_solver",
                                    options=[
                                        {'label': 'Auto', 'value': 'auto'},
                                        {'label': 'Full', 'value': 'full'},
                                        {'label': 'Arpack', 'value': 'arpack'},
                                        {'label': 'Randomized', 'value': 'randomized'}
                                    ],
                                    value='auto',
                                    style={'width': '70%', 'display': 'inline-block', 'margin-left': '10px'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Algorithm to use for the SVD computation. 'Auto' is the default and selects the best method for the given data.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Div([
                                html.Label("Whiten", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id="pca-whiten",
                                    options=[
                                        {'label': 'True', 'value': True},
                                        {'label': 'False', 'value': False}
                                    ],
                                    value=False,
                                    style={'width': '70%', 'display': 'inline-block', 'margin-left': '10px'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "If checked, the components will be scaled to have unit variance (True). Leave unchecked for False.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            )
                        ])
                    ], style={'backgroundColor': '#f8f9fa', 'border': '1px solid #007bff', 'border-radius': '5px'})
                ], width=2),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("t-SNE Parameters", className="text-center", style={'color': '#007bff'})),
                        dbc.CardBody([
                            html.Div([
                                html.Label("Perplexity", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Input(
                                    id="tsne-perplexity",
                                    type="number",
                                    value=30,
                                    min=5,
                                    max=50,
                                    step=1,
                                    placeholder="perplexity",
                                    style={'width': '70%', 'display': 'inline-block', 'margin-left': '10px'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Number of effective neighbors. Controls balance between local and global aspects.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Br(),
                            html.Div([
                                html.Label("learning_rate", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Input(
                                    id="tsne-learning_rate",
                                    type="number",
                                    value=200,
                                    min=10,
                                    max=1000,
                                    step=10,
                                    placeholder="learning_rate",
                                    style={'width': '70%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Controls step size in optimization. Higher values can speed up convergence but may cause instability.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Br(),
                            html.Div([
                                html.Label("n_iter", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Input(
                                    id="tsne-n_iter",
                                    type="number",
                                    value=1000,
                                    min=250,
                                    step=50,
                                    placeholder="n_iter",
                                    style={'width': '70%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Number of iterations for optimization. Larger values can lead to better convergence.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Br(),
                            html.Div([
                                html.Label("metric", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id="tsne-metric",
                                    options=[
                                        {'label': 'euclidean', 'value': 'euclidean'},
                                        {'label': 'cosine', 'value': 'cosine'},
                                        {'label': 'manhattan', 'value': 'manhattan'},
                                        {'label': 'chebyshev', 'value': 'chebyshev'}
                                    ],
                                    value='euclidean',
                                    style={'width': '70%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Metric to compute distances between points. Euclidean is the default.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                        ])
                    ], style={'backgroundColor': '#f8f9fa', 'border': '1px solid #007bff', 'border-radius': '5px'})
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Isomap Parameters", className="text-center", style={'color': '#007bff'})),
                        dbc.CardBody([
                            html.Div([
                                html.Label("n_neighbors", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Input(
                                    id="isomap-n_neighbors",
                                    type="number",
                                    value=10,
                                    min=5,
                                    max=50,
                                    step=1,
                                    placeholder="n_neighbors",
                                    style={'width': '70%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Number of neighbors to consider for the graph construction. Controls local structure.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Br(),
                            html.Div([
                                html.Label("metric", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id="isomap-metric",
                                    options=[
                                        {'label': 'euclidean', 'value': 'euclidean'},
                                        {'label': 'manhattan', 'value': 'manhattan'},
                                        {'label': 'cosine', 'value': 'cosine'},
                                        {'label': 'chebyshev', 'value': 'chebyshev'}
                                    ],
                                    value='euclidean',
                                    style={'width': '70%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Metric to compute distances between points. Euclidean is commonly used.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Br(),
                            html.Div([
                                html.Label("path_method", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id="isomap-path_method",
                                    options=[
                                        {'label': 'auto', 'value': 'auto'},
                                        {'label': 'Floyd-Warshall (FW)', 'value': 'FW'},
                                        {'label': 'Dijkstra (D)', 'value': 'D'}
                                    ],
                                    value='auto',
                                    style={'width': '70%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Method for computing shortest paths in the graph. Defaults to auto.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Br(),
                            html.Div([
                                html.Label("neighbors_algorithm", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id="isomap-neighbors_algorithm",
                                    options=[
                                        {'label': 'auto', 'value': 'auto'},
                                        {'label': 'brute', 'value': 'brute'},
                                        {'label': 'kd_tree', 'value': 'kd_tree'},
                                        {'label': 'ball_tree', 'value': 'ball_tree'}
                                    ],
                                    value='auto',
                                    style={'width': '70%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Algorithm used to find nearest neighbors. Auto selects the best option.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                        ])
                    ], style={'backgroundColor': '#f8f9fa', 'border': '1px solid #007bff', 'border-radius': '5px'})
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("LLE Parameters", className="text-center", style={'color': '#007bff'})),
                        dbc.CardBody([
                            html.Div([
                                html.Label("n_neighbors", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Input(
                                    id="lle-n_neighbors",
                                    type="number",
                                    value=10,
                                    min=5,
                                    max=50,
                                    step=1,
                                    placeholder="n_neighbors",
                                    style={'width': '70%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Number of neighbors to consider for local relationships. Controls the local structure.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Br(),
                            html.Div([
                                html.Label("method", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id="lle-method",
                                    options=[
                                        {'label': 'Standard', 'value': 'standard'},
                                        {'label': 'Modified', 'value': 'modified'},
                                        {'label': 'Hessian', 'value': 'hessian'},
                                        {'label': 'LTSA', 'value': 'ltsa'}
                                    ],
                                    value='standard',
                                    style={'width': '70%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Method for the embedding. 'Standard' is the basic LLE, while others handle complex cases.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Br(),
                            html.Div([
                                html.Label("eigen_solver", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id="lle-eigen_solver",
                                    options=[
                                        {'label': 'Auto', 'value': 'auto'},
                                        {'label': 'Dense', 'value': 'dense'},
                                        {'label': 'ARPACK', 'value': 'arpack'}
                                    ],
                                    value='auto',
                                    style={'width': '70%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Solver to compute eigenvalues. 'Auto' selects the best based on data.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                        ])
                    ], style={'backgroundColor': '#f8f9fa', 'border': '1px solid #007bff', 'border-radius': '5px'})
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("MDS Parameters", className="text-center", style={'color': '#007bff'})),
                        dbc.CardBody([
                            html.Div([
                                html.Label("metric", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id="mds-metric",
                                    options=[
                                        {'label': 'Metric', 'value': True},
                                        {'label': 'Non-Metric', 'value': False}
                                    ],
                                    value=True,
                                    style={'width': '70%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Choose whether to use metric or non-metric MDS. Metric preserves exact distances; non-metric preserves rankings.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Br(),
                            html.Div([
                                html.Label("max_iter", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Input(
                                    id="mds-max_iter",
                                    type="number",
                                    value=300,
                                    min=100,
                                    max=1000,
                                    step=50,
                                    placeholder="max_iter",
                                    style={'width': '70%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Maximum number of iterations for the optimization algorithm.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            )
                        ])
                    ], style={'backgroundColor': '#f8f9fa', 'border': '1px solid #007bff', 'border-radius': '5px'})
                ])
            ]),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Select Algorithm for Embedding Reduction", style={'font-size': '24px', 'font-weight': 'bold', 'align-items': 'center', 'justify-content': 'center'}),
                    dbc.CardBody([
                        dbc.Button("PCA", id="algo-pca", color="primary", style={'margin': '5px'}),
                        dbc.Button("Isomap", id="algo-isomap", color="primary", style={'margin': '5px'}),
                        dbc.Button("t-SNE", id="algo-tsne", color="primary", style={'margin': '5px'}),
                        dbc.Button("LLE", id="algo-lle", color="primary", style={'margin': '5px'}),
                        dbc.Button("MDS", id="algo-mds", color="primary", style={'margin': '5px'}),
                        html.Div([
                            dcc.Loading(
                                id="loading-clustering", 
                                type="circle",
                                children=[
                                    dbc.Button(
                                        "Apply Clustering", 
                                        id="apply-clustering", 
                                        color="success", 
                                        style={
                                            'margin': '5px', 
                                            'padding': '10px 20px', 
                                            'font-size': '16px', 
                                            'border-radius': '8px', 
                                            'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 
                                            'transition': '0.3s ease',
                                            'position': 'relative'
                                        }
                                    ),
                                ]
                            ),
                            
                            dcc.Dropdown(
                                id="clustering-algo",
                                options=[
                                    {'label': 'DBSCAN', 'value': 'dbscan'},
                                    {'label': 'K-Means', 'value': 'kmeans'},
                                    {'label': 'Agglomerative', 'value': 'agglomerative'}
                                ],
                                value='kmeans',
                                style={
                                    'width': '250px', 
                                    'margin-top': '10px', 
                                    'padding': '10px', 
                                    'border-radius': '8px', 
                                    'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 
                                    'font-size': '16px', 
                                    'transition': '0.3s ease'
                                }
                            ),
                        ], style={
                            'display': 'flex', 
                            'align-items': 'center', 
                            'justify-content': 'space-between', 
                            'margin-top': '20px',
                            'padding': '10px',
                            'background-color': '#f8f9fa', 
                            'border-radius': '10px',
                            'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)'
                        })
                    ])
                ]),
                dbc.Card([
                    dbc.CardHeader("Embedding Visualization", style={'font-size': '34px', 'font-weight': 'bold', 'color': '#007bff', 'align-items': 'center', 'justify-content': 'center'}),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-plot",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id="embedding-plot",
                                    figure={},
                                    style={'height': '100%', 'width': '100%'},
                                    config={'displayModeBar': False, 'staticPlot': False}
                                )
                            ]
                        ),
                        html.Div(
                            id='loading-message',
                            children="Waiting for data...",
                            style={'text-align': 'center', 'font-size': '18px', 'margin-top': '20px', 'display': 'none'}
                        ),
                        html.Div(id='image-container', children=[])
                    ])
                ])

            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Select Algorithm for Facial Feature Reduction", style={'font-size': '24px', 'font-weight': 'bold', 'align-items': 'center', 'justify-content': 'center'}),
                    dbc.CardBody([
                        dbc.Button("PCA", id="algo-pca-facial", color="primary", style={'margin': '5px'}),
                        dbc.Button("Isomap", id="algo-isomap-facial", color="primary", style={'margin': '5px'}),
                        dbc.Button("t-SNE", id="algo-tsne-facial", color="primary", style={'margin': '5px'}),
                        dbc.Button("LLE", id="algo-lle-facial", color="primary", style={'margin': '5px'}),
                        dbc.Button("MDS", id="algo-mds-facial", color="primary", style={'margin': '5px'}),
                        html.Div([
                            dcc.Loading(
                                id="loading-clustering",
                                type="circle",
                                children=[
                                    dbc.Button(
                                        "Apply Clustering", 
                                        id="apply-clustering-facial", 
                                        color="success", 
                                        style={
                                            'margin': '5px', 
                                            'padding': '10px 20px', 
                                            'font-size': '16px', 
                                            'border-radius': '8px', 
                                            'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 
                                            'transition': '0.3s ease',
                                            'position': 'relative'
                                        }
                                    ),
                                ]
                            ),
                            
                            dcc.Dropdown(
                                id="clustering-algo-facial",
                                options=[
                                    {'label': 'DBSCAN', 'value': 'dbscan'},
                                    {'label': 'KMeans', 'value': 'kmeans'},
                                    {'label': 'Agglomerative', 'value': 'agglomerative'}
                                ],
                                value='kmeans',
                                style={
                                    'width': '250px', 
                                    'margin-top': '10px', 
                                    'padding': '10px', 
                                    'border-radius': '8px', 
                                    'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)', 
                                    'font-size': '16px', 
                                    'transition': '0.3s ease'
                                }
                            ),
                        ], style={
                            'display': 'flex', 
                            'align-items': 'center', 
                            'justify-content': 'space-between', 
                            'margin-top': '20px',
                            'padding': '10px',
                            'background-color': '#f8f9fa', 
                            'border-radius': '10px',
                            'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)'
                        })
                    ])
                ]),
                dbc.Card([
                    dbc.CardHeader("Facial Feature Visualization", style={'font-size': '34px', 'font-weight': 'bold', 'color': '#007bff', 'align-items': 'center', 'justify-content': 'center'}),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-plot-facial",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id="facial-feature-plot",
                                    figure={},
                                    style={'height': '100%', 'width': '100%'},
                                    config={'displayModeBar': False, 'staticPlot': False}
                                )
                            ]
                        ),
                        html.Div(
                            id='loading-message-facial',
                            children="Waiting for data...",
                            style={'text-align': 'center', 'font-size': '18px', 'margin-top': '20px', 'display': 'none'}
                        ),
                        html.Div(id='image-container')
                    ])
                ]),
                html.Br(),
                html.Br(),
                html.Br()
            ], width=6),

            dbc.Row([
                html.H4(
                    "Set Parameters for CLustering Algorithms", 
                    style={
                        'font-size': '28px',
                        'font-weight': '700',
                        'color': '#2c3e50',
                        'text-align': 'center',
                        'margin-bottom': '25px',
                        'letter-spacing': '1px',
                        'font-family': 'Arial, sans-serif',
                        'text-shadow': '2px 2px 4px rgba(0, 0, 0, 0.1)'
                    }
                ),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("K-Means Parameters", className="text-center", style={'color': '#007bff'})),
                        dbc.CardBody([
                            html.Div([
                                html.Label("n_clusters", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Input(
                                    id="kmeans-n_cluster",
                                    type='number',
                                    value=8,
                                    step=1,
                                    min=2,
                                    max=31000,
                                    style={'width': '20%', 'display': 'inline-block', 'margin-left': '10px'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "The number of clusters to form as well as the number of centroids to generate.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Div([
                                html.Label("init", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id="kmeans-init",
                                    options=[
                                        {'label': 'k-means++', 'value': 'k-means++'},
                                        {'label': 'random', 'value': 'random'}
                                    ],
                                    value='k-means++',
                                    style={'width': '60%', 'display': 'inline-block', 'margin-left': '10px'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Method for initialization. 'k-means++' selects initial cluster centers for faster convergence.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Div([
                                html.Label("max_iter", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Input(
                                    id="kmeans-max_iter",
                                    type='number',
                                    value=300,
                                    step=50,
                                    min=100,
                                    max=1000,
                                    style={'width': '30%', 'display': 'inline-block', 'margin-left': '10px'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Maximum number of iterations of the k-means algorithm for a single run.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Div([
                                html.Label("algorithm", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id="kmeans-algorithm",
                                    options=[
                                        {'label': 'elkan', 'value': 'elkan'},
                                        {'label': 'lloyd', 'value': 'lloyd'}
                                    ],
                                    value='lloyd',
                                    style={'width': '30%', 'display': 'inline-block', 'margin-left': '10px'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "K-means algorithm to use. 'auto' selects the most appropriate algorithm based on the dataset size and sparsity.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            )
                        ])
                    ], style={'backgroundColor': '#f8f9fa', 'border': '1px solid #007bff', 'border-radius': '5px'})
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("DBScan Parameters", className="text-center", style={'color': '#007bff'})),
                        dbc.CardBody([
                            # eps
                            html.Div([
                                html.Label("Eps", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Input(
                                    id="dbscan-eps",
                                    type="number",
                                    value=0.5,
                                    min=0.1,
                                    max=1.0,
                                    step=0.1,
                                    placeholder="Eps value",
                                    style={'width': '10%', 'display': 'inline-block', 'margin-left': '10px'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "The maximum distance between two samples for them to be considered as in the same neighborhood.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Br(),
                            html.Div([
                                html.Label("Min Samples", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Input(
                                    id="dbscan-min_samples",
                                    type="number",
                                    value=5,
                                    min=1,
                                    max=31000,
                                    step=1,
                                    placeholder="Min samples",
                                    style={'width': '20%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "The number of samples in a neighborhood for a point to be considered as a core point.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Br(),
                            html.Div([
                                html.Label("Metric", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id="dbscan-metric",
                                    options=[
                                        {'label': 'euclidean', 'value': 'euclidean'},
                                        {'label': 'manhattan', 'value': 'manhattan'},
                                        {'label': 'cosine', 'value': 'cosine'},
                                        {'label': 'chebyshev', 'value': 'chebyshev'}
                                    ],
                                    value='euclidean',
                                    style={'width': '50%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "The metric to use when calculating distance between instances in a feature array.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                        ])
                    ], style={'backgroundColor': '#f8f9fa', 'border': '1px solid #007bff', 'border-radius': '5px'})
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("AgglomerativeClustering Parameters", className="text-center", style={'color': '#007bff'})),
                        dbc.CardBody([
                            html.Div([
                                html.Label("n_clusters", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Input(
                                    id="agglomerative-n_clusters",
                                    type="number",
                                    value=3,
                                    min=2,
                                    max=31000,
                                    step=1,
                                    placeholder="n_clusters",
                                    style={'width': '20%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "The number of clusters to form. Must be between 2 and n_samples.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Br(),
                            html.Div([
                                html.Label("linkage", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id="agglomerative-linkage",
                                    options=[
                                        {'label': 'ward', 'value': 'ward'},
                                        {'label': 'complete', 'value': 'complete'},
                                        {'label': 'average', 'value': 'average'},
                                        {'label': 'single', 'value': 'single'}
                                    ],
                                    value='ward',
                                    style={'width': '50%'}
                                ),
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "The linkage criterion to use. Ward's minimum variance is the default.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': '10px'}
                            ),
                            html.Br(),
                            html.Div([
                                html.Label("compute_distances", style={'margin-right': '10px', 'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id="agglomerative-compute_distances",
                                    options=[
                                        {'label': "Compute distances", 'value': True},
                                        {'label': "Do not compute distances", 'value': False}
                                    ],
                                    value=True,
                                    style={'width': '60%'}
                                )
                            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '15px'}),
                            html.Div(
                                html.Span(
                                    "Whether to compute distances between points. Defaults to True.",
                                    style={'font-size': '16px', 'color': '#666', 'font-style': 'italic'}
                                ),
                                style={'margin-left': 10}
                            ),
                        ])
                    ], style={'backgroundColor': '#f8f9fa', 'border': '1px solid #007bff', 'border-radius': 5})
                ])
            ]),
        ])
    ], style={'padding': '10px'})
    html.Br(),

    main_layout = html.Div([
        feature_weights_store,
        filter_mode_store,
        run_algo_store,
        embedding_hover_data,
        facial_hover_data,
        reduct_algo_hist_facial,
        reduct_algo_hist_embed,
        section_title,
        dbc.Row([
            dbc.Col(weights_controls, width=4),
            dbc.Col(image_display, width=8)
        ], className="mb-4"),
        title_plots_section,
        plots_section
    ])

    return main_layout

def embeddings_register_callbacks(app):
    from app import facial_features, image_urls, embeddings

    @app.callback(
    [
        Output('embedding-plot', 'figure'),
        Output('reduction-algo-store-embed', 'data')
    ],
    [
        Input('algo-pca', 'n_clicks'),
        Input('algo-isomap', 'n_clicks'),
        Input('algo-tsne', 'n_clicks'),
        Input('algo-lle', 'n_clicks'),
        Input('algo-mds', 'n_clicks'),
        Input('apply-clustering', 'n_clicks')
    ],
    [
        State('reduction-algo-store-embed', 'data'),
        State('clustering-algo', 'value'),
        State('feature-weights-store', 'data'),
        State('filter-mode-store', 'data'),
        State('pca-svd_solver', 'value'),
        State('pca-whiten', 'value'),
        State('tsne-perplexity', 'value'),
        State('tsne-learning_rate', 'value'),
        State('tsne-n_iter', 'value'),
        State('tsne-metric', 'value'),
        State('isomap-n_neighbors', 'value'),
        State('isomap-metric', 'value'),
        State('isomap-path_method', 'value'),
        State('isomap-neighbors_algorithm', 'value'),
        State('lle-n_neighbors', 'value'),
        State('lle-method', 'value'),
        State('lle-eigen_solver', 'value'),
        State('mds-metric', 'value'),
        State('mds-max_iter', 'value'),
        State('kmeans-n_cluster', 'value'),
        State('kmeans-init', 'value'),
        State('kmeans-max_iter', 'value'),
        State('kmeans-algorithm', 'value'),
        State('dbscan-eps', 'value'),
        State('dbscan-min_samples', 'value'),
        State('dbscan-metric', 'value'),
        State('agglomerative-n_clusters', 'value'),
        State('agglomerative-linkage', 'value'),
        State('agglomerative-compute_distances', 'value')
    ],
    prevent_initial_call=True
    )
    def update_embeddings_plot(
        pca_clicks, isomap_clicks, tsne_clicks, lle_clicks, mds_clicks,
        apply_clustering_clicks, last_reduction_algo,
        clustering_algo, feature_weights, filter_mode,
        PCA_svd_solver, PCA_whiten,
        tSNE_perplexity, tSNE_learning_rate, tSNE_n_iter, tSNE_metric,
        ISO_n_neighbors, ISO_metric, ISO_path_method, ISO_neighbors_algorithm,
        LLE_n_neighbors, LLE_method, LLE_eigen_solver,
        MDS_metric, MDS_max_iter,
        n_clusters_kmeans, init_kmeans, max_iter_kmeans, algorithm_kmeans,
        eps_dbscan, min_samples_dbscan, metric_dbscan,
        n_clusters_agglomerative, linkage_agglomerative, compute_distances_agglomerative
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            return {}, last_reduction_algo

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        reduction_algo = last_reduction_algo
        if triggered_id == 'algo-pca':
            reduction_algo = 'PCA'
        elif triggered_id == 'algo-isomap':
            reduction_algo = 'Isomap'
        elif triggered_id == 'algo-tsne':
            reduction_algo = 't-SNE'
        elif triggered_id == 'algo-lle':
            reduction_algo = 'LLE'
        elif triggered_id == 'algo-mds':
            reduction_algo = 'MDS'

        selected_features = [feature for feature, weight in feature_weights.items() if weight > 0]
        if not selected_features:
            return {}, reduction_algo

        if filter_mode == 'AND':
            condition = (facial_features[selected_features] == 1).all(axis=1)
        else:
            condition = (facial_features[selected_features] == 1).any(axis=1)

        indices_true = np.where(condition)[0]
        filtered_embeddings = embeddings[condition]
        filtered_urls = [image_urls[i] for i in indices_true]

        if len(filtered_embeddings) == 0:
            return {}, reduction_algo

        if triggered_id == 'apply-clustering':
            if reduction_algo == 'PCA':
                pca = PCA(n_components=2, svd_solver=PCA_svd_solver, whiten=PCA_whiten)
                filtered_embeddings = pca.fit_transform(filtered_embeddings)
            elif reduction_algo == 'Isomap':
                iso = Isomap(n_components=2, n_neighbors=ISO_n_neighbors, metric=ISO_metric, path_method=ISO_path_method, neighbors_algorithm=ISO_neighbors_algorithm)
                filtered_embeddings = iso.fit_transform(filtered_embeddings)
            elif reduction_algo == 't-SNE':
                tsne = TSNE(n_components=2, perplexity=tSNE_perplexity, learning_rate=tSNE_learning_rate, n_iter=tSNE_n_iter, metric=tSNE_metric)
                filtered_embeddings = tsne.fit_transform(filtered_embeddings)
            elif reduction_algo == 'LLE':
                lle = LocallyLinearEmbedding(n_components=2, n_neighbors=LLE_n_neighbors, method=LLE_method, eigen_solver=LLE_eigen_solver)
                filtered_embeddings = lle.fit_transform(filtered_embeddings)
            elif reduction_algo == 'MDS':
                mds = MDS(n_components=2, metric=MDS_metric, max_iter=MDS_max_iter)
                filtered_embeddings = mds.fit_transform(filtered_embeddings)
            else:
                pca = PCA(n_components=2)
                filtered_embeddings = pca.fit_transform(filtered_embeddings)

            if clustering_algo == 'kmeans':
                return plot_clustered_KMeans_embeddings(filtered_embeddings, filtered_urls, n_clusters_kmeans, init_kmeans, max_iter_kmeans, algorithm_kmeans), reduction_algo
            elif clustering_algo == 'dbscan':
                return plot_clustered_DBScan_embeddings(filtered_embeddings, filtered_urls, eps_dbscan, min_samples_dbscan, metric_dbscan), reduction_algo
            elif clustering_algo == 'agglomerative':
                return plot_clustered_agglomerative_embeddings(filtered_embeddings, filtered_urls, n_clusters_agglomerative, linkage_agglomerative, compute_distances_agglomerative), reduction_algo

        if reduction_algo == 'PCA':
            return plot_PCA_embeddings(filtered_embeddings, filtered_urls, PCA_whiten, PCA_svd_solver), reduction_algo
        elif reduction_algo == 'Isomap':
            return plot_isomap_embeddings(filtered_embeddings, filtered_urls, ISO_n_neighbors, ISO_metric, ISO_path_method, ISO_neighbors_algorithm), reduction_algo
        elif reduction_algo == 't-SNE':
            return plot_tsne_embeddings(filtered_embeddings, filtered_urls, tSNE_perplexity, tSNE_learning_rate, tSNE_n_iter, tSNE_metric), reduction_algo
        elif reduction_algo == 'LLE':
            return plot_lle_embeddings(filtered_embeddings, filtered_urls, LLE_n_neighbors, LLE_method, LLE_eigen_solver), reduction_algo
        elif reduction_algo == 'MDS':
            return plot_mds_embeddings(filtered_embeddings, filtered_urls, MDS_metric, MDS_max_iter), reduction_algo

    @app.callback(
    [
        Output('facial-feature-plot', 'figure'),
        Output('reduction-algo-store-facial', 'data')
    ],
    [
        Input('algo-pca-facial', 'n_clicks'),
        Input('algo-isomap-facial', 'n_clicks'),
        Input('algo-tsne-facial', 'n_clicks'),
        Input('algo-lle-facial', 'n_clicks'),
        Input('algo-mds-facial', 'n_clicks'),
        Input('apply-clustering-facial', 'n_clicks')
    ],
    [
        State('reduction-algo-store-facial', 'data'),
        State('clustering-algo-facial', 'value'),
        State('feature-weights-store', 'data'),
        State('filter-mode-store', 'data'),
        State('pca-svd_solver', 'value'),
        State('pca-whiten', 'value'),
        State('tsne-perplexity', 'value'),
        State('tsne-learning_rate', 'value'),
        State('tsne-n_iter', 'value'),
        State('tsne-metric', 'value'),
        State('isomap-n_neighbors', 'value'),
        State('isomap-metric', 'value'),
        State('isomap-path_method', 'value'),
        State('isomap-neighbors_algorithm', 'value'),
        State('lle-n_neighbors', 'value'),
        State('lle-method', 'value'),
        State('lle-eigen_solver', 'value'),
        State('mds-metric', 'value'),
        State('mds-max_iter', 'value'),
        State('kmeans-n_cluster', 'value'),
        State('kmeans-init', 'value'),
        State('kmeans-max_iter', 'value'),
        State('kmeans-algorithm', 'value'),
        State('dbscan-eps', 'value'),
        State('dbscan-min_samples', 'value'),
        State('dbscan-metric', 'value'),
        State('agglomerative-n_clusters', 'value'),
        State('agglomerative-linkage', 'value'),
        State('agglomerative-compute_distances', 'value')
    ],
    prevent_initial_call=True
    )
    def update_facial_feature_plot(
        pca_clicks, isomap_clicks, tsne_clicks, lle_clicks, mds_clicks,
        apply_clustering_clicks, last_reduction_algo,
        clustering_algo, feature_weights, filter_mode,
        PCA_svd_solver, PCA_whiten,
        tSNE_perplexity, tSNE_learning_rate, tSNE_n_iter, tSNE_metric,
        ISO_n_neighbors, ISO_metric, ISO_path_method, ISO_neighbors_algorithm,
        LLE_n_neighbors, LLE_method, LLE_eigen_solver,
        MDS_metric, MDS_max_iter,
        n_clusters_kmeans, init_kmeans, max_iter_kmeans, algorithm_kmeans,
        eps_dbscan, min_samples_dbscan, metric_dbscan,
        n_clusters_agglomerative, linkage_agglomerative, compute_distances_agglomerative
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            return {}, last_reduction_algo

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        reduction_algo = last_reduction_algo
        if triggered_id == 'algo-pca-facial':
            reduction_algo = 'PCA'
        elif triggered_id == 'algo-isomap-facial':
            reduction_algo = 'Isomap'
        elif triggered_id == 'algo-tsne-facial':
            reduction_algo = 't-SNE'
        elif triggered_id == 'algo-lle-facial':
            reduction_algo = 'LLE'
        elif triggered_id == 'algo-mds-facial':
            reduction_algo = 'MDS'

        selected_features = [feature for feature, weight in feature_weights.items() if weight > 0]
        if not selected_features:
            return {}, reduction_algo

        if filter_mode == 'AND':
            condition = (facial_features[selected_features] == 1).all(axis=1)
        else:
            condition = (facial_features[selected_features] == 1).any(axis=1)

        indices_true = np.where(condition)[0]
        filtered_facial_feats = facial_features[condition]
        filtered_urls = [image_urls[i] for i in indices_true]

        if triggered_id == 'apply-clustering-facial':
            if reduction_algo == 'PCA':
                pca = PCA(n_components=2, svd_solver=PCA_svd_solver, whiten=PCA_whiten)
                filtered_facial_feats = pca.fit_transform(filtered_facial_feats)
            elif reduction_algo == 'Isomap':
                iso = Isomap(n_components=2, n_neighbors=ISO_n_neighbors, metric=ISO_metric, path_method=ISO_path_method, neighbors_algorithm=ISO_neighbors_algorithm)
                filtered_facial_feats = iso.fit_transform(filtered_facial_feats)
            elif reduction_algo == 't-SNE':
                tsne = TSNE(n_components=2, perplexity=tSNE_perplexity, learning_rate=tSNE_learning_rate, n_iter=tSNE_n_iter, metric=tSNE_metric)
                filtered_facial_feats = tsne.fit_transform(filtered_facial_feats)
            elif reduction_algo == 'LLE':
                lle = LocallyLinearEmbedding(n_components=2, n_neighbors=LLE_n_neighbors, method=LLE_method, eigen_solver=LLE_eigen_solver)
                filtered_facial_feats = lle.fit_transform(filtered_facial_feats)
            elif reduction_algo == 'MDS':
                mds = MDS(n_components=2, metric=MDS_metric, max_iter=MDS_max_iter)
                filtered_facial_feats = mds.fit_transform(filtered_facial_feats)
            else:
                pca = PCA(n_components=2)
                filtered_facial_feats = pca.fit_transform(filtered_facial_feats)

            if clustering_algo == 'kmeans':
                return plot_clustered_KMeans_embeddings(filtered_facial_feats, filtered_urls, n_clusters_kmeans, init_kmeans, max_iter_kmeans, algorithm_kmeans), reduction_algo
            elif clustering_algo == 'dbscan':
                return plot_clustered_DBScan_embeddings(filtered_facial_feats, filtered_urls, eps_dbscan, min_samples_dbscan, metric_dbscan), reduction_algo
            elif clustering_algo == 'agglomerative':
                return plot_clustered_agglomerative_embeddings(filtered_facial_feats, filtered_urls, n_clusters_agglomerative, linkage_agglomerative, compute_distances_agglomerative), reduction_algo

        if reduction_algo == 'PCA':
            return plot_PCA_embeddings(filtered_facial_feats, filtered_urls, PCA_whiten, PCA_svd_solver), reduction_algo
        elif reduction_algo == 'Isomap':
            return plot_isomap_embeddings(filtered_facial_feats, filtered_urls, ISO_n_neighbors, ISO_metric, ISO_path_method, ISO_neighbors_algorithm), reduction_algo
        elif reduction_algo == 't-SNE':
            return plot_tsne_embeddings(filtered_facial_feats, filtered_urls, tSNE_perplexity, tSNE_learning_rate, tSNE_n_iter, tSNE_metric), reduction_algo
        elif reduction_algo == 'LLE':
            return plot_lle_embeddings(filtered_facial_feats, filtered_urls, LLE_n_neighbors, LLE_method, LLE_eigen_solver), reduction_algo
        elif reduction_algo == 'MDS':
            return plot_mds_embeddings(filtered_facial_feats, filtered_urls, MDS_metric, MDS_max_iter), reduction_algo


    @app.callback(
        Output('feature-weights-store', 'data'),
        [Input('enable-all', 'n_clicks'), Input('disable-all', 'n_clicks')],
        State('feature-weights-store', 'data')
    )
    def handle_enable_disable_all(enable_all_clicks, disable_all_clicks, current_weights):
        current_weights = current_weights or {feature: 0 for feature in facial_features}
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_weights

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered_id == 'enable-all':
            current_weights = {feature: (1 if weight == 0 else weight) for feature, weight in current_weights.items()}
        elif triggered_id == 'disable-all':
            current_weights = {feature: 0 for feature in current_weights}

        return current_weights

    @app.callback(
        Output('feature-weights-store', 'data', allow_duplicate=True),
        [Input(f'toggle-button-{feature}', 'n_clicks') for feature in facial_features],
        State('feature-weights-store', 'data'),
        prevent_initial_call=True
    )
    def handle_toggle_buttons(*args):
        current_weights = args[-1] or {feature: 0 for feature in facial_features}
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_weights

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered_id.startswith('toggle-button-'):
            feature = triggered_id.replace('toggle-button-', '')
            current_weights[feature] = 1 if current_weights[feature] == 0 else 0

        return current_weights

    @app.callback(
        [Output(f'weight-{feature}', 'value') for feature in facial_features],
        Input('feature-weights-store', 'data'),
    )
    def sync_inputs_with_store(feature_weights):
        feature_weights = feature_weights or {feature: 0 for feature in facial_features}
        return [feature_weights.get(feature, 0) for feature in facial_features]

    @app.callback(
        Output('feature-weights-store', 'data', allow_duplicate=True),
        [Input(f'weight-{feature}', 'value') for feature in facial_features],
        State('feature-weights-store', 'data'),
        prevent_initial_call=True
    )
    def handle_manual_weights(*args):
        current_weights = args[-1] or {feature: 0 for feature in facial_features}
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_weights

        for i, feature in enumerate(facial_features):
            if ctx.triggered[0]['prop_id'] == f'weight-{feature}.value':
                try:
                    weight_value = args[i]
                    if weight_value != '':
                        weight_value = float(weight_value) if isinstance(weight_value, (int, float)) else 0
                        
                        if weight_value >= 0:
                            current_weights[feature] = weight_value
                        else:
                            current_weights[feature] = 0
                    else:
                        current_weights[feature] = 0
                        
                except (ValueError, TypeError):
                    pass

        return current_weights

    @app.callback(
        Output('celeb-count-text', 'children'),
        [Input('feature-weights-store', 'data'),
        Input('filter-mode-store', 'data')]
    )
    def update_filtered_count(feature_weights, filter_mode):
        selected_features = [feature for feature, weight in feature_weights.items() if weight > 0]
        if not selected_features:
            return "Number of Images Present in the Filtered Dataset: 0"
        
        if filter_mode == 'AND':
            condition = (facial_features[selected_features] == 1).all(axis=1)
        else:
            condition = (facial_features[selected_features] == 1).any(axis=1)

        indices_true = np.where(condition)[0]
        filtered_urls = [image_urls[i] for i in indices_true]
        return f"Number of Images Present in the Filtered Dataset: {len(filtered_urls)}"

    @app.callback(
        Output('image-to-display', 'src'),
        [Input('feature-weights-store', 'data')]
    )
    def update_image(feature_weights):
        weighted_embeddings = np.zeros(embeddings.shape[1])
        for feature, weight in feature_weights.items():
            positive_mean, negative_mean = find_embedding_for_attribute(embeddings, facial_features, feature)
            weighted_embeddings += weight * (positive_mean - negative_mean)
        most_similar_index, _ = find_most_similar_embedding(weighted_embeddings, embeddings)
        return image_urls[most_similar_index]

    @app.callback(
        [Output('embedding-hover-data-store', 'data', allow_duplicate=True),
        Output('facial-features-hover-data-store', 'data', allow_duplicate=True)],
        [Input('url', 'pathname')],
        prevent_initial_call=True
    )
    def reset_hover_data_on_load(pathname):
        return {'hover_data': None, 'timestamp': None}, {'hover_data': None, 'timestamp': None}

    @app.callback(
        Output('facial-features-hover-data-store', 'data'),
        [Input('facial-feature-plot', 'hoverData')],
        prevent_initial_call=True
    )
    def update_hover_data(facial_feature_hover_data):
        current_time = datetime.utcnow().isoformat()
        if facial_feature_hover_data:
            return {
                'hover_data': facial_feature_hover_data,
                'timestamp': current_time
            }
        return {
            'hover_data': None,
            'timestamp': current_time
        }
    
    @app.callback(
        Output('embedding-hover-data-store', 'data'),
        [Input('embedding-plot', 'hoverData')],
        prevent_initial_call=True
    )
    def update_hover_data(embedding_hover_data):
        current_time = datetime.utcnow().isoformat()
        if embedding_hover_data:
            return {
                'hover_data': embedding_hover_data,
                'timestamp': current_time
            }
        return {
            'hover_data': None,
            'timestamp': current_time
        }

    @app.callback(
        Output('image-container', 'children'),
        [Input('embedding-hover-data-store', 'data'),
         Input('facial-features-hover-data-store', 'data')]
    )
    def display_image(embedding_hover_data_store, facial_hover_data_store):
        hover_data_embedding = embedding_hover_data_store['timestamp']
        hover_data_facial = facial_hover_data_store['timestamp']

        if hover_data_embedding is None and hover_data_facial is None:
            return html.Div()
        
        elif hover_data_embedding is None and hover_data_facial is not None:
            hover_data = facial_hover_data_store['hover_data']
            
        elif hover_data_facial is None and hover_data_embedding is not None:
            hover_data = embedding_hover_data_store['hover_data']
        
        else:
            if hover_data_embedding <= hover_data_facial:
                hover_data = facial_hover_data_store['hover_data']
            else:
                hover_data = embedding_hover_data_store['hover_data']

        if hover_data is None:
            return html.Div()
    
        point_index = hover_data['points'][0]['text']
        x_position = hover_data['points'][0]['x']
        y_position = hover_data['points'][0]['y']

        image_style = {
            'position': 'relative',
            'top': f'{y_position}px',
            'left': f'{x_position}px',
            'max-height': '150px',
            'max-width': '150px',
            'border': '1px solid #ddd',
            'border-radius': '10px',
            'z-index': '10',
        }

        return html.Img(src=point_index, style=image_style)

    @app.callback(
        [Output('filter-mode-store', 'data'),
        Output('filter-mode-switch', 'children'),
        Output('filter-mode-switch', 'color')],
        [Input('filter-mode-switch', 'n_clicks')],
        State('filter-mode-store', 'data')
    )
    def toggle_filter_mode(n_clicks, current_mode):
        if n_clicks is None:
            return current_mode, f"Filter Mode: {current_mode}", 'info'
        
        if current_mode == 'AND':
            return 'OR', "Filter Mode: OR", 'success'
        else:
            return 'AND', "Filter Mode: AND", 'danger'

    @app.callback(
        Output('clustering-algo', 'style'),
        [Input('apply-clustering', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_clustering_dropdown(n_clicks):
        if n_clicks:
            return {'display': 'block', 'width': '250px', 'margin-top': '10px'}
        return {'display': 'none'}
    
    @app.callback(
        Output("filter-mode-indicator", "style"),
        Input("info-button", "n_clicks"),
        prevent_initial_call=True
    )
    def toggle_info(n_clicks):
        if n_clicks is None:
            return {'display': 'none'} 
        else:
            current_style = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
            if current_style == "filter-mode-indicator":
                return {'display': 'block'} if n_clicks % 2 == 1 else {'display': 'none'}
            return {'display': 'none'}

    @app.callback(
        Output('loading-message', 'style'),
        Input('embedding-plot', 'figure')
    )
    def update_loading_message(figure):
        if not figure or figure == {}:
            return {'text-align': 'center', 'font-size': '30px', 'font-weight': 'bold', 'margin-top': '100px'}
        else:
            return {'display': 'none'}


    @app.callback(
        Output('loading-message-facial', 'style'),
        Input('facial-feature-plot', 'figure')
    )
    def update_loading_message(figure):
        if not figure or figure == {}:
            return {'text-align': 'center', 'font-size': '30px', 'font-weight': 'bold', 'margin-top': '100px'}
        else:
            return {'display': 'none'}
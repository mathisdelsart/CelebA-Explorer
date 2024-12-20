from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc

from utils.data_analysis_utils import plot_corr_matrix_facial_feats, plot_binary_distribution, \
                                    plot_embedding_distribution, plot_feature_cooccurrence, \
                                    plot_attribute_embedding_correlation

def create_data_analysis_page():
    from app import facial_features, image_urls, embeddings

    return html.Div([

        html.H1("Facial Features and Dataset Analysis",
                style={
                    'text-align': 'center', 
                    'margin-bottom': '10px', 
                    'font-size': '48px', 
                    'font-weight': 'bold', 
                    'color': '#4A90E2',  
                    'text-transform': 'uppercase',  
                    'letter-spacing': '3px',  
                    'text-shadow': '3px 3px 5px rgba(0, 0, 0, 0.2)',  
                    'font-family': 'Arial, sans-serif',  
                    'border-bottom': '4px solid #4A90E2',  
                    'padding-bottom': '10px'
                }),

        html.Div([
            html.H3("Understand the dataset and explore the relationships between facial features and embeddings."), 
            html.H3("Use the visualizations below to gain insights into the dataset and its attributes!")
        ], style={
            'padding': '20px', 'background-color': '#f9f9f9', 'border-radius': '10px', 
            'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)', 'margin-bottom': '40px',
            'text-align': 'center', 'color': '#7F8C8D', 'font-family': 'Arial, sans-serif',
        }),

        create_dataset_overview(facial_features, image_urls, embeddings),
        create_graph_selection_dropdown()
    ])

def create_dataset_overview(facial_features, image_urls, embeddings):
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.Strong("Dataset Overview"), className="text-center", style={'font-size': '24px', 'font-weight': 'normal', 'color': '#4A90E2'}),
                dbc.CardBody([
                    html.Div([
                        html.P(["📸 ", html.Strong("Total Images", style={'color': '#4A90E2'}), f": {len(image_urls)}"], style={'font-size': '18px', 'text-align': 'center'}),
                        html.P(["🎭 ", html.Strong("Number of Facial Features", style={'color': '#4A90E2'}), f": {len(facial_features.columns)}"], style={'font-size': '18px', 'text-align': 'center'}),
                        html.P(["🔢 ", html.Strong("Embedding Dimensions", style={'color': '#4A90E2'}), f": {len(embeddings.columns)}"], style={'font-size': '18px', 'text-align': 'center'}),
                    ])
                ])
            ], style={'margin-bottom': '40px', 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'})
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.Strong("Advanced Insights"), className="text-center", style={'font-size': '24px', 'font-weight': 'normal', 'color': '#4A90E2'}),
                dbc.CardBody([
                    html.Div([
                        html.P(["👥 ", html.Strong("Average Attributes per Image", style={'color': '#4A90E2'}), f": {facial_features.sum(axis=1).mean():.2f}"], style={'font-size': '18px', 'text-align': 'center'}),
                        html.P(["🔍 ", html.Strong("Count of All Facial Attributes", style={'color': '#4A90E2'}), f": {facial_features.sum().sum()}"], style={'font-size': '18px', 'text-align': 'center'}),
                        html.P(["🔥 ", html.Strong("Top 5 Most Frequent Features", style={'color': '#4A90E2'}), f": {', '.join((facial_features == 1).sum().sort_values(ascending=False).head(5).index)}"], style={'font-size': '18px', 'text-align': 'center'}),
                        html.P(["⚡ ", html.Strong("Top 5 Least Frequent Features", style={'color': '#4A90E2'}), f": {', '.join((facial_features == 1).sum().sort_values().head(5).index)}"], style={'font-size': '18px', 'text-align': 'center'})
                    ])
                ])
            ], style={'margin-bottom': '40px', 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'})
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.Strong("Feature Presence Insights"), className="text-center", style={'font-size': '24px', 'font-weight': 'normal', 'color': '#4A90E2'}),
                dbc.CardBody([
                    html.Div([
                        html.P(["🌐 ", html.Strong(id="features-present-text", style={'color': '#4A90E2'}), ": ", html.Span(id="features-present-count", style={'font-size': '18px', 'text-align': 'center'})], style={'font-size': '18px', 'text-align': 'center'}),
                        html.P(["🌐 ", html.Strong(id="features-absent-text", style={'color': '#4A90E2'}), ": ", html.Span(id="features-absent-count", style={'font-size': '18px', 'text-align': 'center'})], style={'font-size': '18px', 'text-align': 'center'}),
                        html.P(["📊 ", html.Strong(id="features-present-threshold-text", style={'color': '#4A90E2'}), ": ", html.Span(id="features-present-proportion", style={'font-size': '18px', 'text-align': 'center'})], style={'font-size': '18px', 'text-align': 'center'}),
                        html.P("Adjust the threshold to update the field above.", style={'font-size': '18px', 'text-align': 'center'}),
                        dcc.Slider(id='feature-threshold-slider', min=0, max=100, step=1, value=70, marks={i: f"{i}%" for i in range(0, 101, 10)}, tooltip={"placement": "bottom", "always_visible": True})
                    ])
                ])
            ], style={'margin-bottom': '40px', 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'})
        ], width=4),
    ])

def create_graph_selection_dropdown():
    from app import facial_features
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div(id='graph-title', children="Correlation Between Features", style={
                    'font-size': '28px', 'color': '#4A90E2', 'text-align': 'center', 
                    'font-weight': 'bold', 'border-bottom': '4px solid #4A90E2', 'padding-bottom': '8px',
                    'text-transform': 'uppercase', 'letter-spacing': '1px', 'margin-bottom': '10px'
                }),
            ], width=12),
        ]),

        dbc.Row([
            dbc.Col([
                html.Div(id='graph-description', children="", style={
                    'font-size': '20px', 'color': '#7F8C8D', 'text-align': 'left', 'margin-bottom': '10px', 
                    'line-height': '1.8', 'font-family': 'Arial, sans-serif', 'padding-left': '10px'
                }),
            ], width=8),
        ]),

        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='graph-selection-dropdown',
                    options=[
                        {'label': 'Correlation Between Features', 'value': 'corr_matrix'},
                        {'label': 'Binary Distribution of Features', 'value': 'binary_distribution'},
                        {'label': 'Embedding Values Distribution', 'value': 'embedding_distribution'},
                        {'label': 'Co-occurrence Features Distribution', 'value': 'co_occurrence'},
                        {'label': 'Correlation Between Features and Embeddings', 'value': 'corr_embed_feats'},
                    ],
                    value='corr_matrix',
                    style={'width': '80%', 'text-align': 'center', 'margin-bottom': '20px', 'padding': '10px'}
                ),
            ], width=4),
        ], style={'margin-bottom': '30px'}),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-graph",
                            type="circle",
                            children=dcc.Graph(
                                id="selected-graph", 
                                figure={}, 
                                style={"height": "100%", "width": "100%"}
                            )
                        )
                    ])
                ], style={
                    'box-shadow': '0px 10px 20px rgba(0, 0, 0, 0.1)',
                    'border-radius': '10px',
                    'padding': '20px',
                    'background-color': '#fff',
                    'margin-top': '10px',
                    'display': 'flex',
                    'justify-content': 'center',
                    'align-items': 'center'
                })
            ], width=8, style={
                'display': 'flex',
                'justify-content': 'center',
                'align-items': 'center',
                'padding': '20px',
            })
        ])
    ], style={'margin-top': '40px', 'margin-bottom': '40px'})


def data_analysis_register_callbacks(app):
    from app import facial_features, embeddings

    @app.callback(
        [
            Output('selected-graph', 'figure'),
            Output('graph-title', 'children'),
            Output('graph-description', 'children')
        ],
        [Input('graph-selection-dropdown', 'value')]
    )
    def update_graph(selected_graph):
        if selected_graph == 'corr_matrix':
            figure = plot_corr_matrix_facial_feats(facial_features)
            title = "Correlation Between Features"
            description = (
                "This heatmap visualizes the correlation between facial features. Darker colors indicate strong negative correlation, "
                "while lighter colors show strong positive correlation. This helps identify relationships between features, which is crucial "
                "for tasks like facial recognition. For example, we observe a strong negative correlation between wearing red lipstick and being male, "
                "and between having no beard and a goatee. On the other hand, there is a strong positive correlation between smiling and high cheekbones, "
                "as well as between wearing makeup and lipstick."
            )

        elif selected_graph == 'binary_distribution':
            figure = plot_binary_distribution(facial_features)
            title = "Binary Distribution of Features"
            description = (
                "This visualization shows the binary distribution of facial features, with -1 representing the absence "
                "and 1 representing the presence of a feature. The graph provides insights into the prevalence of each "
                "facial feature within the dataset, highlighting how often certain attributes are present or absent. "
                "It is important to note that the dataset used here is a subset of the CelebA dataset, containing only 30,000 images "
                "instead of the full 200,000+ images. As a result, the dataset is not well balanced, with some features appearing much more frequently than others. "
                "For example, only 2,540 images have narrow eyes, while 24,507 images lack a beard. This shows a significant disparity in feature distribution, "
                "with more negative values (absence) than positive values (presence). However, features like 'Young', 'Wearing Lipstick', and 'No_beard' "
                "have more than 50% presence, suggesting a higher proportion of women in this subset of the dataset."
            )

        elif selected_graph == 'embedding_distribution':
            figure = plot_embedding_distribution(embeddings)
            title = "Embedding Values Distribution"
            description = (
                "This graph displays the distribution of embedding values, which represent the high-dimensional feature vectors "
                "for each individual. Embeddings capture the underlying patterns in facial data, and this distribution allows us to "
                "observe the spread and density of these representations. It can help assess how well the facial features are encoded "
                "in the embedding space and whether certain regions of the embedding space might represent specific groups or classes. "
                "Additionally, the user can select a specific feature to view the distribution of embeddings containing that feature. "
                "This feature-specific view provides insights into how embeddings are structured for particular characteristics. "
                "Overall, the embedding distribution resembles a standard normal distribution (mean of 0 and variance of 1), "
                "indicating that the embeddings are normalized and centered."
            )

        elif selected_graph == 'co_occurrence':
            figure = plot_feature_cooccurrence(facial_features)
            title = "Co-occurrence Features Distribution"
            description = (
                "This visualization shows the co-occurrence of facial features, highlighting how often pairs of features appear together. "
                "By examining the co-occurrence matrix, we can determine which feature combinations are common and which are rare. "
                "This analysis is particularly useful for feature engineering and understanding how the presence of one feature may "
                "influence the presence of another. It can also provide insights into potential correlations between features that may "
                "not be immediately obvious. However, interpreting this graph can be tricky, as features that never appear together will "
                "have a high co-occurrence, even though they may not have a direct relationship. "
                "Nevertheless, when combined with insights from other visualizations, this graph helps draw meaningful conclusions. "
                "For instance, on this plot, 'Bald' has a strong co-occurrence with most features, except 'Young' and 'Heavy Makeup', "
                "which makes sense as bald people are often older and less likely to have heavy makeup, and few young people are bald."
            )

        elif selected_graph == 'corr_embed_feats':
            figure = plot_attribute_embedding_correlation(facial_features, embeddings)
            title = "Correlation Between Features and Embeddings"
            description = (
                "This graph shows the correlation between facial features and their corresponding embedding vectors. "
                "By correlating facial features with the low-dimensional embeddings, we can determine how well the embeddings "
                "capture the underlying relationships between features. This analysis is critical for understanding whether the "
                "embedding space accurately represents the facial attributes and can be used effectively in downstream tasks such as "
                "face recognition and classification. "
                "However, drawing concrete conclusions from this graph is challenging, as the data is largely centered around the value 0, "
                "indicating little direct correlation between the features and the embeddings. That said, we can observe that positive "
                "embedding values tend to correlate with features more commonly associated with women (e.g., Bald, Glasses, Heavy Makeup, Necklace, Young), "
                "while negative values are more linked to features typically associated with men (e.g., Male, Necktie, Receding Hairline). "
                "Features with values closer to 0 tend to represent more neutral facial characteristics, which don't strongly lean towards either gender."
            )

        else:
            figure = {}
            title = "No Graph Selected"
            description = "Please select a graph from the dropdown..."

        return figure, title, description

    @app.callback(
    [
        Output('features-present-count', 'children'),
        Output('features-absent-count', 'children'),
        Output('features-present-proportion', 'children'),
        Output('features-present-text', 'children'),
        Output('features-absent-text', 'children'),
        Output('features-present-threshold-text', 'children')
    ],
    [Input('feature-threshold-slider', 'value')]
    )
    def update_feature_counts(threshold):
        present_count = ((facial_features == 1).mean(axis=1) > threshold / 100).sum()
        absent_count = ((facial_features == -1).mean(axis=1) > threshold / 100).sum()
        present_proportion = ((facial_features == 1).mean(axis=0) > threshold / 100).sum() / len(facial_features.columns) * 100
        return (
            f"{present_count}",
            f"{absent_count}",
            f"{present_proportion:.2f}%",
            f"{threshold}%+ of Features Present in Images",
            f"{threshold}%+ of Features Absent in Images", 
            f"Proportion of Features Present Above {threshold}% Threshold"
        )
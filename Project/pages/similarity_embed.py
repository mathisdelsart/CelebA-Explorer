from utils.similarity_embed_utils import compute_similarity
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import dcc, html

def create_similarity_page():
    from app import image_urls
    return html.Div([
        html.H1(
            "Image Similarity Search Based on Embeddings", 
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
            }
        ),

        html.Div([
            html.H3(
                "Select an image, choose the similarity metric, and find the most similar image based on embeddings!"
            ),
        ], style={
            'padding': '20px', 
            'background-color': '#f9f9f9', 
            'border-radius': '10px', 
            'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
            'margin-bottom': '40px',
            'text-align': 'center',
            'font-size': '22px',
            'color': '#7F8C8D',
            'font-family': 'Arial, sans-serif',
        }),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Search for an Image", style={
                        'font-weight': 'bold', 
                        'text-align': 'center', 
                        'font-size': '36px',
                        'color': '#4A90E2'
                    }),
                    dbc.CardBody([
                        html.Label(f"Select image index (from 0 to {len(image_urls) - 1})", style={'font-size': '16px', 'font-weight': 'normal', 'color': '#555'}),
                        dcc.Slider(
                            id='image-slider',
                            min=0,
                            max=len(image_urls) - 1,
                            step=1,
                            value=0,
                            marks={i: str(i) for i in range(0, len(image_urls), 5000)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Br(),
                        html.Div([
                            html.Label("Enter the image index:", style={'font-size': '16px', 'font-weight': 'normal', 'color': '#555', 'margin-right': '10px'}),
                            dcc.Input(id='image-input', type='number', value=0, debounce=True),
                        ], style={'text-align': 'center', 'margin-top': '10px'}),

                        html.Hr(style={'margin-top': '20px', 'margin-bottom': '20px'}),

                        html.Label("Select similarity metric:", style={'font-size': '16px', 'font-weight': 'normal', 'color': '#555'}),
                        dcc.Dropdown(
                            id='similarity-method',
                            options=[
                                {'label': 'Cosine Similarity', 'value': 'cosine'},
                                {'label': 'Euclidean Distance', 'value': 'euclidean'}
                            ],
                            value='cosine',
                            style={'text-align': 'center', 'width': '100%', 'margin-bottom': '20px'}
                        ),
                        html.Br(),

                        html.Label(f"Select similarity index (from 0 [closest image] to {len(image_urls) - 1} [farthest image])", style={'font-size': '16px', 'font-weight': 'normal', 'color': '#555'}),
                        dcc.Slider(
                            id='image-similarity-slider',
                            min=0,
                            max=len(image_urls) - 1,
                            step=1,
                            value=0,
                            marks={i: str(i) for i in range(10)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Br(),
                        html.Div([
                            html.Label("Enter the similarity index:", style={'font-size': '16px', 'font-weight': 'normal', 'color': '#555', 'margin-right': '10px'}),
                            dcc.Input(id='image-similarity-input', type='number', value=0, debounce=True),
                        ], style={'text-align': 'center', 'margin-top': '10px'}),
                    ])
                ], style={'width': '100%', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'border-radius': '10px'}),
                html.Br(),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Selected Image Attributes", style={
                                'text-align': 'center', 
                                'font-size': '24px',
                                'border-bottom': '2px solid #7F8C8D',
                                'color': '#555',
                                'font-weight': 'normal'
                            }),
                            dcc.Loading(
                                id="loading-selected-image-features",
                                type="circle",
                                children=html.Div(id='selected-image-features', style={'text-align': 'center'}),
                                style={'margin-top': '20px', 'height': '50px', 'width': '50px', 'display': 'flex', 'justify-content': 'center'}  # Centered and larger
                            ),
                        ], width=6),
                        dbc.Col([
                            html.H5("Most Similar Image Attributes", style={
                                'text-align': 'center', 
                                'font-size': '24px',
                                'border-bottom': '2px solid #7F8C8D',
                                'color': '#555',
                                'font-weight': 'normal'
                            }),
                            dcc.Loading(
                                id="loading-similar-image-features",
                                type="circle",
                                children=html.Div(id='similar-image-features', style={'text-align': 'center'}),
                                style={'margin-top': '20px', 'height': '50px', 'width': '50px', 'display': 'flex', 'justify-content': 'center'}  # Centered and larger
                            ),
                        ], width=6),
                    ])
                ], style={'margin-top': '20px', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'border-radius': '10px', 'padding': '15px', 'min-height': '150px'}),  # Increased height
            ], width=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Images", style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '36px', 'color': '#4A90E2'}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H4("Selected Image", style={'text-align': 'center', 'font-size': '28px', 'border-bottom': '2px solid #7F8C8D', 'color': '#555'}),
                                dcc.Loading(
                                    id="loading-image-display",
                                    type="circle",
                                    children=html.Img(id='image-display', style={'width': '100%', 'height': 'auto', 'border-radius': '8px', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)'}),
                                    style={'margin-top': '20px', 'height': '50px', 'width': '50px', 'display': 'flex', 'justify-content': 'center'}  # Centered and larger
                                ),
                            ], width=6),
                            dbc.Col([
                                html.H4("Most Similar Image", style={'text-align': 'center', 'font-size': '28px', 'border-bottom': '2px solid #7F8C8D', 'color': '#555'}),
                                dcc.Loading(
                                    id="loading-image-pair-display",
                                    type="circle",
                                    children=html.Img(id='image-pair-display', style={'width': '100%', 'height': 'auto', 'border-radius': '8px', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)'}),
                                    style={'margin-top': '20px', 'height': '50px', 'width': '50px', 'display': 'flex', 'justify-content': 'center'}  # Centered and larger
                                ),
                            ], width=6),
                        ]),
                        html.Div(style={'margin-top': '20px'}),
                    ])
                ], style={'width': '100%', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'border-radius': '10px'})
            ], width=8)
        ])
    ])

def similarity_embed_register_callbacks(app):
    from app import image_urls, facial_features, embeddings
    @app.callback(
        Output('image-input', 'value'),
        Input('image-slider', 'value')
    )
    def sync_image_input_with_slider(slider_value):
        return slider_value

    @app.callback(
        Output('image-slider', 'value'),
        Input('image-input', 'value')
    )
    def sync_slider_with_image_input(input_value):
        try:
            input_value = int(input_value)
            return max(0, min(input_value, len(image_urls) - 1))
        except ValueError:
            return 0

    @app.callback(
        Output('image-similarity-input', 'value'),
        Input('image-similarity-slider', 'value')
    )
    def sync_similarity_input_with_slider(slider_similarity_value):
        return slider_similarity_value

    @app.callback(
        Output('image-similarity-slider', 'value'),
        Input('image-similarity-input', 'value')
    )
    def sync_slider_with_similarity_input(input_similarity_value):
        try:
            input_similarity_value = int(input_similarity_value)
            return max(0, min(input_similarity_value, len(image_urls) - 1))
        except ValueError:
            return 0

    @app.callback(
        [Output('image-display', 'src'),
        Output('image-pair-display', 'src'),
        Output('selected-image-features', 'children'),
        Output('similar-image-features', 'children')],
        [Input('image-slider', 'value'),
        Input('image-input', 'value'),
        Input('image-similarity-slider', 'value'),
        Input('image-similarity-input', 'value'),
        Input('similarity-method', 'value')]
    )
    def update_image_and_attributes(slider_value, input_value, slider_similarity_value, similarity_input_value, similarity_method):
        image_index = input_value if input_value is not None else slider_value
        image_index = max(0, min(image_index, len(image_urls) - 1))
        similarity_index = similarity_input_value if similarity_input_value is not None else slider_similarity_value
        similarity_index = max(0, min(similarity_index, len(image_urls) - 1))
        image_src = image_urls[image_index]
        selected_facial_features = facial_features.iloc[image_index][facial_features.iloc[image_index] == 1].index.tolist()
        selected_features_text = [html.P(feature) for feature in selected_facial_features]
        similars = compute_similarity(embeddings, image_index, image_urls, method=similarity_method)
        similar_image_src = similars[similarity_index][0]
        similar_idx = similars[similarity_index][1]
        similar_facial_features = facial_features.iloc[similar_idx][facial_features.iloc[similar_idx] == 1].index.tolist()
        similar_features_text = [html.P(feature) for feature in similar_facial_features]

        return image_src, similar_image_src, selected_features_text, similar_features_text
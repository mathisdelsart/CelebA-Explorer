from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
import random

def create_home_page():
    return html.Div([
        html.H1(
            "Visualization Tool For CelebA Dataset", 
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
                "Dive into the CelebA dataset, a rich collection of celebrity images featuring various facial attributes."  
            ),
            html.H4(
                "This dataset also includes advanced 512-dimensional embeddings, enabling in-depth analysis of facial features and recognition."
            ),
        ], style={
            'text-align': 'center',
            'font-size': '22px',
            'font-weight': 'normal',
            'color': '#7F8C8D',
            'font-family': 'Arial, sans-serif',
            'max-width': '80%',
            'margin': '0 auto',
            'line-height': '1.5',
            'transition': 'all 0.3s ease',
            'letter-spacing': '1px',
            'padding': '20px',
            'background-color': '#f9f9f9',
            'border-radius': '10px',
            'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
        }),
        html.Br(),
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H4(html.Strong("Embeddings"), className="card-title", style={'color': '#4A90E2'}),
                        html.P("Explore the 512-dimensional embeddings representing celebrity faces.", className="card-text"),
                        dbc.Button("Go to Embeddings", color="success", href="/embeddings", style={'border-radius': '5px'})
                    ])
                ), width=2, style={
                    'margin-bottom': '20px', 
                    'transition': 'all 0.3s', 
                    'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 
                    'border-radius': '10px', 
                    'background-color': '#ffffff', 
                    'padding': '20px'
                }
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H4(html.Strong("Data Analysis"), className="card-title", style={'color': '#4A90E2'}),
                        html.P("Visualize different statistics and insights from the CelebA dataset.", className="card-text"),
                        dbc.Button("Go to Data Analysis", color="success", href="/data", style={'border-radius': '5px'})
                    ])
                ), width=2, style={
                    'margin-bottom': '20px', 
                    'transition': 'all 0.3s', 
                    'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 
                    'border-radius': '10px', 
                    'background-color': '#ffffff', 
                    'padding': '20px'
                }
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H4(html.Strong("Similarity Search"), className="card-title", style={'color': '#4A90E2'}),
                        html.P("Search for similar images based on embeddings and facial features.", className="card-text"),
                        dbc.Button("Go to Similarity Search", color="success", href="/similarity", style={'border-radius': '5px'})
                    ])
                ), width=2, style={
                    'margin-bottom': '20px', 
                    'transition': 'all 0.3s', 
                    'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 
                    'border-radius': '10px', 
                    'background-color': '#ffffff', 
                    'padding': '20px'
                }
            ),
        ], justify="center", style={
            'margin': '0 auto',
            'gap': '80px'
        }),

        html.Div([
            html.H3(
                "Example of CelebA Images", 
                style={
                    'text-align': 'center', 
                    'margin-top': '40px', 
                    'color': '#4A90E2',
                    'font-size': '36px',
                    'font-weight': 'bold',
                    'text-transform': 'uppercase',
                    'letter-spacing': '3px',
                    'text-shadow': '4px 4px 6px rgba(0, 0, 0, 0.2)',
                    'font-family': 'Arial, sans-serif',
                    'border-bottom': '3px solid #4A90E2',
                    'padding-bottom': '10px',
                    'transition': 'all 0.3s ease'
                }
            ),

            dbc.Row([
                dbc.Col(
                    html.Img(id="image1", style={'width': '80%', 'border-radius': '10px', 'padding': '10px'}),
                    width=3
                ),
                dbc.Col(
                    html.Img(id="image2", style={'width': '80%', 'border-radius': '10px', 'padding': '10px'}),
                    width=3, 
                    style={'display': 'flex', 'justify-content': 'center'}
                ),
                dbc.Col(
                    html.Img(id="image3", style={'width': '80%', 'border-radius': '10px', 'padding': '10px'}),
                    width=3
                ),
            ], justify="center", style={
                'margin-top': '20px', 
                'gap': '20px',
                'display': 'flex',
                'align-items': 'center'
            }),
        ]),

        dcc.Interval(
            id='interval-component',
            interval=3000,
            n_intervals=0
        ),
    ])

def home_register_callbacks(app):
    from app import image_urls
    image_urls = list(image_urls)

    @app.callback(
        [Output('image1', 'src'),
         Output('image2', 'src'),
         Output('image3', 'src')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_images(n_intervals):
        random_images = random.sample(image_urls, 3)
        return random_images[0], random_images[1], random_images[2]
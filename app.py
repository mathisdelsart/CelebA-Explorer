### Import all the required libraries ###
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd

### Import all the utility functions ###
from utils.app_utils import split_dataset, get_navbar

### Import all the pages ###
from pages.home import create_home_page, home_register_callbacks
from pages.embeddings import create_embeddings_page, embeddings_register_callbacks
from pages.data_analysis import create_data_analysis_page, data_analysis_register_callbacks
from pages.similarity_embed import create_similarity_page, similarity_embed_register_callbacks

## Load the dataset ##
dataset = pd.read_csv('Datas/celeba_buffalo_l.csv')
image_urls, facial_features, embeddings = split_dataset(dataset)
image_urls = "/assets/" + image_urls

### App initialization ###
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

### Navbar initialization ###
navbar = get_navbar()

### App layout ###
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])

### Callbacks ###
@app.callback(
    Output('url', 'href'),
    Input('url', 'pathname')
)
def redirect_to_home(pathname):
    if pathname is None or pathname == '/':
        return '/home'
    return pathname

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/home' or pathname == '/':
        return create_home_page()
    elif pathname == '/embeddings':
        return create_embeddings_page()
    elif pathname == '/data':
        return create_data_analysis_page()
    elif pathname == '/similarity':
        return create_similarity_page()
    else:
        return create_home_page()

### Registering Callbacks ###
###   for all the pages   ###
home_register_callbacks(app)             # Home Page Callbacks
data_analysis_register_callbacks(app)    # Data Analysis Page Callbacks
similarity_embed_register_callbacks(app) # Similarity Page Callbacks
embeddings_register_callbacks(app)       # Embeddings Page Callbacks

#### Run the App ####
if __name__ == '__main__':
    app.run_server() # debug=True for development
from dash import html
import dash_bootstrap_components as dbc

def split_dataset(dataset):
    size_embedding = 512
    nb_facial_features = 39
    image_urls = dataset['image_name']
    facial_features = dataset.iloc[:, 1:nb_facial_features+1]
    embeddings = dataset[[f'embedding_{i}' for i in range(size_embedding)]]
    return image_urls, facial_features, embeddings

def get_navbar():
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink(
                [
                    html.I(className='fas fa-home'),
                    html.Img(
                        src='/assets/home_icon.png', 
                        height="40px", 
                        style={'margin-right': '8px', 'filter': 'brightness(0) invert(1)', 'vertical-align': 'middle'}
                    ),
                    ' Home'
                ],
                href="/",
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin-right': '30px'}
            )),
            dbc.NavItem(dbc.NavLink(
                [
                    html.I(className='fas fa-project-diagram'),
                    html.Img(
                        src='/assets/embeddings_icon.webp', 
                        height="60px", 
                        style={'margin-right': '8px', 'filter': 'brightness(0) invert(1)', 'vertical-align': 'middle'}
                    ),
                    ' Embeddings'
                ],
                href="/embeddings",
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin-right': '30px'}
            )),
            dbc.NavItem(dbc.NavLink(
                [
                    html.I(className='fas fa-chart-line'),
                    html.Img(
                        src='/assets/data_analysis_icon.png', 
                        height="50px", 
                        style={'margin-right': '8px', 'filter': 'brightness(0) invert(1)', 'vertical-align': 'middle'}
                    ),
                    ' Data Analysis'
                ],
                href="/data",
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin-right': '30px'}
            )),
            dbc.NavItem(dbc.NavLink(
                [
                    html.I(className='fas fa-search'),
                    html.Img(
                        src='/assets/similarity_icon.png', 
                        height="50px", 
                        style={'margin-right': '8px', 'vertical-align': 'middle'}
                    ),
                    ' Similarity Search'
                ],
                href="/similarity",
                style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
            ))
        ],
        brand="CelebA Dataset Visualization Tool",
        brand_href="/",
        color="primary",
        dark=True,
        style={
            'height': '70px', 
            'font-size': '24px',
            'display': 'flex', 
            'align-items': 'center', 
            'justify-content': 'center',
            'text-align': 'center'
        },
        className="mb-4"
    )
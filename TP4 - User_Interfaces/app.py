# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.express as px
import pandas as pd

# Initialize the Dash app
app = Dash(__name__)

# Load data
df = pd.read_csv('Data/abalone.csv')

# Layout
app.layout = html.Div([
    html.H1("Abalone Data Explorer"),
    
    # Row for Histogram and Scatter Plot
    html.Div([
        html.Div([
            dcc.Graph(id='length-histogram'),
            html.Label("Select Variable for Histogram:"),
            dcc.Dropdown(
                id='histogram-variable-dropdown',
                options=[
                    {'label': 'Length', 'value': 'Length'},
                    {'label': 'Rings', 'value': 'Rings'},
                ],
                value='Length'  # Default selection
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20px'}),
        
        html.Div([
            dcc.Graph(id='scatter-plot'),
            html.Label("Select X-axis:"),
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=[
                    {'label': 'Length', 'value': 'Length'},
                    {'label': 'Diameter', 'value': 'Diameter'},
                    {'label': 'Height', 'value': 'Height'},
                ],
                value='Length'  # Default selection
            ),
            html.Label("Select Y-axis:"),
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=[
                    {'label': 'Whole weight', 'value': 'Whole weight'},
                    {'label': 'Shucked weight', 'value': 'Shucked weight'},
                    {'label': 'Viscera weight', 'value': 'Viscera weight'},
                ],
                value='Whole weight'  # Default selection
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20px'}),
    ]),

    # Row for Box Plot
    html.Div([
        dcc.Graph(id='box-plot'),
        html.Label("Select Variable for Box Plot:"),
        dcc.Dropdown(
            id='box-plot-variable-dropdown',
            options=[
                {'label': 'Length', 'value': 'Length'},
                {'label': 'Diameter', 'value': 'Diameter'},
                {'label': 'Height', 'value': 'Height'},
            ],
            value='Length'  # Default selection
        ),
    ], style={'padding': '20px'}),

    # Row for Data Table
    html.Div([
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.columns],
            style_table={'overflowX': 'auto'},
            page_size=10
        )
    ]),

    # Filter Options
    html.Div([
        html.Label("Filter by Rings:"),
        dcc.RangeSlider(
            id='rings-slider',
            min=df['Rings'].min(),
            max=df['Rings'].max(),
            value=[df['Rings'].min(), df['Rings'].max()],
            marks={i: str(i) for i in range(df['Rings'].min(), df['Rings'].max()+1)},
            step=1
        ),
    ], style={'padding': '20px'})
])

# Callback to update histogram
@app.callback(
    Output('length-histogram', 'figure'),
    Input('histogram-variable-dropdown', 'value')
)
def update_histogram(selected_variable):
    fig = px.histogram(df, x=selected_variable, title=f"Distribution of {selected_variable}")
    return fig

# Callback to update scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis-dropdown', 'value'), Input('y-axis-dropdown', 'value')]
)
def update_scatter_plot(x_column, y_column):
    fig = px.scatter(df, x=x_column, y=y_column, title=f"Scatter Plot of {x_column} vs {y_column}")
    return fig

# Callback to update box plot
@app.callback(
    Output('box-plot', 'figure'),
    Input('box-plot-variable-dropdown', 'value')
)
def update_box_plot(selected_variable):
    fig = px.box(df, x='Sex', y=selected_variable, title=f"Box Plot of {selected_variable} by Sex")
    return fig

# Callback to update data table
@app.callback(
    Output('table', 'data'),
    Input('rings-slider', 'value')
)
def update_table(selected_rings):
    filtered_df = df[(df['Rings'] >= selected_rings[0]) & (df['Rings'] <= selected_rings[1])]
    return filtered_df.to_dict('records')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

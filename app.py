"""
Main Dash application for the Sports Analytics Dashboard.
This application provides a web interface for:
- Data upload and preprocessing
- Data visualization
- Regression modeling
- Interactive AI assistance
"""

import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd
from dash.exceptions import PreventUpdate

# Import our custom modules
from data_processor import DataProcessor
from sports_regressor import SportsRegressor
from sports_chatbot import SportsChatbot

# Initialize components
data_processor = DataProcessor()
regressor = SportsRegressor()
chatbot = SportsChatbot()

# Connect chatbot to data and model
chatbot.connect_to_data(data_processor)
chatbot.connect_to_model(regressor)

# Initialize the Dash app with dark theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Custom styles for dropdown components */
            .Select-control {
                background-color: #2d2d2d !important;
                border-color: #2d2d2d !important;
            }
            .Select-menu-outer {
                background-color: #2d2d2d !important;
                border-color: #2d2d2d !important;
            }
            .Select-option {
                background-color: #2d2d2d !important;
                color: #ffffff !important;
            }
            .Select-option:hover {
                background-color: #3d3d3d !important;
            }
            .Select-value-label {
                color: #ffffff !important;
            }
            .Select-placeholder {
                color: #ffffff !important;
            }
            .Select--single > .Select-control .Select-value {
                color: #ffffff !important;
            }
            .Select--multi .Select-value {
                background-color: #3d3d3d !important;
                border-color: #4d4d4d !important;
                color: #ffffff !important;
            }
            .Select--multi .Select-value-icon {
                border-color: #4d4d4d !important;
            }
            .Select--multi .Select-value-icon:hover {
                background-color: #4d4d4d !important;
                color: #ffffff !important;
            }
            .Select-arrow {
                border-color: #ffffff transparent transparent !important;
            }
            .Select.is-open > .Select-control .Select-arrow {
                border-color: transparent transparent #ffffff !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define custom styles for consistent theming
custom_styles = {
    'background': '#121212',  # Dark background
    'text': '#ffffff',        # White text
    'card': '#1e1e1e',        # Slightly lighter card background
    'border': '#2d2d2d',      # Border color
    'input': '#2d2d2d',       # Input background
    'hover': '#2d2d2d',       # Hover color
}

# Define the layout with primary sections
app.layout = dbc.Container([
    # Header section with title
    dbc.Row([
        dbc.Col([
            html.H1("Smart Sports Analytics Dashboard", className="text-center mb-4 text-white"),
            html.Hr(style={'borderColor': custom_styles['border']})
        ], width=12)
    ]),
    
    # Data Upload Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Upload Sports Data", className="text-white"),
                dbc.CardBody([
                    # File upload component
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select CSV File', style={'color': '#ffffff'})
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px',
                            'backgroundColor': custom_styles['input'],
                            'color': custom_styles['text'],
                            'borderColor': custom_styles['border']
                        },
                        multiple=False
                    ),
                    html.Div(id='output-data-upload'),
                    # Column exclusion section
                    html.Div([
                        html.H5("Select Columns to Exclude from One-Hot Encoding", className="mt-3 text-white"),
                        dcc.Dropdown(
                            id='exclude-columns-dropdown',
                            multi=True,
                            placeholder="Select columns to exclude...",
                            style={
                                'backgroundColor': custom_styles['input'],
                                'color': custom_styles['text']
                            }
                        ),
                        dbc.Button(
                            "Update Excluded Columns",
                            id='update-excluded-button',
                            color="primary",
                            className="mt-2"
                        ),
                        html.Div(id='exclude-columns-status', className="mt-2 text-white")
                    ], id='exclude-columns-section', style={'display': 'none'})
                ])
            ], className="mb-4", style={'backgroundColor': custom_styles['card']})
        ], width=12)
    ]),
    
    # Analysis Tabs (Visualization and ML)
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                # Data Visualization Tab
                dbc.Tab(label="Data Visualization", tab_id="visualization-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Label("Chart Type", className="text-white"),
                            dcc.Dropdown(
                                id='chart-type',
                                options=[
                                    {'label': 'Bar Chart', 'value': 'bar'},
                                    {'label': 'Line Chart', 'value': 'line'},
                                    {'label': 'Scatter Plot', 'value': 'scatter'}
                                ],
                                value='bar',
                                style={
                                    'backgroundColor': custom_styles['input'],
                                    'color': custom_styles['text']
                                }
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("X-Axis", className="text-white"),
                            dcc.Dropdown(
                                id='x-axis',
                                style={
                                    'backgroundColor': custom_styles['input'],
                                    'color': custom_styles['text']
                                }
                            )
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Y-Axis", className="text-white"),
                            dcc.Dropdown(
                                id='y-axis',
                                style={
                                    'backgroundColor': custom_styles['input'],
                                    'color': custom_styles['text']
                                }
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Color Scheme", className="text-white"),
                            dcc.Dropdown(
                                id='color-by',
                                options=[
                                    {'label': 'No Color', 'value': 'None'},
                                    {'label': 'Viridis', 'value': 'Viridis'},
                                    {'label': 'Plasma', 'value': 'Plasma'},
                                    {'label': 'Inferno', 'value': 'Inferno'},
                                    {'label': 'Magma', 'value': 'Magma'},
                                    {'label': 'Cividis', 'value': 'Cividis'},
                                    {'label': 'Rainbow', 'value': 'Rainbow'},
                                    {'label': 'Jet', 'value': 'Jet'},
                                    {'label': 'Hot', 'value': 'Hot'},
                                    {'label': 'Cool', 'value': 'Cool'}
                                ],
                                value='None',
                                style={
                                    'backgroundColor': custom_styles['input'],
                                    'color': custom_styles['text']
                                }
                            )
                        ], width=6)
                    ]),
                    dbc.Button("Update Chart", color="primary", id="update-chart-button", className="mt-2"),
                    dcc.Graph(id='visualization-graph', style={'backgroundColor': custom_styles['card']})
                ]),
                # Regression Modeling Tab
                dbc.Tab(label="Regression Modeling", tab_id="modeling-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Label("Target Variable", className="text-white"),
                            dcc.Dropdown(
                                id='target-variable',
                                style={
                                    'backgroundColor': custom_styles['input'],
                                    'color': custom_styles['text']
                                }
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Model Type", className="text-white"),
                            dcc.Dropdown(
                                id='model-type',
                                options=[
                                    {'label': 'Linear Regression', 'value': 'linear'},
                                    {'label': 'Random Forest', 'value': 'random_forest'}
                                ],
                                value='linear',
                                style={
                                    'backgroundColor': custom_styles['input'],
                                    'color': custom_styles['text']
                                }
                            )
                        ], width=6)
                    ]),
                    html.Label("Feature Variables", className="text-white"),
                    dcc.Dropdown(
                        id='feature-variables',
                        multi=True,
                        style={
                            'backgroundColor': custom_styles['input'],
                            'color': custom_styles['text']
                        }
                    ),
                    dbc.Button("Train Model", color="primary", id="train-model-button", className="mt-2"),
                    html.Div(id='model-metrics', className="mt-3 text-white"),
                    html.Hr(style={'borderColor': custom_styles['border']}),
                    html.H5("Make Predictions", className="text-white"),
                    html.Div(id='prediction-inputs'),
                    dbc.Button("Predict", color="primary", id="predict-button", className="mt-2"),
                    html.Div(id='prediction-output', className="mt-3 text-white")
                ])
            ], id="tabs", active_tab="visualization-tab", style={'backgroundColor': custom_styles['card']})
        ], width=12)
    ]),
    
    # Chatbot Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Marbo - Sports Analytics Assistant", className="mb-0 text-white"),
                    html.Small("Your AI-powered sports data analyst", className="text-light")
                ], className="bg-dark text-white"),
                dbc.CardBody([
                    # Chat history area
                    html.Div(
                        id="chatbot-output",
                        className="chat-history p-3 border rounded",
                        style={
                            'maxHeight': '400px',
                            'overflowY': 'auto',
                            'marginBottom': '15px',
                            'backgroundColor': custom_styles['input'],
                            'color': custom_styles['text'],
                            'borderColor': custom_styles['border']
                        }
                    ),
                    
                    # Chat input area
                    dbc.InputGroup([
                        dbc.Input(
                            id="chatbot-input",
                            type="text",
                            placeholder="Ask Marbo about your sports data...",
                            debounce=True,
                            style={
                                'backgroundColor': custom_styles['input'],
                                'color': custom_styles['text'],
                                'borderColor': custom_styles['border']
                            }
                        ),
                        dbc.Button(
                            "Send",
                            id="chatbot-button",
                            color="primary",
                            n_clicks=0,
                            className="ml-2"
                        )
                    ])
                ]),
                dbc.CardFooter([
                    html.Div([
                        html.Small("Example questions:", className="text-muted"),
                        dbc.Badge("Show me data summary", color="secondary", className="ml-2"),
                        dbc.Badge("What's my model accuracy?", color="secondary", className="ml-2"),
                        dbc.Badge("Show feature importance", color="secondary", className="ml-2"),
                        dbc.Badge("Help with visualization", color="secondary", className="ml-2")
                    ], className="d-flex flex-wrap gap-2")
                ])
            ], className="mt-4 mb-4 shadow", style={'backgroundColor': custom_styles['card']})
        ], width=12)
    ])
], fluid=True, style={'backgroundColor': custom_styles['background'], 'minHeight': '100vh', 'padding': '20px'})

# Callback for data upload
@app.callback(
    [
        Output('output-data-upload', 'children'),
        Output('x-axis', 'options'),
        Output('y-axis', 'options'),
        Output('target-variable', 'options'),
        Output('feature-variables', 'options')
    ],
    [Input('upload-data', 'contents'),
     Input('update-excluded-button', 'n_clicks')],
    [State('upload-data', 'filename'),
     State('exclude-columns-dropdown', 'value')],
    prevent_initial_call=True
)
def update_components(contents, n_clicks, filename, excluded_columns):
    """
    Update components after data upload or column exclusion update.
    
    Args:
        contents: Uploaded file contents
        n_clicks: Number of times update button was clicked
        filename: Name of uploaded file
        excluded_columns: List of columns to exclude from encoding
        
    Returns:
        tuple: Updated components (upload output, x-axis options, y-axis options, target options, feature options)
    """
    if not contents:
        raise PreventUpdate
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if 'csv' in filename:
            # Load and process data
            raw_df = data_processor.load_data(io.StringIO(decoded.decode('utf-8')))
            processed_cols = data_processor.get_processed_columns()
            
            # Get column information
            column_info = data_processor.get_column_info()
            
            # Create options with additional information
            raw_options = [
                {
                    'label': f"{col} ({info['dtype']})",
                    'value': col
                } 
                for col, info in column_info.items()
            ]
            
            # Filter out excluded columns from processed columns
            if excluded_columns:
                processed_cols = [col for col in processed_cols 
                                if not any(excluded.lower() in col.lower() 
                                        for excluded in excluded_columns)]
            
            proc_options = [
                {
                    'label': col,
                    'value': col
                } 
                for col in processed_cols
            ]
            
            # Create data summary
            summary = [
                html.H5(f'Uploaded: {filename}'),
                html.P(f"Rows: {len(raw_df)}, Columns: {len(raw_df.columns)}"),
                html.H6("Data Preview:"),
                dbc.Table.from_dataframe(raw_df.head(3))
            ]
            
            return [
                html.Div(summary),
                raw_options,  # x-axis
                raw_options,  # y-axis
                raw_options,  # target-variable
                proc_options   # feature-variables
            ]
            
    except Exception as e:
        return [
            html.Div([
                html.H5("Error uploading file"),
                html.P(str(e), className="text-danger")
            ]), 
            [], [], [], []
        ]

# Callback for chatbot interaction
@app.callback(
    Output('chatbot-output', 'children'),
    [Input('chatbot-button', 'n_clicks'),
     Input('chatbot-input', 'n_submit')],
    [State('chatbot-input', 'value'),
     State('chatbot-output', 'children')]
)
def update_chatbot_response(n_clicks, n_submit, query, existing_output):
    """
    Update chatbot response based on user input.
    
    Args:
        n_clicks: Number of times send button was clicked
        n_submit: Number of times enter was pressed
        query: User's input query
        existing_output: Current chat history
        
    Returns:
        list: Updated chat history
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    trigger = ctx.triggered[0]['prop_id']
    
    if (trigger == 'chatbot-button.n_clicks' or trigger == 'chatbot-input.n_submit') and query:
        try:
            response = chatbot.get_response(query)
            
            # Create a new conversation entry
            new_entry = html.Div([
                html.P(f"You: {query}", className="user-msg"),
                html.P(f"Marbo: {response}", className="bot-msg")
            ])
            
            # If there's existing output, append to it, otherwise create new
            if existing_output:
                if isinstance(existing_output, list):
                    return existing_output + [new_entry]
                else:
                    return [existing_output, new_entry]
            else:
                return [new_entry]
        except Exception as e:
            return html.Div([
                html.P(f"You: {query}", className="user-msg"),
                html.P(f"Marbo: Error - {str(e)}", className="bot-msg text-danger")
            ])
    
    raise PreventUpdate

# Callback to clear chatbot input after sending
@app.callback(
    Output('chatbot-input', 'value'),
    [Input('chatbot-button', 'n_clicks'),
     Input('chatbot-input', 'n_submit')]
)
def clear_chatbot_input(n_clicks, n_submit):
    """
    Clear chatbot input field after sending a message.
    
    Args:
        n_clicks: Number of times send button was clicked
        n_submit: Number of times enter was pressed
        
    Returns:
        str: Empty string to clear input
    """
    ctx = dash.callback_context
    if ctx.triggered and (ctx.triggered[0]['prop_id'] == 'chatbot-button.n_clicks' or 
                         ctx.triggered[0]['prop_id'] == 'chatbot-input.n_submit'):
        return ''
    raise PreventUpdate

# Callback for visualization
@app.callback(
    Output('visualization-graph', 'figure'),
    [Input('update-chart-button', 'n_clicks')],
    [State('chart-type', 'value'),
     State('x-axis', 'value'),
     State('y-axis', 'value'),
     State('color-by', 'value')]
)
def update_graph(n_clicks, chart_type, x_col, y_col, color_scheme):
    """
    Update visualization graph based on selected options.
    
    Args:
        n_clicks: Number of times update button was clicked
        chart_type: Type of chart to display
        x_col: Column for x-axis
        y_col: Column for y-axis
        color_scheme: Color scheme for visualization
        
    Returns:
        dict: Plotly figure object
    """
    if n_clicks is None or x_col is None or y_col is None:
        raise PreventUpdate
    
    if data_processor.raw_data is None:
        raise PreventUpdate
    
    try:
        df = data_processor.raw_data

        # Create base trace
        trace = {
            'x': df[x_col],
            'y': df[y_col],
            'type': chart_type,
            'mode': 'markers' if chart_type == 'scatter' else 'lines' if chart_type == 'line' else None,
            'name': f'{y_col} vs {x_col}'
        }
        
        # Apply color scheme if selected
        if color_scheme and color_scheme != 'None':
            if chart_type == 'scatter':
                trace['marker'] = {
                    'color': df[y_col],
                    'colorscale': color_scheme,
                    'showscale': True,
                    'colorbar': {
                        'title': y_col
                    }
                }
            elif chart_type == 'line':
                trace['line'] = {
                    'color': df[y_col],
                    'colorscale': color_scheme,
                    'showscale': True,
                    'colorbar': {
                        'title': y_col
                    }
                }
            elif chart_type == 'bar':
                trace['marker'] = {
                    'color': df[y_col],
                    'colorscale': color_scheme,
                    'showscale': True,
                    'colorbar': {
                        'title': y_col
                    }
                }
        
        fig = {
            'data': [trace],
            'layout': {
                'title': f'{chart_type.capitalize()} of {y_col} vs {x_col}',
                'xaxis': {'title': x_col},
                'yaxis': {'title': y_col},
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white'
            }
        }
        
        return fig
    except Exception as e:
        return {
            'data': [],
            'layout': {
                'title': f'Error: {str(e)}',
                'xaxis': {'visible': False},
                'yaxis': {'visible': False}
            }
        }

# Callback for model training
@app.callback(
    Output('model-metrics', 'children'),
    [Input('train-model-button', 'n_clicks')],
    [State('target-variable', 'value'),
     State('feature-variables', 'value'),
     State('model-type', 'value')]
)
def train_model(n_clicks, target, features, model_type):
    """
    Train the regression model and display metrics.
    
    Args:
        n_clicks: Number of times train button was clicked
        target: Target variable for prediction
        features: List of feature variables
        model_type: Type of model to train
        
    Returns:
        html.Div: Model metrics display
    """
    if n_clicks is None or not target or not features:
        raise PreventUpdate
    
    try:
        # Prepare data
        processed_data = data_processor.processed_data
        
        # Ensure target is in processed data
        if target not in data_processor.raw_data.columns:
            return html.Div([
                html.H5("Error training model"),
                html.P(f"Target column '{target}' not found in data")
            ])
            
        # Get target from raw data and add it to processed data for training
        processed_data[target] = data_processor.raw_data[target]
        
        # Train the model
        regressor.prepare_data(processed_data, target)
        regressor.train(model_type)
        
        # Evaluate model
        metrics = regressor.evaluate()
        
        # Update chatbot with metrics
        metrics_interpretation = chatbot.interpret_metrics(metrics)
        
        return html.Div([
            html.H5("Linear Regression Model Metrics"),
            html.P(f"R² Score: {metrics['R²']:.4f}"),
            html.P(f"RMSE: {metrics['RMSE']:.4f}"),
            html.P(f"MAE: {metrics['MAE']:.4f}"),
            html.P(metrics_interpretation)
        ])
    except Exception as e:
        return html.Div([
            html.H5("Error training model"),
            html.P(str(e), className="text-danger")
        ])

# Callback for prediction inputs
@app.callback(
    Output('prediction-inputs', 'children'),
    [Input('train-model-button', 'n_clicks')],
    [State('feature-variables', 'value')]
)
def create_prediction_inputs(n_clicks, features):
    """
    Create input fields for making predictions.
    
    Args:
        n_clicks: Number of times train button was clicked
        features: List of feature variables
        
    Returns:
        list: Input field components
    """
    if n_clicks is None or not features:
        raise PreventUpdate
    
    try:
        inputs = []
        for feature in features:
            inputs.append(
                dbc.Row([
                    dbc.Col([
                        dbc.Label(feature),
                        dbc.Input(id={'type': 'pred-input', 'index': feature}, type='number')
                    ], width=6)
                ], className="mb-2")
            )
        
        return inputs
    except Exception as e:
        return html.Div([
            html.H5("Error creating prediction inputs"),
            html.P(str(e), className="text-danger")
        ])

# Callback for making predictions
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [
        State('feature-variables', 'value'),
        State({'type': 'pred-input', 'index': ALL}, 'value')
    ]
)
def make_prediction(n_clicks, features, pred_values):
    """
    Make predictions using the trained model.
    
    Args:
        n_clicks: Number of times predict button was clicked
        features: List of feature variables
        pred_values: List of input values for prediction
        
    Returns:
        html.Div: Prediction result display
    """
    if n_clicks is None or not features or None in pred_values:
        raise PreventUpdate
    
    try:
        # Create input dictionary
        input_data = {feature: [value] for feature, value in zip(features, pred_values)}
        df = pd.DataFrame(input_data)
        
        # Make prediction
        prediction = regressor.predict(df)
        
        return html.Div([
            html.H5("Prediction Result"),
            html.P(f"Predicted value: {prediction[0]:.4f}")
        ])
    except Exception as e:
        return html.Div([
            html.H5("Error making prediction"),
            html.P(str(e), className="text-danger")
        ])

# Add new callback for updating excluded columns
@app.callback(
    [Output('exclude-columns-section', 'style'),
     Output('exclude-columns-dropdown', 'options'),
     Output('exclude-columns-dropdown', 'value')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_exclude_columns_dropdown(contents, filename):
    """
    Update the excluded columns dropdown after data upload.
    
    Args:
        contents: Uploaded file contents
        filename: Name of uploaded file
        
    Returns:
        tuple: Updated dropdown style, options, and value
    """
    if not contents:
        return {'display': 'none'}, [], []
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if 'csv' in filename:
            # Get column names from the uploaded file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            column_options = [{'label': col, 'value': col} for col in df.columns]
            
            return {'display': 'block'}, column_options, []
            
    except Exception as e:
        return {'display': 'none'}, [], []
    
    return {'display': 'none'}, [], []

# Add callback for updating excluded columns
@app.callback(
    Output('exclude-columns-status', 'children'),
    [Input('update-excluded-button', 'n_clicks')],
    [State('exclude-columns-dropdown', 'value')]
)
def update_excluded_columns(n_clicks, excluded_columns):
    """
    Update the list of excluded columns.
    
    Args:
        n_clicks: Number of times update button was clicked
        excluded_columns: List of columns to exclude
        
    Returns:
        html.Div: Status message
    """
    if n_clicks is None or not excluded_columns:
        raise PreventUpdate
    
    try:
        # Update the data processor with new excluded columns
        data_processor.update_excluded_columns(excluded_columns)
        
        return html.Div([
            html.P(f"Successfully excluded columns: {', '.join(excluded_columns)}", 
                  className="text-success")
        ])
    except Exception as e:
        return html.Div([
            html.P(f"Error updating excluded columns: {str(e)}", 
                  className="text-danger")
        ])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
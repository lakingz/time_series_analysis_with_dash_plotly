from dash import Dash, dcc, html, callback, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_ag_grid as dag
import pandas as pd
import base64
import datetime
import io
import xgboost as xgb


dataset = pd.read_csv('PJME_hourly.csv')
data_tag = dataset.columns
dataset["Datetime"] = pd.to_datetime(dataset["Datetime"])
data_test = pd.DataFrame(pd.date_range(dataset['Datetime'].iloc[-1], dataset['Datetime'].iloc[-1] + (dataset['Datetime'].iloc[-1] - dataset['Datetime'].iloc[0]), freq='H'), columns = ['Datetime'])

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
#
# app framework
#
app.layout = dbc.Container([
        html.H1('Consumption prediction with visualization', style={'textAlign': 'center'}),
        dbc.Row([
            dbc.Col([
                html.Div("Select Test Size %:"),
                dcc.Input(value=90, type='number', debounce=True, id='test-size', min=1, max=100, step=1)
            ], width=3),
            dbc.Col([
                html.Div("Select RandomForest n_estimators:"),
                dcc.Input(value=512, type='number', debounce=True, id='nestimator-size', min=10, max=1000, step=1)
            ], width=3),
            dbc.Col([
                html.Div("Accuracy Score:"),
                html.Div(id='training accuracy', style={'color': 'blue'}, children="")
            ], width=3)
        ], className='mb-3'),

        dag.AgGrid(
            id="grid",
            rowData=dataset.to_dict("records"),
            columnDefs=[{"field": i} for i in dataset.columns],
            columnSize="sizeToFit",
            style={"height": "400px"},
            dashGridOptions={"pagination": True, "paginationPageSize":5},
        ),

        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=px.line(dataset, x=dataset['Datetime'], y="PJME_MW"))
            ], width=6),
            dbc.Col([
                dcc.Graph(id='graph-with-prediction')
            ], width=6)
        ]),
        html.H1('Try out your own data set', style={'textAlign': 'center'}),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
        html.Div(id='output-div'),
        html.Div(id='output-datatable'),
    ])
#
# demo analysis
#
@callback(Output('graph-with-prediction', 'figure'),
          Output('training accuracy', 'children'),
    Input('test-size', 'value'),
    Input('nestimator-size', 'value'))
def update_testing(testsize, n_estimators):
    # Train and Test
    X_train, y_train = create_features(dataset, label=data_tag[-1])
    X_test = create_features(data_test)

    test_size = round(testsize/100 * dataset.shape[0])
    reg = xgb.XGBRegressor(n_estimators=n_estimators)
    reg.fit(X_train[0:test_size], y_train[0:test_size], verbose=True)  # Change verbose to True if you want to see it train
    train_accuracy = round(reg.score(X_train, y_train) * 100, 2)
    y_test = reg.predict(X_test)
    dataset['train/predict'] = 'train'
    data_test["PJME_MW"] = y_test
    data_test['train/predict'] = 'predict'
    data_all = pd.concat([dataset, data_test], sort=False)
    fig = px.line(data_all, x='Datetime', y="PJME_MW", color='train/predict')
    return fig, train_accuracy
def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['hour'] = df['Datetime'].dt.hour
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['quarter'] = df['Datetime'].dt.quarter
    df['month'] = df['Datetime'].dt.month
    df['year'] = df['Datetime'].dt.year
    df['dayofyear'] = df['Datetime'].dt.dayofyear
    df['dayofmonth'] = df['Datetime'].dt.day
    df['weekofyear'] = df['Datetime'].dt.isocalendar().week
    df.set_index('Datetime', drop=False, inplace=True)
    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X
#
# data uploead & display
#
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        html.P("Select data time column"),
        dcc.Dropdown(id='xaxis-data',
                     options=[{'label': x, 'value': x} for x in df.columns]),
        html.P("Select column to prediction"),
        dcc.Dropdown(id='yaxis-data',
                     options=[{'label': x, 'value': x} for x in df.columns]),
        html.Button(id="submit-button", children="start prediction"),
        html.Hr(),

        dbc.Col([
            html.Div("Select Test Size %:"),
            dcc.Input(value=90, type='number', debounce=True, id='test-size-import-data', min=1, max=100, step=1)
        ], width=3),
        dbc.Col([
            html.Div("Select RandomForest n_estimators:"),
            dcc.Input(value=512, type='number', debounce=True, id='nestimator-size-import-data', min=10, max=1000, step=1)
        ], width=3),
        dbc.Col([
            html.Div("Accuracy Score:"),
            html.Div(id='training accuracy-import-data', style={'color': 'blue'}, children="")
        ], width=3),

        html.Div(id='graph-with-prediction-import-data'),  # display prediction graph and accuracy score

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=15
        ),
        dcc.Store(id='stored-data', data=df.to_dict('records')),
        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

@callback(Output('output-datatable', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children
#
# prediction & display
#
@app.callback(Output('graph-with-prediction-import-data', 'children'),
              Output('training accuracy-import-data', 'children'),
              Input('submit-button', 'n_clicks'),
              State('test-size-import-data', 'value'),
              State('nestimator-size-import-data', 'value'),
              State('stored-data', 'data'),
              State('xaxis-data', 'value'),
              State('yaxis-data', 'value'))
def make_graphs(n, testsize_import_data, n_estimator_import_data, import_data_raw, x_data, y_data):
    if n is None:
        return no_update
    import_data = pd.DataFrame(import_data_raw)
    import_data = import_data[[x_data, y_data]].copy()
    import_data_tag = ['Datetime', y_data]
    import_data.columns = import_data_tag
    import_data["Datetime"] = pd.to_datetime(import_data["Datetime"])
    import_data_test = pd.DataFrame(pd.date_range(import_data['Datetime'].iloc[-1], import_data['Datetime'].iloc[-1] + (
            import_data['Datetime'].iloc[-1] - import_data['Datetime'].iloc[0]), freq='H'), columns=['Datetime'])

    # Train and Test
    X_train, y_train = create_features(import_data, label=import_data_tag[-1])
    X_test = create_features(import_data_test)

    test_size = round(testsize_import_data / 100 * dataset.shape[0])
    reg = xgb.XGBRegressor(n_estimators=n_estimator_import_data)
    reg.fit(X_train[0:test_size], y_train[0:test_size],verbose=True)  # Change verbose to True if you want to see it train
    train_accuracy_import_data = round(reg.score(X_train, y_train) * 100, 2)
    y_test = reg.predict(X_test)
    import_data['train/predict'] = 'train'
    import_data_test[y_data] = y_test
    import_data_test['train/predict'] = 'predict'
    import_data_all = pd.concat([import_data, import_data_test], sort=False)
    children = [dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=px.line(import_data_all, x='Datetime', y=y_data, color='train/predict'))
                    ], width=6)
               ])]
    return children, train_accuracy_import_data

#
# run app
#
if __name__=='__main__':
    app.run_server(debug=True, host='0.0.0.0', port=9000)


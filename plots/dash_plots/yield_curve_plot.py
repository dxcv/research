import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output
import plotly.graph_objs as go

from classes.YieldCurve import YieldCurve

# idea create yield curve plot for monthly frequency data where I can plot yc for month end via a slider
df = YieldCurve(ctry='US').load_yield_curve_data().resample('M').last()

# create slider dictionary
slider_dict = dict(zip(range(df.shape[0]), df.index.strftime('%Y/%m/%d')))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
    dcc.Slider(
        id='month-slider',
        min=0,
        max=df.shape[0]-1,
        value=0,
        marks=slider_dict
    ),

])


@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('month-slider', 'value')])
def update_figure(selected_month):
    filtered_yc = df.iloc[selected_month]
    traces = list()
    traces.append(go.Scatter(
        x=filtered_yc.index,
        y=filtered_yc))
    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'maturities'},
            yaxis={'title': 'Yield', 'autorange': False},
            title={'text': 'Yield Curve'}
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)

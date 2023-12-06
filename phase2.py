import math
import time

import dash
from dash import html
from dash import dcc
from dash.dependencies import Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objs as go

external_Style_sheets = ["https://codepen.io/chriddvp/pen/bWLwgP.css"]
df = pd.read_csv("data/downsample.csv")
app = dash.Dash("My app", external_stylesheets=external_Style_sheets)
server = app.server
app.layout = html.Div(
    [
        html.Center(html.H1("Term Project")),
        dcc.Tabs(
            id="select-PAGE",
            children=[
                dcc.Tab(label="Univariate Analysis", value="u"),
                dcc.Tab(label="Multivariate Analysis", value="m"),
            ],
            value="u",
        ),
        html.Div(id="question_ans"),
    ]
)

univariate_layout = html.Div([
    html.Br(),
    dcc.Dropdown(
        id="univariate",
        options=[
            {"label": "Line Plot", "value": "lineplot"},
            {"label": "Histogram", "value": "histplot"},
            {"label": "barplot", "value": "barplot"},
            {"label":"countplot","value":"countplot"}
        ],
        value="lineplot"
    ),
    html.Div(id="univout")]
)

q1_layout = html.Div([
    html.Center(html.H2("Line plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"}
                 ],
                 multi=False,
                 value=" "
                 ),
    html.Br(),
    html.P(html.H5("Select y")),
    dcc.Dropdown(id="y",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"}
                 ],
                 multi=False,
                 value=" "
                 ),
    html.Br(),
    html.Br(),
    html.P(html.H5("Select Marker Style")),
    dcc.RadioItems(id="marker",
                   options=[
                       {"label": "dot marker", "value": "."},
                       {"label": "circle marker", "value": "o"},
                       {"label": "star marker", "value": "*"}
                   ],
                   value="."
                   ),
    html.Br(),
    html.Br(),
    dcc.RadioItems(id="type",
                   options=[
                       {"label": "unique", "value": "unqiue"},
                       {"label": "aggregate", "value": "aggregate"},
                   ],
                   value="aggregate"
                   ),
    html.Br(),
    html.Br(),
    dcc.RangeSlider(id="observations",
                    min=50000,
                    max=200000,
                    step=50000
                    ),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div(dcc.Graph(id="lineplot")),
])


@app.callback(
    Output(component_id="lineplot", component_property="figure"),
    [Input(component_id="x", component_property="value"),
     Input(component_id="y", component_property="value"),
     Input(component_id="marker", component_property="value"),
     Input(component_id="type", component_property="value"),
     Input(component_id="observations", component_property="value")
     ]
)
def plot_line_dy(x, y, marker, agg, observations):
    if x is None and y is None:
        return
    l1 = list(df[x].unique())
    l2 = list(df[y].unique())
    new_df = pd.DataFrame.from_dict({x: sorted(l1[:min(len(l1), len(l2))]), y: sorted(l2[:min(len(l2), len(l1))])})
    if agg == "aggregate":
        fig = px.line(data_frame=new_df,
                      x=x,
                      y=y,
                      markers=marker,
                      title=f"Line plot for {x} and {y}")
    else:
        fig = px.line(data_frame=df.iloc[:observations],
                      x=x,
                      y=y,
                      markers=marker,
                      title=f"Line plot for {x} and {y}")
    return fig


q2_layout = html.Div([
    html.Center(html.H2("Histogram plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"}
                 ],
                 multi=False,
                 value=" "
                 ),
    html.Br(),
    html.Br(),
    html.P(html.H5("Select hue")),
    dcc.RadioItems(id="color",
                   options=[
                       {"label": "yes", "value": "yes"},
                       {"label": "no", "value": "no"},
                   ],
                   value="no"
                   ),
    html.Br(),
    html.Br(),
    html.P(html.H5("Select Binwidth")),
    dcc.Slider(id="binwidth",
               min=0,
               max=50,
               step=5
               ),
    html.Br(),
    html.Br(),
    html.P(html.H5("Select Observations for plotting")),
    dcc.RangeSlider(id="observations",
                    min=0,
                    max=200000,
                    step=50000
                    ),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div(dcc.Graph(id="histplot")),
])


@app.callback(
    Output(component_id="histplot", component_property="figure"),
    [
        Input(component_id="x", component_property="value"),
        Input(component_id="color", component_property="value"),
        Input(component_id="binwidth", component_property="value"),
        Input(component_id="observations", component_property="value")
    ]
)
def plot_histplot(x, color, binwidth, observations):
    if observations is None:
        temp_df = df
    else:
        temp_df = df.iloc[int(observations[0]):int(observations[1])]
    if color == "yes":
        col = "reordered"
    else:
        col = None
    if binwidth is not None and int(binwidth) != 0:
        nbins = math.ceil((temp_df[x].max() - temp_df[x].min()) / int(binwidth))
        fig = px.histogram(data_frame=temp_df,
                           x=x, nbins=nbins,
                           color=col,
                           opacity=0.7,
                           color_discrete_map={'borders': 'rgba(1, 1, 1, 1)'},
                           histnorm='probability density')
    else:
        fig = px.histogram(data_frame=temp_df, x=x, color=col, opacity=0.7,
                           color_discrete_map={'borders': 'rgba(1, 1, 1, 1)'}, histnorm='probability density')
    return fig


q3_layout = html.Div([
    html.Center(html.H2("Bar plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "product_name", "value": "product_name"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"}
                 ],
                 multi=False,
                 value="product_name"
                 ),
    html.Br(),
    html.Br(),
    html.P(html.H5("Select y")),
    dcc.Dropdown(id="y",
                 options=[
                     {"label": "reordered", "value": "reordered"},
                 ],
                 multi=False,
                 value="reordered"
                 ),
    html.Br(),
    html.Br(),
    html.P(html.H5("Select type")),
    dcc.RadioItems(id="plottype",
                   options=[
                       {"label": "stack", "value": "stack"},
                       {"label": "group", "value": "group"},
                       {"label": "relative", "value": "relative"},
                   ],
                   value="relative"
                   ),
    html.Br(),
    html.Br(),
    html.P(html.H5("Select opacity")),
    dcc.Slider(id="barspace",
               min=0.1,
               max=1,
               step=0.1
               ),
    html.Br(),
    html.Br(),
    html.P(html.H5("Select Observations for plotting")),
    dcc.RangeSlider(id="observations",
                    min=0,
                    max=200000,
                    step=50000
                    ),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div(dcc.Graph(id="barplot")),
])


@app.callback(
    Output(component_id="barplot", component_property="figure"),
    [
        Input(component_id="x", component_property="value"),
        Input(component_id="y", component_property="value"),
        Input(component_id="plottype", component_property="value"),
        Input(component_id="barspace", component_property="value"),
        Input(component_id="observations", component_property="value")
    ]
)
def plot_barplot(x, y, plottype, barspace, observations):
    time.sleep(2)
    if observations is None:
        temp_df = df
    else:
        temp_df = df.iloc[int(observations[0]):int(observations[1])]
    temp_1 = temp_df[temp_df[y] == 0][x].value_counts().reset_index().sort_values(
        by=x, ascending=False).iloc[:20].set_index(["index"]).to_dict()
    temp_2 = temp_df[temp_df[y] == 1][x].value_counts().reset_index().sort_values(
        by=x, ascending=False).iloc[:20].set_index(["index"]).to_dict()
    temp_1 = temp_1[x]
    temp_2 = temp_2[x]
    temp_final = pd.DataFrame.from_dict(
        {x: temp_1.keys(), "ordered": temp_1.values(), "reordered": temp_2.values()})
    temp_final = temp_final.set_index([x])
    if barspace:
        barspace = int(barspace)
    else:
        barspace = None
    if plottype is not None:
        plottype = plottype
    else:
        plottype = 'relative'

    fig = px.bar(data_frame=temp_final, barmode=plottype, opacity=barspace)
    return fig


q4_layout = html.Div([
    html.Center(html.H2("Count plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"},
                     {"label": "aisle", "value": "aisle"},
                     {"label": "department", "value": "department"},
                     {"label": "reordered", "value": "reordered"},
                     {"label": "product_name", "value": "product_name"},
                 ],
                 multi=False,
                 value="product_name"
                 ),
    html.Br(),
    html.Br(),
    html.P(html.H5("Select hue")),
    dcc.RadioItems(id="color",
                   options=[
                       {"label": "yes", "value": "yes"},
                       {"label": "no", "value": "no"},
                   ],
                   value="no"
                   ),
    html.Br(),
    html.Br(),
    html.P(html.H5("Select Observations for plotting")),
    dcc.RangeSlider(id="observations",
                    min=0,
                    max=200000,
                    step=50000
                    ),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div(dcc.Graph(id="countplot")),
])


@app.callback(
    Output(component_id="countplot", component_property="figure"),
    [
        Input(component_id="x", component_property="value"),
        Input(component_id="color", component_property="value"),
        Input(component_id="observations", component_property="value")
    ]
)
def plot_countplot(x, color, observations):
    if observations is None:
        temp_df = df
    else:
        temp_df = df.iloc[int(observations[0]):int(observations[1])]
    if color == "yes":
        col = "reordered"
    else:
        col = None

    category_counts = temp_df[x].value_counts()

    # Select the top 20 categories
    top_categories = category_counts.head(20).index

    # Filter the data for the top 20 categories
    filtered_data = temp_df[temp_df[x].isin(top_categories)]

    fig = px.histogram(data_frame=filtered_data, x=x, color=col, opacity=0.7,nbins=20,
                       color_discrete_map={'borders': 'rgba(1, 1, 1, 1)'})
    return fig


@app.callback(
    Output(component_id="question_ans", component_property="children"),
    [Input(component_id="select-PAGE", component_property="value")],
)
def select_function(value):
    if value == "u":
        return univariate_layout
    else:
        return


@app.callback(
    Output(component_id="univout", component_property="children"),
    [Input(component_id="univariate", component_property="value")],
)
def update_layout(value):
    if value == "lineplot":
        return q1_layout
    elif value == "histplot":
        return q2_layout
    elif value == "barplot":
        return q3_layout
    elif value == "countplot":
        return q4_layout
    else:
        return


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8020)

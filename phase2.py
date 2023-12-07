import math
import time
import copy
import scipy.stats as stats
import dash
from dash import html
from dash import dcc
from dash.dependencies import Output, Input
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

from plotly.subplots import make_subplots
title_font_size = 20
title_font_family = "serif"
title_font_color = "blue"
label_font_size = 15
label_font_color = "darkred"
label_font_family="serif"


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
            {"label":"countplot","value":"countplot"},
            {"label":"Pie Chart","value":"Pie Chart"},
            {"label":"Dist plot","value":"Dist plot"},
            {"label":"QQ plot","value":"QQ plot"},
            {"label":"KDE plot","value":"KDE plot"},
            {"label":"Violin plot","value":"Violin plot"},
            {"label":"Box plot","value":"Box plot"},
        ],
        value="lineplot"
    ),
    html.Div(id="univout")]
)

multivariate_layout = html.Div([
    html.Br(),
    dcc.Dropdown(
        id="multivariate",
        options=[
            {"label":"Pair plot","value":"Pair plot"},
            {"label":"Reg plot","value":"Reg plot"},
            {"label":"Joint plot","value":"Joint plot"},
            {"label":"3D plot","value":"3D plot"},
            {"label":"Clustermap","value":"Clustermap"},
            {"label":"Heatmap","value":"Heatmap"},
        ],
        value="Pair plot"
    ),
    html.Div(id="multiout")]
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
                 value="order_dow"
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
                 value="order_dow"
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
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(dcc.Graph(id="lineplot")),
    )
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
    if x is None or y is None:
        return
    if observations is None:
        temp_df = df
    else:
        temp_df = df.iloc[int(observations[0]):int(observations[1])]
    l1 = list(temp_df[x].unique())
    l2 = list(temp_df[y].unique())
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
    fig.update_layout(
        title=f"Line plot for {x} and {y}",
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=0.95,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        height=1000,
    )
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
                 value="order_dow"
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
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(dcc.Graph(id="histplot")),
    )
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
    fig.update_layout(
        title=f"Hist plot for {x}",
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=1,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        height=1000,
    )
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
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(dcc.Graph(id="barplot")),
    )
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
    fig.update_layout(
        title=f"Bar plot for {x}",
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=1,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        height=1000,
    )
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
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(dcc.Graph(id="countplot")),
    )
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
    fig.update_layout(
        title=f"Count plot for {x}",
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=0.95,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        height=1000,
    )
    return fig

q5_layout = html.Div([
    html.Center(html.H2("Pie Chart")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "aisle", "value": "aisle"},
                     {"label": "department", "value": "department"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"}
                 ],
                 multi=False,
                 value="aisle"
                 ),
    html.Br(),
    html.Br(),
    html.P(html.H5("Select hue")),
    dcc.RadioItems(id="color",
                   options=[
                       {"label": "yes", "value": "yes"},
                       {"label": "no", "value": "no"},
                   ],
                   value="yes"
                   ),
    html.Br(),
    html.Br(),
    html.P(html.H5("Select n_components")),
    dcc.Slider(id="components",
               min=10,
               max=50,
               step=5,
               value=10
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
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(dcc.Graph(id="pieplot")),
    )
    ,
])


@app.callback(
    Output(component_id="pieplot", component_property="figure"),
    [
        Input(component_id="x", component_property="value"),
        Input(component_id="color", component_property="value"),
        Input(component_id="components", component_property="value"),
        Input(component_id="observations", component_property="value")
    ]
)
def plot_piechart(x, color, components, observations):
    if observations is None:
        temp_df = df.sample(50000)
    else:
        temp_df = df.iloc[int(observations[0]):int(observations[1])]
    if color == "yes":
        temp_1 = temp_df[temp_df["reordered"] == 1][x].value_counts().reset_index()
        temp_1.columns = [x, "count"]
        temp_1 = temp_1.sort_values(by="count", ascending=False)
        temp_1 = temp_1.set_index(x)
        temp_k = temp_1.iloc[:int(components)].__deepcopy__()
        temp_k["count"] = temp_k["count"] / temp_k["count"].sum()
        temp_k = list(temp_k["count"])
        # products sold in aisle reordered vs not reordered
        # fig,ax = plt.subplots(1,2)
        temp_2 = temp_df[temp_df["reordered"] == 0][x].value_counts().reset_index()
        temp_2.columns = [x, "count"]
        temp_2 = temp_2.sort_values(by="count", ascending=False)
        temp_2 = temp_2.set_index(x)
        temp_k2 = temp_2.iloc[:int(components)].__deepcopy__()
        temp_k2["count"] = temp_k2["count"] / temp_k2["count"].sum()
        temp_k2 = list(temp_k2["count"])
        temp_1 = temp_1.sort_values(by="count", ascending=False).iloc[:int(components)]
        temp_2 = temp_2.sort_values(by="count",ascending=False).iloc[:int(components)]
        explode_1 = []
        explode_2 = []
        for i in range(len(temp_k)):
            if temp_k[i] < 0.02:
                explode_1.append(0.1)
            else:
                explode_1.append(0)
        for i in range(len(temp_k2)):
            if temp_k2[i] < 0.02:
                explode_2.append(0.1)
            else:
                explode_2.append(0)
        fig = make_subplots(1,2,specs=[[{'type':'domain'}, {'type':'domain'}]])
        fig.add_trace( go.Pie(labels=temp_1.index,values=temp_1["count"],pull=explode_1,name="reordered"),row=1,col=1)
        fig.add_trace(go.Pie(labels=temp_2.index,values=temp_2["count"],pull=explode_2,name="not reordered"),row=1,col=2)
        fig.update_layout(
        title=f"Pie chart for {x}",
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=0.95,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        height=1000,
        )

    else:
        temp_1 = temp_df[x].value_counts().reset_index()
        temp_1.columns = [x, "count"]
        temp_1 = temp_1.sort_values(by="count", ascending=False)
        temp_1 = temp_1.set_index(x)
        temp_k = temp_1.iloc[:int(components)].__deepcopy__()
        temp_k["count"] = temp_k["count"] / temp_k["count"].sum()
        temp_k = list(temp_k["count"])
        temp_1 = temp_1.sort_values(by="count", ascending=False).iloc[:int(components)]
        explode_1 = []
        for i in range(len(temp_k)):
            if temp_k[i] < 0.02:
                explode_1.append(0.1)
            else:
                explode_1.append(0)
        fig = go.Figure(data = [go.Pie(labels=temp_1.index,values=temp_1["count"],pull=explode_1)])
        fig.update_layout(
        title=f"Pie chart for {x}",
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=0.95,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        height=1000,
    )
        
    
    
    return fig

q6_layout = html.Div([
    html.Center(html.H2("Dist plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"},
                     {"label":"order_id","value":"order_id"},
                     {"label":"aisle","value":"aisle"},
                     {"label":"product_name","value":"product_name"}
                 ],
                 multi=False,
                 value="order_dow"
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
    html.P(html.H5("Select Binsize")),
    dcc.Slider(id="binwidth",
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
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(dcc.Graph(id="distplot")),
    )
    ,
])


@app.callback(
    Output(component_id="distplot", component_property="figure"),
    [
        Input(component_id="x", component_property="value"),
        Input(component_id="color", component_property="value"),
        Input(component_id="binwidth", component_property="value"),
        Input(component_id="observations", component_property="value")
    ]
)
def plot_distplot(x, color, binwidth, observations):
    if observations is None:
        temp_df = df.sample(50000)
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
    density = stats.gaussian_kde(temp_df[x])
    xs = np.linspace(temp_df[x].min(),temp_df[x].max(),len(temp_df))
    ys = density(xs)
    fig.add_trace(px.line(x=xs,y=ys,labels=f"distribution_{x}").data[0])
    fig.update_layout(
        title = f"Distribution Plot for {x}",
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=0.95,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        height=1000,
        )
    return fig

q7_layout = html.Div([
    html.Center(html.H2("Pair plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Checklist(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"}
                 ],
                 value=["order_dow","order_hour_of_day"]
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
    
    html.P(html.H5("Select Observations for plotting")),
    dcc.RangeSlider(id="observations",
                    min=0,
                    max=200000,
                    step=50000
                    ),
    html.Br(),
    html.Br(),
    html.Br(),
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(dcc.Graph(id="pairplot")),
    )
])


@app.callback(
    Output(component_id="pairplot", component_property="figure"),
    [
        Input(component_id="x", component_property="value"),
        Input(component_id="color", component_property="value"),
        Input(component_id="observations", component_property="value")
    ]
)
def plot_pairplot(x, color, observations):
    time.sleep(5)
    if observations is None:
        temp_df = df[x].sample(50000)
    else:
        temp_df = df[x].iloc[int(observations[0]):int(observations[1])]
    if color == "yes":
        col = "reordered"
    else:
        col = None
    fig = make_subplots(rows=len(x), cols=len(x))
    color_dict = {1:"red",2:"blue"}
    # Populate the subplots
    for i, var1 in enumerate(temp_df.columns):
        for j, var2 in enumerate(temp_df.columns):
            if col is not None:
                color = dict(color=[color_dict[cat] for cat in temp_df["reordered"]])
            else:
                color=None
            if i == j:  # Diagonal - histogram
                fig.add_trace(go.Histogram(x=temp_df[var1], nbinsx=20,name=f"histogram_{var1}"), row=i+1, col=j+1)
            else:       # Off-diagonal - scatter plot
                fig.add_trace(go.Scatter(x=temp_df[var1], y=temp_df[var2],marker=color, mode='markers',name=f"scatter_{var1}_{var2}"), row=i+1, col=j+1)

    # Customize and show the plot
    fig.update_layout(
        title = f"Pairplot for {x}",
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=0.95,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        height=1000,
        )
    return fig

q8_layout = html.Div([
    html.Center(html.H2("QQ plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"},
                 ],
                 multi=False,
                 value="order_dow"
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
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(dcc.Graph(id="qqplot")),
    )
])


@app.callback(
    Output(component_id="qqplot", component_property="figure"),
    [
        Input(component_id="x", component_property="value"),
        Input(component_id="observations", component_property="value")
    ]
)
def plot_qq(x, observations):
    time.sleep(3)
    if observations is None:
        temp_df = df.sample(50000)
    else:
        temp_df = df.iloc[int(observations[0]):int(observations[1])]

    data_sorted = np.sort(temp_df[x])
    norm_quantiles = stats.norm.ppf((np.arange(len(temp_df)) + 0.5) / len(temp_df))

    # Create a QQ plot
    fig = go.Figure(go.Scatter(x=norm_quantiles, y=data_sorted, mode='markers',name=f"{x}"))

    # Add a reference line
    fig.add_trace(go.Scatter(x=[min(norm_quantiles), max(norm_quantiles)], 
                            y=[min(norm_quantiles), max(norm_quantiles)],
                            mode='lines',
                            name='Reference Line'))

    # Customize and show the plot
    fig.update_layout(
        title = f"QQ plot for {x}",
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles',
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=0.95,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        height=1000,
        )
    return fig

q9_layout = html.Div([
    html.Center(html.H2("KDE plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"},
                 ],
                 multi=False,
                 value="order_dow"
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
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(dcc.Graph(id="kdeplot")),
    )
])


@app.callback(
    Output(component_id="kdeplot", component_property="figure"),
    [
        Input(component_id="x", component_property="value"),
        Input(component_id="observations", component_property="value")
    ]
)
def plot_kde(x, observations):
    time.sleep(3)
    if observations is None:
        temp_df = df
    else:
        temp_df = df.iloc[int(observations[0]):int(observations[1])]

    data_sorted = np.sort(temp_df[x])
    kde_val = stats.gaussian_kde(temp_df[x])
    col_name = copy.deepcopy(x)
    x = np.linspace(min(temp_df[x]),max(temp_df[x]),1000)
    y = kde_val(x)

    # Create a QQ plot
    fig = go.Figure(go.Scatter(x=x, y=y,fill = "tozeroy"))

    # Add a reference line

    # Customize and show the plot
    fig.update_layout(
        title = f"KDE Plot for {col_name}",
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=0.95,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        height=1000,
        )
    return fig

q10_layout = html.Div([
    html.Center(html.H2("Violin plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"}
                 ],
                 multi=True,
                 value="order_dow"
                 ),
    html.Br(),
    html.Br(),
    html.P(html.H5("Select opacity")),
    dcc.Slider(id="opacity",
               min=0.1,
               max=1,
               step=0.1,
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
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(dcc.Graph(id="violinplot")),
    )
    ,
])


@app.callback(
    Output(component_id="violinplot", component_property="figure"),
    [
        Input(component_id="x", component_property="value"),
        Input(component_id="opacity", component_property="value"),
        Input(component_id="observations", component_property="value")
    ]
)
def plot_violinplot(x, binwidth, observations):
    if observations is None:
        temp_df = df
    else:
        temp_df = df.iloc[int(observations[0]):int(observations[1])]
    if isinstance(x,str):
        fig_ = go.Figure(data=go.Violin(y=temp_df[x], box_visible=True, meanline_visible=True,opacity=binwidth))
    else:
        fig_ = make_subplots(rows=1, cols=len(x))
        for idx in x:
            if binwidth is not None:
                fig = go.Figure(data=go.Violin(y=temp_df[idx], box_visible=True, meanline_visible=True,opacity=binwidth,name=idx))
            else:
                fig = go.Figure(data=go.Violin(y=temp_df[idx], box_visible=True, meanline_visible=True,name=idx))
            fig_.add_trace(fig.data[0],row=1,col=x.index(idx)+1)
    fig_.update_layout(
        title = f"Violin Plot for {x}",
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=0.95,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        height=1000,
        )
    return fig_

q11_layout = html.Div([
    html.Center(html.H2("Reg plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"}
                 ],
                 multi=False,
                 value="days_since_prior_order"
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
                 value="days_since_prior_order"
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
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(html.Div(dcc.Graph(id="regrplot"))),
    )
    
])


@app.callback(
    Output(component_id="regrplot", component_property="figure"),
    [Input(component_id="x", component_property="value"),
     Input(component_id="y", component_property="value"),
     Input(component_id="observations", component_property="value")
     ]
)
def plot_regr_line(x, y,observations):
    if x is None and y is None:
        return
    if observations is None:
        temp_df = df
    else:
        temp_df = df.iloc[int(observations[0]):int(observations[1])]
    fig = px.scatter(temp_df, x=x, y=y,trendline="ols")
    fig.update_layout(
        title = f"Regression Line Plot for {x} and {y}",
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=0.95,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        height=1000,
        )
    return fig

q12_layout = html.Div([
    html.Center(html.H2("Joint plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"}
                 ],
                 multi=False,
                 value="days_since_prior_order"
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
                 value="days_since_prior_order"
                 ),
    html.Br(),
    html.Br(),
    dcc.RadioItems(id="type_",
                    options=[
                        {"label": "scatter", "value": "scatter"},
                        {"label": "kde", "value": "kde"}
                    ],
                    value="scatter"),
    html.Br(),
    html.Br(),
    html.B(html.H2("Enter number of observations")),
    dcc.Input(id="observations", type="number", placeholder=f"{len(df)}"),
    html.Br(),
    html.Br(),
    html.Br(),
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(dcc.Graph(id="jointplot")),
    )
])


@app.callback(
    Output(component_id="jointplot", component_property="figure"),
    [Input(component_id="x", component_property="value"),
     Input(component_id="y", component_property="value"),
     Input(component_id="observations", component_property="value"),
     Input(component_id="type_", component_property="value")
     ]
)
def update_graph(x, y,observations,type_):
    time.sleep(5)
    if observations is None:
        observations = 50000
    else:
        observations = observations
    new_df = df[:int(observations)]
    if type_ == "scatter":
        fig = make_subplots(rows=2, cols=2, 
                            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                                [{"type": "histogram"}, None]])

        fig.add_trace(go.Scatter(x=new_df[x], y=new_df[y], mode='markers',name="scatter_{x}_{y}"), row=1, col=1)
        fig.add_trace(go.Histogram(x=new_df[x],name="histogram_{x}"), row=1, col=2)
        fig.add_trace(go.Histogram(y=new_df[y],name="histogram_{y}"), row=2, col=1)

    else:
        fig = make_subplots(rows=2, cols=2, 
                            specs=[[{"type": "scatter"}, {"type": "contour"}],
                                [{"type": "histogram"}, None]])
        x_bins = np.linspace(new_df[x].min(), new_df[x].max(), 30)
        y_bins = np.linspace(new_df[y].min(), new_df[y].max(), 30)
        histogram,x_edges,y_edges = np.histogram2d(new_df[x], new_df[y], bins=[x_bins, y_bins],density=True)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        fig.add_trace(go.Contour(z=histogram,
                                         x=x_centers,
                                         y=y_centers,
                                         contours_coloring='heatmap',name="contour_{x}_{y}"),row=1,col=1)
        fig.add_trace(go.Histogram(x=new_df[x],name="histogram_{x}"), row=1, col=2)
        fig.add_trace(go.Histogram(y=new_df[y],name="histogram_{y}"), row=2, col=1)
        fig.update_layout()
    fig.update_layout(
        title='Joint plot',
        xaxis_title=x,
        yaxis_title=y,
        width=1600,
        height=1600,
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=0.95,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        )
    return fig
q13_layout = html.Div([
    html.Center(html.H2("Joint plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"}
                 ],
                 multi=False,
                 value="days_since_prior_order"
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
                 value="days_since_prior_order"
                 ),
    html.Br(),
    html.Br(),
    html.P(html.H5("Select z")),
    dcc.Dropdown(id="z",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"}
                 ],
                 multi=False,
                 value="days_since_prior_order"
                 ),
    html.Br(),
    html.Br(),
    html.B(html.H2("Enter number of observations")),
    dcc.Input(id="observations", type="number", placeholder=f"{len(df)}"),
    html.Br(),
    html.Br(),
    html.Br(),
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(dcc.Graph(id="3Dplot")),
    )
])

@app.callback(
    Output(component_id="3Dplot", component_property="figure"),
     [Input(component_id="x", component_property="value"),
     Input(component_id="y", component_property="value"),
     Input(component_id="z", component_property="value"),
     Input(component_id="observations", component_property="value"),
     ]
)
def update_graph_3d(x, y, z,observations):
    time.sleep(5)
    if observations is None:
        observations = len(df)
    else:
        observations = observations
    new_df = df.iloc[:int(observations)]
    fig = go.Figure(data=[go.Scatter3d(
    x=new_df[x],
    y=new_df[y],
    z=df[z],
    mode='markers',
    marker=dict(
        size=3,
        color=new_df[z],  # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    ),name=f"3Dplot_{x}_{y}_{z}")])

    # tight layout
    fig.update_layout(
        title = f"Regression Line Plot for {x} and {y}",
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=0.95,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        height=1000,
        margin=dict(l=0, r=0, b=0, t=0)
        )
    return fig

q14_layout = html.Div([
    html.Center(html.H2("Clustermap")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"}
                 ],
                 multi=True,
                 value=["order_dow","order_hour_of_day","days_since_prior_order"],
                 ),
    html.Br(),
    html.Br(),
    html.B(html.H2("Enter number of observations")),
    dcc.Input(id="observations", type="number", placeholder=f"{100}"),
    html.Br(),
    html.Br(),
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(dcc.Graph(id="clustermap")),
    )
    
])


@app.callback(
    Output(component_id="clustermap", component_property="figure"),
    [
        Input(component_id="x", component_property="value"),
        Input(component_id="observations", component_property="value")
    ]
)
def plot_clustermap(x, observations):
    if observations is None:
        temp_df = df.sample(100)
    else:
        temp_df = df.sample(int(observations))
    df_temp = temp_df[x]

    # Calculate the distance between each sample
    dists = pdist(df_temp.values)
    print(dists)

    # Perform hierarchical/agglomerative clustering
    linkage_matrix = linkage(dists, method='centroid')

    # Create a dendrogram
    dendro = ff.create_dendrogram(linkage_matrix, orientation='left')

    # Update layout
    dendro['layout'].update({'width':800, 'height':600})
    dendro.update_layout(
        title = f"Cluster plot for {x}",
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=0.95,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        height=1000,
        )
    return dendro

q15_layout = html.Div([
    html.Center(html.H2("Box plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"},
                     {"label": "order_number", "value": "order_number"}
                 ],
                 multi=True,
                 value=["order_dow","order_hour_of_day","days_since_prior_order"],
                 ),
    html.Br(),
    html.Br(),
    html.B(html.H2("Enter number of observations")),
    dcc.Input(id="observations", type="number", placeholder=f"{len(df)}"),
    html.Br(),
    html.Br(),
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(dcc.Graph(id="boxplot"))
    )
])


@app.callback(
    Output(component_id="boxplot", component_property="figure"),
    [
        Input(component_id="x", component_property="value"),
        Input(component_id="observations", component_property="value")
    ]
)
def plot_boxplot(x, observations):
    if observations is None:
        temp_df = df
    else:
        temp_df = df.iloc[:int(observations)]
    df_temp = temp_df[x]

    # Create a box plot
    fig = go.Figure()
    for col in x:
        fig.add_trace(go.Box(y=temp_df[col], name=col))

    # Update layout

    fig.update_layout(
        title = f"Box plot for {x}",
        autosize=False,
        width=1500, 
        height=500,
        margin=dict(l=65, r=50, b=65, t=90),
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=0.95,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        )
    return fig

q16_layout = html.Div([
    html.Center(html.H2("Heatmap plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"},
                     {"label": "order_number", "value": "order_number"}
                 ],
                 multi=True,
                 value=["order_dow","order_hour_of_day","days_since_prior_order"],
                 ),
    html.Br(),
    html.Br(),
    html.B(html.H2("Enter number of observations")),
    dcc.Input(id="observations", type="number", placeholder=f"{len(df)}"),
    html.Br(),
    html.Br(),
    dcc.Loading(
        id="loading",
        type="circle",
        children=html.Div(dcc.Graph(id="heatmap"))
    )
])


@app.callback(
    Output(component_id="heatmap", component_property="figure"),
    [
        Input(component_id="x", component_property="value"),
        Input(component_id="observations", component_property="value")
    ]
)
def plot_heatmap(x, observations):
    if observations is None:
        temp_df = df.sample(50000)
    else:
        temp_df = df.iloc[:int(observations)]
    df_temp = temp_df[x]
    df_corr = df_temp.corr()
    # Create a box plot
    fig = px.imshow(df_corr,labels=dict(color="Correlation Coefficients"),x=df_corr.columns,y=df_corr.columns,zmin=-0.5,zmax=0.6,color_continuous_scale="RdBu_r")
    annot = []
    for i,row_ in enumerate(df_corr.values):
        for j,val in enumerate(row_):
            annot.append(dict(x=df_corr.columns[j],y=df_corr.columns[i],text=np.round(val,2),showarrow=False))

    # Update layout

    fig.update_layout(
        title = f"Correlation Heatmap plot",
        autosize=False,
        margin=dict(l=65, r=50, b=65, t=90),
        width=1500, 
        height=500,
        title_font_family=title_font_family,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        title_x=0.5,
        title_y=0.95,
        font_family = label_font_family,
        font_color = label_font_color,
        font_size = label_font_size,
        annotations=annot,
        )
    return fig

@app.callback(
    Output(component_id="question_ans", component_property="children"),
    [Input(component_id="select-PAGE", component_property="value")],
)
def select_function(value):
    if value == "u":
        return univariate_layout
    elif value == "m":
        return multivariate_layout
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
    elif value == "Pie Chart":
        return q5_layout
    elif value == "Dist plot":
        return q6_layout
    elif value == "QQ plot":
        return q8_layout
    elif value == "KDE plot":
        return q9_layout
    elif value == "Violin plot":
        return q10_layout
    elif value == "Box plot":
        return q15_layout
    else:
        return
    
@app.callback(
    Output(component_id="multiout", component_property="children"),
    [Input(component_id="multivariate", component_property="value")],
)
def update_layout(value):
    if value == "Pair plot":
        return q7_layout
    elif value == "Reg plot":
        return q11_layout
    elif value == "Joint plot":
        return q12_layout
    elif value == "3D plot":
        return q13_layout
    elif value == "Clustermap":
        return q14_layout
    elif value == "Heatmap":
        return q16_layout
    else:
        return


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8020)

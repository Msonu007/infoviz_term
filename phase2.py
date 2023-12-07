import math
import time
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

from plotly.subplots import make_subplots

external_Style_sheets = ["https://codepen.io/chriddvp/pen/bWLwgP.css"]
df = pd.read_csv("downsample.csv")
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
            {"label":"Pair plot","value":"Pair plot"},
            {"label":"QQ plot","value":"QQ plot"},
            {"label":"KDE plot","value":"KDE plot"}
        ],
        value="lineplot"
    ),
    html.Div(id="univout")]
)

multivariate_layout = html.Div([
    html.Br(),
    dcc.Dropdown(
        id="univariate",
        options=[
            {"label":"Pair plot","value":"Pair plot"},
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
    html.Div(dcc.Graph(id="pieplot")),
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
        fig.add_trace( go.Pie(labels=temp_1.index,values=temp_1["count"],pull=explode_1),row=1,col=1)
        fig.add_trace(go.Pie(labels=temp_2.index,values=temp_2["count"],pull=explode_2),row=1,col=2)
        fig.update_layout(
            title="Pie Plot",
            title_font_family="Times New Roman",
            title_font_color="red",
            title_font_size=30,
            title_x=0.5,
            title_y=0.95,
            font_family="Courier New",
            legend_title_font_color="green",
            legend_font_color="black",
            legend_font_size=30,
            font_size=15,
            font_color="black",
        )

    else:
        temp_df = df[int(observations[0]):int(observations[1])]
        temp_1 = df[x].value_counts().reset_index()
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
    html.Div(dcc.Graph(id="distplot")),
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
    density = stats.gaussian_kde(temp_df[x])
    xs = np.linspace(temp_df[x].min(),temp_df[x].max(),len(temp_df))
    ys = density(xs)
    fig.add_trace(px.line(x=xs,y=ys).data[0])
    fig.update_layout(title = "Distribution Plot")
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
    html.Div(dcc.Graph(id="pairplot")),
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
        temp_df = df[x]
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
                fig.add_trace(go.Histogram(x=temp_df[var1], nbinsx=20), row=i+1, col=j+1)
            else:       # Off-diagonal - scatter plot
                fig.add_trace(go.Scatter(x=temp_df[var1], y=temp_df[var2],marker=color, mode='markers'), row=i+1, col=j+1)

    # Customize and show the plot
    fig.update_layout(title_text='Pairplot')
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
    html.Div(dcc.Graph(id="qqplot")),
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
        temp_df = df
    else:
        temp_df = df.iloc[int(observations[0]):int(observations[1])]

    data_sorted = np.sort(temp_df[x])
    norm_quantiles = stats.norm.ppf((np.arange(len(temp_df)) + 0.5) / len(temp_df))

    # Create a QQ plot
    fig = go.Figure(go.Scatter(x=norm_quantiles, y=data_sorted, mode='markers'))

    # Add a reference line
    fig.add_trace(go.Scatter(x=[min(norm_quantiles), max(norm_quantiles)], 
                            y=[min(norm_quantiles), max(norm_quantiles)],
                            mode='lines',
                            name='Reference Line'))

    # Customize and show the plot
    fig.update_layout(title='QQ Plot',
                    xaxis_title='Theoretical Quantiles',
                    yaxis_title='Sample Quantiles')
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
    html.Div(dcc.Graph(id="kdeplot")),
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
    x = np.linspace(min(temp_df[x]),max(temp_df[x]),1000)
    y = kde_val(x)

    # Create a QQ plot
    fig = go.Figure(go.Scatter(x=x, y=y,fill = "tozeroy"))

    # Add a reference line

    # Customize and show the plot
    fig.update_layout(title='KDE Plot')
    return fig



@app.callback(
    Output(component_id="question_ans", component_property="children"),
    [Input(component_id="select-PAGE", component_property="value")],
)
def select_function(value):
    if value == "u":
        return univariate_layout
    else:
        return multivariate_layout


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
    else:
        return
    
@app.callback(
    Output(component_id="multiout", component_property="children"),
    [Input(component_id="multivariate", component_property="value")],
)
def update_layout(value):
    if value == "Pair plot":
        return q7_layout
    else:
        return


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8020)

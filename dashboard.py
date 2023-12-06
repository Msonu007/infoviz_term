import dash
from dash import html
from dash import dcc
from dash.dependencies import Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
import scipy
from scipy.fft import fft
import plotly.graph_objs as go
####loading dataset
df = pd.read_csv(
 "https://github.com/rjafari979/Information-Visualization-Data-AnalyticsDataset-/raw/main/CONVENIENT_global_confirmed_cases.csv"
)
external_Style_sheets = ["https://codepen.io/chriddvp/pen/bWLwgP.css"]
app = dash.Dash("My app", external_stylesheets=external_Style_sheets)
server = app.server
app.layout = html.Div(
     [
         html.Center(html.H1("Homework4")),
         dcc.Tabs(
         id="hw4-select",
         children=[
         dcc.Tab(label="Q1", value="q1"),
         dcc.Tab(label="Q2", value="q2"),
         dcc.Tab(label="Q3", value="q3"),
         dcc.Tab(label="Q4", value="q4"),
         dcc.Tab(label="Q5", value="q5"),
         dcc.Tab(label="Q6", value="q6"),
         ],
         value="q1",
         ),
         html.Div(id="question_ans"),
     ]
)
q1_layout = html.Div(
     [
         html.Center(html.H3("Plotting covid19 cases for different countries")),
         html.P("Pick the country Name"),
         dcc.Dropdown(
         id="country-name-q1",
         options=[
             {"label": "US", "value": "US"},
             {"label": "Brazil", "value": "Brazil"},
             {"label": "United Kingdom_sun", "value": "United Kingdom_sun"},
             {"label": "China_sum", "value": "China_sum"},
             {"label": "India", "value": "India"},
             {"label": "Italy", "value": "Italy"},
             {"label": "Germany", "value": "Germany"},
         ],
         multi=True,
         value="",
         ),
         html.Div(dcc.Graph(id="out-graph-q1")),
     ]
)
@app.callback(
 Output(component_id="out-graph-q1", component_property="figure"),
 [Input(component_id="country-name-q1", component_property="value")],
)
def render_q1(values):
     dates = df["Country/Region"][1:]
     df_new_dict = {"date": dates}
     for i in range(len(values)):
         if values[i] == "China_sum":
            out_ = df["China"][1:].astype(float)
         for j in range(1, 33):
            out_ = out_ + df[f"China.{j}"][1:].astype(float)
         elif values[i] == "United Kingdom_sun":
            out_ = df["United Kingdom"][1:].astype(float)
            for j in range(1, 11):
                out_ = out_ + df[f"United Kingdom.{j}"][1:].astype(float)
         else:
            out_ = df[values[i]][1:]
            df_new_dict[values[i]] = out_
            df_new = pd.DataFrame.from_dict(df_new_dict)
            k = list(df_new.columns)
            k.pop(k.index("date"))
            fig = px.line(
             data_frame=df_new,
             x="date",
             y=k,
             title="covid cases countrywise",
             width=2000,
             height=800,
             )
     return fig
q2_layout = html.Div(
 [
 html.Center(html.H3("Plotting a Quadratic Function")),
 html.P("Select a"),
 dcc.Slider(id="a", min=-10, max=10, step=0.5, value=10),
 html.Br(),
 html.P("select b"),
 dcc.Slider(id="b", min=-10, max=10, step=0.5, value=10),
 html.Br(),
 html.P("select c"),
 dcc.Slider(id="c", min=-10, max=10, step=0.5, value=10),
 html.Br(),
 html.Div(dcc.Graph(id="out-graph-q2")),
 ]
)
@app.callback(
 Output(component_id="out-graph-q2", component_property="figure"),
 [
 Input(component_id="a", component_property="value"),
 Input(component_id="b", component_property="value"),
 Input(component_id="c", component_property="value"),
 ],
)
def plot_q2(a, b, c):
     x = np.linspace(-2, 2, 1000)
     out_ = (a * (np.square(x))) + (b * x) + c
     df_ = pd.DataFrame.from_dict({"x": x, "y": out_})
     print(df_.head())
     fig = px.line(data_frame=df_, x="x", y="y", title=r"quadratic function ax^2 +bx + c ")
     return fig
q3_layout = html.Div(
 [
 html.Center(html.H2("Caluculator")),
 html.P("Enter a"),
 dcc.Input(id="q3_a", type="number", placeholder=""),
 html.Br(),
 html.P("Enter b"),
 dcc.Input(id="q3_b", type="number", placeholder=""),
 html.Br(),
 html.Br(),
 html.Br(),
 dcc.Dropdown(
 id="dropdown_q3",
 options=[
 {"label": "Addition", "value": "add"},
 {"label": "Subtraction", "value": "sub"},
 {"label": "Multiplication", "value": "mul"},
 {"label": "Division", "value": "div"},
 {"label": "Log", "value": "log"},
 {"label": "bthRootofa", "value": "root"},
 ],
 ),
 html.Br(),
 html.Div(html.P(id="res-q3")),
 ]
)
@app.callback(
 Output(component_id="res-q3", component_property="children"),
 [
 Input(component_id="q3_a", component_property="value"),
 Input(component_id="q3_b", component_property="value"),
 Input(component_id="dropdown_q3", component_property="value"),
 ],
)
def plot_q3(q3_a, q3_b, dropdown_q3):
     out_ = None
     if dropdown_q3 == "add":
        out_ = q3_a + q3_b
     elif dropdown_q3 == "sub":
        out_ = q3_a - q3_b
     elif dropdown_q3 == "mul":
        out_ = q3_a * q3_b
     elif dropdown_q3 == "div":
        out_ = q3_a / q3_b
     elif dropdown_q3 == "log":
         if q3_a >= 1 and q3_b > 1:
            out_ = np.log(q3_a) / np.log(q3_b)
         else:
            out_ = "inf please check your inputs"
     elif dropdown_q3 == "root":
         out_ = np.power(q3_a, 1 / q3_b)
     return f"The output is {out_}"
q4_layout = html.Div(
 [
 html.Center(html.H4("Plotting polynomial with a degree")),
 html.B(html.H2("Please enter the polynomial order")),
 html.Br(),
 dcc.Input(id="x", type="number", placeholder=""),
 html.Br(),
 html.Div(dcc.Graph(id="graph-q4")),
 ]
)
@app.callback(
 Output(component_id="graph-q4", component_property="figure"),
 [Input(component_id="x", component_property="value")],
)
def q4(x):
     k = np.linspace(-2, 2, 1000)
     print(np.max(k**x))
     out_ = k**x
     data = pd.DataFrame.from_dict({"x": k, "y": out_})
     fig = px.line(data_frame=data, x="x", y="y")
     return fig
q5_layout = html.Div(
 [
 html.Center(html.H3("Fast fourier Transform")),
 html.P(html.B("Please enter the number of sinusodal cycle")),
 dcc.Input(id="q5_s", type="number", placeholder=""),
 html.Br(),
 html.Br(),
 html.P(html.B("Please enter the number of white noise")),
 dcc.Input(id="q5_m", type="number", placeholder=""),
 html.Br(),
 html.Br(),
 html.P(html.B("Please enter the standard deviation of the white noise")),
 dcc.Input(id="q5_std", type="number", placeholder=""),
 html.Br(),
 html.Br(),
 html.P(html.B("Please enter the number of samples")),
 dcc.Input(id="q5_n", type="number", placeholder=""),
 html.Br(),
 html.Br(),
 dcc.Graph(id="input_g"),
 html.Br(),
 html.P(html.B("The fast fourier transform of above generated data is")),
 html.Br(),
 dcc.Graph(id="output_g"),
 ]
)
@app.callback(
 [
 Output(component_id="input_g", component_property="figure"),
 Output(component_id="output_g", component_property="figure"),
 ],
 [
 Input(component_id="q5_s", component_property="value"),
 Input(component_id="q5_m", component_property="value"),
 Input(component_id="q5_std", component_property="value"),
 Input(component_id="q5_n", component_property="value"),
 ],
)
def q5(q5_s, q5_m, q5_std, q5_n):
     x = np.linspace(-1 * np.pi, np.pi, q5_n)
     x = q5_s * x
     noise_ = np.random.normal(q5_m, q5_std, q5_n)
     out_ = np.sin(x) + noise_
     data_df = pd.DataFrame.from_dict({"x": x, "noise": out_})
     fig1 = px.line(data_frame=data_df, x="x", y="noise")
     out_1 = fft(noise_)
     freq = np.fft.fftfreq(q5_n, d=(2 * np.pi) / q5_n)
     for i in range(len(out_1)):
     out_1[i] = np.absolute(out_1[i])
     data_df_1 = pd.DataFrame.from_dict({"x": x, "fft": out_1.astype(float)})
     fig2 = px.line(data_frame=data_df_1, x="x", y="fft").data[0]
     fig = go.Figure()
     fig.add_trace(fig2)
     return [fig1, fig]
q6_layout = html.Div(
 [
 html.Center(html.H3("Neural network charecterstic graph")),
 html.Div([
 html.Div(style={'width': '30%', 'display': 'inline-block', 'verticalalign':'middle'}),

html.Img(src="https://www.researchgate.net/profile/Betim-Cico/publication/265992319/figure/fig2/AS:669022924337162@1536518903423/Example-functionapproximation-network.ppm",style={'width': '30%', 'display': 'inline-block',
'vertical-align':'middle'}),
 html.Div(style={'width': '30%', 'display': 'inline-block', 'verticalalign':'middle'}),
 ]),
 html.Div([dcc.Graph(id="sigmoid")]),
 html.Div(
 [
 html.P(dcc.Markdown('b<sup>1</sup><sub>1</sub>',
dangerously_allow_html=True)),
 dcc.Slider(id="b11", min=-10, max=10, step=1, value=10),
 html.Br(),
 html.P(dcc.Markdown('b<sup>1</sup><sub>2</sub>',
dangerously_allow_html=True)),
 dcc.Slider(id="b12", min=-10, max=10, step=1, value=10),
 html.Br(),
 html.P(dcc.Markdown('w<sup>1</sup><sub>1,1</sub>',
dangerously_allow_html=True)),
 dcc.Slider(id="w11_1", min=-10, max=10, step=1, value=10),
 html.Br(),
 html.P(dcc.Markdown('w<sup>1</sup><sub>2,1</sub>',
dangerously_allow_html=True)),
 dcc.Slider(id="w21_1", min=-10, max=10, step=1, value=10),
 html.Br(),
 html.P(dcc.Markdown('b<sup>2</sup>', dangerously_allow_html=True)),
 dcc.Slider(id="b2", min=-10, max=10, step=1, value=10),
 html.Br(),
 html.P(dcc.Markdown('w<sup>2</sup><sub>1,1</sub>',
dangerously_allow_html=True)),
 dcc.Slider(id="w11_2", min=-10, max=10, step=1, value=10),
 html.Br(),
 html.P(dcc.Markdown('w<sup>2</sup><sub>2,1</sub>',
dangerously_allow_html=True)),
 dcc.Slider(id="w21_2", min=-10, max=10, step=1, value=10),
 ]
 ),
 ],
)
@app.callback(
 Output(component_id="sigmoid", component_property="figure"),
 [
 Input(component_id="w11_1", component_property="value"),
 Input(component_id="w21_1", component_property="value"),
 Input(component_id="w11_2", component_property="value"),
 Input(component_id="w21_2", component_property="value"),
 Input(component_id="b11", component_property="value"),
 Input(component_id="b12", component_property="value"),
 Input(component_id="b2", component_property="value"),
 ],
)
def q6(w11_1, w21_1, w11_2, w21_2, b11, b12, b2):
     def sigmoid(x):
        return 1 / (1 + (1 / np.exp(x)))
     p = np.linspace(-5, 5, 1000)
     k1 = (w11_1 * p) + b11
     k2 = (w21_1 * p) + b12
     k1 = sigmoid(k1)
     k2 = sigmoid(k2)
     k3 = (w11_2 * k1) + (w21_2 * k2) + b2
     df_ = pd.DataFrame.from_dict({"p": p, "y": k3})
     fig = px.line(data_frame=df_, x="p", y="y")
     fig.update_layout(yaxis_title=r"a^2")
     return fig
@app.callback(
 Output(component_id="question_ans", component_property="children"),
 [Input(component_id="hw4-select", component_property="value")],
)
def select_function(value):
     if value == "q1":
        return q1_layout
     elif value == "q2":
        return q2_layout
     elif value == "q3":
        return q3_layout
     elif value == "q4":
        return q4_layout
     elif value == "q5":
         return q5_layout
     elif value == "q6":
        return q6_layout
if __name__ == "__main__":
 app.run_server(debug=True, host='0.0.0.0', port=8020)
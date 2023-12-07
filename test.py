q15_layout = html.Div([
    html.Center(html.H2("Box plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"}
                     {"label": "order_number", "value": "order_number"}
                 ],
                 multi=True,
                 value="order_dow"
                 ),
    html.Br(),
    html.Br(),
    html.B(html.H2("Enter number of observations")),
    dcc.Input(id="observations", type="number", placeholder=f"{len(df)}"),
    html.Br(),
    html.Br()
    html.Div(dcc.Graph(id="boxplot")),
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
    fig.add_trace(go.Box(y=df_temp, name=x))

    # Update layout
    fig.update_layout(title='Box plot', autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))

    return fig
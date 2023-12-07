q17_layout = html.Div([
    html.Center(html.H2("Area plot")),
    html.Br(),
    html.P(html.H5("Select x")),
    dcc.Dropdown(id="x",
                 options=[
                     {"label": "order_dow", "value": "order_dow"},
                     {"label": "order_hour_of_day", "value": "order_hour_of_day"},
                     {"label": "days_since_prior_order", "value": "days_since_prior_order"},
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
        children=html.Div(dcc.Graph(id="areaplot"))
    )
])


@app.callback(
    Output(component_id="areaplot", component_property="figure"),
    [
        Input(component_id="x", component_property="value"),
        Input(component_id="observations", component_property="value")
    ]
)
def plot_area(x, observations):
    if observations is None:
        temp_df = df.sample(50000)
    else:
        temp_df = df.iloc[:int(observations)]
    
    fig = go.Figure()

    for var in x:
        fig.add_trace(go.Scatter(
            x=temp_df.index, 
            y=temp_df[var],
            fill='tozeroy',
            mode='lines', 
            name=var,
        ))

    fig.update_layout(
        title="Area Plot",
        xaxis_title="Index",
        yaxis_title="Value",
        autosize=False,
        width=1500,
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
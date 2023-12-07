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
    Output(component_id="qqplot", component_property="figure"),
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
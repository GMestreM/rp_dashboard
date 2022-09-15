import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc 
from dash.dependencies import Input, Output, State

import os
import time

import pandas as pd 
import numpy as np 

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from utils.metrics import (
    get_historical_drawdown,
    get_summary_metrics,
)
from utils.utils import (
    simple_returns,
)

from database import (
    get_strategy_prices,
)

DEFAULT_PLOTLY_COLORS = px.colors.qualitative.Plotly

# Load data
# =========================
dict_strategy_pnl = get_strategy_prices()
df = pd.concat([
    dict_strategy_pnl['df_perf_rp'],
    dict_strategy_pnl['df_perf_ew'],
],axis = 1)
df.columns = ['Strategy','Benchmark']
df.index.name = 'Dates'

# Get model weights and asset returns
df_weights   = dict_strategy_pnl['df_model_weights']
df_asset_ret = dict_strategy_pnl['df_asset_return_model']

# Rename columns
dict_rename  = dict_strategy_pnl['dict_asset_name']
df_weights.rename(columns = dict_rename, inplace = True)
df_asset_ret.rename(columns = dict_rename, inplace = True)

# Define initial metrics dataframe
df_metrics = get_summary_metrics(df_prices_1 = df[df.columns[0]].to_frame(),
                                 df_prices_2 = df[df.columns[1]].to_frame())

# Define app subcomponents
# ==============================
# Create plotly figure
df_plot = df.copy()
# Get drawdown of both strategies
df_plot_dd = get_historical_drawdown(df_plot)

# fig = go.Figure()
fig = make_subplots(
    rows = 2, cols = 1, shared_xaxes=True,
    subplot_titles = ['Performance','Drawdown'],
    vertical_spacing = 0.05,
    # specs=[{"type":'xy'}, {"type":'xy'}]
)

trace_1_1 = go.Scattergl(
        x = df_plot.index,
        y = df_plot['Strategy'],
        mode = "lines+markers",
        name = 'Strategy',
        legendgroup='group1',
        opacity=1,
        line = dict(color = DEFAULT_PLOTLY_COLORS[0]),
)

trace_1_2 = go.Scattergl(
        x = df_plot.index,
        y = df_plot['Benchmark'],
        mode = "lines+markers",
        name = 'Benchmark',
        legendgroup='group2',
        opacity=1,
        line = dict(color = DEFAULT_PLOTLY_COLORS[1]),
)

trace_2_1 = go.Scattergl(
        x = df_plot_dd.index,
        y = df_plot_dd['Strategy'],
        mode = "lines+markers",
        name = 'Strategy',
        legendgroup='group1',
        showlegend=False,
        opacity=1,
        line = dict(color = DEFAULT_PLOTLY_COLORS[0]),
)

trace_2_2 = go.Scattergl(
        x = df_plot_dd.index,
        y = df_plot_dd['Benchmark'],
        mode = "lines+markers",
        name = 'Benchmark',
        legendgroup='group2',
        showlegend=False,
        opacity=1,
        line = dict(color = DEFAULT_PLOTLY_COLORS[1]),
)

fig.add_trace(
    trace_1_1,
    row = 1,col = 1,
)

fig.add_trace(
    trace_1_2, 
    row = 1,col = 1,
    
)

fig.add_trace(
    trace_2_1,
    row = 2,col = 1,
)

fig.add_trace(
    trace_2_2, 
    row = 2,col = 1, 
)


# Change grid color and axis colors
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#343a40')

# Add Range selector suttons
fig.update_xaxes(
    matches='x',
    showgrid=True, 
    gridwidth=1, 
    gridcolor='#343a40',
    # rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1M", step="month", stepmode="backward",),
            dict(count=6, label="6M", step="month", stepmode="backward",),
            dict(count=1, label="YTD", step="year", stepmode="todate",),
            dict(count=1, label="1Y", step="year", stepmode="backward",),
            # dict(step="all", label = 'All'), # Remove this button, leaver undesired margin
        ]),
        # type="date",
    ),
    # rangeselector_font = dict(size = 5),
    rangeselector_bgcolor = '#0C4160',
    autorange = False,
    range=(df_plot_dd.index.min(), df_plot_dd.index.max()),
    type="date",
    rangeselector_xanchor='right',
    rangeselector_x = 1,#1
    rangeselector_yanchor='top',
    rangeselector_y = 1.15,
)

# fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)



# Generate spiderchart with metrics
df_metrics_spiderchart = df_metrics.copy()
# Normalize each column between 0 and 1
for nrow in range(len(df_metrics_spiderchart)):
    # df_metrics_spiderchart.iloc[nrow, :] = (df_metrics_spiderchart.iloc[nrow, :] - np.min(df_metrics_spiderchart.iloc[nrow, :]))/(np.max(df_metrics_spiderchart.iloc[nrow, :]) - np.min(df_metrics_spiderchart.iloc[nrow, :]))

    # Use 0 as min value
    df_metrics_spiderchart.iloc[nrow, :] = (df_metrics_spiderchart.iloc[nrow, :] - 0)/(np.max(df_metrics_spiderchart.iloc[nrow, :]) - 0)
#Duplicate last row
df_metrics_spiderchart = pd.concat([df_metrics_spiderchart, df_metrics_spiderchart.iloc[0,:].to_frame().T], axis = 0)
# print(df_metrics_spiderchart)
fig_radar = go.Figure()

categories = df_metrics_spiderchart.index.tolist()

for col_name in df_metrics_spiderchart.columns:
    # print(categories)
    # print(df_metrics_spiderchart[col_name].values)
    fig_radar.add_trace(go.Scatterpolar(
            r = df_metrics_spiderchart[col_name].values,
            theta = categories,
            fill='toself',
            name = col_name,
        )
    )
fig_radar.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True, # If True, display the radial axis
      range=[0, 1],
    )),
  showlegend=False,
)

#Generate pie charts with TAA
labels_pie = df_weights.columns
values_pie = df_weights.iloc[-1,:].values

# Define colors
dict_colors_assets = {}
for i, label_pie in enumerate(labels_pie):
    dict_colors_assets[label_pie] = DEFAULT_PLOTLY_COLORS[i]
# Now to color list in the same order
list_colors_assets = [dict_colors_assets[name] for name in labels_pie]

# fig_pie = make_subplots(rows=1, cols=2, specs=[[
#         {'type':'domain'},# type domain for pie charts
#         {'type':'bar'},# type domain for bar charts
#     ]],
#     subplot_titles=('Allocation', 'Contribution'),
# )
fig_pie = go.Figure()

# Add pie chart
fig_pie.add_trace(
    go.Pie(
        labels = labels_pie,
        values = values_pie,
        name = f'{df_weights.index.max():%Y-%m-%d}',#'Asset Allocation',
        hole=.3, 
        showlegend = False,
        marker_colors = list_colors_assets,
    )#,1,1,
)

fig_waterfall = go.Figure()

# Get last 3M contribution
df_weights_3M = df_weights# .loc[(-21*3):,:].copy()
df_contrib_returns_assets = (
    df_weights_3M*df_asset_ret
).dropna()

asset_contrib_returns = df_contrib_returns_assets.sum(axis = 0).values

# Add bar chart
name_aux = f'{df_contrib_returns_assets.index.min():%Y-%m-%d} - {df_contrib_returns_assets.index.max():%Y-%m-%d}'
fig_waterfall.add_trace(
    go.Waterfall(
        x = df_contrib_returns_assets.columns,
        y = asset_contrib_returns,
        measure = ['relative']*len(labels_pie),
        showlegend = False,
        name = name_aux,
        connector = {"line":{"color":"rgb(0, 0, 0)", 'width':5}},
    )#,1,2,
)

# Update dates in subtitles
# dates_names = {
#         'Allocation':f'{df_weights.index.max():%Y-%m-%d}',
#         'Contribution':f'{df_contrib_returns_assets.index.min():%Y-%m-%d} - {df_contrib_returns_assets.index.max():%Y-%m-%d}',
#     }
# fig_pie.for_each_annotation(lambda a: a.update(text = a.text + ' (' + dates_names[a.text] + ') <br>'))

# https://stackoverflow.com/questions/66064756/how-to-change-subplot-size-and-decrease-space-between-subplot
# fig_pie.layout.annotations[0].update(y=1.05, x = 0)
# fig_pie.layout.annotations[1].update(y=1.05, x = 0)

# fig_pie.update_layout(margin=dict(t=20000), autosize=False,)

# fig_pie.update_layout(title_pad=dict(t=20), margin=dict(pad=15))

fig_waterfall.update_layout(yaxis={'tickformat': ',.1%'},)


# Initialize Dash app
# =========================
app = dash.Dash(
    assets_folder='./assets',
    include_assets_files=True,
    title='Risk Parity dashboard',
    external_stylesheets=[dbc.icons.BOOTSTRAP, dbc.themes.DARKLY],
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1",
        },
    ],
    # external_stylesheets=[dbc.themes.BOOTSTRAP]  # If not included, use the .css files located in assets folder
)

server = app.server

html_main_title = html.H1('Risk Parity Dashboard')

table_metrics = dbc.Table.from_dataframe(df = df_metrics.round(decimals = 2), 
                                         id='table_metrics',
                                         responsive=True,
                                         striped=True,
                                         bordered=False, 
                                         index=True,
                                         hover=True)

figure_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    dcc.Graph(figure = fig.update_layout(
                        template='plotly_dark',
                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                        autosize = True,
                        legend=dict(
                            orientation="v",
                            y=1.05,      # y-coordinate for the anchor point (1 means top of the y axis in the figure)
                            x=0.0,       # x-coordinate for the anchor point (1 means top of the x axis in the figure) 
                            # y=.9,      # y-coordinate for the anchor point (1 means top of the y axis in the figure)
                            # x=0.02,       # x-coordinate for the anchor point (1 means top of the x axis in the figure) 
                            yanchor="bottom",
                            xanchor="left",
                            font=dict(size= 10),
                        ),
                        margin=dict(l=0, r=0, t=0, b=0, autoexpand = True),
                    ),config = {'displayModeBar':False},id='graph-figure',
                    ),
                ], )
            ),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H3('Allocation'),
                            # html.H6(id = 'contrib_returns'),
                            # html.Br(),
                            dcc.Graph(figure = fig_pie.update_layout(
                                template='plotly_dark',
                                plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                autosize = True,
                                # legend = None,
                                # legend=dict(
                                #     orientation="h",
                                #     y=1.05,      # y-coordinate for the anchor point (1 means top of the y axis in the figure)
                                #     x=0.0,       # x-coordinate for the anchor point (1 means top of the x axis in the figure) 
                                #     # y=.9,      # y-coordinate for the anchor point (1 means top of the y axis in the figure)
                                #     # x=0.02,       # x-coordinate for the anchor point (1 means top of the x axis in the figure) 
                                #     yanchor="bottom",
                                #     xanchor="left",
                                #     font=dict(size= 10),
                                # ),
                                margin=dict(l=0, r=0, t=25, b=0, autoexpand = True),
                                ),config = {'displayModeBar':False},id='graph-figure-pie',
                            ),
                        ]),
                    ),
                ], md = {'size':6}, sm = {'size' : 12}),
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H3('Contribution'),
                            # html.H6(id = 'contrib_returns'),
                            # html.Br(),
                            dcc.Graph(figure = fig_waterfall.update_layout(
                                template='plotly_dark',
                                plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                autosize = True,
                                # legend = None,
                                # legend=dict(
                                #     orientation="h",
                                #     y=1.05,      # y-coordinate for the anchor point (1 means top of the y axis in the figure)
                                #     x=0.0,       # x-coordinate for the anchor point (1 means top of the x axis in the figure) 
                                #     # y=.9,      # y-coordinate for the anchor point (1 means top of the y axis in the figure)
                                #     # x=0.02,       # x-coordinate for the anchor point (1 means top of the x axis in the figure) 
                                #     yanchor="bottom",
                                #     xanchor="left",
                                #     font=dict(size= 10),
                                # ),
                                margin=dict(l=0, r=0, t=25, b=0, autoexpand = True),
                                ),config = {'displayModeBar':False},id='graph-figure-waterfall',
                            ),
                        ]),
                    ),
                ], md = {'size':6}, sm = {'size' : 12}),
            ]),
        ], md = {'size':7}, sm = {'size' : 12}),
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                        html.H3('Risk Metrics'),
                        #dcc.Loading([
                                html.H6(id = 'risk_metrics_dates'),
                                html.Div(children = [table_metrics], id = 'table_metrics_div'),
                        #    ], type = 'dot',
                        #),
                        #dcc.Loading([
                            dcc.Graph(figure = fig_radar.update_layout(
                                template='plotly_dark',
                                plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                margin=dict(l=100, r=100, t=0, b=0),
                            ),config = {'displayModeBar':False},id = 'graph-figure-radar-chart',),
                        #], type = 'dot',),
                    ])
            )
        ], md = {'size':5}, sm = {'size' : 12}),
    ])
], fluid=True)

# Define layout
app.layout = dbc.Container(
    children=[
        dbc.Row([
            dbc.Col(
                html_main_title,
                # width={'size':11}
            ),
        ]),
        html.Hr(),
        figure_layout,
    ], 
    fluid=True,
)

# Callbacks
# ==================================
# Callback to update table using figure zoom
# TO-DO: base 100 each time the user zooms on data
@app.callback(
    Output("risk_metrics_dates","children"), 
    [
        Input('graph-figure','relayoutData'),
        Input('graph-figure','figure'),
    ]
)
def display_zoom_dates(relayout_data, figure_val):
    # print(relayout_data)
    # print(figure_val['layout']["xaxis"]["range"])

    try:
        ini_date = pd.to_datetime(figure_val['layout']["xaxis"]["range"][0], format = '%Y-%m-%d %H:%M:%S.%f')
        end_date = pd.to_datetime(figure_val['layout']["xaxis"]["range"][1], format = '%Y-%m-%d %H:%M:%S.%f')

        # If the date displayed in the zoomed figure is grater than the range of data, fix it
        if ini_date < df.index.min():
            ini_date = df.index.min()
        if end_date > df.index.max():
            end_date = df.index.max()

        output_range = f"{ini_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
    except (KeyError, TypeError):
        # output_range = f'None'
        output_range = f"{df.index.min().strftime('%Y-%m-%d')} - {df.index.max().strftime('%Y-%m-%d')}"

    return output_range


# callback to update radar chart
@app.callback(
    Output("graph-figure-radar-chart","figure"), 
    [
        Input('graph-figure','relayoutData'),
        Input('graph-figure','figure'),
    ]
)
def display_radar_chart_metrics(relayout_data, figure_val):
    # Retrieve initial and end dates
    try:
        ini_date = pd.to_datetime(figure_val['layout']["xaxis"]["range"][0], format = '%Y-%m-%d %H:%M:%S.%f')
        end_date = pd.to_datetime(figure_val['layout']["xaxis"]["range"][1], format = '%Y-%m-%d %H:%M:%S.%f')

        # If the date displayed in the zoomed figure is grater than the range of data, fix it
        if ini_date < df.index.min():
            ini_date = df.index.min()
        if end_date > df.index.max():
            end_date = df.index.max()

        output_range = f"{ini_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
    except (KeyError, TypeError):
        # output_range = f'None'
        output_range = f"{df.index.min().strftime('%Y-%m-%d')} - {df.index.max().strftime('%Y-%m-%d')}"
    date_range_str = output_range.split(' - ')
    ini_date_filt = pd.to_datetime(date_range_str[0], format = '%Y-%m-%d')
    end_date_filt = pd.to_datetime(date_range_str[1], format = '%Y-%m-%d')

    df_filt_aux = df.loc[(df.index >= ini_date_filt)&(df.index <= end_date_filt),:].copy()
    df_metrics_filt = get_summary_metrics(df_prices_1 = df_filt_aux[df_filt_aux.columns[0]].to_frame(),
                                          df_prices_2 = df_filt_aux[df_filt_aux.columns[1]].to_frame())

    df_metrics_spiderchart = df_metrics_filt.copy()                              
    # Normalize each column between 0 and 1
    for nrow in range(len(df_metrics_spiderchart)):
        # df_metrics_spiderchart.iloc[nrow, :] = (df_metrics_spiderchart.iloc[nrow, :] - np.min(df_metrics_spiderchart.iloc[nrow, :]))/(np.max(df_metrics_spiderchart.iloc[nrow, :]) - np.min(df_metrics_spiderchart.iloc[nrow, :]))

        # Use 0 as min value
        df_metrics_spiderchart.iloc[nrow, :] = (df_metrics_spiderchart.iloc[nrow, :] - 0)/(np.max(df_metrics_spiderchart.iloc[nrow, :]) - 0)
    #Duplicate last row
    df_metrics_spiderchart = pd.concat([df_metrics_spiderchart, df_metrics_spiderchart.iloc[0,:].to_frame().T], axis = 0)
    # print(df_metrics_spiderchart)
    fig_radar_aux = go.Figure()

    categories = df_metrics_spiderchart.index.tolist()
    categories = [x.replace(' ','<br>') for x in categories]

    for col_name in df_metrics_spiderchart.columns:
        # print(categories)
        # print(df_metrics_spiderchart[col_name].values)
        fig_radar_aux.add_trace(go.Scatterpolar(
                r = df_metrics_spiderchart[col_name].values,
                theta = categories,
                fill='toself',
                name = col_name,
            )
        )
    fig_radar_aux.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True, # If True, display the radial axis
          range=[0, 1],
          showticklabels = False,
        )),
      showlegend=False,
    )
    fig_radar_aux.update_layout(
        template='plotly_dark',
        font_size = 11,
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
        autosize = True,
        margin=dict(l=30, r=30, t=0, b=0,autoexpand = True),
        # margin=dict(autoexpand = True)
    )

    return fig_radar_aux


@app.callback(
    Output("table_metrics_div","children"), 
    [
        Input('graph-figure','relayoutData'),
        Input('graph-figure','figure'),
    ]
)
def update_table_risk_metrics(relayout_data, figure_val):
    # Retrieve initial and end dates
    try:
        ini_date = pd.to_datetime(figure_val['layout']["xaxis"]["range"][0], format = '%Y-%m-%d %H:%M:%S.%f')
        end_date = pd.to_datetime(figure_val['layout']["xaxis"]["range"][1], format = '%Y-%m-%d %H:%M:%S.%f')

        # If the date displayed in the zoomed figure is grater than the range of data, fix it
        if ini_date < df.index.min():
            ini_date = df.index.min()
        if end_date > df.index.max():
            end_date = df.index.max()

        output_range = f"{ini_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
    except (KeyError, TypeError):
        # output_range = f'None'
        output_range = f"{df.index.min().strftime('%Y-%m-%d')} - {df.index.max().strftime('%Y-%m-%d')}"
    date_range_str = output_range.split(' - ')
    ini_date_filt = pd.to_datetime(date_range_str[0], format = '%Y-%m-%d')
    end_date_filt = pd.to_datetime(date_range_str[1], format = '%Y-%m-%d')

    df_filt_aux = df.loc[(df.index >= ini_date_filt)&(df.index <= end_date_filt),:].copy()
    df_metrics_filt = get_summary_metrics(df_prices_1 = df_filt_aux[df_filt_aux.columns[0]].to_frame(),
                                          df_prices_2 = df_filt_aux[df_filt_aux.columns[1]].to_frame())

    table_metrics_aux = dbc.Table.from_dataframe(df = df_metrics_filt.round(decimals = 2),
                                         id='table_metrics',
                                         responsive=True,
                                         striped=True,
                                         bordered=False, 
                                         index=True,
                                         hover=True)
                                          
    return table_metrics_aux

# callback to update pie chart
@app.callback(
    Output("graph-figure-pie","figure"), 
    [
        Input('graph-figure','relayoutData'),
        Input('graph-figure','figure'),
    ]
)
def display_pie_chart_contribution(relayout_data, figure_val):
    # Retrieve initial and end dates
    try:
        ini_date = pd.to_datetime(figure_val['layout']["xaxis"]["range"][0], format = '%Y-%m-%d %H:%M:%S.%f')
        end_date = pd.to_datetime(figure_val['layout']["xaxis"]["range"][1], format = '%Y-%m-%d %H:%M:%S.%f')

        # If the date displayed in the zoomed figure is grater than the range of data, fix it
        if ini_date < df.index.min():
            ini_date = df.index.min()
        if end_date > df.index.max():
            end_date = df.index.max()

        output_range = f"{ini_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
    except (KeyError, TypeError):
        # output_range = f'None'
        output_range = f"{df.index.min().strftime('%Y-%m-%d')} - {df.index.max().strftime('%Y-%m-%d')}"
    date_range_str = output_range.split(' - ')
    ini_date_filt = pd.to_datetime(date_range_str[0], format = '%Y-%m-%d')
    end_date_filt = pd.to_datetime(date_range_str[1], format = '%Y-%m-%d')


    #Filter weights
    df_weights_filt = df_weights.loc[(df_weights.index >= ini_date_filt)&(df_weights.index <= end_date_filt),:].copy()
    # print(df_weights_filt)
    
    #Generate pie charts with TAA
    values_pie = df_weights_filt.iloc[-1,:].values

    # fig_pie_aux = make_subplots(rows=1, cols=2, specs=[[
    #         {'type':'domain'},# type domain for pie charts
    #         {'type':'bar'},# type domain for bar charts
    #     ]],
    #     subplot_titles=('Allocation', 'Contribution'),
    # )
    fig_pie_aux = go.Figure()

    # Add pie chart
    fig_pie_aux.add_trace(
        go.Pie(
            labels = labels_pie,
            values = values_pie,
            name = f'{df_weights_filt.index.max():%Y-%m-%d}',#'Asset Allocation',
            hole=.3, 
            showlegend = False,
            marker_colors = list_colors_assets,
        )#,1,1,
    )

    # # Get last 3M contribution
    # df_weights_3M = df_weights_filt.copy()# .loc[(-21*3):,:].copy()
    # df_contrib_returns_assets = (
    #     df_weights_3M*df_asset_ret
    # ).dropna()

    # asset_contrib_returns = df_contrib_returns_assets.sum(axis = 0).values

    # # Add bar chart
    # name_aux = f'{df_contrib_returns_assets.index.min():%Y-%m-%d} - {df_contrib_returns_assets.index.max():%Y-%m-%d}'
    # fig_pie_aux.add_trace(
    #     go.Waterfall(
    #         x = df_contrib_returns_assets.columns,
    #         y = asset_contrib_returns,
    #         measure = ['relative']*len(labels_pie),
    #         showlegend = False,
    #         name = name_aux,
    #         connector = {"line":{"color":"rgb(0, 0, 0)", 'width':5}},
    #     ),1,2,
    # )

    # # Update dates in subtitles
    # dates_names = {
    #         'Allocation':f'{df_weights_filt.index.max():%Y-%m-%d}',
    #         'Contribution':f'{df_contrib_returns_assets.index.min():%Y-%m-%d} - {df_contrib_returns_assets.index.max():%Y-%m-%d}',
    #     }
    # fig_pie_aux.for_each_annotation(lambda a: a.update(text = a.text + ' (' + dates_names[a.text] + ') <br>'))

    # https://stackoverflow.com/questions/66064756/how-to-change-subplot-size-and-decrease-space-between-subplot
    # fig_pie.layout.annotations[0].update(y=1.05, x = 0)
    # fig_pie.layout.annotations[1].update(y=1.05, x = 0)

    # fig_pie.update_layout(margin=dict(t=20000), autosize=False,)

    # fig_pie.update_layout(title_pad=dict(t=20), margin=dict(pad=15))

    fig_pie_aux.update_layout(yaxis={'tickformat': ',.1%'},title = f'{df_weights_filt.index.max():%Y-%m-%d}')

    fig_pie_aux.update_layout(
                        template='plotly_dark',
                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                        autosize = True,
                        # legend = None,
                        # legend=dict(
                        #     orientation="h",
                        #     y=1.05,      # y-coordinate for the anchor point (1 means top of the y axis in the figure)
                        #     x=0.0,       # x-coordinate for the anchor point (1 means top of the x axis in the figure) 
                        #     # y=.9,      # y-coordinate for the anchor point (1 means top of the y axis in the figure)
                        #     # x=0.02,       # x-coordinate for the anchor point (1 means top of the x axis in the figure) 
                        #     yanchor="bottom",
                        #     xanchor="left",
                        #     font=dict(size= 10),
                        # ),
                        margin=dict(l=0, r=0, t=25, b=0, autoexpand = True),
                        )

    return fig_pie_aux


# callback to update waterfall chart
@app.callback(
    Output("graph-figure-waterfall","figure"), 
    [
        Input('graph-figure','relayoutData'),
        Input('graph-figure','figure'),
    ]
)
def display_pie_chart_contribution(relayout_data, figure_val):
    # Retrieve initial and end dates
    try:
        ini_date = pd.to_datetime(figure_val['layout']["xaxis"]["range"][0], format = '%Y-%m-%d %H:%M:%S.%f')
        end_date = pd.to_datetime(figure_val['layout']["xaxis"]["range"][1], format = '%Y-%m-%d %H:%M:%S.%f')

        # If the date displayed in the zoomed figure is grater than the range of data, fix it
        if ini_date < df.index.min():
            ini_date = df.index.min()
        if end_date > df.index.max():
            end_date = df.index.max()

        output_range = f"{ini_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
    except (KeyError, TypeError):
        # output_range = f'None'
        output_range = f"{df.index.min().strftime('%Y-%m-%d')} - {df.index.max().strftime('%Y-%m-%d')}"
    date_range_str = output_range.split(' - ')
    ini_date_filt = pd.to_datetime(date_range_str[0], format = '%Y-%m-%d')
    end_date_filt = pd.to_datetime(date_range_str[1], format = '%Y-%m-%d')


    #Filter weights
    df_weights_filt = df_weights.loc[(df_weights.index >= ini_date_filt)&(df_weights.index <= end_date_filt),:].copy()
    # print(df_weights_filt)
    
    #Generate pie charts with TAA
    values_pie = df_weights_filt.iloc[-1,:].values

    # fig_pie_aux = make_subplots(rows=1, cols=2, specs=[[
    #         {'type':'domain'},# type domain for pie charts
    #         {'type':'bar'},# type domain for bar charts
    #     ]],
    #     subplot_titles=('Allocation', 'Contribution'),
    # )
    fig_pie_aux = go.Figure()

    # Add pie chart

    # # Get last 3M contribution
    # df_weights_3M = df_weights_filt.copy()# .loc[(-21*3):,:].copy()
    df_contrib_returns_assets = (
        df_weights_3M*df_asset_ret
    ).dropna()

    asset_contrib_returns = df_contrib_returns_assets.sum(axis = 0).values

    # Add bar chart
    name_aux = f'{df_contrib_returns_assets.index.min():%Y-%m-%d} - {df_contrib_returns_assets.index.max():%Y-%m-%d}'
    fig_pie_aux.add_trace(
        go.Waterfall(
            x = df_contrib_returns_assets.columns,
            y = asset_contrib_returns,
            measure = ['relative']*len(labels_pie),
            showlegend = False,
            name = name_aux,
            connector = {"line":{"color":"rgb(0, 0, 0)", 'width':5}},
        )#,1,2,
    )

    # Update dates in subtitles
    # dates_names = {
    #         'Allocation':f'{df_weights_filt.index.max():%Y-%m-%d}',
    #         'Contribution':f'{df_contrib_returns_assets.index.min():%Y-%m-%d} - {df_contrib_returns_assets.index.max():%Y-%m-%d}',
    #     }
    # fig_pie_aux.for_each_annotation(lambda a: a.update(text = a.text + ' (' + dates_names[a.text] + ') <br>'))

    # https://stackoverflow.com/questions/66064756/how-to-change-subplot-size-and-decrease-space-between-subplot
    # fig_pie.layout.annotations[0].update(y=1.05, x = 0)
    # fig_pie.layout.annotations[1].update(y=1.05, x = 0)

    # fig_pie.update_layout(margin=dict(t=20000), autosize=False,)

    # fig_pie.update_layout(title_pad=dict(t=20), margin=dict(pad=15))

    fig_pie_aux.update_layout(yaxis={'tickformat': ',.1%'},title = f'{df_contrib_returns_assets.index.min():%Y-%m-%d} - {df_contrib_returns_assets.index.max():%Y-%m-%d}')

    fig_pie_aux.update_layout(
                        template='plotly_dark',
                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                        autosize = True,
                        # legend = None,
                        # legend=dict(
                        #     orientation="h",
                        #     y=1.05,      # y-coordinate for the anchor point (1 means top of the y axis in the figure)
                        #     x=0.0,       # x-coordinate for the anchor point (1 means top of the x axis in the figure) 
                        #     # y=.9,      # y-coordinate for the anchor point (1 means top of the y axis in the figure)
                        #     # x=0.02,       # x-coordinate for the anchor point (1 means top of the x axis in the figure) 
                        #     yanchor="bottom",
                        #     xanchor="left",
                        #     font=dict(size= 10),
                        # ),
                        margin=dict(l=0, r=0, t=25, b=0, autoexpand = True),
                        )

    return fig_pie_aux



if __name__ == "__main__":
    app.run(
       # debug=True, # Do not use in production!
    )
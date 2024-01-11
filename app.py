from dash import Dash, dcc, html, Input, Output, callback
import os
import json
from urllib.request import urlopen
import plotly.express as px
import pandas as pd
import geopandas as gpd
import dash_bootstrap_components as dbc
import dash
from itertools import product 
import datetime

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server


accidents_data = pd.read_csv("./data/accidents_by_zone.csv", index_col=[0])
accidents_data['index'] = accidents_data.index

arrondis_gdf = gpd.read_file("./data/arrondissements.geojson")

zones_gdf = gpd.read_file("./data/zones.geojson")

accidents_paris_ll = gpd.read_file("./data/accidents_paris_ll.geojson")
lats = accidents_paris_ll.get_coordinates().y.to_list()
lons = accidents_paris_ll.get_coordinates().x.to_list()

months_order = ['janvier', 'février', 'mars', 'avril',
          'mai', 'juin', 'juillet', 'août',
          'septembre', 'octobre', 'novembre', 'décembre']


with open("./data/zones.json") as response:
    zones = json.load(response)

# APP Layout

SIDEBAR_STYLE = {
    "left": 0,
    "top": 0,
    "align": "left",
    "padding": "2rem 1rem",
}



available_years = [years for years in accidents_paris_ll['an'].unique()]

year_month_dict = {}

for year in available_years:
    months = accidents_paris_ll[accidents_paris_ll['an'] == year]['mois'].unique().tolist()
    year_month_dict[year] = [months for _, months in sorted(zip(months_order, months))]

names = list(year_month_dict.keys())

controls = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Year"),
                dcc.Dropdown(
                    id='year_dropdown',
                    options=[{'label':year, 'value': year} for year in available_years],
                    multi=False,
                    clearable=False,
                    value = list(year_month_dict.keys())[0],
                    style={"width": "40%","height": "100%"}
                ),
                dbc.Label("Month"),
                dcc.Dropdown(
                    id='month_dropdown',
                    multi=False,
                    clearable=False,
                    style={"width": "40%","height": "100%"}
                ),
            ], style={'top': '0'}
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Gravity"),
                dbc.Checklist(
                    id='radio_gravity', 
                    options=["1", "2", "3","4"],
                    value="2",
                    inline=True
                ),
                dbc.Label("Show accident's locations ?"),
                # Switch to enable/disable accidents point layer
                dbc.Checklist(
                    options=[
                        {"label": "Enable / Disable", "value": True},
                    ],
                    value=False,
                    id="accidents_switch",
                    switch=True
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Cluster count"),
                dbc.Input(id="cluster-count", type="number", value=3),
            ]
        ),
    ],
    body=True,
    style={'height': "100%"}
)


app.layout = dbc.Container(
    [
        html.H1("Bicycle accidents in Paris", style={'textAlign': 'center'}),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=5),
                dbc.Col(dcc.Graph(id="map_graph"), md=7),
            ],
            align="top",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="month_graph"), md=6),
                dbc.Col(dcc.Graph(id="hours_graph"), md=6),

            ]
        ),
    ],
    fluid=True,
)

@app.callback(
    Output('month_dropdown', 'options'),
    [Input('year_dropdown', 'value')]
)
def update_date_dropdown(selected_year):
    months_dict = {'janvier': 0, 'février': 1, 'mars': 2, 'avril': 3, 'mai': 4, 'juin': 5, 'juillet': 6, 
    'août': 7, 'septembre': 8, 'octobre': 9, 'novembre': 10, 'décembre': 11}
    ordered_month_list = sorted(year_month_dict[selected_year], key=lambda x: months_dict[x.lower()])

    return [{'label': i, 'value': i} for i in ordered_month_list]

@callback(
    Output('month_dropdown', 'value'),
    Input('month_dropdown', 'options'))
def update_month_drowdown(available_options):
    return available_options[0]['value']


# Callback functions
@app.callback(
    [Output(component_id="month_graph", component_property="figure"),
     Output(component_id="hours_graph", component_property="figure")],
    [Input("month_dropdown", "value"),
     Input("year_dropdown", "value")])
def months_and_hours_graph(month,year):
    selected_year = year if year else 2011
    selected_month = month if month else year_month_dict[selected_year][0]

    print("Selected year and month : " ,selected_year,selected_month)
    # Filter year
    accidents_paris_ll_by_year = accidents_paris_ll.loc[(accidents_paris_ll['an'] == int(selected_year))]
    # Filter month
    accidents_paris_ll_by_year_month = accidents_paris_ll.loc[(accidents_paris_ll['an'] == int(selected_year)) & (accidents_paris_ll['mois'] == selected_month)]
    # Group and sort accidents by month
    accidents_paris_sorted_month = accidents_paris_ll_by_year.sort_values('mois', ascending=True).groupby('mois').size().reset_index(name ='Accidents')
    accidents_paris_sorted_month_grouped = accidents_paris_sorted_month.sort_values('mois', key=lambda s: s.apply(months_order.index), ignore_index=True)
    # Group and sort accidents by hour
    accidents_paris_grouped_hour = accidents_paris_ll_by_year_month.sort_values('hour', ascending=True).groupby('hour').size().reset_index(name ='Accidents')
    
    months_graph = px.line(accidents_paris_sorted_month_grouped, x='mois', y='Accidents')
    months_graph.update_layout(
        title={
            'text': "Accidents in <b>" + str(selected_year) + "</b>",
            'y':0.94,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    hours_graph = px.line(accidents_paris_grouped_hour, x='hour', y='Accidents')
    hours_graph.update_layout(
        title={
            'text': "Accidents in <b>" + selected_month + "</b> per day hour",
            'y':0.94,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    
    return months_graph, hours_graph
    


@app.callback(
     Output("map_graph", "figure"),
     [Input("month_dropdown", "value"),
     Input("accidents_switch", "value")],
 )
def display_selected_data(points_month,accidents_switch):
    color_continuous_scale=["green", "yellow", "orange","red"]
    #month = selected_month if selected_month else "janvier"
    #points_month = zones_gdf.query("month_list == @month_list")
    #if selectedData:
    #    indices = [point["customdata"][0] for point in selectedData["points"]]
    #    points_month = points_month.loc[indices]
    fig = px.choropleth_mapbox(accidents_data, geojson=zones_gdf,
                        locations="index",
                        labels={"num_acc_by_area": "Accidents"},
                        color_continuous_scale=color_continuous_scale,
                        color='num_acc_by_area',
                        zoom=11, center = {"lat": 48.85848828830715, "lon": 2.351379571148244},
                        template='seaborn',
                        mapbox_style="open-street-map",
                        opacity=0.6,
                        hover_name=None,
                        #animation_frame="grav_mean",
                        hover_data={'index':False})            
    fig.update_geos(fitbounds="locations")
    fig.update_traces(marker_line_width = 1, marker_line_color = 'black')
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    if accidents_switch:
        fig.add_scattermapbox(lat=lats, lon=lons, marker_size=6, marker_color='rgb(0, 0, 0)',opacity=0.3,hoverinfo = "skip")
    
    return fig


if __name__ == "__main__":
    app.run(debug=True)

from dash import Dash, dcc, html, Input, Output, callback
import json
import plotly.express as px
import pandas as pd
import geopandas as gpd
import dash_bootstrap_components as dbc
import calendar
from flask import Flask
import waitress

external_stylesheets = [dbc.themes.BOOTSTRAP]


flask_app = Flask(__name__)
dash_app = Dash(__name__, server=flask_app, 
                external_stylesheets=external_stylesheets)
#server = flask_app.server

accidents_data = pd.read_csv("accidents_by_zone.csv", index_col=[0])
accidents_data['index'] = accidents_data.index

arrondis_gdf = gpd.read_file("arrondissements.geojson")

zones_gdf = gpd.read_file("zones.geojson")

accidents_paris_ll = gpd.read_file("accidents_paris_ll.geojson")
lats = accidents_paris_ll.get_coordinates().y.to_list()
lons = accidents_paris_ll.get_coordinates().x.to_list()

months_order_fr = ['janvier', 'février', 'mars', 'avril',
          'mai', 'juin', 'juillet', 'août',
          'septembre', 'octobre', 'novembre', 'décembre']
months_dict = {'janvier': 0, 'février': 1, 'mars': 2, 'avril': 3, 'mai': 4, 'juin':5, 'juillet': 6, 
'août': 7, 'septembre': 8, 'octobre': 9, 'novembre': 10, 'décembre': 11}

month_names_en = [calendar.month_name[i] for i in range(1, 13)]

ordered_month_list_translated = {month_names_en[i]:month_fr for i,month_fr in enumerate(months_order_fr)}

with open("./zones.json") as response:
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
    year_month_dict[year] = [months for _, months in sorted(zip(months_order_fr, months))]

# for year, month_list in year_month_dict.items():
#    new_month_list = []
#    for month in month_list:
#        new_month_list.append(months_fr_en[month])
#    year_month_dict[year] = new_month_list

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
                    style={"width": "70%"}
                ),
                html.Br(),
                dbc.Label("Month"),
                dcc.Dropdown(
                    id='month_dropdown',
                    multi=False,
                    clearable=False,
                    style={"width": "70%"}
                ),
                html.Br(),
                dbc.Label("Aggregate accidents by : "),
                dbc.RadioItems(
                    id="aggmap_radioitems",
                    options=[
                        {"label": "Total number", "value": 'num_acc_by_area'},
                        {"label": "Gravity", "value": 'grav_mean'},
                    ],
                value='num_acc_by_area',
                ),
                html.Br(),
                dbc.Label("Show accident's locations ?"),
                # Switch to enable/disable accidents point layer
                dbc.Checklist(
                    options=[
                        {"label": "Enable / Disable", "value": True},
                    ],
                    value=False,
                    id="accidents_switch",
                    switch=True
                )
            ]
        )
    ],
    body=True,
    style={"height": "100%"}
),

dash_app.layout = dbc.Container(
    [
        html.H1("Bicycle accidents in Paris", style={'textAlign': 'center'}),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=3),
                dbc.Col(dcc.Graph(id="map_graph"), md=9),
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

@dash_app.callback(
    Output('month_dropdown', 'options'),
    [Input('year_dropdown', 'value')]
)
def update_date_dropdown(selected_year):
    ordered_month_list = sorted(year_month_dict[selected_year], key=lambda x: months_dict[x.lower()])
    # Translate to english
    translated_month_list = {month_names_en[months_dict[month_fr]]: month_fr for month_fr in ordered_month_list}
    return [{'label': en, 'value': fr} for en, fr in translated_month_list.items()]

@callback(
    Output('month_dropdown', 'value'),
    Input('month_dropdown', 'options'))
def update_month_drowdown(available_options):
    return available_options[0]['value']
waitress

# Callback functions
@dash_app.callback(
    [Output(component_id="month_graph", component_property="figure"),
     Output(component_id="hours_graph", component_property="figure")],
    [Input("month_dropdown", "value"),
     Input("year_dropdown", "value")])
def months_and_hours_graph(month, year):
    selected_year = year if year else 2011
    selected_month = month if month else year_month_dict[selected_year][0]
    selected_month_en = month_names_en[months_dict[month]]
    print("Selected year and month(en) and month (fr) : " , selected_year, selected_month, selected_month_en)
    # Filter year
    accidents_paris_ll_by_year = accidents_paris_ll.loc[(accidents_paris_ll['an'] == int(selected_year))]
    # Filter month
    accidents_paris_ll_by_year_month = accidents_paris_ll.loc[(accidents_paris_ll['an'] == int(selected_year)) & (accidents_paris_ll['mois'] == selected_month)]
    # Group and sort accidents by month
    accidents_paris_sorted_month = accidents_paris_ll_by_year.sort_values('mois', ascending=True).groupby('mois').size().reset_index(name ='Accidents')
    accidents_paris_sorted_month_grouped = accidents_paris_sorted_month.sort_values('mois', key=lambda s: s.apply(months_order_fr.index), ignore_index=True)
    # Group and sort accidents by hour
    accidents_paris_grouped_hour = accidents_paris_ll_by_year_month.sort_values('hour', ascending=True).groupby('hour').size().reset_index(name ='Accidents')
    
    months_graph = px.line(accidents_paris_sorted_month_grouped, x='mois', y='Accidents')
    months_graph.update_layout(
        title={
            'text': "Accidents in <b>" + str(selected_year) + "</b>",
            'y': 0.94,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    hours_graph = px.line(accidents_paris_grouped_hour, x='hour', y='Accidents')
    hours_graph.update_layout(
        title={
            'text': "Accidents in <b>" + selected_month_en + "</b> by hour of day",
            'y': 0.94,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    
    return months_graph, hours_graph
    


@dash_app.callback(
     Output("map_graph", "figure"),
     [Input("month_dropdown", "value"),
     Input("accidents_switch", "value"),
     Input("aggmap_radioitems", "value")],
 )
def display_selected_data(points_month,accidents_switch,agg_data_radioitem):
    color_continuous_scale=["green", "yellow", "orange","red"]
    print("Selected agg map : ", agg_data_radioitem)
    label_hover = "Accidents" if agg_data_radioitem == 'num_acc_by_area' else "Gravity"
    fig = px.choropleth_mapbox(accidents_data, geojson=zones_gdf,
                        locations="index",
                        labels={agg_data_radioitem: label_hover},
                        color_continuous_scale=color_continuous_scale,
                        color=agg_data_radioitem,
                        zoom=11, center={"lat": 48.85848828830715, "lon": 2.351379571148244},
                        template='seaborn',
                        mapbox_style="open-street-map",
                        opacity=0.6,
                        hover_name=None,
                        #animation_frame="grav_mean",
                        hover_data={'index': False})            
    fig.update_geos(fitbounds="locations")
    fig.update_traces(marker_line_width = 1, marker_line_color = 'black')
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    if accidents_switch:
        fig.add_scattermapbox(lat=lats, lon=lons, marker_size=6, marker_color='rgb(0, 0, 0)', opacity=0.3, hoverinfo = "skip")

    return fig


def create_app():
    return flask_app


if __name__ == "__main__":
    from waitress import serve
    serve(flask_app, host="app2.ouicodedata.com", port=80)

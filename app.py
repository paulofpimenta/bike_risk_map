from dash import Dash, dcc, html, Input, Output, callback
import os
import json
from urllib.request import urlopen
import plotly.express as px
import pandas as pd
import geopandas as gpd
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server


accidents_data = pd.read_csv("./data/accidents_by_zone.csv", index_col=[0])
accidents_data['index'] = accidents_data.index
print(accidents_data)
arrondis_gdf = gpd.read_file("./data/arrondissements.geojson")

zones_gdf = gpd.read_file("./data/zones.geojson")
paris_accidents_ll = gpd.read_file("./data/paris_accidents.geojson")
month_list = paris_accidents_ll.an.unique()
# zones_df = pd.read_csv()
# 48.86260697107489, 2.346794478444374
with open("./data/zones.json") as response:
    zones = json.load(response)

print(month_list)
# APP Layout
#'' 'mai' '' '' 'janvier' 'mars' '' 'juin'
# 'août' '' '' 'décembre'

SIDEBAR_STYLE = {
    "left": 0,
    "top": 0,
    "align": "left",
    "padding": "2rem 1rem",
}


controls = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("X variable"),
                dcc.Dropdown(
                    id="selected_month",
                    options=[
                        {"label": "January", "value": "janvier"},
                        {"label": "February", "value": "février"},
                        {"label": "Mars", "value": "mars"},
                        {"label": "April", "value": "avril"},
                        {"label": "May", "value": "mai"},
                        {"label": "June", "value": "juin"},
                        {"label": "July", "value": "juillet"},
                        {"label": "August", "value": "août"},
                        {"label": "September", "value": "septembre"},
                        {"label": "October", "value": "octobre"},
                        {"label": "November", "value": "novembre"},
                        {"label": "December", "value": "décembre"},
                    ],
                    multi=False,
                    value="janvier",
                    style={"width": "40%"}
                ),
            ],
        ),
        html.Div(
            [
                dbc.Label("Y variable"),
                dcc.RadioItems(
                    id='year', 
                    options=["Joly", "Coderre", "Bergeron"],
                    value="Coderre",
                    inline=True
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Cluster count"),
                dbc.Input(id="cluster-count", type="number", value=3),
            ]
        ),
    ],
    body=True,
)

app.layout = dbc.Container(
    [
        html.H1("Bicycle accidents in Paris"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=5),
                dbc.Col(dcc.Graph(id="graph2"), md=7),
            ],
            align="center",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="graph1"), md=6),

            ]
        ),
    ],
    fluid=True,
)
    

# Callback functions
@app.callback(Output("graph1", "figure"), [Input("selected_month", "value")])
def cb(selected_month):
    month = selected_month if selected_month else "janvier"
    
    # Group and sort accidents by month
    #accidents_paris_grouped = accidents_paris.sort_values('mois', ascending=True).groupby('mois').size().reset_index(name ='Accidents')
    #accidents_paris_sorted_grouped = accidents_paris_grouped.sort_values('mois', key=lambda s: s.apply(months_order.index), ignore_index=True)

    # Group and sort accidents by hour
    #accidents_paris_grouped_hour = accidents_paris.sort_values('hour', ascending=True).groupby('hour').size().reset_index(name ='Accidents')
    
    points_month_selected = paris_accidents_ll[paris_accidents_ll["mois"] == month]
    points_years_of_selected_months = points_month_selected[['an']]
    grouped_by_year = points_years_of_selected_months.groupby(["an"]).agg({'an': "sum"}).rename(columns={"an": "total"})
    print(grouped_by_year)
    #points_accidents_sum_month = points_month_selected[paris_accidents_ll["mois"] == month]

    return px.line(grouped_by_year, title='Accidents per year')


@app.callback(
     Output("graph2", "figure"),
     [Input("selected_month", "value")],
     #[Input("graph", "selectedData")],
 )
def display_selected_data(points_month):
    #month = selected_month if selected_month else "janvier"
    #points_month = zones_gdf.query("month_list == @month_list")
    #if selectedData:
    #    indices = [point["customdata"][0] for point in selectedData["points"]]
    #    points_month = points_month.loc[indices]
    print(accidents_data)
    fig = px.choropleth_mapbox(accidents_data, geojson=zones_gdf,
                        locations="index",
                        labels={"num_acc_by_area": "number of accidents"},
                        color_continuous_scale="Viridis",
                        color='num_acc_by_area',
                        mapbox_style="carto-positron",
                        zoom=11, center = {"lat": 48.85848828830715, "lon": 2.351379571148244},
                        template='seaborn',
                        opacity=0.6,
                        hover_name="index", 
                        hover_data={'index':False})
                        
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(legend_xref="paper")
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_coloraxes(cauto=False)
    fig.update_coloraxes(colorbar_ypad=0)
    return fig


if __name__ == "__main__":
    app.run(debug=True)

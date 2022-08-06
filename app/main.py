#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import plotly.express as px
from dash import html, dcc
import dash
from dash.dependencies import Input, Output, State
import json
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from jupyter_dash import JupyterDash
import datashader as ds
from datashader import transfer_functions as tf
from colorcet import bgy, fire, blues, CET_L18, dimgray, kgy, rainbow, bmy
from datashader.colors import colormap_select, viridis
from pyproj import Transformer
from functools import partial
cm = partial(colormap_select)
from functions import * 
import plotly.graph_objects as go
import dash_auth
import geopandas as gpd
from datetime import datetime
import pymysql
from sqlalchemy import create_engine


# In[2]:


#df_analitica = pd.read_csv('data/properties_202207311801.csv')
#map_df = gpd.read_file('data/ign_departamento/ign_departamento.shp', encoding='utf-8')

host='database-3.cyb8qmtjr8ga.us-east-1.rds.amazonaws.com'
port=int(3306)
user="admin"
passw="chachechichochu"
database="db_cde"
table = 'properties'
engine = create_engine('mysql+pymysql://' + user + ':' + passw + '@' + host + ':' + str(port) + '/' + database , echo=False)
df_analitica = pd.read_sql_table(
    table,
    con=engine
)

map_df = gpd.read_file("s3://cde-m1/resources/ign_departamento/ign_departamento.shp")


# In[3]:


pot_cmap = bmy
def_mapbox_style = "carto-positron" 
metric = "property_type_med_price_m2"
init_zoom = 4
init_lon, init_lat, default_position = init_position(df_analitica, init_zoom)
#mapboxt = 'pk.eyJ1IjoibmlzYW50aWwiLCJhIjoiY2pnNTlyem5xN2hvMDMzczJjbDlncTA5ZSJ9.G4poDRUAwKLBYoKHaSlw7A'

# In[6]:


init_fig = build_base_map(df_analitica[:1], default_position, mapbox_style=def_mapbox_style)


# In[7]:

#df_analitica = df_analitica[(df_analitica.operation_type=='Venta') & (df_analitica.currency=='USD') & (df_analitica.l1=='Argentina')]
df_analitica['Provincia'] = df_analitica.Provincia.fillna('Sin Agrupar')
df_analitica['Localidad'] = df_analitica.Localidad.fillna('Sin Agrupar')
l_rubro = list(df_analitica.property_type.unique())
l_rubro = list(filter(None, l_rubro))
l_rubro.insert(0, "All")
l_provincia = list(df_analitica.Provincia.unique())
l_provincia.insert(0, "All")
l_localidad = list(df_analitica.Localidad.unique())
l_localidad.insert(0, "All")
header_class = "pl-3 pt-3 mt-2 pb-1 border-bottom border-dark"
subheader_class = "ml-3 mt-3 text-dark"

LOGO = "assets/timit.png"
loading_style = {'align-self': 'center'}

header = html.Div(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Div(
                html.Img(src=LOGO, height="40px"),  style={"float": "left"}), width=3),
            dbc.Col(html.Div(
                html.H3([
                    "GIS".upper(),
                    html.I(className="fas fa-regular fa-map-location-dot"),
                    " establecimientos Prisma".upper()
                ], style={'textAlign': 'center'}))
            ),
            dbc.Col([
                html.Small(["Created by Nikoloide"],className="py-2 px-3 text-light")
            ], style={'textAlign': 'right'},  width=3),
        ], justify='center'),

        html.P([
            html.Span("La Visualizacion permite explorar las propiedades en Venta"),
           # html.I(className="fas fa-regular fa-store"),
            html.Span(" en base su la ubicación geográfica. "),
            html.Span("Esto nos permite detectar áreas de oportunidad en base a la performance analizada"),
        ], className="lead mb-1"),
    ]), className="bg-dark text-light py-3"
)


card_icon = {
    "color": "white",
    "textAlign": "center",
    "fontSize": 30,
    "margin": "auto",
}


kpi = dbc.Row([
        dbc.Col([dbc.CardGroup(
    [
            dbc.Card(
            [
                dbc.CardHeader("Propiedades"),
                dbc.CardBody(
                    [
                        html.H4(children='Propiedades', id="q_terminales", className="card-title"),
                        html.P("En el área seleccionada.", className="card-text"),
                    ]
            )],style={"width": "25rem"}),
                dbc.Card(
                    html.Div(className="fa fa-store", style=card_icon),
                    className="bg-dark",
                    style={"maxWidth": 75},
             ),
            ])],className="mt-4"),
        dbc.Col([dbc.CardGroup(
    [
            dbc.Card(
            [
                dbc.CardHeader("Precio del m2 Promedio"),
                dbc.CardBody(
                    [
                        html.H4(children='Precio Promedio', id="avg_monto", className="card-title"),
                        html.P("En el área seleccionada", className="card-text"),
                    ]
            )],style={"width": "25rem"}),
                dbc.Card(
                    html.Div(className="fa fa-credit-card", style=card_icon),
                    className="bg-dark",
                    style={"maxWidth": 75},
             ),
            ])],className="mt-4"),
        dbc.Col([dbc.CardGroup(
    [
            dbc.Card(
            [
                dbc.CardHeader("Precio Promedio"),
                dbc.CardBody(
                    [
                        html.H4(children='m2 Promedio', id="tkt_avg", className="card-title"),
                        html.P("En el área seleccionada", className="card-text"),
                    ]
            )],style={"width": "25rem"}),
                dbc.Card(
                    html.Div(className="fa fa-receipt", style=card_icon),
                    className="bg-dark",
                    style={"maxWidth": 75},
             ),
            ])],className="mt-4"),
]) 

body = html.Div([dbc.Container(
    [   

        dbc.Row([html.H5("Summary".upper(), className=header_class),
            #     dbc.Row([dcc.Markdown('''
            # #### 
            # **Segmento: **Small y Micro-Merchants - **Período: **Q1-2022
            # ''')]),
            dbc.Row([
                html.P([
                    html.I(className="fas fa-regular fa-store"),
                    " Segmento: ".upper(),
                    " Propiedades en Venta"
                ])]),
            dbc.Row([
                html.P([
                    html.I(className="fas fa-regular fa-calendar"),
                    " Periodo: ".upper(),
                    " Q3-2022"
                ])]),
                kpi]),
            dbc.Row([dcc.Markdown('''
    #### 
    **Intrucciones** Realizar zoom sobre la zona geográfica de interés, aguardar y se adaptaran los valores de la capa del mapa y se ajustaran los Kpi's y graficos del dashboard
    ''')]),
        dbc.Row([
                dbc.Col([
                    html.H5("Explorer".upper(), className=header_class),
                    dbc.Card(
                        [
                            # dbc.CardHeader([
                            #     dbc.Row([
                            #         dbc.Col([
                            #             html.H3("Legend".upper(), className=header_class),
                            #             dbc.Card([
                            #                 dbc.CardBody(
                            #                     build_legend(scale_min=0, scale_max=100),
                            #                     className="p-1 m-1", id="legends-card"
                            #                 ),
                            #             ])])
                            #     ])
                            # ], className="py-0"),

                            dbc.CardBody([
                                dcc.Loading(id='loading', 
                                children=[dash.dcc.Graph(id="fig_map", figure=init_fig)],type='circle', fullscreen=False),#, parent_style=loading_style),
                                dash.html.Div(
                                    id="debug_container",
                                ),
                                dash.dcc.Store(
                                    id="points-store",
                                    data={
                                        "lat": [],
                                        "lon": [],
                                    },
                                ),
                                dcc.Store(id='intermediate-value'),
                                ]),  
                             #html.Div([
                             #   html.P("...")
                            #], className="p-1 m-1", id="map-notes") 
                        ]
                    ),
                    
                ]
            ),
                dbc.Col([
                    dbc.Row([
                        html.H5("Legend".upper(), className=header_class),
                        dbc.Card([
                            dbc.CardBody(
                                build_legend(scale_min=0, scale_max=100),
                                className="p-1 m-1", id="legends-card"
                            ),
                        ])]),
                    dbc.Row(
                        [html.H5("Filters".upper(), className=header_class),
                        html.P()
                        ,html.H5('Provincia')
                        , dcc.Dropdown(id = 'provincia-drop'
                            ,options=[
                                {'label': i, 'value': i} for i in l_provincia],
                            value=['All'],
                            multi=True
                        )] + [ dbc.Tooltip(f"Seleccionar primero Provincia para filtrar Localidad y/o CP", target="provincia-drop")
                        ]),
                    html.Br(),
                    dbc.Row(
                        [html.H5('Localidad')
                        , dcc.Dropdown(id = 'localidad-drop',
                            value=['All'],
                            multi=True,
                            searchable=True,
                            options=[{'label': k, 'value': k} for k in sorted(df_analitica['Localidad'].astype(str).unique())] +
                                [{'label': 'All', 'value': 'All'}],
                        )]+ [ dbc.Tooltip(f"Localidades de Provincia seleccionada", target="localidad-drop")
                        ]),
                    html.Br(),
                    dbc.Row(
                        [html.H5('Barrio')
                        , dcc.Dropdown(id = 'cp-drop',
                            value=['All'],
                            multi=True,
                            searchable=True,
                            options=[{'label': k, 'value': k} for k in sorted(df_analitica['id'].astype(str).unique())] +
                                [{'label': 'All', 'value': 'All'}],
                        )]+ [ dbc.Tooltip(f"Seleccionar primero Provincia para filtrar CP", target="cp-drop")
                        ]),
                    html.Br(),
                    dbc.Row(
                        [html.H5('Tipo Propiedad')
                        , dcc.Dropdown(id = 'rubro-drop'
                            ,options=[
                                {'label': i, 'value': i} for i in l_rubro],
                            value=['All'],
                            multi=True
                        )]),
                    html.Br(),
                    dbc.Row(
                        [html.H5('Layer'),
                        dbc.Container(
    [
                        dbc.RadioItems(id="layer-checklist",
                            options=[
                            {'label': 'Precio_m2', 'value': 'property_type_med_price_m2', "label_id": "property_type_med_price_m2"},
                            {'label': 'Precio', 'value': 'property_type_med_price', "label_id": "property_type_med_price"},
                            {'label': 'Superficie', 'value': 'property_type_med_surface_total', "label_id": "property_type_med_surface_total"},
                        ], labelStyle = dict(display='block'),
                        value='property_type_med_price_m2'
                        ) ]
                     +  [ dbc.Tooltip(f"Facturación media del período de análisis", target="Precio_m2"),
                          dbc.Tooltip(f"Score en base a su nivel de facturación, probabilidad de Churn y de adquisición de CA", target="m"),
                          dbc.Tooltip(f"Probabilidad de inactividad para los proximos 3 meses ", target="Superficie")
                        ])
                        ]
                    )
                ])
        ]),
    ]
),


    dbc.Container([
                    html.H5("Charts".upper(), className=header_class),
                    dbc.Row([

                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H6("Localidad: Cantidad y facturación promedio", className="card-title mb-0 mt-0")),
                                dbc.CardBody(dash.dcc.Graph(figure=px.scatter(height=500), id="grid-localidad"))
                            ])
                        ], className="col-sm-12 col-md-6"),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H6("Tipo de Propiedad", className="card-title mb-0 mt-0")),
                                dbc.CardBody([dash.dcc.Graph(figure=px.scatter(height=500), id="pie-segmento")]),
                            ], outline=True),
                        ], className="col-sm-12 col-md-6"),
                   ]),
                   html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H6("Distribución de Precio m2", className="card-title mb-0 mt-0")),
                                dbc.CardBody([dash.dcc.Graph(figure=px.scatter(height=500), id="histogram-potentials")]),
                            ], outline=True),
                        ], className="col-sm-12 col-md-6"),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H6("Propiedad: Cantidad y facturación promedio", className="card-title mb-0 mt-0")),
                                #dbc.CardBody(dash.dcc.Graph(figure=px.scatter(height=500), id="grid-rubro"))
                            dbc.Tabs([
                                dbc.Tab(label='Propiedad', children=[
                                dcc.Graph(figure=px.scatter(height=490), id="grid-rubro")]),
                                dbc.Tab(label='Rooms', children=[
                                    dcc.Graph(figure=px.scatter(height=490), id="tree-terminal")]),
                                ])
                            ])
                        ], className="col-sm-12 col-md-6"),
                    ]),
                    html.Br(),
                    dbc.Row([
                        #dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H6("Superficie", className="card-title mb-0 mt-0")),
                                #dbc.CardBody([dash.dcc.Graph(figure=px.scatter(height=500), id="")]),
                            dbc.Tabs([
                                dbc.Tab(label='Habitaciones', children=[
                                dcc.Graph(figure=px.scatter(height=490), id="pie-terminal")]),
                                dbc.Tab(label='Superficie por Propiedad', children=[
                                    dcc.Graph(figure=px.scatter(height=490), id="hist-terminal")]),
                                ])
                             ])
                        #], className="col-sm-12 col-md-6")
                    ]),
                    html.Br(),
                    dbc.Container(
                        [dbc.Button(id="btn_csv", 
                            children=[html.I(className="fa fa-download mr-1"), "Download"],
                            color="light",
                            className="mt-1",
                            n_clicks=0
                            ), dcc.Download(id="download-dataframe-csv"),
                    ],),
                    html.Br(),

    ])
,
###This container is to callbacks, not show
    dbc.Container([
                    dbc.Row(
                                    [html.P(id="prev-zoom", children=init_zoom)], style={"display": "none"}
                                ),
                    dbc.Row(
                                    [html.P(id="prev-center", children=json.dumps([init_lon, init_lat]))],
                                    style={"display": "none"},
                                ),
                    dbc.Row(
                                    [html.P(id="relayout-text-old"), html.Span("", id="placeholder")],
                                    style={"display": "none"},
                                ),])
])

# Build App
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    {
        'href': 'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-50oBUHEmvpQ+1lW4y57PTFmhCaXp0ML5d60M1M7uH2+nqUivzIebhndOJK28anvf',
        'crossorigin': 'anonymous'
    }
])

app.title = 'GIS Agentes'
auth = dash_auth.BasicAuth(
    app,
    {'prismamp':'invitado'})

app.layout = html.Div([header, body])

def render_content(tab, ch1, ch2):
    if tab == 'tab-1-example-graph':
                tabla = html.Div([
                    dcc.Graph(figure=ch1)
                    ])
    elif tab == 'tab-2-example-graph':
                tabla = html.Div([
                    dcc.Graph(figure=ch2)
                    ])
    return tabla

#def generate_csv(df, n_clicks):
#    return dcc.send_data_frame(df.to_csv, filename="some_name.csv")

@app.callback(
    Output('localidad-drop', 'options'),
    Input('provincia-drop', 'value'))
def set_cities_options(selected_provincia):
    if selected_provincia != ['All']:
        raca_options = df_analitica[df_analitica['l2'].isin(selected_provincia)]
        #print(f'DEBUG 1.1: L1 options "NOT ALL": {raca_options}')
    else:
        raca_options = df_analitica
        #print(f'DEBUG 1.2: L1 options "ALL": {raca_options}')
    return [{'label': i, 'value': i} for i in sorted(np.append(raca_options['l3'].astype(str).unique(), 'All'))]

@app.callback(
    Output('cp-drop', 'options'),
    Input('provincia-drop', 'value'))
def set_cp_options(selected_localidad):
    if selected_localidad != ['All']:
        raca_options = df_analitica[df_analitica['l2'].isin(selected_localidad)]
        #print(f'DEBUG 1.1: L1 options "NOT ALL": {raca_options}')
    else:
        raca_options = df_analitica
        #print(f'DEBUG 1.2: L1 options "ALL": {raca_options}')
    return [{'label': i, 'value': i} for i in sorted(np.append(raca_options['l4'].astype(str).unique(), 'All'))]



@app.callback(
    [
        Output("legends-card", "children"),
        Output("fig_map", "figure"),
        Output('loading', 'parent_style'),
        Output("prev-center", "children"),
        Output("prev-zoom", "children"),
        Output("relayout-text-old", "children"),
        Output("histogram-potentials", "figure"),
        Output("grid-rubro", "figure"),
        Output("grid-localidad", "figure"),
        Output("pie-segmento", "figure"),
        Output('tree-terminal', 'figure'),
        Output('pie-terminal', 'figure'),
        Output('hist-terminal', 'figure'),
        Output('q_terminales', "children"),
        Output('avg_monto', "children"),
        Output('tkt_avg', "children"),
        Output('intermediate-value','data')
    ],
    [
        Input("fig_map", "relayoutData"),
        State("prev-center", "children"),
        State("prev-zoom", "children"),
        Input('rubro-drop', 'value'),
        Input('provincia-drop', 'value'),
        Input('localidad-drop', 'value'),
        Input('cp-drop', 'value'),
        Input('layer-checklist', 'value'),
    ],
)
def update_charts(relayout_data, prev_center_json, prev_zoom, rubros_l, provincia_l, l_localidad, l_cp, metric_raster):
    new_loading_style = loading_style
    prev_center = json.loads(prev_center_json)
    relayout_lon, relayout_lat, relayout_zoom = get_lon_lat_zoom(relayout_data, prev_center, prev_zoom)
    new_center = {"lon": relayout_lon, "lat": relayout_lat}

    if provincia_l==['All']:
        df_analitica_x = df_analitica
    elif provincia_l == ['']:
        df_analitica_x = df_analitica
    else:
        df_analitica_x = df_analitica[df_analitica.l2.isin(provincia_l)]

    if provincia_l==['All']:
        df_analitica_x = df_analitica_x
    elif provincia_l!=['All'] and l_localidad ==['All']:
        df_analitica_x = df_analitica_x
    elif l_localidad == '':
        df_analitica_x = df_analitica_x
    else:
        df_analitica_x = df_analitica_x[df_analitica_x.l3.isin(l_localidad)]

    if provincia_l==['All']:
        df_analitica_x = df_analitica_x
    elif provincia_l!=['All'] and l_localidad ==['All'] and l_cp==['All']:
        df_analitica_x = df_analitica_x
    elif l_cp == '':
        df_analitica_x = df_analitica_x
    else:
        df_analitica_x = df_analitica_x[df_analitica_x.l4.isin(l_cp)]

    if rubros_l==['All']:
        df_analitica_x = df_analitica_x
    elif rubros_l == '':
        df_analitica_x = df_analitica_x
    else:
        df_analitica_x = df_analitica_x[df_analitica_x.property_type.isin(rubros_l)]
    
    tmp_pot_df, points_a, raster, img_out, points, agg,  tmp_pot_df, agg_min, agg_max = update_layer(df_analitica_x,relayout_zoom, relayout_lon, relayout_lat, metric_raster)
 
    #Filter show boundaries states
    geo_df_z = gpd.GeoDataFrame(tmp_pot_df, geometry=gpd.points_from_xy(tmp_pot_df.nr_longitud, tmp_pot_df.nr_latitud))
    geo_df_z = geo_df_z.set_crs(epsg=4326, inplace=True)
    map_df_f = map_df
    inp, res = geo_df_z.sindex.query_bulk(map_df_f.geometry, predicate='contains') #{'contains', 'overlaps', 'intersects', 'covered_by', None, 'within', 'covers', 'touches', 'contains_properly', 'crosses'}
    map_df_f['intersects'] = np.isin(np.arange(0, len(map_df_f)), inp)
    map_df_f = map_df_f[map_df_f['intersects'] == True]

    mapbox_layers = list()
    mapbox_layers.append(img_out)

   #mapbox_layers.append(points)

    fig = build_base_map(tmp_pot_df[:1], default_position, mapbox_style=def_mapbox_style)
    fig.update_layout(
        mapbox_layers=mapbox_layers

    )


    if relayout_zoom > 8:
        layer = {
                    "source": json.loads(map_df_f.geometry.to_json()),
                    "below": "traces",
                    "type": "line",
                    "color": "grey",
                    "line": {"width": 1.1},
                }
        mapbox_layers.append(layer)
        fig.update_layout(
        mapbox_layers=mapbox_layers )
    else:
        pass



    position = {"zoom": relayout_zoom, "center": new_center}
    fig["layout"]["mapbox"].update(position)
    legends_div = []
    legend = build_legend(scale_min=agg_min, scale_max=agg_max, legend_title="legend", cmap=pot_cmap, dec_pl=0)
    legends_div.append(legend)

    q_terminales = tmp_pot_df.shape[0]
    avg_monto = tmp_pot_df.property_type_med_price_m2.median()
    tkt_avg = tmp_pot_df.property_type_med_price.median()
    hist_f = build_hist_vol(tmp_pot_df, 'property_type_med_price_m2', 20)
    over_rubro = build_bar_fig(tmp_pot_df, 'property_type', metric, metric,'property_type', 'h')
    over_localidad = build_bar_fig(tmp_pot_df, 'l3', metric, metric,'l3', 'h')
    over_segmento = build_pie_fig(tmp_pot_df, 'property_type')
    by_terminal = build_tree_fig(tmp_pot_df, 'rooms', metric)

    #tmp_pot_df['tipo_terminal_desc'] = np.where(tmp_pot_df.ds_tipo_terminal_active.str.contains('Newland', regex=False), 'Newland', 'Legacy')
    pie_over_terminal = build_pie_fig(tmp_pot_df, 'rooms')
    hist_terminal = build_hist_vol(tmp_pot_df, 'property_type_med_surface_total', 20, color = 'property_type', marginal="box")

    #tabla = render_content(tab, over_rubro,over_localidad)
    columns = ['l2','price_m2']
    df = tmp_pot_df[columns].head(10)

    return (
        legends_div,
        fig,
        new_loading_style,
        json.dumps([relayout_lon, relayout_lat]),
        relayout_zoom,
        json.dumps([relayout_data]),
        hist_f,
        over_rubro,
        over_localidad,
        over_segmento,
        by_terminal,
        pie_over_terminal,
        hist_terminal,
        f"{q_terminales:,.0f}",
        '$' + f"{round(avg_monto, 2):,.0f}",
        '$' + f"{round(tkt_avg, 2):,.0f}",
        df.to_json(date_format='iso', orient='records')
    )

@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("btn_csv", "n_clicks"),
    Input("intermediate-value", "data")],
    prevent_initial_call=False,
)

def func(n_clicks, df_json):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if "btn_csv" in changed_id:
        df_f = pd.read_json(df_json, orient='records')
        today = datetime.now()
        return dcc.send_data_frame(df_f.to_csv, "DS_Agencias_"+str(today.strftime("%d-%m-%Y %H:%M:%S"))+"_"+str(n_clicks)+".csv")


# app.run_server()
if __name__ == "__main__":
    #app.run_server(debug=True)
    app.run_server(host='0.0.0.0', port=80, debug=False)

 
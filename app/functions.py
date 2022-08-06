
import pandas as pd
from colorcet import bgy, fire, blues, CET_L18, dimgray, kgy, rainbow, bmy
import plotly.express as px
from dash import html
import numpy as np
from numpy import median
from jupyter_dash import JupyterDash
import datashader as ds
from pyproj import Transformer
from datashader import transfer_functions as tf
from functools import partial
from datashader.colors import colormap_select, viridis
from datashader import transfer_functions as tf, reductions as rd
import json
import plotly.graph_objects as go
import plotly.figure_factory as ff

cm = partial(colormap_select)
def_cmap = bmy
# Coordinate transformations
transformer_4326_to_3857 = Transformer.from_crs("epsg:4326", "epsg:3857")
transformer_3857_to_4326 = Transformer.from_crs("epsg:3857", "epsg:4326")


def init_position(df, init_zoom):
    init_zoom = init_zoom
    init_lon = df[:1].nr_longitud[0]#.mean() median(df.nr_longitud) #
    init_lat = df[:1].nr_latitud[0]#.mean() median(df.nr_latitud) #
    default_position = {
        "zoom": init_zoom,
        "pitch": 0,
        "bearing": 0,
        "center": {"lon": init_lon, "lat": init_lat},
    }
    return init_lon, init_lat, default_position

def_mapbox_style = "carto-positron" 

def mask_df_ll(df_in, lons, lats):
    # Mask dataframe based on lon/lat corners
    lon0, lon1 = (min(lons), max(lons))
    lat0, lat1 = (min(lats), max(lats))
    tmp_df = (df_in.query(f"nr_latitud > {lat0}").query(f"nr_latitud < {lat1}")
              .query(f"nr_longitud > {lon0}").query(f"nr_longitud < {lon1}"))
    return tmp_df

def filter_df(df_in, zoom_in, lon_in, lat_in, volt_range=None, cap_range=None, pp_sectors=None):
    # Mask DF with centre coordinate & zoom level
    m_per_px = 156543.03  # Conversion from "zoom level" to pixels by metres
    lon_offset = m_per_px / (2 ** zoom_in) * np.cos(np.radians(lon_in)) / 111111 * 600
    lat_offset = m_per_px / (2 ** zoom_in) * np.cos(np.radians(lat_in)) / 111111 * 300
    relayout_corners_ll = [
        [lon_in - lon_offset, lat_in + lat_offset],
        [lon_in + lon_offset, lat_in + lat_offset],
        [lon_in + lon_offset, lat_in - lat_offset],
        [lon_in - lon_offset, lat_in - lat_offset],
    ]
    lons, lats = zip(*relayout_corners_ll)
    tmp_df = mask_df_ll(df_in, lons, lats)

    return tmp_df



# epsg4326: Lon/lat
# epsg3857: Easting/Northing (Spherical Mercator)
def ll2en(coords):  # epsg_4326_to_3857
    return [transformer_4326_to_3857.transform(*reversed(row)) for row in coords]


def en2ll(coords):  # epsg_3857_to_4326
    return [list(reversed(transformer_3857_to_4326.transform(*row))) for row in coords]


def get_cnr_coords(agg, coord_params):
    # Get corners of aggregated image, which need to be passed to mapbox
    coords_lon, coords_lat = agg.coords[coord_params[0]].values, agg.coords[coord_params[1]].values  # agg is an xarray object, see http://xarray.pydata.org/en/stable/ for more details
    coords_ll = [
        [coords_lon[0], coords_lat[0]],
        [coords_lon[-1], coords_lat[0]],
        [coords_lon[-1], coords_lat[-1]],
        [coords_lon[0], coords_lat[-1]],
    ]
    curr_coords_ll_out = en2ll(coords_ll)
    return curr_coords_ll_out


def update_layer(df,zoom, lon, lat, metric):
    tmp_pot_df = filter_df(df, zoom, lon, lat)
    x_range = tmp_pot_df['x'].min(), tmp_pot_df['x'].max()
    y_range = tmp_pot_df['y'].min(), tmp_pot_df['y'].max()
    

    res = np.log(zoom) + 1 ** np.log(tmp_pot_df.shape[0])
    # if zoom<10:
    #     res = 2.6
    # elif ((zoom >= 10) and (zoom < 12)):
    #         res = 2.9
    # elif ((zoom >= 12) and (zoom < 12.5)):
    #         res = 3.5
    # elif ((zoom >= 12.5) and (zoom < 13)):
    #         res = 4.5
    # elif ((zoom >= 13) and (zoom < 50)):
    #         res = 5
    # else:
    #         res = 6
    
    if ((tmp_pot_df["nr_longitud"].nunique() > 0)&(zoom<14.5)):
        meshgrid_cols = [int(18 * np.log(zoom)), int(18 * np.log(zoom))]
        #meshgrid_cols=[int(tmp_pot_df["nr_longitud"].count() / (tmp_pot_df["nr_longitud"].count() / zoom**res)), int(tmp_pot_df["nr_latitud"].nunique() / (tmp_pot_df["nr_latitud"].count() / zoom**res))]        
        #meshgrid_cols=[int(tmp_pot_df["nr_longitud"].count() / zoom**res), int(tmp_pot_df["nr_latitud"].count() / zoom**res)]        
        #meshgrid_cols=[int(tmp_pot_df["nr_longitud"].count() / (tmp_pot_df["nr_longitud"].count() / res)), int(tmp_pot_df["nr_latitud"].count() / (tmp_pot_df["nr_latitud"].count() / res))]        
    elif ((tmp_pot_df["nr_longitud"].count() > 0)&(zoom>=14.5)):
        #meshgrid_cols=[int(tmp_pot_df["nr_longitud"].count()) ,int(tmp_pot_df["nr_latitud"].count()) ]
        meshgrid_cols = [int(18 * np.log(zoom)), int(18 * np.log(zoom))]
    else:
        meshgrid_cols = [1,1]
        
    if tmp_pot_df.shape[0]>1000:
        cvs_p = ds.Canvas(1000, 1000)
    else:
        cvs_p = ds.Canvas(tmp_pot_df.shape[0], tmp_pot_df.shape[0])
        
    cvs = ds.Canvas(plot_width=meshgrid_cols[0], plot_height=meshgrid_cols[1]) #, x_range=x_range, y_range=y_range
    agg_point = cvs_p.points(tmp_pot_df,  x='x', y='y')
    agg = cvs.points(tmp_pot_df, agg=rd.mean(metric), x='x', y='y')
    #agg = cvs.raster(agg, interpolate='nearest') ##nearest, 'linear'
    curr_coords_ll_out_points = get_cnr_coords(agg_point, ['x', 'y'])
    curr_coords_ll_out = get_cnr_coords(agg, ['x', 'y'])
    cmap=[ 'blue' ]*(int(256/5)) + [ 'violet' ]*(int(256/5)) + ['red']*(int (256/5)) + ['orange' ]*(int (256/5)) + [ 'yellow' ]*(int(256/5))
    points = tf.shade(agg_point, cmap=cm(def_cmap), how='eq_hist')[::-1].to_pil() ##'eq_hist', 'cbrt' ,'log'
    raster = tf.shade(cvs.raster(agg,agg=rd.mean(), interpolate='nearest'),cmap=cm(def_cmap), how='eq_hist')
    #raster = tf.shade(agg, cmap=cm(def_cmap),  how='eq_hist') ##'eq_hist', 'cbrt' ,'log'  
    raster = tf.dynspread(raster, threshold=0.0, max_px=5)[::-1].to_pil()
    
    mapbox_layers = list()
    pot_layer = {"sourcetype": "image", "opacity": 0.8, "source": raster, "coordinates": curr_coords_ll_out}
    mapbox_layers.append(pot_layer)

   # polygon_fig = build_polygons(df, gdf, mapboxt)

    return tmp_pot_df, points, raster, {"sourcetype": "image", "opacity": 0.6, "source": raster, "coordinates": curr_coords_ll_out}, {"sourcetype": "image", "opacity": 0.2, "source": points, "coordinates": curr_coords_ll_out_points}, agg, tmp_pot_df, float(agg.min().values), float(agg.max().values)#, {'mapbox.layers': polygon_fig}


def build_polygons(df, gdf, mapboxt):
    gdf = gdf[gdf.NAM.isin(df.Localidad.unique())]
    gdf.to_file('arg-geo.json', driver = 'GeoJSON')
    with open('arg-geo.json') as geofile:
        jdataNo = json.load(geofile) 
    
    for k in range(len(jdataNo['features'])):
        jdataNo['features'][k]['id'] = k

    locations = [k for k in range(gdf.shape[0])]
    z = [k*1.05 for k in range(gdf.shape[0])]
    name = [jdataNo['features'][i]['properties']['NAM'] for i in range(gdf.shape[0])]

    fig = go.Figure(go.Choroplethmapbox(z=z,
                            locations = locations,
                            geojson = jdataNo,
                            text = name,#df['geo-name'],
                            hovertemplate = '<b>State</b>: <b>%{text}</b>'+
                                            '<br> <b>Val </b>: %{z}<br>',
                            marker_line_width=0.1, marker_opacity=0.7))

    fig.update_layout(mapbox = dict(center= dict(lat= -35.0902, lon= -50.7129),            
                                accesstoken= mapboxt,
                                zoom=3
                                ))
    
    return fig

def build_base_map(df, default_position, mapbox_style=def_mapbox_style):
    # Build the underlying map that the Datashader overlay will be on top of
    fig = px.scatter_mapbox(df, lat="nr_latitud", lon="nr_longitud")#, color='mediana_fact_21', color_continuous_scale=px.colors.sequential.Rainbow)
    fig["layout"]["mapbox"].update(default_position)
    fig.update_layout(mapbox_style=mapbox_style, width=850, height=800, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    return fig

def get_lon_lat_zoom(relayout_data, prev_center, prev_zoom):
    # If there is a zoom level or relayout_data["mapbox.center"] - Update map based on the info. Otherwise - use default
    if relayout_data:  # Center point loc will not always be in relayout data
        relayout_lon = relayout_data.get("mapbox.center", {}).get("lon", prev_center[0])
        relayout_lat = relayout_data.get("mapbox.center", {}).get("lat", prev_center[1])
        relayout_zoom = relayout_data.get("mapbox.zoom", float(prev_zoom))
    else:
        relayout_lon = prev_center[0]
        relayout_lat = prev_center[1]
        relayout_zoom = float(prev_zoom)
    return relayout_lon, relayout_lat, relayout_zoom

def build_legend(scale_min=0.0, scale_max=1.0, colorscale_n=5, cmap=def_cmap, legend_title="Legend", dec_pl=0):
    colorscale_int = int((len(def_cmap) - 1) / (colorscale_n - 1))
    legend_headers = list()
    legend_colors = list()
    colwidth = int(100 / (colorscale_n))
    for i in range(colorscale_n):
        tmp_col = def_cmap[i * colorscale_int]  # Color
        tmp_num = round(scale_min + (scale_max - scale_min) / (colorscale_n - 1) * i, dec_pl)  # Number
        legend_headers.append(
            html.Th(
                f" ",
                style={
                    "background-color": tmp_col,
                    "color": "black",
                    "fontSize": 10,
                    "height": "0.9em",
                    "width": str(colwidth) + "%"
                },
            ),
        )  # Build the color boxes
        legend_colors.append(html.Td(tmp_num, style={"fontSize": 10}))  # Build the text legend

    legend_body = html.Table([
        html.Tr(legend_headers),
        html.Tr(legend_colors),
    ], style={"width": "90%"})
    legend = html.Table([
        html.Tr([html.Td(html.Strong(f"{legend_title}:", style={"fontSize": 13}))]),
        html.Tr([html.Td(legend_body)])
    ], style={"width": "90%"})
    return legend


def get_select_est_para(select_power, var1, var2):
    if np.isnan(select_power):
        return html.Div([
            html.P([
                html.Span("The selected area is outside the view window, try moving back or selecting a new area."),
            ]),
        ])
    else:
        avg_resource_sent = html.Span([
        f" con un promedio de facturación de",
        dbc.Badge(f"{round(avg_var, 1)} $", color="warning"),
        "."
        ])

        return html.Div([
            html.P([
                html.Span("El área seleccionada contiene "),
                html.Span(dbc.Badge(f"{int(q_terminales):,}" + " terminales", color=highlight_col)),
                avg_resource_sent
            ])
        ])


def build_hist_vol(df, metric, bins, color=None, marginal=None):
    q_low = df[metric].quantile(0.05)
    q_hi  = df[metric].quantile(0.95)
    df_filtered = df[(df[metric] < q_hi) & (df[metric] > q_low)]
    fig = px.histogram(df_filtered, x=metric, nbins=bins, color=color, marginal=marginal, color_discrete_sequence= px.colors.sequential.Sunset_r)
    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
    return fig

def build_bar_fig(df, cat, metric, x, y, orientation):
    #tmp_df = df.groupby(cat)[metric].count().reset_index()
    tmp_df = df.groupby(cat).agg({'id': 'count', metric: 'median'}).reset_index()
    tmp_df[metric] = round(tmp_df[metric],0)
    tmp_df.rename(columns = {'id':'cantidad', metric:'median'}, inplace = True)
    tmp_df.sort_values('cantidad', ascending=True, inplace=True)
    tmp_df = tmp_df.tail(10)
    fig = px.bar(tmp_df, x='cantidad', y=cat, color='median', #cat, 
                orientation=orientation,
                 #color_discrete_sequence=px.colors.qualitative.D3,
                 template="plotly_white",
                 #, labels={"MWh": "Energy (MWh)", "Series": "Group"}
                 #hover_name=cat
                )
    #fig.update_traces(hovertemplate='Cantidad: %{x:0.f}' + '<br>' +str(metric) + ': %{y:0.f}') 
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
    return fig

def build_pie_fig(df, cat):
    tmp_df = df.groupby(cat).agg({'id': 'count'}).reset_index()
    tmp_df.rename(columns = {'id':'cantidad'}, inplace = True)
    fig = px.pie(tmp_df, values='cantidad', names=cat, color_discrete_sequence= px.colors.sequential.Sunset_r)
    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
    return fig

def build_tree_fig(df, cat, metric):
    #l = list(df[pd.notnull(df[cat])].ds_tipo_terminal_active)
    #res = [i.strip("[]").split(", ") for i in l]
    #a = pd.Series([item for sublist in res for item in sublist])
    #df_plot = a.value_counts().sort_index().rename_axis('ds_tipo_terminal_active').reset_index(name='q')
    df_plot = df.groupby(['Provincia', cat]).agg({'id': 'count', metric: 'median'}).reset_index()
    df_plot.rename(columns = {'id':'cantidad', metric:'median'}, inplace = True)
    df_plot.sort_values('cantidad', ascending=True, inplace=True)
    df_plot = df_plot.tail(20)
    fig = px.treemap(df_plot, path=['Provincia', cat], values='cantidad',
                    color='median',
                     template="plotly_white")
    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
    return fig
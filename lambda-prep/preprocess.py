import pandas as pd
import json
import os
import numpy as np
from sklearn.neighbors import BallTree
from operator import itemgetter
import geopandas as gpd
import boto3
import io
from statistics import median
import logging
#import psycopg2
#import os
import pymysql
from sqlalchemy import create_engine

# create logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

print('ok import')

import pygeos
gpd.options.use_pygeos = True

distance_in_km = 2
earth_radius_in_km = 6367
radius = distance_in_km / earth_radius_in_km

#Tranformo lat/lon a points
def wgs84_to_web_mercator(df, lon="LON", lat="LAT"):

      k = 6378137
      df["x"] = df[lon] * (k * np.pi/180.0)
      df["y"] = np.log(np.tan((90 + df[lat]) * np.pi/360.0)) * k

      return df


#Funcion de busqueda vecinos
def vecinos_radio (df, distance_in_km,k, metrica):
    for column in df[["nr_latitud", "nr_longitud"]]:
        rad = np.deg2rad(df[column].values)
        df[f'{column}_rad'] = rad
    # Takes the first group's latitude and longitude values to construct
    # the ball tree.
    ball = BallTree(df[["nr_latitud_rad", "nr_longitud_rad"]].values, metric='haversine')
    is_within, distances = ball.query_radius(df[["nr_latitud_rad", "nr_longitud_rad"]].values, r=radius, count_only=False, return_distance=True)
    distances_k, is_within_k = ball.query(df[["nr_latitud_rad", "nr_longitud_rad"]].values, k = k)
    
    for i in range(0, len(is_within)):
        if len(is_within[i]) >= k:
            break
        else:
            is_within[i] = is_within_k[i]
    
    #Calculamos la mediana de lo grupos
    res_list = list(itemgetter(*is_within)(df[metrica].values))
    #ind_med = [np.median(lst, axis=0) for lst in res_list]

    res_list_un = [x for x in res_list if len(x) >= 0]
    ind_med = [np.median(lst, axis=0) for lst in res_list_un]
    is_within_un = [x for x in is_within if len(x) >= 0]
    
    w = []
    for i in range(0, len(is_within_un)):
        w.append(1+(len(is_within_un[i])/len(df)))
        
    ind_med = [a * b for a, b in zip(ind_med, w)]
    
    quart = pd.qcut(ind_med, 5, labels=False)

    return is_within_un#, ind_med, quart
    
    return df#Tranformo lat/lon a points
def wgs84_to_web_mercator(df, lon="LON", lat="LAT"):

      k = 6378137
      df["x"] = df[lon] * (k * np.pi/180.0)
      df["y"] = np.log(np.tan((90 + df[lat]) * np.pi/360.0)) * k

      return df


def agrupar_medianas(df, agrupador, l2, distance_in_km, k, metrica):
    is_within_un = vecinos_radio(df, distance_in_km,k, metrica)
    res_list = list(itemgetter(*is_within_un)(df[[agrupador, metrica]].values))
    #value_list = list(itemgetter(*is_within_un)(df[metrica].values))
    #rubro_list = list(itemgetter(*is_within_un)(df[agrupador].values))
    
    list_dicts = []

    for i in range (0,len(res_list)):
        my_dict = {}
        for k, v in res_list[i]:
            my_dict.setdefault(k, []).append(v)
        list_dicts.append(my_dict)
    
    list_rubros = []
    for i in range(0, len(list_dicts)): 
        avgDict = {}
        for k,v in list_dicts[i].items():
            # v is the list of grades for student k
            avgDict[k] = median(v)
        list_rubros.append(avgDict)
    
    l1 = range(0, len(list_rubros))
    
    list_vol_rubro = []
    for i, j in zip(l1, l2): 
        a = df.iloc[[i]].apply(lambda row: list_rubros[i][row.property_type], axis=1)
        list_vol_rubro.insert(j, a[j])
    
    df[f'property_type_med_{metrica}'] = list_vol_rubro
    
    return df

def handler(event, context):

    #other option
    #filename = 'input/dataset.parquet/b7ede4002c2a45f094fbabd8e6bf179d.snappy.parquet'
    #bucket = 'app-cde'
    #buffer = io.BytesIO()
    #client = boto3.resourse('s3')
    #object = client.Object(bucket, filename)
    #object.download_fileobj(buffer)
    #df = pd.read_parquet(buffer)

    # Read the parquet file
    df = pd.read_parquet('s3://cde-m1/raw-data/5d8bdd2fb0f14fff889fa736b16108dd.snappy.parquet')
    print('ok import')
    map_df = gpd.read_file("s3://cde-m1/resources/ign_departamento/ign_departamento.shp")
    print('ok import')
    provincia_sh = gpd.read_file("s3://cde-m1/resources/provincia/provincia.shp")

    print('carga-ok')
    #df = df[:1000]
    df = df[(df.operation_type=='Alquiler') & (df.currency=='ARS') & (df.l2=='Capital Federal') & (df.l1=='Argentina')]
    df = df[:25000]
    df.dropna(subset=['lat'], inplace=True)
    df.dropna(subset=['lon'], inplace=True)
    df.dropna(subset=['price'], inplace=True)
    df.rename(columns = {'lat':'nr_latitud', 'lon':'nr_longitud'}, inplace = True) 
    df["surface_total"] = df['surface_total'].fillna(df.groupby('property_type')['surface_total'].transform('mean'))
    df['price_m2'] = df.price / df.surface_total

    print('feats ok')

    #Definimos los vecinos por radio en km
    k = 30
    puntos_zona = 100
    

    df = wgs84_to_web_mercator(df, 'nr_longitud', 'nr_latitud')

    print('conv-geo')

    agrupador = 'property_type'
    l2 = list(df.index)
    distance_in_km = 1

    df = agrupar_medianas(df, agrupador, l2, distance_in_km, k, 'price_m2')
    df = agrupar_medianas(df, agrupador, l2, distance_in_km, k, 'price')
    df = agrupar_medianas(df, agrupador, l2, distance_in_km, k, 'surface_total')

    print('paso1')

    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.nr_longitud, df.nr_latitud))

    map_df = map_df.set_crs(epsg=4326, inplace=True, allow_override=True)
    provincia_sh = provincia_sh.set_crs(epsg=4326, inplace=True, allow_override=True)
    df = df.set_crs(epsg=4326, inplace=True)

    df_join = df.sjoin(provincia_sh[['nam','geometry']], how="inner", predicate='intersects') #Binary predicate, one of {‘intersects’, ‘contains’, ‘within’}
    df_join.rename(columns = {'nam':'Provincia'}, inplace = True)

    l_localidad = df.sjoin(map_df[['NAM','geometry']], how="inner", predicate='intersects')['NAM'] 
    df_join['Localidad'] = l_localidad
    df_join = df_join.drop(['index_right'], axis=1)

    print('paso2')

    #wr.s3.to_parquet(
    #        df_join,
    #        path="s3://app-cde/input",
    #        dataset=True,
    #        index=False,compression = 'snappy'
    #        )

    host= os.environ['HOST']
    port=int(3306)
    user= os.environ['USER']
    passw= os.environ['PASS']
    database= os.environ['DATABASE']
    mydb = create_engine('mysql+pymysql://' + user + ':' + passw + '@' + host + ':' + str(port) + '/' + database , echo=False)
    
    df_join.to_sql(name="properties", con=mydb, if_exists = 'replace', index=False)

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }

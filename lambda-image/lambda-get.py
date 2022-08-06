import boto3
import logging
import pandas as pd
import requests
from typing import Tuple
import awswrangler as wr
import os
import json

# create logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# get client
s3_client = boto3.client('s3')
sns_client = boto3.client('sns')


def create_data(
    source_url: str,
    df_prop: pd.DataFrame
) -> Tuple[pd.DataFrame, int]:
    """
    Execute logic that produces database.

    Args:
        source_url: string with url for data.

    Returns:
        df_input: pandas dataframe with data;
    """

    # read url data
    base = pd.read_csv(source_url,compression='gzip')
    df_prop = base 

    return df_prop


def handler(event, context):
    # ---- read environment variables
    bucket_name = 'app-cde' #os.environ['BUCKET_NAME']
    key_name_output = 'input/dataset.parquet' # os.environ['OUTPUT_KEY_NAME']
    #source_url = os.environ['SOURCE_URL']
    source_url = 'https://storage.googleapis.com/properati-data-public/ar_properties.csv.gz'

    # download data
    s3 = boto3.client('s3')


    try:
        response = requests.get(source_url, timeout=10)
        logger.info("URL válida")
        success = True
    
        # create table
        df_input = pd.DataFrame()
        df_input = create_data(source_url,df_input)

        print(df_input.shape)
        logging.info('dataframe head - {}'.format(df_input.head()))
    
        # write output to tmp directory and then upload to s3
        logger.info('convert file...')

        wr.s3.to_parquet(
        df_input,
        path="s3://cde-m1/raw-data/",
        dataset=True,
        index=False,compression = 'snappy'
        )
        
        logger.info('converted ok')

        #logger.info('upload')

        #with open('/tmp/base.parquet', 'rb') as f:
        #    s3_client.upload_fileobj(
        #        Fileobj=f,
        #        Bucket=bucket_name,
        #        Key=key_name_output)


        logger.info('upload ok')
        
        #with open('/tmp/base_props.parquet', 'rb') as f:
        #    s3_client.upload_fileobj(
        #        Fileobj=f,
        #        Bucket=bucket_name,
        #        Key=key_name_output)

    
        #logger.info('write file ok')
    
            # publish message to SNS Topic
           
            #response = sns_client.publish(
            #    TopicArn=os.environ['SNS_TOPIC_ARN'],
            #    Message='Actualización de la tabla de inflación exitosa',
            #    Subject=f'AWS: INFLATION DATA EXTRACTION - Actualización de datos exitosa')
    
        return {
            'statusCode': 200,
            'body': 'Success'
        }

    except:
        logger.info("Error en la url. Verificar")
        
        # publish message to SNS Topic
        #response = sns_client.publish(
        #    TopicArn=os.environ['SNS_TOPIC_ARN'],
        #    Message='Error en la url. Verificar.',
        #    Subject=f'AWS: INFLATION DATA EXTRACTION - Error en la URL'
        

        # mandar mail
        return {
            'statusCode': 400,
            'body': 'Error'
        }

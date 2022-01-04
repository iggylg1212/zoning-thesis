import boto3
import shapefile
import pandas as pd
import geopandas as gpd
import io

S3 = boto3.client("s3")
S3_PATH_NAME = "thesis1212"
S3_PATH = f"s3://{S3_PATH_NAME}/"

def read_s3_file(key):
    return io.BytesIO(S3.get_object(Bucket=S3_PATH_NAME, Key=key)['Body'].read())

def read_shapefile(key, crs):
    sf = shapefile.Reader(shp=io.BytesIO(S3.get_object(Bucket=S3_PATH_NAME, Key=f'{key}.shp')['Body'].read()),
                              shx=io.BytesIO(S3.get_object(Bucket=S3_PATH_NAME, Key=f'{key}.shx')['Body'].read()),
                              dbf=io.BytesIO(S3.get_object(Bucket=S3_PATH_NAME, Key=f'{key}.dbf')['Body'].read()),
                              prj=io.BytesIO(S3.get_object(Bucket=S3_PATH_NAME, Key=f'{key}.prj')['Body'].read()),
                              cpg=io.BytesIO(S3.get_object(Bucket=S3_PATH_NAME, Key=f'{key}.cpg')['Body'].read()))

    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s for s in sf.shapes()]

    #write into a dataframe
    df = gpd.GeoDataFrame(pd.DataFrame(columns=fields, data=records).assign(geometry=shps), crs=crs)
    
    return df

def write_csv(key, df):
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, header=True, index=False)
    csv_buf.seek(0)
    S3.put_object(Bucket=S3_PATH_NAME, Body=csv_buf.getvalue(), Key=key)

def write_gpd(key, df, driver):
    gpkg_buf = io.BytesIO()
    df.to_file(gpkg_buf, driver=driver)
    gpkg_buf.seek(0)
    S3.put_object(Bucket=S3_PATH_NAME, Body=gpkg_buf.getvalue(), Key=key)
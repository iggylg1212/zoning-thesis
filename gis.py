from concurrent.futures import process
from venv import create
from global_var import *
import pymysql
import os
from config import *
from geopandas.io.file import read_file
import pandas as pd  # provides interface for interacting with tabular data
import geopandas as gpd  # combines the capabilities of pandas and shapely for geospatial operations
from shapely.geometry import Point, Polygon, MultiPolygon, mapping  # for manipulating text data into geospatial shapes
from shapely import wkt  # stands for "well known text," allows for interchange across GIS programs
import pickle
import rtree
import numpy as np
from scipy.sparse.csgraph import connected_components
from sqlitedict import SqliteDict
import math
from tqdm import tqdm
import rasterio
import rasterio.plot
from multiprocessing.pool import ThreadPool as Pool
from rasterio import features
import numpy as np
import os.path as path

def create_connection():
    conn = None
    errors = 0
    while conn is None and errors<10: 
      try:
          conn = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)
      except:
          errors += 1
    return conn

files1 = ['.dbf','.prj','.shp','.shx','.cpg']
files2 = ['.dbf', '.cpg', '.sbn', '.sbx', '.shp', '.shp.xml', '.shx', '.prj']

# ########## Undevelopable Land
# # Water and Wetlands
# land_use = {'Open Water':11, 'Perennial Ice/Snow':12, 'Developed, Open Space':21, 'Developed, Low Intensity':22, 
#             'Developed, Medium Intensity':23, 'Developed, High Intensity':24, 'Barren Land (Rock/Sand/Clay)':31, 
#             'Deciduous Forest':41, 'Evergreen Forest':42, 'Mixed Forest':43, 'Dwarf Scrub':51, 'Shrub/Scrub':52, 
#             'Grasslands/Herbaceous':71, 'Sedge/Herbaceous':72, 'Lichens':73, 'Moss':74, 'Pasture/Hay':81, 'Cultivated Crops':82, 
#             'Woody Wetlands':90, 'Emergent Herbaceous Wetlands':95}

# with rasterio.Env():
#     with rasterio.open(f'{ROOT_DIR}1data/gis/2001_land_cover/nlcd_2001_land_cover_l48_20210604.img') as src:
#         image = src.read(1)
#         mask = (image==11) | (image==90) | (image==95) # Open Water or Wetlands
#         results = SqliteDict(f'{ROOT_DIR}1data/pickle/undev_land_cover.sqlite', autocommit=True)
#         for i, (s, v) in enumerate(features.shapes(image, mask=mask, transform=src.transform)):
#             results[i]= {'properties': {'raster_val': v}, 'geometry': s}

# print('Geopandas')
# with SqliteDict(f'{ROOT_DIR}1data/pickle/undev_land_cover.sqlite', autocommit=True) as results:
#     undev = gpd.GeoDataFrame.from_features(results.values(), crs = src.crs)[['geometry']]
#     undev = undev.to_crs('EPSG:4326')
#     print(len(undev))
#     print('Writing File')
#     undev.to_file(f'{S3_PATH}gis/gis/undev_land_cover/undev_land_cover.gpkg', driver='GPKG')

# # Elevation
# S3FS.get(f'{S3_PATH}gis/gis/National_Slope/National_Slope.img', 'National_Slope.img' )
# with rasterio.Env():
#     with rasterio.open('National_Slope.img') as src:
#         image = src.read(1)
#         mask = image >=20 # Slope greater than 20 degrees
#         fp = np.memmap('1data/pickle/numpy.memmap', mode='w+', shape=image.shape)
#         with SqliteDict(f'{ROOT_DIR}1data/pickle/undev_slope.sqlite', autocommit=True) as results:
#             for i, (s, v) in enumerate(features.shapes(fp, mask=mask, transform=src.transform)):
#                 results[i] = {'properties': {'raster_val': v}, 'geometry': s}
# print('Geopandas')
# undev  = gpd.GeoDataFrame.from_features( results.values(), crs=src.crs)[['geometry']]
# undev = undev.to_crs('EPSG:4326')
# print('Writing File')
# undev.to_file('{S3_PATH}gis/gis/undev_slope/undev_slope.gpkg', driver='GPKG')
# S3FS.put('undev_slope.gpkg','{S3_PATH}gis/gis/undev_slope/undev_slope.gpkg')
# os.remove('undev_slope.gpkg')

# # Public Lands
# def pad_processor(region, set):
#     if set == 'Marine':
#         try:
#             for file in files2:
#                 if file == '.prj':
#                     S3FS.get(f'{S3_PATH}gis/gis/PAD/PADUS2_1_Region{region}_Shapefile/PADUS2_1Marine_Region{region}{file}', f'temporary{file}')
#                 else:
#                     S3FS.get(f'{S3_PATH}gis/gis/PAD/PADUS2_1_Region{region}_Shapefile/PADUS2_1Marine{file}', f'temporary{file}')
#             pad = gpd.read_file(f'temporary.shp')
#             for file in files2:
#                 os.remove(f'temporary{file}')
#             pad = pad.set_crs('PROJCS["USA_Contiguous_Albers_Equal_Area_Conic_USGS_version",GEOGCS["GCS_North_American_1983",DATUM["North_American_Datum_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433],AUTHORITY["EPSG","4269"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["longitude_of_center",-96.0],PARAMETER["Standard_Parallel_1",29.5],PARAMETER["Standard_Parallel_2",45.5],PARAMETER["latitude_of_center",23.0],UNIT["Meter",1.0]]')
#         except:
#             return None
#     else:
#         for file in files2:
#             if file == '.prj':
#                 S3FS.get(f'{S3_PATH}gis/gis/PAD/PADUS2_1_Region{region}_Shapefile/PADUS2_1{set}_Region{region}{file}', f'temporary{file}')
#             else:
#                 S3FS.get(f'{S3_PATH}gis/gis/PAD/PADUS2_1_Region{region}_Shapefile/PADUS2_1{set}_Region{region}{file}', f'temporary{file}')
#         pad = gpd.read_file(f'temporary.shp')
#         for file in files2:
#             os.remove(f'temporary{file}')
    
#     none_pad = pad[ ~(pad['Date_Est'].apply(type) == str) ]
#     old_pad = pad[ pad[ 'Date_Est'].apply(type) == str ]
#     old_pad = old_pad[ old_pad['Date_Est'].apply(int) <= 1990 ]
#     pad = none_pad.append(old_pad)[['SHAPE_Leng', 'SHAPE_Area', 'geometry']]
#     return pad

# sets = ['Designation','Easement','Fee']
# regions = list(range(1,13))

# public_lands = pad_processor(1,'Marine')
# for set in sets:
#     for region in regions:
#         public_lands = public_lands.append(pad_processor(region, set))

# public_lands = public_lands.to_crs('EPSG:4326')
# public_lands.to_file('PAD_Concat.gpkg', driver='GPKG')
# S3FS.put('PAD_Concat.gpkg', f'{S3_PATH}gis/gis/PAD/PAD_Concat/PAD_Concat.gpkg')
# os.remove('PAD_Concat.gpkg')

# # Concatentate
# S3FS.get(f'{S3_PATH}gis/gis/PAD/PAD_Concat/PAD_Concat.gpkg', f'public.gpkg')
# public_lands = gpd.read_file('public.gpkg', crs='EPSG:4326')[['geometry']]
# public_lands['source'] = 'public lands'
# os.remove('public.gpkg')
# print('Public Lands')
# S3FS.get(f'{S3_PATH}gis/gis/undev_land_cover/undev_land_cover.gpkg', 'water.gpkg')
# water_wet = gpd.read_file('water.gpkg', crs='EPSG:4326')[['geometry']]
# water_wet['source'] = 'water and wetlands'
# os.remove('water.gpkg')
# print('Water')
# S3FS.get(f'{S3_PATH}gis/gis/undev_slope/undev_slope.gpkg', 'elev.gpkg')
# elev = gpd.read_file('elev.gpkg', crs='EPSG:4326')[['geometry']]
# elev['source'] = 'elevation'
# os.remove('elev.gpkg')
# print('Elevation')

# undev = water_wet.append(elev, ignore_index=True).append(public_lands, ignore_index=True)

# # Undevelopable in urban areas
# for file in files1:
#     S3FS.get(f'{S3_PATH}gis/gis/us_urb_area_1990/reprojection_urb_area_1990{file}', f'temporary{file}')
# uas = gpd.read_file('temporary.shp', crs='EPSG:4326')
# for file in files1:
#     os.remove(f'temporary{file}')

# def worker(index):
#     cities = uas
#     row = undev.iloc[index]
#     try:
#         cities['bool'] = cities.intersects(row['geometry'])
#         set = cities[cities['bool']==True].reset_index(drop=True)
#         if len(set)==0:
#             return
#         else:
#             for idx in list(range(len(set))):
#                 subset = set.iloc[idx]
#                 conn = create_connection()
#                 conn.__enter__()
#                 cur = conn.cursor()
#                 # cur.execute(''' CREATE TABLE IF NOT EXISTS undev (
#                 #                     source TEXT NOT NULL,
#                 #                     city TEXT NOT NULL,
#                 #                     geometry LONGTEXT NOT NULL 
#                 #                 ); ''')
#                 # conn.commit()
#                 cur.execute(f"INSERT INTO undev (source, city, geometry) values ('{str(row['source'])}','{str(subset['UACODE'])}','{str(row['geometry'])}')")
#                 conn.commit()
#                 conn.__exit__()
#     except:
#         print(f'broken geom:{index}')

# tasks = list(range(len(undev)))

# if __name__ == "__main__":
#     pool= Pool(processes=1)
#     for _ in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)):
#         pass

# conn = create_connection()
# with conn:
#     cur = conn.cursor()
#     cur.execute('CREATE TABLE undev_copy SELECT DISTINCT source, city, geometry FROM undev;')
#     conn.commit()
#     cur.execute('DROP TABLE undev')
#     conn.commit()
#     cur.execute('ALTER TABLE undev_copy RENAME TO undev;')
#     conn.commit()

# conn = create_connection()
# with conn:
#     cur = conn.cursor()
#     cur.execute('SELECT * FROM undev')
#     results = cur.fetchall()
#     source = []
#     city = []
#     geometry = []
#     for result in results:
#         source.append(result[0])
#         city.append(result[1])
#         geometry.append(wkt.loads(result[2]))

# gpkg = gpd.GeoDataFrame(pd.DataFrame({'source':source,'city':city,'geometry':geometry}), geometry ='geometry', crs='EPSG:4326')
# gpkg.to_file('undev_city1990.gpkg')
# S3FS.put('undev_city1990.gpkg',f'{S3_PATH}gis/gis/undev_city1990.gpkg')
# os.remove('undev_city1990.gpkg')

# # Difference Urban Areas by undevelopable
# for file in files1:
#     S3FS.get(f'{S3_PATH}gis/gis/us_urb_area_1990/reprojection_urb_area_1990{file}', f'temporary{file}')
# uas = gpd.read_file('temporary.shp', crs='EPSG:4326')
# for file in files1:
#     os.remove(f'temporary{file}')

# conn = create_connection()
# with conn:
#     cur = conn.cursor()
#     cur.execute('CREATE TABLE IF NOT EXISTS undev_city (uacode TEXT, name TEXT, gisjoin TEXT, geometry LONGTEXT);')
#     conn.commit()
#     cur.execute('SELECT uacode FROM undev_city')
#     results = cur.fetchall()
#     done_list = []
#     for result in results:
#         done_list.append(result[0])

# def clip_worker(row):
#     conn = create_connection()
#     conn.__enter__()
#     cur = conn.cursor()
#     code = uas.iloc[row]['UACODE']
#     if code in done_list:
#         return
#     cur.execute(f'SELECT geometry FROM undev where city={code}')
#     geom_groups = cur.fetchall()
#     conn.__exit__()
#     undev = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
#     for geom_group in geom_groups:
#         geom = wkt.loads(geom_group[0])
#         undev = undev.append({'geometry':geom}, ignore_index=True)
#     dev_city = uas[uas['UACODE']==code]
#     try:
#         dev_city = dev_city.overlay(undev, how='difference', keep_geom_type=True)
#     except:
#         for und in undev:
#             try:
#                 dev_city = dev_city.overlay(und, how='difference', keep_geom_type=True)
#             except:
#                 print('broken geom')
#     for row in list(range(len(dev_city))):
#         subset = dev_city.iloc[row]
#         conn = create_connection()
#         conn.__enter__()
#         cur = conn.cursor()
#         cur.execute(f"INSERT INTO undev_city (uacode, name, gisjoin, geometry) values ('{subset['UACODE']}','{subset['NAME']}','{subset['GISJOIN']}','{str(subset['geometry'])}')")
#         conn.commit()
#         conn.__exit__()

# tasks = list(range(len(uas)))
# tasks.reverse()
# if __name__ == "__main__":
#     pool= Pool()
#     for _ in tqdm(pool.imap_unordered(clip_worker, tasks), total=len(tasks)):
#         pass

# conn = create_connection()
# with conn:
#     cur = conn.cursor()
#     cur.execute('CREATE TABLE undev_city_copy SELECT DISTINCT uacode, name, gisjoin, geometry FROM undev_city;')
#     conn.commit()
#     cur.execute('DROP TABLE undev_city')
#     conn.commit()
#     cur.execute('ALTER TABLE undev_city_copy RENAME TO undev_city;')
#     conn.commit()

# conn = create_connection()
# with conn:
#     cur = conn.cursor()
#     cur.execute('SELECT * FROM undev_city')
#     results = cur.fetchall()
#     uacode = []
#     name = []
#     gisjoin = []
#     geometry = []
#     for result in results:
#         uacode.append(result[0])
#         name.append(result[1])
#         gisjoin.append(result[2])
#         geometry.append(wkt.loads(result[3]))

# gpkg = gpd.GeoDataFrame(pd.DataFrame({'uacode':uacode,'name':name,'gisjoin':gisjoin,'geometry':geometry}), geometry ='geometry', crs='EPSG:4326')
# gpkg.to_file('dev_city1990.gpkg')
# S3FS.put('dev_city1990.gpkg',f'{S3_PATH}gis/gis/dev_city1990.gpkg')
# os.remove('dev_city1990.gpkg')

# Obtain Urbanized Block Groups
# S3FS.get(f'{S3_PATH}gis/gis/1990_blckgrp','1990_blkckgrp', recursive=True)
# S3FS.get(f'{S3_PATH}gis/gis/1990_uas','1990_uas', recursive=True)

blckgrp = gpd.read_file('1990_blkckgrp/US_blck_grp_1990.shp')[['GISJOIN','geometry']]
blckgrp['UACODE'] = np.nan
uas = gpd.read_file('1990_uas/US_urb_area_1990.shp')[['UACODE','geometry']]

for index, row in tqdm(blckgrp.iterrows(), total = len(blckgrp)):
    uas['bool'] = uas.intersects(row['geometry'])
    subset = uas[uas['bool']==True]
    if len(subset)!=0:
        code = subset['UACODE'].iloc[0]
        blckgrp.loc[index, 'UACODE'] = code

urb_blckgrp = blckgrp[blckgrp['UACODE'].notna()].to_crs('EPSG:4326')
urb_blckgrp.to_file('urb_blckgrp.gpkg')

# Find distribution of commutes
comm_blckgrp = pd.read_csv(f'{S3_PATH}gis/csv/1990_blckgrp_commutes/nhgis0034_ds123_1990_blck_grp.csv')
comm_blckgrp = urb_blckgrp.merge(comm_blckgrp, how='left', on='GISJOIN')

def comm_distribution(x, alpha):
    # alpha is a tuning parameter for logistic function through which `diff` is transformed. Recommended setting ~3
    ranges = [(0,5),(5,9),(10,14),(15,19),(20,24),(25,29),(30,34),(35,39),(40,44),(45,59),(60,89),(90,100)]
    pop = 0  
    for num in list(range(1,13)):
        if len(str(num)) == 1:
            num = f'0{num}'
        pop += x[f'E3W0{num}']
    
    if pop == 0:
        return [np.nan, np.nan, np.nan] 

    values = []
    probs = []
    for i in list(range(1,13)):
        if len(str(i)) == 1:
            i = f'0{i}'
        pop_dup = pop + x[f'E3W0{i}']
        prob = x[f'E3W0{i}'] / pop
        probs.append(prob)
        min_share = 0 
        for min in list(range(1,int(i)+1)):
            if len(str(min)) == 1:
                min = f'0{min}'
            min_share += x[f'E3W0{min}']/pop_dup
        max_share = 0 
        for max in list(range( int(i), 13)):
            if len(str(max)) == 1:
                max = f'0{max}'
            max_share += x[f'E3W0{max}']/pop_dup
        diff = max_share - min_share
        diff = (2/(1+math.exp(-alpha*diff)))-1
        
        value = (ranges[int(i)-1][1]+ranges[int(i)-1][0])/2 + diff*(ranges[int(i)-1][1]-ranges[int(i)-1][0])/2
        values.append(value)
    
    results = [values, probs, pop]
    return results 

comm_blckgrp['Values'] = None
comm_blckgrp['Probs'] = None
comm_blckgrp['Pop'] = None
for index, row in tqdm(comm_blckgrp.iterrows(), total=len(comm_blckgrp)):
    results = comm_distribution(row, 3)
    comm_blckgrp.loc[index, "Values"] = str(results[0]) 
    comm_blckgrp.loc[index, "Probs"] = str(results[1]) 
    comm_blckgrp.loc[index, "Pop"] = str(results[2]) 
comm_blckgrp = comm_blckgrp[['GISJOIN','UACODE','Values','Probs','Pop']]

comm_blckgrp.to_csv(f'{S3_PATH}gis/csv/1990_comm_blckgrp_dist.csv')
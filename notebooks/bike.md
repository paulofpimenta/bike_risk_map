---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
from osgeo import gdal
import pandas as pd
import geopandas as gp
from matplotlib import pyplot as plt
from matplotlib import colors
from shapely.geometry import Point
import py7zr
```

# 1. Import files from Google Drive and extract them into data folder

```python
import gdown
# Google drive public shared folder ID
id = "1vdm8b0ewDslPVvubA8emwDYHUGtwx5pT"
gdown.download_folder(id=id,output="../data", quiet=True, use_cookies=False)
```

```python
import py7zr
# Extract files from 7zip file
with py7zr.SevenZipFile('../data/bike_data.7z', mode='r') as z:
    z.extractall("../data")
```

## 2. Read files

```python
df = pd.read_csv('../data/accidentsVelo.csv')
df.head()
```

# Removing NA's on lat and long long columns only

```python
num_rows = len(df)
df = df.loc[(df["lat"].notna() & df["long"].notna())]
num_rows_not_na = len(df)
print("Removed NA's rows : %d" % (num_rows -num_rows_not_na) )
```

# Check column types and convert if necessary

```python
df.info()
```

### Convert lat and long cols to float

```python
df['lat'] = df['lat'].apply(lambda x: float(str(x).replace(',', '.')))
df['long'] = df['long'].apply(lambda x: float(str(x).replace(',', '.')))
```

```python
print("Lat and long colum type are |{}| and |{}| ".format(df.dtypes['lat'],df.dtypes['long'] ))
```

### Checking zeros on lat and long cols and total number of cols

```python
count_lat = df['lat'].value_counts()[0]
count_long = df['long'].value_counts()[0]
rows = len(df)

results = pd.DataFrame(data={"lat_with_zero":[count_lat], "long_with_zero":[count_long],"Total number of rows": [rows]})
results
```

### Remove rows where lat and long columns are not 0

```python
df_filtered = df.loc[(df['lat'] != 0) & (df['long'] != 0)].copy()
print("Non zero coordinates data frame has %d rows" %(len(df_filtered)))
```

### Convert to a proper geodataframe

```python
# Convert longitude and latitude to a geometric Point object
points_gdf = gp.GeoDataFrame(df_filtered, geometry=gp.points_from_xy(df_filtered.lat, df_filtered.long))
# Convert DataFrame to GeoDataFrame
points_gdf = points_gdf.set_crs('epsg:4326')
points_gdf.plot(aspect='equal')
print(points_gdf.crs)
```

### Its seems the coordinates are inverted, let's fix this

```python
# Convert longitude and latitude to a geometric Point object
points_gdf = gp.GeoDataFrame(df_filtered, geometry=gp.points_from_xy(df_filtered.long, df_filtered.lat))
# Convert DataFrame to GeoDataFrame
points_gdf = points_gdf.set_crs('epsg:4326').to_crs('epsg:27561')
points_gdf.plot()
print(points_gdf.crs)
```

### Check columns

```python
points_gdf.head()
```

### Plot France department limits and bike accident points

```python
france_gdf = gp.read_file("../data/france_dep.geojson")
france_gdf = france_gdf.to_crs('epsg:27561')

ax = points_gdf.plot(marker='o', color='red', markersize=5,aspect=1)
france_gdf.plot(ax=ax, color='white', edgecolor='black')
plt.show()
```

### Let's grab only the points inside France and plot bicycle accident points together

```python
points_within = gp.sjoin(points_gdf,france_gdf,predicate='within')
```

```python
ax = points_within.plot(legend=True, markersize=3,alpha=0.5,figsize=(12,12))
france_gdf.plot(ax=ax, color='white', alpha=0.5, edgecolor="k")
plt.show()
```

# Checking if the gravity of the accident can be spatially explained

```python
fig, axs = plt.subplots(2, 2)


points_within_grav_1 = points_within.loc[points_within['grav'] == 1]
points_within_grav_1.plot(ax=axs[0, 0],markersize=3,alpha=0.5,color='green')
axs[0, 0].set_title("Low")

points_within_grav_2 = points_within.loc[points_within['grav'] == 2]
points_within_grav_2.plot(ax=axs[0, 1],markersize=3,alpha=0.5,color='yellow')
axs[0, 1].set_title("Mild")

axs[1, 0].sharex(axs[0, 0])

points_within_grav_3 = points_within.loc[points_within['grav'] == 3]
points_within_grav_3.plot(ax=axs[1, 0],markersize=3,alpha=0.5,color='orange')
axs[1, 0].set_title("High")

points_within_grav_4 = points_within.loc[points_within['grav'] == 4]
points_within_grav_4.plot(ax=axs[1, 1],markersize=3,alpha=0.5,color='red')
axs[1, 1].set_title("Very high")

fig.tight_layout()

```

## It seems that the accidents of types are well distributed across the country and concentrated on major cities. Also, it may indicate that the gravity of an accident is subjective.


## Lets check the density of accidents (regardless of it's gravity) by region population

```python
accidents_by_department = points_gdf.sjoin(france_gdf,predicate='within')
accidents_by_department.plot()
```

### Plot accidents by gravity and by department

```python
accidents_by_region = accidents_by_department.dissolve(by='nom', aggfunc='sum',numeric_only=False)

# Remove index field names
france_gdf=france_gdf.loc[:,~france_gdf.columns.str.startswith('index_')]
accidents_by_region=accidents_by_region.loc[:,~accidents_by_region.columns.str.startswith('index_')]

departments_by_accidents = gp.sjoin(france_gdf,accidents_by_region,how="inner",predicate="intersects")

cmap = colors.LinearSegmentedColormap.from_list("", ["green","yellow","orange","red"])

departments_by_accidents.plot(column = 'grav', scheme='quantiles', cmap=cmap,figsize=(10,10),legend=True);
plt.title("Number of accidents by gravity and department")
plt.show()
```

```python
departments_by_accidents[["nom","grav"]].sort_values("grav",ascending=False)
```

### By using a quantilles representation, we  notice that touristic zones have a high number of accidents. But Paris, on absolute numbers, is still way higher than other departments


# Removing spaces and upper cases from cols of both dataframes

```python
accidents_by_region.columns = [x.lower().replace(' ','') for x in accidents_by_region.columns]
accidents_by_department.columns = [x.lower().replace(' ','') for x in accidents_by_department.columns]
```

# Calculate the occurences of accidents by region and add this info to the geodataframe with polygons

```python
num_accidents_per_region = pd.DataFrame({'total':accidents_by_department['nom'].value_counts()}).reset_index().rename(columns={'index': 'nom'})
num_accidents_per_region
```

```python
# Merge
accidents_by_region_and_name = accidents_by_department.merge(num_accidents_per_region, on='nom')
region_by_accidents = france_gdf.merge(num_accidents_per_region, on='nom')
```

```python
accidents_by_region_and_name.plot(column='grav')
```

```python
region_by_accidents.plot(column='total',cmap=cmap,scheme='equalinterval',legend=True,figsize=(10,10))
#boxplot', 'equalinterval', 'fisherjenks', 'fisherjenkssampled', 'headtailbreaks', 'jenkscaspall', 'jenkscaspallforced', 
#'jenkscaspallsampled', 'maxp', 'maximumbreaks', 'naturalbreaks', 'quantiles', 'percentiles', 'prettybreaks', 'stdmean', 'userdefined'
```

# By using equal intervals, the total number of accidents in Paris is way higher than in most regions of france. Let's make the same plot by using region population instead

```python
population = pd.read_excel('../data/TCRD_004.xlsx',index_col=[0])
population_filtered = population[['nom','2023 (p)']].copy().rename(columns = {'2023 (p)':'pop_2023'})
population_filtered
```

```python
region_by_accidents_pop2023 = region_by_accidents.merge(population_filtered,on='nom')
region_by_accidents_pop2023['acc_per_hab'] = (region_by_accidents_pop2023['total'] / region_by_accidents_pop2023['pop_2023']) * 1000
```

```python
region_by_accidents_pop2023.sort_values(by=['acc_per_hab'],ascending=[False]).head(10)
```

```python
region_by_accidents_pop2023.plot(column='acc_per_hab',scheme='equalinterval',k=3,cmap=cmap,legend=True,figsize=(10,10))
```

# Maybe the cyclable rods can provide more info

```python
pistes_cyclable = gp.read_file("../data/france-20230901.geojson")
pistes_cyclable = pistes_cyclable.to_crs('epsg:27561')
pistes_cyclable.plot()
```

```python
pistes_cyclable.head(1)
```

# Check some colums and unique values

```python
ame = pd.Series(u for u in pistes_cyclable['ame_d'].unique())
sens_d = pd.Series(u for u in pistes_cyclable['sens_d'].unique())
revet_g = pd.Series(u for u in pistes_cyclable['revet_g'].unique())
access_ame = pd.Series(u for u in pistes_cyclable['access_ame'].unique())
lumiere = pd.Series(u for u in pistes_cyclable['lumiere'].unique())
```

```python
df_unique = pd.DataFrame({'track_type': ame, 'track_direction':sens_d,'track_material':revet_g,'track_acess':access_ame,'lumiere':lumiere})
df_unique
```

# Plot proportions of NA for each col

```python
null_track_type_freq = pistes_cyclable['ame_d'].isnull().sum();
not_null_track_type_freq = pistes_cyclable['ame_d'].notnull().sum();

null_track_direction_freq = pistes_cyclable['sens_d'].isnull().sum();
not_null_track_direction_freq = pistes_cyclable['sens_d'].notnull().sum();

null_track_material_freq = pistes_cyclable['revet_g'].isnull().sum();
not_null_track_material_freq = pistes_cyclable['revet_g'].notnull().sum();

null_track_acess_freq = pistes_cyclable['access_ame'].isnull().sum();
not_null_track_acess_freq = pistes_cyclable['access_ame'].notnull().sum();

null_lumiere_freq = pistes_cyclable['lumiere'].isnull().sum();
not_null_lumiere_freq = pistes_cyclable['lumiere'].notnull().sum();

stats = pd.DataFrame({'Col' : ['Track type','Track dir','Track material','Track acess','Lumiere'],
                     'NA': [null_track_type_freq,null_track_direction_freq,null_track_material_freq,null_track_acess_freq,null_lumiere_freq],
                     'Not NA': [not_null_track_type_freq,not_null_track_direction_freq,not_null_track_material_freq,not_null_track_acess_freq,not_null_lumiere_freq]})

stats.set_index('Col')

stats
```

# Since there are too many null values regarding track material, acess type and presence of ligtht, let's explore track type and track direction

```python
# Starting by track type

pistes_cyclable_type_not_na = pistes_cyclable.loc[pistes_cyclable['ame_d'].notnull()]
print("Number of track types with null values : ", pistes_cyclable_type_not_na['ame_d'].isnull().sum())
```

## Visuzalize the top 5 departments by accidents and extract the first one

```python
top5 = region_by_accidents_pop2023.sort_values(by=['acc_per_hab'],ascending=[False]).head(5)
print(top5[['nom','pop_2023']])
paris = top5.head(1)
paris
```

```python

ax = paris.plot(color="white", edgecolor="black", figsize=(20, 10))

# Drop cols with index_ suffix
accidents_by_region_and_name=accidents_by_region_and_name.loc[:,~accidents_by_region_and_name.columns.str.startswith('index_')]
# Intersect tracks with paris
region_intersect_track_type = gp.sjoin(pistes_cyclable_type_not_na, paris, predicate='intersects')
# Intersect accidents with paris
accidents_intersect_type = gp.sjoin(accidents_by_region_and_name, paris, predicate='within')
# Define a colormap
region_intersect_cmap = colors.LinearSegmentedColormap.from_list(region_intersect_track_type["ame_d"].unique(), list(reversed(["green","yellow","orange","red"])))

region_intersect_track_type.plot(ax=ax,cmap='turbo',column='ame_d',legend=True)
accidents_intersect_type.plot(ax=ax,color="blue",legend=True,alpha=0.5)
```

# There doesnt seem to have a direct relation between accidents and track type. Lets see about the track direction

```python
# Starting by track type

pistes_cyclable_direction_not_na = pistes_cyclable.loc[pistes_cyclable['ame_d'].notnull()]
print("Number of track types with null values : ", pistes_cyclable_direction_not_na['ame_d'].isnull().sum())
```

```python
# Intersect tracks with paris

ax = paris.plot(color="white", edgecolor="black",alpha=0.5, figsize=(20, 10))

region_intersect_track_direction = gp.sjoin(pistes_cyclable_direction_not_na, paris, predicate='intersects')
# Intersect accidents with paris
accidents_intersect_track_direction = gp.sjoin(accidents_by_region_and_name, paris, predicate='within')
# Define a colormap
region_intersect_cmap = colors.LinearSegmentedColormap.from_list(region_intersect_track_direction["sens_d"].unique(), list(reversed(["green","red"])))

region_intersect_track_direction.plot(ax=ax,column='sens_d',cmap=region_intersect_cmap,legend=True)
accidents_intersect_track_direction.plot(ax=ax,column="age",color='blue',legend=True,alpha=0.5)


```

# Accidents by period of the year

```python
months = list(accidents_intersect_track_direction['mois'].unique()).sort
months_order = ['janvier', 'février', 'mars', 'avril',
          'mai', 'juin', 'juillet', 'août',
          'septembre', 'octobre', 'novembre', 'décembre']
```

```python
accidents_intersect_track_direction_grouped = accidents_intersect_track_direction.sort_values('mois', ascending=True).groupby('mois').size().reset_index(name ='Accidents')

accidents_intersect_track_direction_sorted_grouped = accidents_intersect_track_direction_grouped.sort_values('mois', key=lambda s: s.apply(months_order.index), ignore_index=True)
accidents_intersect_track_direction_sorted_grouped
```

```python
import seaborn as sns
sns.set_theme()

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=accidents_intersect_track_direction_sorted_grouped, x="mois",y='Accidents').set_title("Accidents per month in paris")
ax.tick_params(axis='x', labelrotation = 45)
plt.show()
```

# It's cleaar that population has a hight weight on the bicycle , since during the month June and July, we have a incresed number of tourists in Paris. The month of September, is the first after vaccation in France


## Let's make cloroplets, using arrondissement zones in paris

```python
import geoplot as gplt


# Read polygons of arrondissement
arrondis_gdf = gp.read_file("../data/arrondissements.geojson")
arrondis_gdf = arrondis_gdf.to_crs('epsg:27561')

# Drop cols with index_ suffix
#paris_accidents = paris_accidents.loc[:,~paris_accidents.columns.str.startswith('index_')]

# Extract only accidents in paris (by name)
paris_accidents = accidents_by_department.loc[accidents_by_department['nom'] == 'Paris']

# Merge paris accidents with population data and remove cols with index_ (from merge result)
paris_accidents_by_pop2023 = paris_accidents.merge(population_filtered,on='nom')
paris_accidents_by_pop2023 = paris_accidents_by_pop2023.loc[:,~paris_accidents_by_pop2023.columns.str.startswith('index_')]

# Spatial join between arrondissement (polygon) and  accidents in paris (points), using 'intersects predicate
paris_arrondis_accidents_by_pop2023 = gp.sjoin(arrondis_gdf, paris_accidents_by_pop2023, predicate='intersects')

# Group by arronsidement names and merge with popsulation
population_by_arrondissement = paris_arrondis_accidents_by_pop2023.groupby('l_ar').size().reset_index(name='num_accidents')
paris_accidents_by_pop2023_arrondi = paris_arrondis_accidents_by_pop2023.merge(population_by_arrondissement,on='l_ar')

# Ploting result
ax = arrondis_gdf.plot(color="white",edgecolor="black",alpha=0.5, figsize=(20, 12))
paris_accidents_by_pop2023_arrondi.plot(ax=ax,column='num_accidents',cmap=region_intersect_cmap.reversed(),legend=True)

# Plot arrondissement labels based on polygons centroids
paris_accidents_by_pop2023_arrondi.apply(lambda x: ax.annotate(text= x['l_ar'].split(' ')[0], xy=x.geometry.centroid.coords[0], ha='center'), axis=1);
ax.set_title('Number of accidents by arrondissement', fontsize=13);
```

## And plot data

```python
# Order by accident
population_by_arrondissement_ordered = population_by_arrondissement.sort_values('num_accidents')
# Split X axis labels to extract first word
labels = list(population_by_arrondissement_ordered['l_ar'].apply(lambda x: x.split(' ')[0]))

# Define an axis with plot and its parameters
ax = population_by_arrondissement.sort_values('num_accidents').plot(kind='bar',x='l_ar',
                                                               xlabel="Arrondisement",
                                                               ylabel="Number of accidents")
ax.set_xticklabels(labels,rotation=45)

# Plot
plt.show()
```

### They are not so representative, since we cant see where are the zones where accidents occur more ofter. Lets make a grid. A greate function can be found here : https://pygis.io/docs/e_summarize_vector.html

```python
import math
import numpy as np
from shapely.geometry import Polygon, box

def create_grid(feature, shape, side_length):
    '''Create a grid consisting of either rectangles or hexagons with a specified side length that covers the extent of input feature.'''

    # Slightly displace the minimum and maximum values of the feature extent by creating a buffer
    # This decreases likelihood that a feature will fall directly on a cell boundary (in between two cells)
    # Buffer is projection dependent (due to units)
    feature = feature.buffer(20)

    # Get extent of buffered input feature
    min_x, min_y, max_x, max_y = feature.total_bounds

    # Shape area
    area = 0


    # Create empty list to hold individual cells that will make up the grid
    cells_list = []

    # Create grid of squares if specified
    if shape in ["square", "rectangle", "box"]:

        # Adapted from https://james-brennan.github.io/posts/fast_gridding_geopandas/
        # Create and iterate through list of x values that will define column positions with specified side length
        for x in np.arange(min_x - side_length, max_x + side_length, side_length):

            # Create and iterate through list of y values that will define row positions with specified side length
            for y in np.arange(min_y - side_length, max_y + side_length, side_length):

                # Create a box with specified side length and append to list
                cells_list.append(box(x, y, x + side_length, y + side_length))
        est = (max_x - min_x) / length(cells_list)
        north = (max_y - min_y) / length(cells_list)
        area = (est * north)

    # Otherwise, create grid of hexagons
    elif shape == "hexagon":

        # Set horizontal displacement that will define column positions with specified side length (based on normal hexagon)
        x_step = 1.5 * side_length

        # Set vertical displacement that will define row positions with specified side length (based on normal hexagon)
        # This is the distance between the centers of two hexagons stacked on top of each other (vertically)
        y_step = math.sqrt(3) * side_length

        # Get apothem (distance between center and midpoint of a side, based on normal hexagon)
        apothem = (math.sqrt(3) * side_length / 2)

        # Set column number
        column_number = 0

        # Create and iterate through list of x values that will define column positions with vertical displacement
        for x in np.arange(min_x, max_x + x_step, x_step):

            # Create and iterate through list of y values that will define column positions with horizontal displacement
            for y in np.arange(min_y, max_y + y_step, y_step):

                # Create hexagon with specified side length
                hexagon = [[x + math.cos(math.radians(angle)) * side_length, y + math.sin(math.radians(angle)) * side_length] for angle in range(0, 360, 60)]

                # Append hexagon to list
                cells_list.append(Polygon(hexagon))

            # Check if column number is even
            if column_number % 2 == 0:

                # If even, expand minimum and maximum y values by apothem value to vertically displace next row
                # Expand values so as to not miss any features near the feature extent
                min_y -= apothem
                max_y += apothem

            # Else, odd
            else:

                # Revert minimum and maximum y values back to original
                min_y += apothem
                max_y -= apothem

            # Increase column number by 1
            column_number += 1
        area  = (3 * math.sqrt(3) * pow(side_length,2)) / 2

    # Else, raise error
    else:
        raise Exception("Specify a rectangle or hexagon as the grid shape.")

    # Create grid from list of cells
    grid = gp.GeoDataFrame(cells_list, columns = ['geometry'], crs = "epsg:27561")

    # Create a column that assigns each grid a number
    grid["Grid_ID"] = np.arange(len(grid))

    # Return grid
    return grid,area
```

### Create hexagons

```python
# Create heaxagon
area_grid,area = create_grid(feature = paris_arrondis_accidents_by_pop2023, shape = 'hexagon', side_length = 200)
#cell_grid["cell_id"] = cell_grid.index + 1
#cell_grid.head(5)
area_grid.plot()
```

### Remove cells outside Paris

```python
area_grid_paris = gp.sjoin(area_grid, arrondis_gdf, how='inner', predicate='intersects')
# Remove fields from spatial join
area_grid_paris = area_grid_paris.loc[:,~area_grid_paris.columns.str.startswith('index_')]
area_grid_paris.reset_index(inplace=True)
area_grid_paris.plot()
```

### Later, we fuse the cells with the accidents layer

```python
## Remove index columns from previous merge operations, if necessary

paris_accidents = paris_accidents.loc[:,~paris_accidents.columns.str.startswith('index_')]
paris_accidents_by_pop2023_arrondi = paris_accidents_by_pop2023_arrondi.loc[:,~paris_accidents_by_pop2023_arrondi.columns.str.startswith('index_')]
print("Dataframes have columns with name index ? :", any(paris_accidents.columns.str.startswith('index_')) 
                                                      and any(paris_accidents_by_pop2023_arrondi.columns.str.startswith('index_')))
```

```python
# Grab all dataset of accidents, within paris
accidents_paris = gp.sjoin(accidents_by_region_and_name, paris_accidents_by_pop2023_arrondi, how='inner', predicate='within')
accidents_paris = accidents_paris.loc[:,~accidents_paris.columns.str.startswith('index_')]
accidents_paris.columns = accidents_paris.columns.str.rstrip("_left")
accidents_paris.columns = accidents_paris.columns.str.rstrip("_right")

accidents_paris.plot()
```

```python
accidents_paris.head()
```

## Aggreate accidents by cell grid

```python
#######
# Perform spatial join, merging attribute table of wells point and that of the cell with which it intersects
# op = "intersects" also counts those that fall on a cell boundary (between two cells)
# op = "within" will not count those fall on a cell boundary

points_within = points_within.loc[:,~points_within.columns.str.startswith('index_')]

# Merging accidents data with grid data by spatial intersection boundary
grid_accidents = gp.sjoin(points_within, area_grid_paris, how='left', predicate='within')

# Add a field with constant value of 1
grid_accidents['n_acc'] = 1

# Compute stats per grid cell -- aggregate fires to grid cells with dissolve
dissolve = grid_accidents.dissolve(by="index_right", aggfunc="count")

# put this into cell
area_grid_paris.loc[dissolve.index, 'n_acc'] = dissolve.n_acc.values

# Fill the NaN values (cells without any points) with 0 if we want to see
area_grid_paris['n_acc'] = area_grid_paris['n_acc'].fillna(0)
#cell_grid = cell_grid.within(paris_accidents_by_pop2023_arrondi)]

```

```python
# Plot data
ax = paris_accidents_by_pop2023_arrondi.plot(markersize=.1, figsize=(15, 10),color="None",edgecolor="red",legend=True)
legend_intervals = [int(area_grid_paris["n_acc"].min()),5,10,15,int(area_grid_paris["n_acc"].max())]
accidents_paris.plot(ax = ax,marker = 'o', color = 'dimgray', markersize = 3)
area_grid_paris.plot(ax = ax, column = "n_acc", 
                cmap=cmap,edgecolor="lightseagreen", linewidth = 0.5, alpha = 0.8,legend = True,
                legend_kwds={
                    "shrink":.68,
                    "format": "%g",
                    'label': "Accidents",
                    "pad": 0.01,
                    #"ticks" : legend_intervals
                })
# Set title
ax.set_title(f'Grid of accidents per ±{area:.0f} m2', fontdict = {'fontsize': '15', 'fontweight' : '3'})
plt.show()
```

# Lets test some interpolation methods to fill the empty cells and get better results


## Give a small subset of points to train on

```python
def f(x):
    """Function to be approximated by polynomial interpolation."""
    return x * np.sin(x)

```

```python
samples = area_grid_paris["n_acc"].to_list()

# whole range we want to plot
x_plot = np.linspace(min(samples), max(samples), len(samples))

# To make it interesting, we only give a small subset of points to train on.
x_train =  samples
rng = np.random.RandomState(0)
x_train = np.sort(rng.choice(x_train, size=10, replace=False))
y_train = f(x_train)

# create 2D-array versions of these arrays to feed to transformers
X_train = x_train[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]
```

```python

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

# plot function
lw = 2
fig, ax = plt.subplots()
ax.set_prop_cycle(
    color=["black", "teal", "yellowgreen", "gold", "darkorange", "tomato"]
)
ax.plot(x_plot, f(x_plot), linewidth=lw, label="ground truth")

# plot training points
ax.scatter(x_train, y_train, label="training points")

# polynomial features
for degree in [3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3))
    model.fit(X_train, y_train)
    y_plot = model.predict(X_plot)
    ax.plot(x_plot, y_plot, label=f"degree {degree}")

# B-spline with 4 + 3 - 1 = 6 basis functions
model = make_pipeline(SplineTransformer(n_knots=4, degree=3), Ridge(alpha=1e-3))
model.fit(X_train, y_train)

y_plot = model.predict(X_plot)
ax.plot(x_plot, y_plot, label="B-spline")
ax.legend(loc="lower center")
ax.set_ylim(-40, 40)
plt.show()

```

### Splines make a good job on fitting well the data, and provide the same time, some paramenters to control extrapolation. However, seasonal effects may cut an expected periodic continuation of the underlying signal. Periodic splines could be used in such case

```python
def g(x):
    """Function to be approximated by periodic spline interpolation."""
    return np.sin(x) - 0.7 * np.cos(x * 3)
```

```python

y_train = g(x_train)

# Extend the test data into the future:
x_plot_ext = np.linspace(min(samples), max(samples) + 10, len(samples) + 100)
X_plot_ext = x_plot_ext[:, np.newaxis]

lw = 2
fig, ax = plt.subplots()
ax.set_prop_cycle(color=["black", "tomato", "teal"])
ax.plot(x_plot_ext, g(x_plot_ext), linewidth=lw, label="ground truth")
ax.scatter(x_train, y_train, label="training points")

for transformer, label in [
    (SplineTransformer(degree=3, n_knots=10), "spline"),
    (
        SplineTransformer(
            degree=3,
            knots=np.linspace(0, 2 * np.pi, 10)[:, None],
            extrapolation="periodic",
        ),
        "periodic spline",
    ),
]:
    model = make_pipeline(transformer, Ridge(alpha=1e-3))
    model.fit(X_train, y_train)
    y_plot_ext = model.predict(X_plot_ext)
    ax.plot(x_plot_ext, y_plot_ext, label=label)

ax.legend()
fig.show()
```

```python

```

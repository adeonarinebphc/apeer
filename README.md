# The aPEER Algorithm

July 13, 2023
Updated: Oct 25, 2023

By: Andrew D

## Background
This script will reproduce the analysis in Figure 4B, which illustrates how the Stroke Belt
can be assembled from 177 different pollution indicators in the EPA AirToxScreen Database.

## STEP 1: you will need to download the following libraries:
```
seaborn
pandas
sklearn
matplotlib
numpy
descartes
openpyxl
```

## STEP 2: download the data files to perform the analysis:

```
EJSCREEN_2021_USPR_Tracts.csv
https://drive.google.com/file/d/1siIosFHP9JK8VsjY6DHcoQ8Kd_llHF6t/view?usp=sharing

PLACES__Census_Tract_Data__GIS_Friendly_Format___2021_release.csv
https://drive.google.com/file/d/1ftVtQEAFJ3MqLWD-eh3_ipCOy5H2GAoA/view?usp=sharing

PLACES__County_Data__GIS_Friendly_Format___2021_release.tsv
https://drive.google.com/file/d/1ftVtQEAFJ3MqLWD-eh3_ipCOy5H2GAoA/view?usp=sharing

2018_Toxics_Ambient_Concentrations.updated.tract.tsv
https://drive.google.com/file/d/1wcBGlTlS2ZJSgk2_BQlV4SdBofHW_yLF/view?usp=sharing

2018_Toxics_Ambient_Concentrations.updated.county.tsv
https://drive.google.com/file/d/1KnHRpmv2Z_ee6nDfN-agO95qdmit20Mx/view?usp=sharing

GeoJSON File - Tracts:
https://drive.google.com/file/d/1v0uOkMNQJDr1F2UxN_pUbvcafLw2GqAw/view?usp=sharing

GeoJSON File - Counties:
https://drive.google.com/file/d/1T1djNPuDqTkSlD2kRyhHw3coxuBjgLpx/view?usp=sharing

```

## STEP 3: now run the script using the command:

```
python3 apeer.py
```

## STEP 4: 
The output from the script includes the Jaccard Index and p-values for
different clusters, with maps generated for each cluster such as 
apeer_county.2.pdf.map.png (for k=2 with counties).


# The aPEER Algorithm

July 13, 2023

By: Andrew D

# STEP 1: 
- you will need to download the following libraries:
- seaborn
- pandas
- sklearn
- matplotlib
- numpy
- descartes
- openpyxl

# STEP 2: 
- download AirToxScreen data (2018) from the EPA website, and save it as a tsv "2018_Toxics_Ambient_Concentrations.tract.tsv"
- Download the file from here: https://drive.google.com/file/d/1IAwhEqD-DuBShfXH3Q1uUcY2y2bckTr2/view?usp=share_link

# STEP 3: 
- download EJSCREEN data (https://www.epa.gov/ejscreen/download-ejscreen-data) and calculate the population at the tract level, saving the file as "EJSCREEN_2021_USPR_Tracts.csv" (2nd column = FIPS code, 3rd column = population)
- Download the file from here: https://drive.google.com/file/d/1H4bod9ZcS7XjV5tV5kZVfZiwk6Cj4YfD/view?usp=sharing

# STEP 4: 
- now run the script using the command:

```
python3 apeer.py
```

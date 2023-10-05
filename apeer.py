import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from descartes import PolygonPatch
from json import dumps
import json
from collections import defaultdict
import csv
import openpyxl
import os
import scipy.stats as stats

import matplotlib
matplotlib.use("Agg")

def GetMainLists():

	# print out CDC PLACES counties / tracts
	countylist = LoadPLACES("county")
	tractlist = LoadPLACES("tract")
	epacounties = LoadEPAData("county")
	epatracts = LoadEPAData("tract")
	chemdata_county = LoadChemicalData("county")
	chemdata_tract = LoadChemicalData("tract")
	master_counties = defaultdict(str)
	master_tracts = defaultdict(str)

	tcnt1 = 1
	hcountylist = defaultdict(str)
	for tcounty in countylist.index:
		#print(str(tcnt) + "\t" + tcounty)
		hcountylist[tcounty] = ""
	for tcounty in hcountylist:
		master_counties[tcounty] = ""
		tcnt1 += 1
	print("Number of CDC counties: " + str(tcnt1))

	tcnt2 = 1
	htractlist = defaultdict(str)
	for ttract in tractlist.index:
		#print(str(tcnt) + "\t" + ttract)
		htractlist[ttract] = ""
	for ttract in htractlist:
		master_tracts[ttract] = ""
		tcnt2 += 1
	print("Number of CDC counties: " + str(tcnt2))

	# EPA EJScreen
	ecnt1 = 1
	hcountylist2 = defaultdict(str)
	for tcounty in epacounties.index:
		#print(str(tcnt) + "\t" + tcounty)
		hcountylist2[tcounty] = ""
	for tcounty in hcountylist2:
		master_counties[tcounty] = ""
		ecnt1 += 1
	print("Number of EPA counties: " + str(ecnt1))

	ecnt2 = 1
	htractlist2 = defaultdict(str)
	for ttract in epatracts.index:
		#print(str(tcnt) + "\t" + tcounty)
		htractlist2[ttract] = ""
	for ttract in htractlist2:
		master_tracts[ttract] = ""
		ecnt2 += 1
	print("Number of EPA tracts: " + str(ecnt2))
	
	# EPA EJScreen
	ccnt1 = 1
	hcountylist3 = defaultdict(str)
	for tcounty in chemdata_county.index:
		#print(str(tcnt) + "\t" + tcounty)
		hcountylist3[tcounty] = ""
	for tcounty in hcountylist3:
		master_counties[tcounty] = ""
		ccnt1 += 1
	print("Number of Chem counties: " + str(ccnt1))

	ccnt2 = 1
	htractlist3 = defaultdict(str)
	for ttract in chemdata_tract.index:
		#print(str(tcnt) + "\t" + tcounty)
		htractlist3[ttract] = ""
	for ttract in htractlist3:
		master_tracts[ttract] = ""
		ccnt2 += 1
	print("Number of Chem tracts: " + str(ccnt2))
	
	# get intersections
	mcounty = 0
	mtract = 0
	finalcounties = []
	finaltracts = []
	for tcounty in master_counties:
		if (tcounty in hcountylist) and (tcounty in hcountylist2) and (tcounty in hcountylist3):
			finalcounties.append(tcounty)
			if len(tcounty.strip()) != 5:
				print("County Potential Error: " + tcounty)
			mcounty += 1
	for ttract in master_tracts:
		if (ttract in htractlist) and (ttract in htractlist2) and (ttract in htractlist3):
			finaltracts.append(ttract)
			if len(ttract.strip()) != 11:
				print("Tract Potential Error: " + ttract)
			mtract += 1
	print("Number of common counties: " + str(mcounty))
	print("Number of common tracts: " + str(mtract))
	
	return finalcounties, finaltracts

def LoadPLACES(tmode):

	# load counties
	includeCols = ["ACCESS2_CrudePrev","ARTHRITIS_CrudePrev","BINGE_CrudePrev","BPHIGH_CrudePrev","BPMED_CrudePrev","CANCER_CrudePrev","CASTHMA_CrudePrev","CERVICAL_CrudePrev","CHD_CrudePrev","CHECKUP_CrudePrev","CHOLSCREEN_CrudePrev","COLON_SCREEN_CrudePrev","COPD_CrudePrev","COREM_CrudePrev","COREW_CrudePrev","CSMOKING_CrudePrev","DENTAL_CrudePrev","DEPRESSION_CrudePrev","DIABETES_CrudePrev","GHLTH_CrudePrev","HIGHCHOL_CrudePrev","KIDNEY_CrudePrev","LPA_CrudePrev","MAMMOUSE_CrudePrev","MHLTH_CrudePrev","OBESITY_CrudePrev","PHLTH_CrudePrev","SLEEP_CrudePrev","STROKE_CrudePrev","TEETHLOST_CrudePrev"]

	cdcdata = defaultdict(lambda: defaultdict(str))
	#bpath = "c:\\Apache24\\htdocs\\bphc\\data"
	bpath = ""
	popdata = defaultdict(int)
	rowcnt = 0
	
	tfile = "PLACES__County_Data__GIS_Friendly_Format___2021_release.tsv"
	tsep = "\t"
	tcol = "CountyFIPS"
	if tmode == "tract":
		tfile = "PLACES__Census_Tract_Data__GIS_Friendly_Format___2021_release.csv"
		tsep = ","
		tcol = "TractFIPS"
	
	#print("Loading " + tfile)
	tplaces = pd.read_table(tfile, delimiter=tsep, dtype=str)
	tplaces = tplaces.set_index(tcol)
	#tplaces.index = tplaces.index.map(str)
	tindex = tplaces.index
	tplaces[includeCols] = tplaces[includeCols].apply(pd.to_numeric)
	# check to see if fips is 5 digits
	fipscodes = tplaces.index
	for x in range(0, len(fipscodes)):
	
		tindex = tplaces.index[x]
		
		if tmode == "county":
			if len(str(tindex)) == 4:
				nindex = "0" + tindex
				#print("Renaming " + tindex + " to " + nindex)
				tplaces = tplaces.rename(index={tindex: nindex})

		if tmode == "tract":
			if len(str(tindex)) == 10:
				#print("Padding zero: " + tindex)
				nindex = "0" + tindex
				#print("Renaming " + tindex + " to " + nindex)
				tplaces = tplaces.rename(index={tindex: nindex})
				
		#if ((x % 100) == 0):
		#	print(str(x) + ". Index: " + tindex)

	tfinal = tplaces[includeCols]
	#print("LoadPLACES:")
	#print(tfinal)

	return tfinal

def LoadEPAData(tmode):

	# get file
	tfile = "EJSCREEN_2021_USPR_Tracts.csv"
	
	# load into dataframe
	# https://www.epa.gov/ejscreen/overview-environmental-indicators-ejscreen
	
	#6	REGION	US EPA Region number
	#7	ACSTOTPOP	Total population
	#8	MINORPCT	% people of color
	#9	LOWINCPCT	% low income
	#10	LESSHSPCT	% less than high school education
	#11	LINGISOPCT	% linguistically isolated
	#12	UNDER5PCT	% under age 5
	#13	OVER64PCT	% over age 64
	#14	PRE1960PCT	Lead paint
	#15	UNEMPPCT	Unemployment rate
	#16	VULEOPCT	Demographic Index
	#17	DSLPM	2017 Diesel particulate matter
	#18	CANCER	2017 Air toxics cancer risk
	#19	RESP	2017 Air toxics respiratory HI
	#20	PTRAF	Traffic proximity
	#21	PWDIS	Wastewater discharge
	#22	PNPL	Superfund proximity
	#23	PRMP	RMP facility proximity
	#24	PTSDF	Hazardous waste proximity
	#25	OZONE	Ozone
	#26	PM25	Particulate Matter 2.5
	#27	UST	Underground storage tanks
	
	#collist = ['ACSTOTPOP', "CANCER", "DSLPM", "RESP", "PTRAF", "PWDIS", "PNPL", "PRMP", "PTSDF", "OZONE", "PM25", "UST", "PRE1960PCT", "MINORPCT", "LOWINCPCT", "LESSHSPCT", "LINGISOPCT", "UNDER5PCT", "OVER64PCT", "UNEMPPCT"]
	collist = ['ACSTOTPOP', "MINORPCT", "LOWINCPCT", "LESSHSPCT", "LINGISOPCT", "UNDER5PCT", "OVER64PCT", "UNEMPPCT", \
	"PRE1960PCT", "DSLPM", "CANCER", "RESP", "PTRAF", "PWDIS", "PNPL", "PRMP", "PTSDF", "OZONE", "PM25", "UST"]
	
	collist_final = []
	for tid in collist:
		if tid != "CANCER":
			collist_final.append(tid)
		if tid == "CANCER":
			collist_final.append("CANCER_AIR")
	
	#including PTRAF or PWDIS removes the clusters
	raw_data = pd.read_csv(tfile, header=0, low_memory=False)
	raw_data.set_index("ID", inplace=True)
	raw_data = raw_data[collist]
	
	# average county data
	ttable = []
	tempCountyData = defaultdict(lambda: defaultdict(float))
	tempTractData = defaultdict(lambda: defaultdict(float))
	county_data = pd.DataFrame()	
	countytotal = defaultdict(float)
	tractdata = defaultdict(float)

	# get county pop totals for weights
	for x in range(0, len(raw_data.index)):
		tfips = str(raw_data.index[x])
		nfips = str(tfips)
		if len(nfips) == 10:
			nfips = "0" + nfips
		tcounty = nfips[0:5]
		countytotal[tcounty] += raw_data.loc[raw_data.index[x], "ACSTOTPOP"]
	
	for x in range(0, len(raw_data.index)):
		tfips = raw_data.index[x]
		nfips = str(tfips)
		if len(nfips) == 10:
			nfips = "0" + nfips
		tcounty = str(nfips)[0:5]
		tstate = str(nfips)[0:2]
		# ignore protectorates
		if int(tstate) < 60:
			weight = raw_data.loc[raw_data.index[x], "ACSTOTPOP"] / countytotal[tcounty]
			for tcol in collist:
				ncol = tcol
				if ncol == "CANCER":
					ncol = "CANCER_AIR"
				if tmode == "county":
					tempCountyData[tcounty][ncol] += raw_data.loc[tfips, tcol] * weight
				if tmode == "tract":
					tempCountyData[nfips][ncol] += raw_data.loc[tfips, tcol]

	for tfips in tempCountyData:
		trow = [tfips]
		for tcol in collist:
			ncol = tcol
			if ncol == "CANCER":
				ncol = "CANCER_AIR"
			trow.append(tempCountyData[tfips][ncol])
		ttable.append(trow)

	clist = ["fips"] + collist_final
	epa_data = pd.DataFrame(ttable, columns=clist)
	epa_data.set_index("fips", inplace=True)
	epa_data[collist_final] = epa_data[collist_final].apply(pd.to_numeric)
	epa_data = epa_data.replace(np.nan, 0)
	epa_data = epa_data.drop("ACSTOTPOP", axis=1)
	
	#print(epa_data)

	return epa_data

def LoadChemicalData(tmode):

	if tmode == "county":
		chemdata_county = pd.read_csv("2018_Toxics_Ambient_Concentrations.updated.county.tsv", header=0, index_col=0, sep="\t", low_memory=False)
		tindex = []
		for tid in chemdata_county.index:
			nid = str(tid)
			if len(nid) == 4:
				nid = "0" + nid
			tindex.append(nid)
		chemdata_county.index = tindex
		
		return chemdata_county

	if tmode == "tract":
		chemdata_tract = pd.read_csv("2018_Toxics_Ambient_Concentrations.updated.tract.tsv", header=0, index_col=0, sep="\t")
		tindex = []
		for tid in chemdata_tract.index:
			nid = str(tid)
			if len(nid) == 10:
				nid = "0" + nid
			tindex.append(nid)
		chemdata_tract.index = tindex
		
		return chemdata_tract


def CalcFisherPVal(list1, list2, overlap, popsize):
	
	a = popsize - list1 - list2 + overlap
	b = list2 - overlap
	c = list1 - overlap
	d = overlap

	odds_ratio = 1
	p_value = 1
	if (a > 0) and (b > 0) and (c > 0) and (d > 0):
		data = [[a, b], [c, d]]	  
		odd_ratio, p_value = stats.fisher_exact(data)
	
	return p_value	

def GetCanonicalStrokeBelt(tmode, countylist, tractlist):

	# Download from here for 2017-2019 using Stroke mortality, and use the export function
	# Texas: 48
	# Florida: 12
	strokebelt_states = ["01", "05", "13", "18", "21", "22", "28", "37", "45", "47", "51"]
	
	tcnt = 0
	#tcutoff = 70.6
	tcutoff = 78
	tfile = "cdc_stroke_mort_2017_2019.csv"
	beltlist = defaultdict(float)
	with open(tfile, "r") as infile:
		csv_reader = csv.reader(infile, delimiter=",")
		for row in csv_reader:
			if tcnt == 0:
				theader = row
			if tcnt > 0:
				for row in csv_reader:
					tfips = row[0]
					if len(tfips) == 4:
						tfips = "0" + tfips
					tstate = tfips[0:2]
					
					inState = 1
					if tmode == "canonical":
						if tstate not in strokebelt_states:
							inState = 0
					
					if inState == 1:
						tvalue = 0
						if (row[2] == "") or (row[2] == "-1"):
							tvalue = 0
						else:
							tvalue = float(row[2])
						if tvalue > tcutoff:
							beltlist[tfips] = tvalue
			tcnt += 1
	infile.close()
	
	# now compare beltlist to counties, tracts
	county_belt = defaultdict(str)
	tract_belt = defaultdict(str)
	for tfips in beltlist:
		county_belt[tfips] = "1"
	for tfips in tractlist:
		tcounty = tfips[0:5]
		if tcounty in beltlist:
			tract_belt[tfips] = "1"
		
	return county_belt, tract_belt

def LoadClusterData(tfile):

	numlines = 0
	clusterdata = defaultdict(str)
	with open(tfile, "r") as infile:
		theader = infile.readline()
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			tfips = ldata[0]
			tclust = ldata[1]
			clusterdata[tfips] = tclust
			numlines += 1
	infile.close()
	
	print("Loaded: " + str(numlines))

	return clusterdata

def CalcFastJaccard(clustdata, countybelt, totalcnt):

	# Jaccard = intersection / union
	jaccard_clust = defaultdict(str)
	list2cnt = 0
	for tid in countybelt:
		list2cnt += 1

	# get list of clusters, counts
	clustlist = defaultdict(int)
	tintersect_table = defaultdict(int)
	tunion_table = defaultdict(int)
	list1cnt_table = defaultdict(int)
	clustlist_table = defaultdict(str)
	curr_clustdata = defaultdict(lambda: defaultdict(int))
	unionlist_table = defaultdict(lambda: defaultdict(int))

	# get cluster data
	for tfips in clustdata:
		tclust = clustdata[tfips]
		clustlist[tclust] = 1
		curr_clustdata[tclust][tfips] = 1
		list1cnt_table[tclust] += 1
		unionlist_table[tclust][tfips] = 1
		tunion_table[tclust] += 1
		if tfips in countybelt:
			tintersect_table[tclust] += 1

	# calculate jaccard / p-value for each cluster
	max_jaccard = 0
	pval_max = 1
	for tclust in clustlist:
		
		# get union
		for tfips in countybelt:
			if unionlist_table[tclust][tfips] != 1:
				tunion_table[tclust] += 1

		# calc p-value
		pval = CalcFisherPVal(list1cnt_table[tclust], list2cnt, tintersect_table[tclust], totalcnt)		

		# calculate Jaccard
		tjaccard = tintersect_table[tclust] / tunion_table[tclust]
		jaccard_clust[tclust] = str(tjaccard)
                #print("Cluster: " + clustnum + "\t" + str(tjaccard))
		if tjaccard > max_jaccard:
			max_jaccard = tjaccard
			pval_max = pval
	
	return max_jaccard, pval_max


def LoadMapJSON(tmode):

	# Map the data
	tfile = "us_counties.geojson"
	if tmode == "tract":
		tfile = "tracts.geojson"

	print("LoadMapJSON: " + tfile)
	with open(tfile, "r") as json_file:
		json_data = json.load(json_file)

	return json_data

def GetPopData():

	# get file
	tcnt = 0
	popdata = defaultdict(float)
	tfile = "EJSCREEN_2021_USPR_Tracts.csv"
	with open(tfile, "r") as infile:
		for line in infile:
			ldata = line.split(',')
			#print(str(line))
			if tcnt > 0:
				tfips = ldata[1]
				tpop = ldata[2]
				popdata[tfips] = float(tpop)
			tcnt += 1

	return popdata

def GetCountyData(popdata, tfile):
	
	# calculate county-level populations
	countypop = defaultdict(float)
	for tfips in popdata:
		nfips = tfips
		if len(nfips) == 10:
			nfips = "0" + nfips
		tcounty = nfips[0:5]
		countypop[tcounty] += popdata[tfips]
	
	# now convert the chemical data to county level
	lcnt = 0
	countydata = defaultdict(lambda: defaultdict(float))
	with open(tfile, "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			if lcnt == 0:
				theader = ldata
				#print("County header: " + str(theader))
				print("Debug 1b - Number of cols: " + str(len(theader)))
			if lcnt > 0:
				#tdesc = ldata[2]
				#if tdesc.find("Entire") == -1:
				tfips = ldata[0]
				if len(tfips) == 10:
					tfips = "0" + tfips
				tcounty = tfips[0:5]
				if tcounty in countypop:
					tweight = popdata[tfips] / countypop[tcounty]
					for x in range(1, len(theader)):
						nval = float(ldata[x]) * float(tweight)
						countydata[tcounty][theader[x]] += nval
			lcnt += 1
	infile.close()
	
	# write out county data
	print("Debug 1x - Number of columns: " + str(len(theader)))
	ofile = "2018_Toxics_Ambient_Concentrations.county.tsv"
	f = open(ofile, "w")
	tline = "FIPS\t"
	for x in range(1, len(theader)):
		print(str(x) + "\t" + theader[x])
		tline = tline + theader[x] + "\t"
	tline = tline.strip()
	f.write(tline + "\n")
	for tfips in countydata:
		tline = tfips + "\t"
		for x in range(1, len(theader)):
			tline = tline + str(countydata[tfips][theader[x]]) + "\t"
		tline = tline.strip()
		f.write(tline + "\n")
	f.close()
	
	return ofile
	

def RunPCA(pcadata, tfile, ttitle):

	pca = PCA()
	pcadata = pcadata.replace(np.nan, 0)
	
	Xt = pca.fit_transform(pcadata)
	# print PC variance explained
	exp_var_pca = pca.explained_variance_ratio_
	plt.close()

	dropdata = []

	return Xt

def PCA_Kmeans(Xt_pca, numclust, pcadata, pcafile, clustfile, clust_tsv):

	cluster_colors = ["red", "green", "blue", "purple", "yellow", "violet", "orange"]

	PCA_components = pd.DataFrame(Xt_pca)
	model = KMeans(n_clusters = numclust)
	model.fit(PCA_components.iloc[:,:2])
	labels = model.predict(PCA_components.iloc[:,:2])
	
	colorlist = []
	for x in range(0, len(labels)):
		colorlist.append(cluster_colors[labels[x]])
	
	plt.close()
	covidtable = []
	covidindex = []
	clustertable = []
	for i, name in enumerate(pcadata.index):
	
		# combine with COVID-19 data
		tfips = str(name)
		if len(tfips) == 4:
			tfips = '0' + tfips
		if len(tfips) == 10:
			tfips = '0' + tfips		

		# list of clusters
		trow = []
		tclust = []
		tclust.append(tfips)
		tclust.append(labels[i])			
		clustertable.append(tclust)
			
	clusterframe = pd.DataFrame(clustertable, columns=["county", "cluster"])
	clusterframe = clusterframe.set_index("county")
	clusterframe.to_csv(clust_tsv, sep="\t")

	return clusterframe

def ShowBinaryMap(beltlist, mapfile, json_data, backcolor, forecolor):

	# set colors
	tcolors = ["#ff0000", "#00ff00", "#0000ff", "#6a0dad", "#ffff00"]

	plt.close()
	fig = plt.figure(figsize=(25, 16))
	ax = fig.gca()
	for titem in json_data["features"]:
		tfips = titem["properties"]["GEOID"]
		if int(tfips[0:1]) < 6:
			get_color = backcolor
			if tfips in beltlist:
				get_color = forecolor
			#get_color = tcolors[color_index]
			poly = titem["geometry"]
			if str(poly) != "None":
				#print(tfips + "\t" + str(get_color))
				ax.add_patch(PolygonPatch(poly, fc=get_color, ec=get_color, alpha=1, zorder=2))
	
	ax.set_xlim(-130, -60)
	ax.set_ylim(23, 50)
	ax.set_facecolor('xkcd:white')

	fig.tight_layout()
	plt.axis("off")
	plt.savefig(mapfile)

def ShowClustersOnMap(tmode, datatable, numclust, tfile, json_data):

	# make filenames
	#bpath = "data"
	pcafile = tfile + ".pca.pdf"
	kfile = tfile + ".kmeans.pca.pdf"
	cluster_file = tfile + ".kmeans.pca.clusters.pdf"
	cluster_tsv = tfile + "." + str(numclust) + ".cluster.tsv"
	mapfile = tfile + ".map.png"
	
	# PCA
	datatable = datatable.replace(np.nan, 0)
	Xt_epa = RunPCA(datatable, pcafile, tfile)
	epa_clust = PCA_Kmeans(Xt_epa, numclust, datatable, kfile, cluster_file, cluster_tsv)
	epa_clust.to_csv(cluster_tsv, sep="\t")

	plt.close()
	plt.axis("off")
	fig = plt.figure(figsize=(5.5, 3.5))
	ax = fig.gca()
	
	# get which cluster has more FIPS codes in it (larger in size) - set this as the gray background
	# and set the smaller cluster as blue
	light_gray = "#f5f5f5"
	blue_color = "blue"
	clust0 = 0
	clust1 = 0
	if numclust == 2:
		for tfips in epa_clust.index:
			tclust = epa_clust.loc[tfips, "cluster"]
			if (tclust == 0):
				clust0 += 1
			if (tclust == 1):
				clust1 += 1
		if (clust0 >= clust1):
			tcolors = ["#f5f5f5", "blue"]
		if (clust0 < clust1):
			tcolors = ["blue", "#f5f5f5"]
	
	# assign colors - stroke belt = red, other = gray
	# if iowa (fips=19) is in the cluster, assign it gray
	#tcolors = ["blue", "red", "green", "purple", "orange", "yellow"]
	if numclust == 3:
		tcolors = ["red", "blue", "green"]
	if numclust == 4:
		tcolors = ["blue", "red", "green", "purple", "orange", "#f5f5f5"]
		#tcolors = ["#f5f5f5", "#f5f5f5", "#f5f5f5", "blue"]
	if numclust == 5:
		tcolors = ["blue", "red", "green", "purple", "orange", "#f5f5f5"]
	#tcolors = ["red", "green"]
	for titem in json_data["features"]:
		tfips = titem["properties"]["GEOID"]
		if tfips in epa_clust.index:
			if (tfips[0:2] == "19") and ((len(tfips) == 5) or (len(tfips) == 11)):
				if epa_clust.loc[tfips, "cluster"] == 0:
					#tcolors = ["green", "red"]
					if numclust == 3:
						tcolors = ["red", "blue", "green"]
					if numclust == 4:
						tcolors = ["red", "green", "purple", "blue"]
					if numclust == 5:
						tcolors = ["red", "blue", "green", "purple", "orange", "yellow"]
	
	# now plot on map
	for titem in json_data["features"]:
		tfips = titem["properties"]["GEOID"]
		if tfips in epa_clust.index:			
			color_index = epa_clust.loc[tfips, "cluster"]
			get_color = tcolors[color_index]
			poly = titem["geometry"]
			if poly is not None:
				ax.add_patch(PolygonPatch(poly, fc=get_color, ec=get_color, alpha=1, zorder=2))
	
	#ax.set_xlim(-180, -60)
	#ax.set_ylim(20, 80)
	ax.set_xlim(-130, -60)
	ax.set_ylim(23, 50)
	ax.set_facecolor('xkcd:white')
	#plt.xlabel("Longitude")
	#plt.ylabel("Latitude")
	fig.tight_layout()
	plt.axis("off")

	fig.tight_layout()
	plt.savefig(mapfile, dpi=1200)

	return cluster_tsv


#########################
### Run main analysis ###
#########################

# Background
# This script will reproduce the analysis in Figure 4B, which illustrates how the Stroke Belt
# can be assembled from 177 different pollution indicators in the EPA AirToxScreen Database.

# STEP 1: you will need to download the following libraries:
#seaborn
#pandas
#sklearn
#matplotlib
#numpy
#descartes
#openpyxl

# STEP 2: download the data files to perform the analysis:

# EJSCREEN_2021_USPR_Tracts.csv
# https://drive.google.com/file/d/1siIosFHP9JK8VsjY6DHcoQ8Kd_llHF6t/view?usp=sharing

# PLACES__Census_Tract_Data__GIS_Friendly_Format___2021_release.csv
# https://drive.google.com/file/d/1ftVtQEAFJ3MqLWD-eh3_ipCOy5H2GAoA/view?usp=sharing

# PLACES__County_Data__GIS_Friendly_Format___2021_release.tsv
# https://drive.google.com/file/d/1ftVtQEAFJ3MqLWD-eh3_ipCOy5H2GAoA/view?usp=sharing

# 2018_Toxics_Ambient_Concentrations.updated.tract.tsv
# https://drive.google.com/file/d/1wcBGlTlS2ZJSgk2_BQlV4SdBofHW_yLF/view?usp=sharing

# 2018_Toxics_Ambient_Concentrations.updated.county.tsv
# https://drive.google.com/file/d/1KnHRpmv2Z_ee6nDfN-agO95qdmit20Mx/view?usp=sharing

# STEP 3: now run the script using the command:
# python3 apeer.py

# STEP 4: the output from the script includes the Jaccard Index and p-values for
# different clusters, with maps generated for each cluster such as 
# apeer_county.2.pdf.map.png (for k=2 with counties).

####################
### Main Program ###
####################
chemdata_tract = LoadChemicalData("tract")
chemdata_county = LoadChemicalData("county")
finalcounties, finaltracts = GetMainLists()

# use the controlled list of counties and tracts
chemdata_tract = chemdata_tract[chemdata_tract.index.isin(finaltracts)]
chemdata_county = chemdata_county[chemdata_county.index.isin(finalcounties)]

# print out data
print("Number of counties in airtoxscreendata: " + str(len(chemdata_county.index)))
print("Number of tracts in airtoxscreendata: " + str(len(chemdata_tract.index)))
print("Number of county columns in airtoxscreendata: " + str(len(chemdata_county.columns)))
print("Number of tract columns in airtoxscreendata: " + str(len(chemdata_tract.columns)))

# get map data
county_json = LoadMapJSON("county")
tract_json = LoadMapJSON("tract")

# show maps for different clustering
lowclust = 2
hiclust = 6

filelist = []
for numclust in range(lowclust, hiclust):

	print("Completing clustering for k=" + str(numclust))
	tfile = ShowClustersOnMap("tract", chemdata_tract, numclust, "apeer_tract." + str(numclust), tract_json)
	filelist.append(tfile)
	tfile = ShowClustersOnMap("county", chemdata_county, numclust, "apeer_county." + str(numclust), county_json)
	filelist.append(tfile)

# count up tracts / counties for Jaccard
countycnt = 0
for tfips in finalcounties:
	countycnt += 1
tractcnt = 0
for tfips in finaltracts:
	tractcnt += 1

print("All Counties: " + str(countycnt) + " All Tracts: " + str(tractcnt))

countymort, tractmort = GetCanonicalStrokeBelt("all", finalcounties, finaltracts)

# check cluster counts
for tfile in filelist:
	stats_data = defaultdict(int)
	print("Calculating cluster file: " + tfile)
	clustdata = LoadClusterData(tfile)
	for tid in clustdata:
		tclust = clustdata[tid]
		stats_data[tclust] += 1
	for tid in stats_data:
		print(str(tid) + "\t" + str(stats_data[tid]))

# Jaccard indices are stored in apeer_jaccard.tsv
ofile = "apeer_jaccard.tsv"
chronic_disease = defaultdict(lambda: defaultdict(float))
f = open(ofile, "w")
for tfile in filelist:
	print("Calculating Jaccard Indices for " + tfile)
	clustdata = LoadClusterData(tfile)
	if tfile.find('county') > -1:
		tjaccard, tpval = CalcFastJaccard(clustdata, countymort, countycnt)
	if tfile.find('tract') > -1:
		tjaccard, tpval = CalcFastJaccard(clustdata, tractmort, tractcnt)
	print(tfile + "\t" + str(tjaccard) + "\t" + str(tpval))
	f.write(tfile + "\t" + str(tjaccard) + "\t" + str(tpval) + "\n")
f.close()

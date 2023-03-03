from HARMONY_LUTI_ATH.globals import *

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
import networkx as nx
import osmnx as ox

def population_map_creation(inputs, outputs, logger):
    logger.warning("Saving maps...")

    df_pop19 = pd.read_csv(outputs["EjOi2019"], usecols=['zone', 'OiPred_19'], index_col='zone')
    df_pop30 = pd.read_csv(outputs["EjOi2030"], usecols=['zone', 'OiPred_30'], index_col='zone')
    df_pop45 = pd.read_csv(outputs["EjOi2045"], usecols=['zone', 'OiPred_45'], index_col='zone')
    df_pop_merged = pd.merge(pd.merge(df_pop19, df_pop30, on='zone'), df_pop45, on='zone')

    df_pop_merged['PopCh19_30'] = ((df_pop30['OiPred_30'] - df_pop19['OiPred_19']) / df_pop19['OiPred_19']) * 100.0
    df_pop_merged['PopCh19_45'] = ((df_pop45['OiPred_45'] - df_pop19['OiPred_19']) / df_pop19['OiPred_19']) * 100.0
    df_pop_merged['PopCh30_45'] = ((df_pop45['OiPred_45'] - df_pop30['OiPred_30']) / df_pop30['OiPred_30']) * 100.0
    df_pop_merged.to_csv(Pop_Change)

    pop_ch = pd.read_csv(Pop_Change)

    map_df = gpd.read_file(inputs["DataZonesShapefile"])
    ath_map_popch_df = map_df.merge(pop_ch, left_on='NO', right_on='zone')

    # Plotting the Population change between 2019 - 2030
    fig1, ax1 = plt.subplots(1, figsize=(20, 10))
    ath_map_popch_df.plot(column='PopCh19_30', cmap='Reds', ax=ax1, edgecolor='darkgrey', linewidth=0.1)
    ax1.axis('off')
    ax1.set_title('Population Change 2019 - 2030 in the Attica Region', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=None)
    sm._A = []
    cbar = fig1.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label='Scale 1:400000', dimension="si-length", units="m", location='lower left', pad=5, border_pad=2)
    ax1.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax1.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax1.transAxes)
    plt.savefig(outputs["MapPopChange20192030"], dpi = 600)
    #plt.show()

    # Plotting the Population change between 2030 - 2045
    fig2, ax2 = plt.subplots(1, figsize=(20, 10))
    ath_map_popch_df.plot(column='PopCh30_45', cmap='Reds', ax=ax2, edgecolor='darkgrey', linewidth=0.1)
    ax2.axis('off')
    ax2.set_title('Population Change 2030 - 2045 in the Attica Region', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=None)
    sm._A = []
    cbar = fig2.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label='Scale 1:400000', dimension="si-length", units="m", location='lower left', pad=5, border_pad=2)
    ax2.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax2.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax2.transAxes)
    plt.savefig(outputs["MapPopChange20302045"], dpi = 600)
    #plt.show()

    # Plotting the Population change between 2019 - 2045
    fig3, ax3 = plt.subplots(1, figsize=(20, 10))
    ath_map_popch_df.plot(column='PopCh19_45', cmap='Reds', ax=ax3, edgecolor='darkgrey', linewidth=0.1)
    ax3.axis('off')
    ax3.set_title('Population Change 2019 - 2045 in the Attica Region', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=None)
    sm._A = []
    cbar = fig3.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label='Scale 1:400000', dimension="si-length", units="m", location='lower left', pad=5, border_pad=2)
    ax3.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax3.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8),ha='center', va='center', fontsize=15, xycoords=ax3.transAxes)
    plt.savefig(outputs["MapPopChange20192045"], dpi = 600)
    #plt.show()

    #Housing Accessibility Change
    df_HA_2019 = pd.read_csv(outputs["HousingAccessibility2019"], usecols=['zone', 'HApu19', 'HApr19'])
    df_HA_2030 = pd.read_csv(outputs["HousingAccessibility2030"], usecols=['zone','HApu30', 'HApr30'])
    df_HA_2045 = pd.read_csv(outputs["HousingAccessibility2045"], usecols=['zone','HApu45', 'HApr45'])

    #Merging the DataFrames
    df_HA_merged = pd.merge(pd.merge(df_HA_2019, df_HA_2030, on='zone'), df_HA_2045, on='zone')

    df_HA_merged['HACh1930pu'] = ((df_HA_2030['HApu30'] - df_HA_2019['HApu19']) / df_HA_2019['HApu19']) * 100.0
    df_HA_merged['HACh1930pr'] = ((df_HA_2030['HApr30'] - df_HA_2019['HApr19']) / df_HA_2019['HApr19']) * 100.0
    df_HA_merged['HACh1945pu'] = ((df_HA_2045['HApu45'] - df_HA_2019['HApu19']) / df_HA_2019['HApu19']) * 100.0
    df_HA_merged['HACh1945pr'] = ((df_HA_2045['HApr45'] - df_HA_2019['HApr19']) / df_HA_2019['HApr19']) * 100.0
    df_HA_merged['HACh3045pu'] = ((df_HA_2045['HApu45'] - df_HA_2030['HApu30']) / df_HA_2030['HApu30']) * 100.0
    df_HA_merged['HACh3045pr'] = ((df_HA_2045['HApr45'] - df_HA_2030['HApr30']) / df_HA_2030['HApr30']) * 100.0
    df_HA_merged.to_csv(HA_Change)

    # Plotting the Housing Accessibility change
    HousingAcc_change = pd.read_csv(HA_Change)
    ath_map_HAch_df = map_df.merge(HousingAcc_change, left_on='NO', right_on='zone')

    # Producing Maps for Housing Accessibility Change 2019 - 2030/ 2030 - 2045 / 2019 - 2045 using public/private transport in the Attica Region
    fig4, ax4 = plt.subplots(1, figsize=(20, 10))
    ath_map_HAch_df.plot(column='HACh1930pu', cmap='OrRd', ax=ax4, edgecolor='darkgrey', linewidth=0.1)
    ax4.axis('off')
    ax4.set_title('Housing Accessibility Change 2019 - 2030 using public transport in the Attica Region', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=None)
    sm._A = []
    cbar = fig4.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:400000',dimension="si-length", units="m", location='lower left', pad = 5, border_pad = 2)
    ax4.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax4.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax4.transAxes)
    plt.savefig(outputs["MapHousingAccChange20192030Public"], dpi = 600)
    #plt.show()

    fig5, ax5 = plt.subplots(1, figsize=(20, 10))
    ath_map_HAch_df.plot(column='HACh1930pr', cmap='OrRd', ax=ax5, edgecolor='darkgrey', linewidth=0.1)
    ax5.axis('off')
    ax5.set_title('Housing Accessibility Change 2019 - 2030 using private transport in the Attica Region', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=None)
    sm._A = []
    cbar = fig5.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:400000',dimension="si-length", units="m", location='lower left', pad = 5, border_pad = 2)
    ax5.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax5.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax5.transAxes)
    plt.savefig(outputs["MapHousingAccChange20192030Private"], dpi = 600)
    #plt.show()

    fig6, ax6 = plt.subplots(1, figsize=(20, 10))
    ath_map_HAch_df.plot(column='HACh1945pu', cmap='OrRd', ax=ax6, edgecolor='darkgrey', linewidth=0.1)
    ax6.axis('off')
    ax6.set_title('Housing Accessibility Change 2019 - 2045 using public transport in the Attica Region', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=None)
    sm._A = []
    cbar = fig6.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:400000',dimension="si-length", units="m", location='lower left', pad = 5, border_pad = 2)
    ax6.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax6.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax6.transAxes)
    plt.savefig(outputs["MapHousingAccChange20192045Public"], dpi = 600)
    #plt.show()

    fig7, ax7 = plt.subplots(1, figsize=(20, 10))
    ath_map_HAch_df.plot(column='HACh1945pr', cmap='OrRd', ax=ax7, edgecolor='darkgrey', linewidth=0.1)
    ax7.axis('off')
    ax7.set_title('Housing Accessibility Change 2019 - 2045 using private transport in the Attica Region', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=None)
    sm._A = []
    cbar = fig7.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:400000',dimension="si-length", units="m", location='lower left', pad = 5, border_pad = 2)
    ax7.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax7.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax7.transAxes)
    plt.savefig(outputs["MapHousingAccChange20192045Private"], dpi = 600)
    #plt.show()

    fig8, ax8 = plt.subplots(1, figsize=(20, 10))
    ath_map_HAch_df.plot(column='HACh3045pu', cmap='OrRd', ax=ax8, edgecolor='darkgrey', linewidth=0.1)
    ax8.axis('off')
    ax8.set_title('Housing Accessibility Change 2030 - 2045 using public transport in the Attica Region', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=None)
    sm._A = []
    cbar = fig8.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:400000',dimension="si-length", units="m", location='lower left', pad = 5, border_pad = 2)
    ax8.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax8.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax8.transAxes)
    plt.savefig(outputs["MapHousingAccChange20302045Public"], dpi = 600)
    #plt.show()

    fig9, ax9 = plt.subplots(1, figsize=(20, 10))
    ath_map_HAch_df.plot(column='HACh3045pr', cmap='OrRd', ax=ax9, edgecolor='darkgrey', linewidth=0.1)
    ax9.axis('off')
    ax9.set_title('Housing Accessibility Change 2030 - 2045 using private transport in the Attica Region', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=None)
    sm._A = []
    cbar = fig9.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:400000',dimension="si-length", units="m", location='lower left', pad = 5, border_pad = 2)
    ax9.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax9.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax9.transAxes)
    plt.savefig(outputs["MapHousingAccChange201302045Private"], dpi = 600)
    #plt.show()

    #Jobs Accessibility Change
    df_JobAcc_2019 = pd.read_csv(outputs["JobsAccessibility2019"], usecols=['zone', 'JobsApu19', 'JobsApr19'])
    df_JobAcc_2030 = pd.read_csv(outputs["JobsAccessibility2030"], usecols=['zone', 'JobsApu30', 'JobsApr30'])
    df_JobAcc_2045 = pd.read_csv(outputs["JobsAccessibility2045"], usecols=['zone', 'JobsApu45', 'JobsApr45'])

    # Merging the DataFrames
    df_JobAcc_merged = pd.merge(pd.merge(df_JobAcc_2019, df_JobAcc_2030, on='zone'), df_JobAcc_2045, on='zone')

    df_JobAcc_merged['JACh1930pu'] = ((df_JobAcc_2030['JobsApu30'] - df_JobAcc_2019['JobsApu19']) / df_JobAcc_2019['JobsApu19']) * 100.0
    df_JobAcc_merged['JACh1930pr'] = ((df_JobAcc_2030['JobsApr30'] - df_JobAcc_2019['JobsApr19']) / df_JobAcc_2019['JobsApr19']) * 100.0
    df_JobAcc_merged['JACh1945pu'] = ((df_JobAcc_2045['JobsApu45'] - df_JobAcc_2019['JobsApu19']) / df_JobAcc_2019['JobsApu19']) * 100.0
    df_JobAcc_merged['JACh1945pr'] = ((df_JobAcc_2045['JobsApr45'] - df_JobAcc_2019['JobsApr19']) / df_JobAcc_2019['JobsApr19']) * 100.0
    df_JobAcc_merged['JACh3045pu'] = ((df_JobAcc_2045['JobsApu45'] - df_JobAcc_2030['JobsApu30']) / df_JobAcc_2030['JobsApu30']) * 100.0
    df_JobAcc_merged['JACh3045pr'] = ((df_JobAcc_2045['JobsApr45'] - df_JobAcc_2030['JobsApr30']) / df_JobAcc_2030['JobsApr30']) * 100.0
    df_JobAcc_merged.to_csv(Job_Change)

    # Plotting the Jobs Accessibility change
    JobAcc_change = pd.read_csv(Job_Change)
    ath_map_JAch_df = map_df.merge(JobAcc_change, left_on='NO', right_on='zone')

    # Producing Maps for Jobs Accessibility Change 2019 - 2030/ 2030 - 2045 / 2019 - 2045 using public/private transport in the Attica Region
    fig10, ax10 = plt.subplots(1, figsize=(20, 10))
    ath_map_JAch_df.plot(column='JACh1930pu', cmap='Greens', ax=ax10, edgecolor='darkgrey', linewidth=0.1)
    ax10.axis('off')
    ax10.set_title('Jobs Accessibility Change 2019 - 2030 using public transport in the Attica Region', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Greens', norm=None)
    sm._A = []
    cbar = fig10.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:400000',dimension="si-length", units="m", location='lower left', pad = 5, border_pad = 2)
    ax10.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax10.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax10.transAxes)
    plt.savefig(outputs["MapJobsAccChange20192030Public"], dpi = 600)
    #plt.show()

    fig11, ax11 = plt.subplots(1, figsize=(20, 10))
    ath_map_JAch_df.plot(column='JACh1930pr', cmap='Greens', ax=ax11, edgecolor='darkgrey', linewidth=0.1)
    ax11.axis('off')
    ax11.set_title('Jobs Accessibility Change 2019 - 2030 using private transport in the Attica Region', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Greens', norm=None)
    sm._A = []
    cbar = fig11.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:400000',dimension="si-length", units="m", location='lower left', pad = 5, border_pad = 2)
    ax11.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax11.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax11.transAxes)
    plt.savefig(outputs["MapJobsAccChange20192030Private"], dpi = 600)
    #plt.show()

    fig12, ax12 = plt.subplots(1, figsize=(20, 10))
    ath_map_JAch_df.plot(column='JACh1945pu', cmap='Greens', ax=ax12, edgecolor='darkgrey', linewidth=0.1)
    ax12.axis('off')
    ax12.set_title('Jobs Accessibility Change 2019 - 2045 using public transport in the Attica Region', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Greens', norm=None)
    sm._A = []
    cbar = fig12.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:400000',dimension="si-length", units="m", location='lower left', pad = 5, border_pad = 2)
    ax12.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax12.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax12.transAxes)
    plt.savefig(outputs["MapJobsAccChange20192045Public"], dpi = 600)
    #plt.show()

    fig13, ax13 = plt.subplots(1, figsize=(20, 10))
    ath_map_JAch_df.plot(column='JACh1945pr', cmap='Greens', ax=ax13, edgecolor='darkgrey', linewidth=0.1)
    ax13.axis('off')
    ax13.set_title('Jobs Accessibility Change 2019 - 2045 using private transport in the Attica Region', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Greens', norm=None)
    sm._A = []
    cbar = fig13.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:400000',dimension="si-length", units="m", location='lower left', pad = 5, border_pad = 2)
    ax13.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax13.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax13.transAxes)
    plt.savefig(outputs["MapJobsAccChange20192045Private"], dpi = 600)
    #plt.show()

    fig14, ax14 = plt.subplots(1, figsize=(20, 10))
    ath_map_JAch_df.plot(column='JACh3045pu', cmap='Greens', ax=ax14, edgecolor='darkgrey', linewidth=0.1)
    ax14.axis('off')
    ax14.set_title('Jobs Accessibility Change 2030 - 2045 using public transport in the Attica Region', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Greens', norm=None)
    sm._A = []
    cbar = fig14.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:400000',dimension="si-length", units="m", location='lower left', pad = 5, border_pad = 2)
    ax14.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax14.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax14.transAxes)
    plt.savefig(outputs["MapJobsAccChange20302045Public"], dpi = 600)
    #plt.show()

    fig15, ax15 = plt.subplots(1, figsize=(20, 10))
    ath_map_JAch_df.plot(column='JACh3045pr', cmap='Greens', ax=ax15, edgecolor='darkgrey', linewidth=0.1)
    ax15.axis('off')
    ax15.set_title('Jobs Accessibility Change 2030 - 2045 using private transport in the Attica Region', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Greens', norm=None)
    sm._A = []
    cbar = fig15.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:400000',dimension="si-length", units="m", location='lower left', pad = 5, border_pad = 2)
    ax15.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax15.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax15.transAxes)
    plt.savefig(outputs["MapJobsAccChange20302045Private"], dpi = 600)
    #plt.show()

    #Create a common shapefile (polygon) that contains:
    # 1. Population change (2019-2030-2045)
    # 2. Housing Accessibility change pu and pr (2019-2030-2045)
    # 3. Jobs Accessibility change pu and pr (2019-2030-2045)
    tot_shp_df = map_df.merge(pd.merge(pd.merge(HousingAcc_change, JobAcc_change, on='zone'), df_pop_merged, on='zone'), left_on='NO', right_on='zone')
    #Drop unsuseful columns
    tot_shp_df.drop(columns=['NAME', 'CODE', 'TYPENO', 'EMME_DATA1', 'EMME_DATA2', 'EMME_DATA3', 'zone'], inplace = True, axis = 1)
    #Save the shapefile
    tot_shp_df.to_file(outputs["MapResultsShapefile"])

def flows_map_creation(inputs, outputs, flows_output_keys): # Using OSM

    Zone_nodes = nx.read_shp(inputs["ZoneCentroidsShapefileWGS84"]) # Must be in epsg:4326 (WGS84)

    Case_Study_Zones = ['Attica']

    X = ox.graph_from_place(Case_Study_Zones, network_type='drive')
    # crs = X.graph["crs"]
    # print('Graph CRS: ', crs)
    # print()

    # ox.plot_graph(X) # test plot

    X = X.to_undirected()

    # Calculate the origins and destinations for the shortest paths algorithms to be run on OSM graph
    OD_list = calc_shortest_paths_ODs_osm(Zone_nodes, X)

    Flows = []

    for kk, flows_output_key in enumerate(flows_output_keys):
        Flows.append(pd.read_csv(outputs[flows_output_key], header=None))

        # Initialise weights to 0:
        for source, target in X.edges():
            X[source][target][0]["Flows_" + str(kk)] = 0

    TOT_count = len(OD_list)
    # print(OD_list)

    for n, i in enumerate(OD_list):
        print("Flows maps creation - iteration ", n + 1, " of ", TOT_count)
        sssp_paths = nx.single_source_dijkstra_path(X, i, weight='length')  # single source shortest paths from i to all nodes of the network
        for m, j in enumerate(OD_list):
            shortest_path = sssp_paths[j]  # shortest path from i to j
            path_edges = zip(shortest_path, shortest_path[1:])  # Create edges from nodes of the shortest path

            for edge in list(path_edges):
                for cc in range(len(Flows)):
                    X[edge[0]][edge[1]][0]["Flows_" + str(cc)] += Flows[cc].iloc[n, m]

    # save graph to shapefile
    output_folder_path = "./outputs-Athens/" + "Flows_shp"
    ox.save_graph_shapefile(X, filepath=output_folder_path)

''' Computationaly very demanding
def flows_map_creation_HEAVY(inputs, outputs, flows_output_keys): # Using OSM

    Zone_nodes = nx.read_shp(inputs["ZoneCentroidsShapefile"]) # Must be in epsg:4326 (WGS84)

    Case_Study_Zones = ['Attica']

    X = ox.graph_from_place(Case_Study_Zones, network_type='drive')

    # ox.plot_graph(X) # test plot

    X = X.to_undirected()

    # Calculate the origins and destinations for the shortest paths algorithms to be run on OSM graph
    OD_list = calc_shortest_paths_ODs_osm(Zone_nodes, X)

    Flows = []

    for kk, flows_output_key in enumerate(flows_output_keys):
        Flows.append(pd.read_csv(outputs[flows_output_key], header=None))

        # Initialise weights to 0:
        for source, target in X.edges():
            X[source][target][0]["Flows_" + str(kk)] = 0

    # shortest_paths_nodes = []  # List of shortest paths nodes
    # shortest_paths_edges = []  # List of shortest paths edges

    TOT_count = len(OD_list)

    for n, i in enumerate(OD_list):
        print("Flows maps creation - iteration ", n+1, " of ", TOT_count)
        for m, j in enumerate(OD_list):
            shortest_path = ox.shortest_path(X, i, j, weight='length', cpus=16) # cpus: if "None" use all available
            # Save the nodes and the edges of the shortest path into two lists
            # shortest_paths_nodes.append(shortest_path)  # Save nodes
            path_edges = zip(shortest_path, shortest_path[1:])  # Create edges from nodes of the shortest path
            # shortest_paths_edges.append(path_edges)  # Save edges

            # Add the weight of each edge of the shortest path = flows
            for edge in list(path_edges):
                for cc in range(len(Flows)):
                    X[edge[0]][edge[1]][0]["Flows_" + str(cc)] += Flows[cc].iloc[n, m]

    # save graph to shapefile
    # output_folder_path = "./outputs-Turin/" + flows_output_key + "_shp"
    output_folder_path = "./outputs-Athens/" + "Flows_shp"
    ox.save_graph_shapefile(X, filepath=output_folder_path)
'''
def calc_closest(new_node, node_list):
    # Calculate the closest node in the network
    best_diff = 100000000
    closest_node = [0, 0]
    for comp_node in node_list.nodes():

        diff = (abs(comp_node[0] - new_node[0]) + abs(comp_node[1] - new_node[1]))
        if abs(diff) < best_diff:
            best_diff = diff
            closest_node = comp_node

    return closest_node

def calc_shortest_paths_ODs_osm(zones_centroids, network):
    # For each zone centroid, this function calculates the closest node in the OSM graph.
    # These nodes will be used as origins and destinations in the shortest paths calculations.
    list_of_ODs = []
    for c in zones_centroids:
        graph_clostest_node = ox.nearest_nodes(network, c[0], c[1], return_dist=False)
        list_of_ODs.append(graph_clostest_node)
    return list_of_ODs
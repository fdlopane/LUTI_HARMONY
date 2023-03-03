from HARMONY_LUTI_TUR.globals import *
import os
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
import csv
import networkx as nx
import osmnx as ox
from shapely.geometry import shape

def population_map_creation(inputs, outputs):

    df_pop19 = pd.read_csv(outputs["JobsDjOi2019"], usecols=['zone', 'OiPred_Tot_19'], index_col='zone')
    df_pop30 = pd.read_csv(outputs["JobsDjOi2030"], usecols=['zone', 'OiPred_Tot_30'], index_col='zone')
    df_pop_merged = pd.merge(df_pop19, df_pop30, on='zone')
    print(df_pop_merged)
    df_pop_merged['PopCh19_30'] = ((df_pop30['OiPred_Tot_30'] - df_pop19['OiPred_Tot_19']) / df_pop19['OiPred_Tot_19']) * 100.0
    df_pop_merged.to_csv(Pop_Change)

    pop_ch = pd.read_csv(Pop_Change)

    map_df = gpd.read_file(inputs["DataZonesShapefile"])
    print(map_df)
    tur_map_popch_df = map_df.merge(pop_ch, left_on='NO', right_on='zone')

    # Plotting the Population change between 2019 - 2030
    fig1, ax1 = plt.subplots(1, figsize=(20, 10))
    tur_map_popch_df.plot(column='PopCh19_30', cmap='Reds', ax=ax1, edgecolor='darkgrey', linewidth=0.1)
    ax1.axis('off')
    ax1.set_title('Population Change 2019 - 2030 in Turin', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=None)
    sm._A = []
    cbar = fig1.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label='Scale 1:250000', dimension="si-length", units="m", location='lower left', box_color=None)
    ax1.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax1.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax1.transAxes)
    plt.savefig(outputs["MapPopChange20192030"], dpi = 600)
    #plt.show()

    #Housing Accessibility Change
    df_HA_2019 = pd.read_csv(outputs["HousingAccessibility2019"], usecols=['zone', 'HAcar19', 'HAbus19', 'HArail19'])
    df_HA_2030 = pd.read_csv(outputs["HousingAccessibility2030"], usecols=['zone','HAcar30', 'HAbus30', 'HArail30'])

    #Merging the DataFrames
    df_HA_merged = pd.merge(df_HA_2019, df_HA_2030, on='zone')
    print(df_HA_merged)

    df_HA_merged['HAC1930car'] = ((df_HA_2030['HAcar30'] - df_HA_2019['HAcar19']) / df_HA_2019['HAcar19']) * 100.0
    df_HA_merged['HAC1930bus'] = ((df_HA_2030['HAbus30'] - df_HA_2019['HAbus19']) / df_HA_2019['HAbus19']) * 100.0
    df_HA_merged['HAC1930rai'] = ((df_HA_2030['HArail30'] - df_HA_2019['HArail19']) / df_HA_2019['HArail19']) * 100.0

    df_HA_merged.to_csv(HA_Change)

    # Plotting the Housing Accessibility change
    HousingAcc_change = pd.read_csv(HA_Change)
    tur_map_HAch_df = map_df.merge(HousingAcc_change, left_on='NO', right_on='zone')

    # Producing Maps for Housing Accessibility Change 2019 - 2030 using car, bus and rail transport in the Turin
    fig2, ax2 = plt.subplots(1, figsize=(20, 10))
    tur_map_HAch_df.plot(column='HAC1930car', cmap='OrRd', ax=ax2, edgecolor='darkgrey', linewidth=0.1)
    ax2.axis('off')
    ax2.set_title('Housing Accessibility Change 2019 - 2030 using car in Turin', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=None)
    sm._A = []
    cbar = fig2.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:250000',dimension="si-length", units="m", location='lower left', box_color=None)
    ax2.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax2.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax2.transAxes)
    plt.savefig(outputs["MapHousingAccChange20192030Roads"], dpi = 600)
    #plt.show()

    fig3, ax3 = plt.subplots(1, figsize=(20, 10))
    tur_map_HAch_df.plot(column='HAC1930bus', cmap='OrRd', ax=ax3, edgecolor='darkgrey', linewidth=0.1)
    ax3.axis('off')
    ax3.set_title('Housing Accessibility Change 2019 - 2030 using bus in Turin', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=None)
    sm._A = []
    cbar = fig3.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:250000',dimension="si-length", units="m", location='lower left', box_color=None)
    ax3.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax3.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax3.transAxes)
    plt.savefig(outputs["MapHousingAccChange20192030Bus"], dpi = 600)
    #plt.show()

    fig4, ax4 = plt.subplots(1, figsize=(20, 10))
    tur_map_HAch_df.plot(column='HAC1930rai', cmap='OrRd', ax=ax4, edgecolor='darkgrey', linewidth=0.1)
    ax4.axis('off')
    ax4.set_title('Housing Accessibility Change 2019 - 2030 using rail in Turin', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=None)
    sm._A = []
    cbar = fig4.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:250000',dimension="si-length", units="m", location='lower left', box_color=None)
    ax4.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax4.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax4.transAxes)
    plt.savefig(outputs["MapHousingAccChange20192030Rail"], dpi = 600)
    #plt.show()


    #Jobs Accessibility Change
    df_JobAcc_2019 = pd.read_csv(outputs["JobsAccessibility2019"], usecols=['zone', 'JAcar19', 'JAbus19', 'JArail19'])
    df_JobAcc_2030 = pd.read_csv(outputs["JobsAccessibility2030"], usecols=['zone', 'JAcar30', 'JAbus30', 'JArail30'])

    # Merging the DataFrames
    df_JobAcc_merged = pd.merge(df_JobAcc_2019, df_JobAcc_2030, on='zone')
    print(df_JobAcc_merged)

    df_JobAcc_merged['JAC1930car'] = ((df_JobAcc_2030['JAcar30'] - df_JobAcc_2019['JAcar19']) / df_JobAcc_2019['JAcar19']) * 100.0
    df_JobAcc_merged['JAC1930bus'] = ((df_JobAcc_2030['JAbus30'] - df_JobAcc_2019['JAbus19']) / df_JobAcc_2019['JAbus19']) * 100.0
    df_JobAcc_merged['JAC1930rai'] = ((df_JobAcc_2030['JArail30'] - df_JobAcc_2019['JArail19']) / df_JobAcc_2019['JArail19']) * 100.0
    df_JobAcc_merged.to_csv(Job_Change)

    # Plotting the Jobs Accessibility change
    JobAcc_change = pd.read_csv(Job_Change)
    tur_map_JAch_df = map_df.merge(JobAcc_change, left_on='NO', right_on='zone')

    # Producing Maps for Jobs Accessibility Change 2019 - 2030 using car, bus, rail in the Turin
    fig5, ax5 = plt.subplots(1, figsize=(20, 10))
    tur_map_JAch_df.plot(column='JAC1930car', cmap='Greens', ax=ax5, edgecolor='darkgrey', linewidth=0.1)
    ax5.axis('off')
    ax5.set_title('Jobs Accessibility Change 2019 - 2030 using car in Turin', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Greens', norm=None)
    sm._A = []
    cbar = fig5.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:250000',dimension="si-length", units="m", location='lower left', box_color=None)
    ax5.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax5.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax5.transAxes)
    plt.savefig(outputs["MapJobsAccChange20192030Roads"], dpi = 600)
    #plt.show()

    fig6, ax6 = plt.subplots(1, figsize=(20, 10))
    tur_map_JAch_df.plot(column='JAC1930bus', cmap='Greens', ax=ax6, edgecolor='darkgrey', linewidth=0.1)
    ax6.axis('off')
    ax6.set_title('Jobs Accessibility Change 2019 - 2030 using bus in Turin', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Greens', norm=None)
    sm._A = []
    cbar = fig6.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:250000',dimension="si-length", units="m", location='lower left', box_color=None)
    ax6.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax6.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax6.transAxes)
    plt.savefig(outputs["MapJobsAccChange20192030Bus"], dpi = 600)
    #plt.show()

    fig7, ax7 = plt.subplots(1, figsize=(20, 10))
    tur_map_JAch_df.plot(column='JAC1930rai', cmap='Greens', ax=ax7, edgecolor='darkgrey', linewidth=0.1)
    ax7.axis('off')
    ax7.set_title('Jobs Accessibility Change 2019 - 2030 using rail in Turin', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Greens', norm=None)
    sm._A = []
    cbar = fig7.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label= 'Scale 1:250000',dimension="si-length", units="m", location='lower left', box_color=None)
    ax7.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax7.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax7.transAxes)
    plt.savefig(outputs["MapJobsAccChange20192030Rail"], dpi = 600)
    #plt.show()

    #Create a common shapefile (polygon) that contains:
    # 1. Population change (2019-2030)
    # 2. Housing Accessibility change for car, bus and rail (2019-2030)
    # 3. Jobs Accessibility change for car, bus and rail (2019-2030)
    tot_shp_df = map_df.merge(pd.merge(pd.merge(HousingAcc_change, JobAcc_change, on='zone'), df_pop_merged, on='zone'), left_on='NO', right_on='zone')
    #Drop unsuseful columns
    tot_shp_df.drop(columns=['NAME', 'FUA', 'MACROCODE', 'MACRONAME', 'zone'], inplace = True, axis = 1)
    #Save the shapefile
    tot_shp_df.to_file(outputs["MapResultsShapefile"])

def flows_map_creation(inputs, outputs, flows_output_keys): # Using OSM

    Zone_nodes = nx.read_shp(inputs["ZonesCentroidsShapefile"])  # Must be in epsg:4326 (WGS84)

    Case_Study_Zones = ['Torino', 'Alpignano', 'Baldissero Torinese', 'Beinasco', 'Borgaro Torinese', 'Cambiano',
                        'Candiolo',
                        'Carignano', 'Caselle Torinese', 'Chieri', 'Collegno', 'Druento', 'Grugliasco', 'La Loggia',
                        'Leini',
                        'Moncalieri', 'Nichelino', 'Orbassano', 'Pecetto Torinese', 'Pianezza', 'Pino Torinese',
                        'Piobesi Torinese', 'Piossasco', 'Rivalta di Torino', 'Rivoli', 'San Mauro Torinese',
                        'Settimo Torinese', 'Trofarello', 'Venaria Reale', 'Vinovo', 'Airasca', 'Almese', 'Avigliana',
                        'Bosconero', 'Brandizzo', 'Bruino', 'Buttigliera Alta', 'Cafasse', 'Caprie', 'Casalborgone',
                        'Caselette', 'Castagneto Po', 'Castagnole Piemonte', 'Castiglione Torinese', 'Chivasso',
                        'Cinzano',
                        'Cumiana', 'Fiano', 'Foglizzo', 'Front', 'Gassino Torinese', 'Givoletto', 'La Cassa',
                        'Lauriano',
                        'Marentino', 'Montaldo Torinese', 'Montanaro', 'Monteu da Po', 'None', 'Pavarolo', 'Reano',
                        'Rivalba',
                        'Rivarossa', 'Robassomero', 'Rosta', 'Rubiana', 'San Benigno Canavese',
                        'San Francesco al Campo',
                        'Sangano', 'San Gillio', 'San Maurizio Canavese', 'San Raffaele Cimena', 'San Sebastiano da Po',
                        'Sciolze', 'Torrazza Piemonte', 'Trana', 'Val della Torre', 'Vallo Torinese', 'Varisella',
                        'Verolengo',
                        'Villarbasse', 'Villar Dora', 'Villastellone', 'Volpiano', 'Volvera', 'Lombardone', 'Cantalupa',
                        'Virle Piemonte']

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
        print("Flows maps creation - iteration ", n+1, " of ", TOT_count)
        sssp_paths = nx.single_source_dijkstra_path(X, i, weight='length') # single source shortest paths from i to all nodes of the network
        for m, j in enumerate(OD_list):
            shortest_path = sssp_paths[j] # shortest path from i to j
            path_edges = zip(shortest_path, shortest_path[1:])  # Create edges from nodes of the shortest path

            for edge in list(path_edges):
                for cc in range(len(Flows)):
                    X[edge[0]][edge[1]][0]["Flows_" + str(cc)] += Flows[cc].iloc[n, m]

    # save graph to shapefile
    output_folder_path = "./outputs-Turin/" + "Flows_shp"
    ox.save_graph_shapefile(X, filepath=output_folder_path)

'''
def flows_map_creation_HEAVY(inputs, outputs, flows_output_keys): # Using OSM

    Zone_nodes = nx.read_shp(inputs["ZonesCentroidsShapefile"]) # Must be in epsg:4326 (WGS84)

    Case_Study_Zones = ['Torino', 'Alpignano', 'Baldissero Torinese', 'Beinasco', 'Borgaro Torinese', 'Cambiano', 'Candiolo',
                 'Carignano', 'Caselle Torinese', 'Chieri', 'Collegno', 'Druento', 'Grugliasco', 'La Loggia', 'Leini',
                 'Moncalieri', 'Nichelino', 'Orbassano', 'Pecetto Torinese', 'Pianezza', 'Pino Torinese',
                 'Piobesi Torinese', 'Piossasco', 'Rivalta di Torino', 'Rivoli', 'San Mauro Torinese',
                 'Settimo Torinese', 'Trofarello', 'Venaria Reale', 'Vinovo', 'Airasca', 'Almese', 'Avigliana',
                 'Bosconero', 'Brandizzo', 'Bruino', 'Buttigliera Alta', 'Cafasse', 'Caprie', 'Casalborgone',
                 'Caselette', 'Castagneto Po', 'Castagnole Piemonte', 'Castiglione Torinese', 'Chivasso', 'Cinzano',
                 'Cumiana', 'Fiano', 'Foglizzo', 'Front', 'Gassino Torinese', 'Givoletto', 'La Cassa', 'Lauriano',
                 'Marentino', 'Montaldo Torinese', 'Montanaro', 'Monteu da Po', 'None', 'Pavarolo', 'Reano', 'Rivalba',
                 'Rivarossa', 'Robassomero', 'Rosta', 'Rubiana', 'San Benigno Canavese', 'San Francesco al Campo',
                 'Sangano', 'San Gillio', 'San Maurizio Canavese', 'San Raffaele Cimena', 'San Sebastiano da Po',
                 'Sciolze', 'Torrazza Piemonte', 'Trana', 'Val della Torre', 'Vallo Torinese', 'Varisella', 'Verolengo',
                 'Villarbasse', 'Villar Dora', 'Villastellone', 'Volpiano', 'Volvera', 'Lombardone', 'Cantalupa',
                 'Virle Piemonte']

    # Case_Study_Zones = ['Alpignano'] # Smaller network for test purposes

    X = ox.graph_from_place(Case_Study_Zones, network_type='drive')

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
            shortest_path = ox.shortest_path(X, i, j, weight='length', cpus=None) # cpus: if "None" use all available
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
    output_folder_path = "./outputs-Turin/" + "Flows_shp"
    ox.save_graph_shapefile(X, filepath=output_folder_path)
'''

def calc_closest(new_node, node_list):
    # Calculate the closest node in the network
    best_diff = 10000
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
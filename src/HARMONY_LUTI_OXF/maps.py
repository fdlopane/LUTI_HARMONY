from HARMONY_LUTI_OXF.globals import *
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
import networkx as nx
import os
import osmnx as ox

def population_map_creation(inputs, outputs):

    df_pop11 = pd.read_csv(Jobs_DjOi_2011, usecols=['msoa', 'OiPred_Tot_11'], index_col='msoa')
    df_pop19 = pd.read_csv(outputs["JobsDjOi2019"], usecols=['msoa', 'OiPred_Tot_19'], index_col='msoa')
    df_pop30 = pd.read_csv(outputs["JobsDjOi2030"], usecols=['msoa', 'OiPred_Tot_30'], index_col='msoa')
    df_pop_merged = pd.merge(pd.merge(df_pop11, df_pop19, on='msoa'), df_pop30, on='msoa')

    df_pop_merged['PopCh11_30'] = ((df_pop30['OiPred_Tot_30'] - df_pop11['OiPred_Tot_11']) / df_pop11['OiPred_Tot_11']) * 100.0
    df_pop_merged['PopCh11_19'] = ((df_pop19['OiPred_Tot_19'] - df_pop11['OiPred_Tot_11']) / df_pop11['OiPred_Tot_11']) * 100.0
    df_pop_merged['PopCh19_30'] = ((df_pop30['OiPred_Tot_30'] - df_pop19['OiPred_Tot_19']) / df_pop19['OiPred_Tot_19']) * 100.0

    df_pop_merged.to_csv(Pop_Change)
    pop_ch = pd.read_csv(Pop_Change)

    map_df = gpd.read_file(inputs["MsoaShapefile"])
    oxf_map_popch_df = map_df.merge(pop_ch, left_on='msoa11cd', right_on='msoa')

    # Plotting the Population change between 2019 - 2030
    fig3, ax3 = plt.subplots(1, figsize=(20, 10))
    oxf_map_popch_df.plot(column='PopCh19_30', cmap='Reds', ax=ax3, edgecolor='darkgrey', linewidth=0.1)
    ax3.axis('off')
    ax3.set_title('Population Change 2019 - 2030 in Oxfordshire', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=None)
    sm._A = []
    cbar = fig3.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label='Scale 1:400000', dimension="si-length", units="m", location='lower left', box_color=None)
    ax3.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax3.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8),ha='center', va='center', fontsize=15, xycoords=ax3.transAxes)
    plt.savefig(outputs["MapPopChange20192030"], dpi = 600)
    # plt.show()

    #Housing Accessibility Change
    df_HA_2011 = pd.read_csv(data_housing_accessibility_2011, usecols=['areakey', 'HAcar11', 'HAbus11', 'HArail11'])
    df_HA_2019 = pd.read_csv(outputs["HousingAccessibility2019"], usecols=['areakey', 'HAcar19', 'HAbus19', 'HArail19'])
    df_HA_2030 = pd.read_csv(outputs["HousingAccessibility2030"], usecols=['areakey', 'HAcar30', 'HAbus30', 'HArail30'])

    #Merging the DataFrames
    df_HA_merged = pd.merge(pd.merge(df_HA_2011, df_HA_2019, on='areakey'), df_HA_2030, on='areakey')

    df_HA_merged['HAC1930car'] = ((df_HA_2030['HAcar30'] - df_HA_2019['HAcar19']) / df_HA_2019['HAcar19']) * 100.0
    df_HA_merged['HAC1930bus'] = ((df_HA_2030['HAbus30'] - df_HA_2019['HAbus19']) / df_HA_2019['HAbus19']) * 100.0
    df_HA_merged['HAC1930rai'] = ((df_HA_2030['HArail30'] - df_HA_2019['HArail19']) / df_HA_2019['HArail19']) * 100.0

    df_HA_merged.to_csv(HA_Change)

    # Plotting the Housing Accessibility change
    HousingAcc_change = pd.read_csv(HA_Change)
    oxf_map_HAch_df = map_df.merge(HousingAcc_change, left_on='msoa11cd', right_on='areakey')

    # Producing Maps for Housing Accessibility Change 2019 - 2030 using car/bus/rail in the Oxfordshire

    fig7, ax7 = plt.subplots(1, figsize=(20, 10))
    oxf_map_HAch_df.plot(column='HAC1930car', cmap='OrRd', ax=ax7, edgecolor='darkgrey', linewidth=0.1)
    ax7.axis('off')
    ax7.set_title('Housing Accessibility Change 2019 - 2030 using car in Oxfordshire', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=None)
    sm._A = []
    cbar = fig7.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label='Scale 1:400000', dimension="si-length", units="m", location='lower left', box_color=None)
    ax7.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax7.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax7.transAxes)
    plt.savefig(outputs["MapHousingAccChange20192030Roads"], dpi=600)
    # plt.show()

    fig8, ax8 = plt.subplots(1, figsize=(20, 10))
    oxf_map_HAch_df.plot(column='HAC1930bus', cmap='OrRd', ax=ax8, edgecolor='darkgrey', linewidth=0.1)
    ax8.axis('off')
    ax8.set_title('Housing Accessibility Change 2019 - 2030 using bus in Oxfordshire', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=None)
    sm._A = []
    cbar = fig8.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label='Scale 1:400000', dimension="si-length", units="m", location='lower left', box_color=None)
    ax8.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax8.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax8.transAxes)
    plt.savefig(outputs["MapHousingAccChange20192030Bus"], dpi=600)
    # plt.show()

    fig9, ax9 = plt.subplots(1, figsize=(20, 10))
    oxf_map_HAch_df.plot(column='HAC1930rai', cmap='OrRd', ax=ax9, edgecolor='darkgrey', linewidth=0.1)
    ax9.axis('off')
    ax9.set_title('Housing Accessibility Change 2019 - 2030 using rail in Oxfordshire', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=None)
    sm._A = []
    cbar = fig9.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label='Scale 1:400000', dimension="si-length", units="m", location='lower left', box_color=None)
    ax9.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax9.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax9.transAxes)
    plt.savefig(outputs["MapHousingAccChange20192030Rail"], dpi=600)
    # plt.show()


    #Jobs Accessibility Change
    df_JobAcc_2011 = pd.read_csv(data_jobs_accessibility_2011, usecols=['areakey', 'JAcar11', 'JAbus11', 'JArail11'])
    df_JobAcc_2019 = pd.read_csv(outputs["JobsAccessibility2019"], usecols=['areakey', 'JAcar19', 'JAbus19', 'JArail19'])
    df_JobAcc_2030 = pd.read_csv(outputs["JobsAccessibility2030"], usecols=['areakey', 'JAcar30', 'JAbus30', 'JArail30'])

    # Merging the DataFrames
    df_JobAcc_merged = pd.merge(pd.merge(df_JobAcc_2011, df_JobAcc_2019, on='areakey'), df_JobAcc_2030, on='areakey')

    df_JobAcc_merged['JAC1930car'] = ((df_JobAcc_2030['JAcar30'] - df_JobAcc_2019['JAcar19']) / df_JobAcc_2019['JAcar19']) * 100.0
    df_JobAcc_merged['JAC1930bus'] = ((df_JobAcc_2030['JAbus30'] - df_JobAcc_2019['JAbus19']) / df_JobAcc_2019['JAbus19']) * 100.0
    df_JobAcc_merged['JAC1930rai'] = ((df_JobAcc_2030['JArail30'] - df_JobAcc_2019['JArail19']) / df_JobAcc_2019['JArail19']) * 100.0

    df_JobAcc_merged.to_csv(Job_Change)

    # Plotting the Jobs Accessibility change
    JobAcc_change = pd.read_csv(Job_Change)
    oxf_map_JAch_df = map_df.merge(JobAcc_change, left_on='msoa11cd', right_on='areakey')

    # Producing Maps for Jobs Accessibility Change 2019 - 2030 using car/bus/rail in the Oxfordshire

    fig13, ax13 = plt.subplots(1, figsize=(20, 10))
    oxf_map_JAch_df.plot(column='JAC1930car', cmap='Greens', ax=ax13, edgecolor='darkgrey', linewidth=0.1)
    ax13.axis('off')
    ax13.set_title('Jobs Accessibility Change 2019 - 2030 using car in Oxfordshire', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Greens', norm=None)
    sm._A = []
    cbar = fig13.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label='Scale 1:400000', dimension="si-length", units="m", location='lower left', box_color=None)
    ax13.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax13.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax13.transAxes)
    plt.savefig(outputs["MapJobsAccChange20192030Roads"], dpi=600)
    # plt.show()

    fig14, ax14 = plt.subplots(1, figsize=(20, 10))
    oxf_map_JAch_df.plot(column='JAC1930bus', cmap='Greens', ax=ax14, edgecolor='darkgrey', linewidth=0.1)
    ax14.axis('off')
    ax14.set_title('Jobs Accessibility Change 2019 - 2030 using bus in Oxfordshire', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Greens', norm=None)
    sm._A = []
    cbar = fig14.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label='Scale 1:400000', dimension="si-length", units="m", location='lower left', box_color=None)
    ax14.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax14.annotate('N', xy=(x, y), xytext=(x, y - arrow_length), arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax14.transAxes)
    plt.savefig(outputs["MapJobsAccChange20192030Bus"], dpi=600)
    # plt.show()

    fig15, ax15 = plt.subplots(1, figsize=(20, 10))
    oxf_map_JAch_df.plot(column='JAC1930rai', cmap='Greens', ax=ax15, edgecolor='darkgrey', linewidth=0.1)
    ax15.axis('off')
    ax15.set_title('Jobs Accessibility Change 2019 - 2030 using rail in Oxfordshire', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap='Greens', norm=None)
    sm._A = []
    cbar = fig15.colorbar(sm)
    scalebar = ScaleBar(dx=0.1, label='Scale 1:400000', dimension="si-length", units="m", location='lower left', box_color=None)
    ax15.add_artist(scalebar)
    x, y, arrow_length = 0, 1, 0.06
    ax15.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),arrowprops=dict(facecolor='black', width=2, headwidth=8), ha='center', va='center', fontsize=15, xycoords=ax15.transAxes)
    plt.savefig(outputs["MapJobsAccChange20192030Rail"], dpi=600)
    # plt.show()


    #Create a common shapefile (polygon) that contains:
    # 1. Population change (2019-2030)
    # 2. Housing Accessibility change for car, bus and rail (2019-2030)
    # 3. Jobs Accessibility change for car, bus and rail (2019-2030)
    tot_shp_df = oxf_map_popch_df.merge(pd.merge(HousingAcc_change, JobAcc_change, on='areakey'), left_on='msoa11cd', right_on='areakey')
    #Drop unsuseful columns
    tot_shp_df.drop(columns=['objectid','msoa11nm', 'msoa11nmw', 'st_areasha', 'st_lengths', 'msoa', 'areakey'], inplace = True, axis = 1)
    #Save the shapefile
    tot_shp_df.to_file(outputs["MapResultsShapefile"])

def flows_map_creation(inputs, outputs, flows_output_keys): # Using OSM

    Zone_nodes = nx.read_shp(inputs["MSOACentroidsShapefileWGS84"]) # Must be in epsg:4326 (WGS84)

    Case_Study_Zones = ['Oxfordshire']
    # Case_Study_Zones = ['Oxford'] # Smaller network for test purposes

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
    output_folder_path = "./outputs-Oxfordshire/" + "Flows_shp"
    ox.save_graph_shapefile(X, filepath=output_folder_path)


def flows_map_creation_HEAVY(inputs, outputs, flows_output_keys): # Using OSM

    Zone_nodes = nx.read_shp(inputs["MSOACentroidsShapefile"]) # Must be in epsg:4326 (WGS84)

    Case_Study_Zones = ['Oxfordshire']

    # Case_Study_Zones = ['Oxford'] # Smaller network for test purposes

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
    output_folder_path = "./outputs-Oxfordshire/" + "Flows_shp"
    ox.save_graph_shapefile(X, filepath=output_folder_path)

def flows_map_creation_light(inputs, outputs, flows_output_key, network_input):
    Flows = pd.read_csv(outputs[flows_output_key], header=None)

    Network = nx.read_shp(network_input) # Read network shapefile
    Zone_nodes = nx.read_shp(inputs["MSOACentroidsShapefile"])
    map_zones_shp = inputs["MsoaShapefile"]

    # Add Zone centroids to the road network
    for node in Zone_nodes:
        # Calculate the closest road node
        closest_node = calc_closest(node, Network)
        Network.add_node(node) # adds node to network
        Network.add_edge(node, closest_node) # adds edge between nodes

    Network = Network.to_undirected()

    pos = {k: v for k,v in enumerate(Network.nodes())} # Map the nodes of the network into a dictionary

    X = nx.Graph()  # Empty graph
    X.add_nodes_from(pos.keys())  # Add nodes preserving coordinates
    l = [set(x) for x in Network.edges()]  # To speed things up in case of large objects
    edg = [tuple(k for k, v in pos.items() if v in sl) for sl in l]  # Map the Network.edges start and endpoints onto pos
    X.add_edges_from(edg)

    OD_nodes = [] # List of pos keys of Zone centroids
    # shortest_paths_nodes = [] # List of shortest paths nodes
    # shortest_paths_edges = [] # List of shortest paths edges

    # Initialise weights to 0
    nx.set_edge_attributes(X, values=0, name='weight')

    c1 = 0
    for i in Zone_nodes:
        OD_nodes.append(list(pos.keys())[list(pos.values()).index(i)])
        c2 = 0
        for j in Zone_nodes:
            # shortest_path = nx.shortest_path(Network, i, j, weight=None)
            shortest_path = nx.shortest_path(X, list(pos.keys())[list(pos.values()).index(i)], list(pos.keys())[list(pos.values()).index(j)], weight=None)

            # Save the nodes and the edges of the shortest path into two lists
            # shortest_paths_nodes.append(shortest_path) # Save nodes
            path_edges = zip(shortest_path, shortest_path[1:]) # Create edges from nodes of the shortest path
            # shortest_paths_edges.append(path_edges) # Save edges

            # Add 1 to the weight of each edge of the shortest path
            for edge in list(path_edges):
                # X[edge[0]][edge[1]]['weight'] += 1
                X[edge[0]][edge[1]]['weight'] = X[edge[0]][edge[1]]['weight'] + Flows.iloc[c1, c2]

            c2 += 1
        c1 += 1

    # Save the weights into a variable for plotting
    edges, weights = zip(*nx.get_edge_attributes(X, 'weight').items())

    # Normalise the weights:
    weights = weights / max(weights)

    # Draw base road network
    fig, ax = plt.subplots()

    # Background map
    map_df = gpd.read_file(map_zones_shp)
    # map_df.plot(ax=ax, facecolor="none", edgecolor='0.6', linewidth=0.2)  # edgecolor='0.8' as light gray; [0, 1] for grayscale values.
    map_df.plot(ax=ax, facecolor="0.3", edgecolor='0.5', linewidth=0.2)  # edgecolor='0.8' as light gray; [0, 1] for grayscale values.

    # Plot all the nodes of the network:
    # nx.draw_networkx_nodes(X, pos, node_size=0.01, node_color='k')
    nx.draw_networkx_nodes(X, pos, node_size=0.001, node_color='k', alpha=0.0)

    # Plots zone centroids:
    # nx.draw_networkx_nodes(X, pos, nodelist=OD_nodes, node_size=5, node_color='w')

    # Draw all network edges:
    # nx.draw_networkx_edges(X, pos, edge_color='w')
    nx.draw_networkx_edges(X, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.inferno)

    # nx.draw(X, pos, node_size=0.001, node_color='k', edgelist=edges, edge_color=weights, edge_cmap=plt.cm.inferno) # Fixed edges' widths
    # nx.draw(X, pos, node_size=0.001, node_color='white', edgelist=edges, edge_color=weights, edge_cmap=plt.cm.YlOrBr)  # Fixed edges' widths
    # nx.draw(X, pos, node_size=0.001, node_color='k', edgelist=edges, edge_color=weights, edge_cmap=plt.cm.inferno, width=list(5*weights)) # Edges' widths proportional to weights

    ax.axis('off') # Turn axes off
    ax.set_aspect(aspect='equal') # Plot the figure with actual proportions

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmin=0, vmax=1))
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrBr, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cb = plt.colorbar(sm)

    # set colorbar label plus label color:
    cb.set_label('Normalised flows', color='white')
    # cb.set_label('Normalised flows', color='black')

    # set colorbar tick colour:
    cb.ax.yaxis.set_tick_params(color='white')
    # cb.ax.yaxis.set_tick_params(color='black')

    # set colorbar edgecolor:
    cb.outline.set_edgecolor('white')
    # cb.outline.set_edgecolor('black')

    # set colorbar ticklabels:
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    # plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='black')

    # Title
    # ax.set_title('Flows', color='white')

    # Background colour
    fig.set_facecolor('black')
    # fig.set_facecolor('white')

    ax.set_facecolor('black')
    # ax.set_facecolor('white')

    # plt.show()

    # Save figure
    plt.savefig(outputs[flows_output_key+"FlowMap"], dpi=2400, facecolor='black', edgecolor='black')  # save as png
    # plt.savefig(outputs[flows_output_key + "FlowMap"], dpi=2400, facecolor='white', edgecolor='white')  # save as png

    # Write shapefile
    # outdir = "./Outputs-Oxfordshire/"+ flows_output_key + "FlowMap_shape"
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir) # Create a directory for each map

    # Write shpaefile:
    # H = nx.DiGraph()
    # mapping = dict()
    #
    # for node in list(X.nodes()):
    #     xx, yy = float(X.nodes[str(node)]['x']), float(X.nodes[str(node)]['y'])
    #     H.add_node(str(node))
    #     # nx.set_node_attributes(H, {str(node): (xx, yy)}, 'loc')
    #     mapping[node] = (xx, yy)
    #
    # H1 = nx.relabel_nodes(H, mapping)
    # for edge in list(X.edges()):
    #     e = (mapping[str(edge[0])], mapping[str(edge[1])])
    #     H1.add_edge(*e)

    # X.to_directed()
    # nx.write_shp(X, outdir)

def calc_closest(new_node, node_list):
    # Calculate the closest node in the network
    best_diff = 100000
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
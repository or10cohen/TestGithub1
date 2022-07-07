import osmnx as ox
import networkx as nx

ox.config(log_console=True, use_cache=True)

G_walk = ox.graph_from_place('Manhattan Island, New York City, New York, USA',
                             network_type='walk')

orig_node = ox.get_nearest_node(G_walk,
                                (40.748441, -73.985664))

dest_node = ox.get_nearest_node(G_walk,
                                (40.748441, -73.4))

route = nx.shortest_path(G_walk,
                         orig_node,
                         dest_node,
                         weight='length')

route_map = ox.plot_route_folium(G_walk, route)

route_map.save('route.html')
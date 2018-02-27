import csv
import time
import pprint as pp
import networkx as nx

import Network_Based_Recommendation_System_FUNCTIONS as homework_2



print
print ("Current time: " + str(time.asctime(time.localtime())))
print
print


all_groups = [
	{1701: 1, 1703: 1, 1705: 1, 1707: 1, 1709: 1}, ### Movie night with friends.
	{1701: 1, 1702: 4}, ### First appointment scenario: the preferences of the girl are 4 times more important than those of the man.
	{1701: 1, 1702: 2, 1703: 1, 1704: 2}, ### Two couples scenario: the preferences of girls are still more important than those of the men...
	{1701: 1, 1702: 1, 1703: 1, 1704: 1, 1705: 1, 1720:10}, ### Movie night with a special guest.
	{1701: 1, 1702: 1, 1703: 1, 1704: 1, 1705: 1, 1720:10, 1721:10, 1722:10}, ### Movie night with 3 special guests.
]
print
pp.pprint(all_groups)
print


graph_file = "./input_data/u_data_homework_format.txt"

pp.pprint("Load Graph.")
print ("Current time: " + str(time.asctime(time.localtime())))
graph_users_items = homework_2.create_graph_set_of_users_set_of_items(graph_file)
print (" #Users in Graph= " + str(len(graph_users_items['users'])))
print (" #Items in Graph= " + str(len(graph_users_items['items'])))
print (" #Nodes in Graph= " + str(len(graph_users_items['graph'])))
print (" #Edges in Graph= " + str(graph_users_items['graph'].number_of_edges()))
print ("Current time: " + str(time.asctime(time.localtime())))
print
print


pp.pprint("Create Item-Item-Weighted Graph.")
print ("Current time: " + str(time.asctime(time.localtime())))
item_item_graph = homework_2.create_item_item_graph(graph_users_items)
print (" #Nodes in Item-Item Graph= " + str(len(item_item_graph)))
print (" #Edges in Item-Item Graph= " + str(item_item_graph.number_of_edges()))
print ("Current time: " + str(time.asctime(time.localtime())))
print
print


### Conversion of the 'Item-Item-Graph' to a scipy sparse matrix representation.
### This reduces a lot the PageRank running time ;)
print
print (" Conversion of the 'Item-Item-Graph' to a scipy sparse matrix representation.")
N = len(item_item_graph)
nodelist = item_item_graph.nodes()
M = nx.to_scipy_sparse_matrix(item_item_graph, nodelist=nodelist, weight='weight', dtype=float)
print (" Done.")
print
#################################################################################################


output_file = open("./Output_Recommendation_for_Group.tsv", 'w')
output_file_csv_writer = csv.writer(output_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
print
for current_group in all_groups:
    print ("Current group: ")
    pp.pprint(current_group)
    print ("Current time: " + str(time.asctime(time.localtime())))
    
    sorted_list_of_recommended_items_for_current_group = []

    # Creating super user
    super_user = {}
    total_weight = sum(current_group.values())

    for u in current_group:
        w = current_group.get(u)
        user_weight = w/total_weight
        ratings = graph_users_items['graph'][u]
        for movie_id in ratings:
            r = ratings.get(movie_id)
            if super_user.get(movie_id) == None:
                super_user[movie_id] = [(user_weight, r)]
            else:
                super_user.get(movie_id).append((user_weight, r))
                
    for m in super_user.keys():
        total_relative_weight = 0
        final_rating = 0
        for w, r in super_user.get(m):
            total_relative_weight += w
            
        for w, r in super_user.get(m):
            final_rating += (w/total_relative_weight) * r['weight']

        super_user[m] = final_rating

    # Creating preference vector
    preference_vector = {}

    items_iig = graph_users_items.get('items_iig')

    gui = graph_users_items.get('graph')
    items_from_user = set(super_user.keys()).intersection(items_iig)
    sum_all_scores = sum(super_user.values())

    for i in items_from_user:
        preference_vector[i] = super_user[i]/sum_all_scores
    
    for i in items_iig:
        if preference_vector.get(i) == None:
            preference_vector[i] = 0

    personalized_pagerank_vector_of_items = homework_2.pagerank(M, N, nodelist, alpha=0.85, personalization=preference_vector)
    
    # Creating ranked list  
    ranked_list = list(personalized_pagerank_vector_of_items.items())
    ranked_list.sort(key=lambda x:x[1], reverse = True)
    for t in ranked_list:
        if super_user.get(t[0]) == None:
            sorted_list_of_recommended_items_for_current_group.append(t[0])
    
    print ("Recommended Sorted List of Items:")
    print(str(sorted_list_of_recommended_items_for_current_group[:30]))
    print
    output_file_csv_writer.writerow(sorted_list_of_recommended_items_for_current_group)
	
output_file.close()	
	
	




print
print
print ("Current time: " + str(time.asctime(time.localtime())))
print ("Done ;)")
print

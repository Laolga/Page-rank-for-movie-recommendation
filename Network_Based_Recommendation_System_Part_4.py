import csv
import time
import pprint as pp
import networkx as nx

import Network_Based_Recommendation_System_FUNCTIONS as homework_2


def create_category_movies(category_movies_file):
    category_movies = {}
    movie_categories = {}
    result = {}

    c = 1

    input_file = open(category_movies_file, 'r')
    input_file_csv_reader = csv.reader(input_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
    for line in input_file_csv_reader:
        for i in range(len(line)):
            if category_movies.get(int(line[i])) == None:
                category_movies[int(line[i])] = []
                category_movies.get(int(line[i])).append(c)
            else:
                category_movies.get(int(line[i])).append(c)

            if movie_categories.get(c) == None:
                movie_categories[c] = [int(line[i])]
            else:
                movie_categories.get(c).append(int(line[i]))

        c += 1
    input_file.close()
    
    result['category_movies'] = category_movies
    result['movie_categories'] = movie_categories

    return(result)
    

def create_preference_vector_for_teleporting_categories(user_preferences):
    preference_vector = {}

    items_iig = graph_users_items.get('items_iig')
    result = create_category_movies('./datasets/category_movies.txt')
    category_movies = result.get('category_movies') 
    movie_categories = result.get('movie_categories')
    
    for i in items_iig:
        total_prob = 0
        for c in category_movies.get(i):
            total_prob += user_preferences.get(c)/len(movie_categories.get(c))
        preference_vector[i] = total_prob
    
    return preference_vector



print
print ("Current time: " + str(time.asctime(time.localtime())))
print
print


graph_file = "./datasets/user_movie_rating.txt"

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

print
print ("Current time: " + str(time.asctime(time.localtime())))

sorted_list_of_recommended_items = []

user_preferences = {1: 0.2, 2: 0.6, 3: 0.1, 4: 0.1, 5: 0}

# Creating preference vector
preference_vector = create_preference_vector_for_teleporting_categories(user_preferences)
personalized_pagerank_vector_of_items = homework_2.pagerank(M, N, nodelist, alpha=0.85, personalization=preference_vector)

# Creating ranked list  
ranked_list = list(personalized_pagerank_vector_of_items.items())
ranked_list.sort(key=lambda x:x[1], reverse = True)

print ("Recommended Sorted List of Items:")
for i in ranked_list[1:30]:
    print(str(i[0]) + ", " + str(i[1]))
print
	

print
print
print ("Current time: " + str(time.asctime(time.localtime())))
print ("Done ;)")
print

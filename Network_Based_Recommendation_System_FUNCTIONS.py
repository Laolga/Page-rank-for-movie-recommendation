import csv
import pprint as pp
import networkx as nx
import itertools as it
import math
import scipy.sparse
import random




def pagerank(M, N, nodelist, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, dangling=None):
	if N == 0:
		return {}
	S = scipy.array(M.sum(axis=1)).flatten()
	S[S != 0] = 1.0 / S[S != 0]
	Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
	M = Q * M
	
	# initial vector
	x = scipy.repeat(1.0 / N, N)
	
	# Personalization vector
	if personalization is None:
		p = scipy.repeat(1.0 / N, N)
	else:
		missing = set(nodelist) - set(personalization)
		if missing:
			#raise NetworkXError('Personalization vector dictionary must have a value for every node. Missing nodes %s' % missing)
			print
			print('Error: personalization vector dictionary must have a value for every node')
			print
			exit(-1)
		p = scipy.array([personalization[n] for n in nodelist], dtype=float)
		#p = p / p.sum()
		sum_of_all_components = p.sum()
		if sum_of_all_components > 1.001 or sum_of_all_components < 0.999:
			print
			print("Error: the personalization vector does not represent a probability distribution :(")
			print
			exit(-1)
	
	# Dangling nodes
	if dangling is None:
		dangling_weights = p
	else:
		missing = set(nodelist) - set(dangling)
		if missing:
			#raise NetworkXError('Dangling node dictionary must have a value for every node. Missing nodes %s' % missing)
			print
			print('Error: dangling node dictionary must have a value for every node.')
			print
			exit(-1)
		# Convert the dangling dictionary into an array in nodelist order
		dangling_weights = scipy.array([dangling[n] for n in nodelist], dtype=float)
		dangling_weights /= dangling_weights.sum()
	is_dangling = scipy.where(S == 0)[0]

	# power iteration: make up to max_iter iterations
	for _ in range(max_iter):
         xlast = x
         x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
         # check convergence, l1 norm
         err = scipy.absolute(x - xlast).sum()
         if err < N * tol:
             return dict(zip(nodelist, map(float, x)))
	#raise NetworkXError('power iteration failed to converge in %d iterations.' % max_iter)
	print
	print('Error: power iteration failed to converge in '+str(max_iter)+' iterations.')
	print
	exit(-1)




def create_graph_set_of_users_set_of_items(user_item_ranking_file):
	graph_users_items = {}
	all_users_id = set()
	all_items_id = set()
	g = nx.DiGraph()
	input_file = open(user_item_ranking_file, 'r')
	input_file_csv_reader = csv.reader(input_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
	for line in input_file_csv_reader:
		user_id = int(line[0])
		item_id = int(line[1])
		rating = int(line[2])
		g.add_edge(user_id, item_id, weight=rating)
		all_users_id.add(user_id)
		all_items_id.add(item_id)
	input_file.close()
	graph_users_items['graph'] = g
	graph_users_items['users'] = all_users_id
	graph_users_items['items'] = all_items_id
	return graph_users_items


def create_item_item_graph(graph_users_items):
    g = nx.Graph()
    items = list(graph_users_items.get('items'))
    gui = graph_users_items.get('graph').to_undirected()
    
    for i in range(len(items)):
        for j in range(i):
            movie1 = gui[items[i]]
            movie2 = gui[items[j]]
            users1 = set(movie1.keys())
            users2 = set(movie2.keys())
            weight = len(users1.intersection(users2))
            if weight > 0:
                g.add_edge(items[i], items[j], weight=weight)
    
    graph_users_items['items_iig'] = set(g.nodes())
    return g




def create_preference_vector_for_teleporting(user_id, graph_users_items):
    preference_vector = {}

    items_iig = graph_users_items.get('items_iig')

    gui = graph_users_items.get('graph')
    items_from_user = set(gui[user_id]).intersection(items_iig)
    sum_all_scores = 0
    for i in items_from_user:
        sum_all_scores = sum_all_scores + gui[user_id][i]['weight']

    for i in items_from_user:
        preference_vector[i] = gui[user_id][i]['weight']/sum_all_scores
    
    for i in items_iig:
        if preference_vector.get(i) == None:
            preference_vector[i] = 0
    
    return preference_vector
	



def create_ranked_list_of_recommended_items(page_rank_vector_of_items, user_id, training_graph_users_items):
    # This is a list of 'item_id' sorted in descending order of score.
    sorted_list_of_recommended_items = []
  
    already_rated = training_graph_users_items['graph'][user_id]
    ranked_list = list(page_rank_vector_of_items.items())
    ranked_list.sort(key=lambda x:x[1], reverse = True)
    for t in ranked_list:
        if already_rated.get(t[0]) == None:
            sorted_list_of_recommended_items.append(t[0])
    
    return sorted_list_of_recommended_items




def discounted_cumulative_gain(user_id, sorted_list_of_recommended_items, test_graph_users_items):
    dcg = 0.
    raitings = test_graph_users_items['graph'][user_id]
    k = 1
    for i in sorted_list_of_recommended_items:
        if raitings.get(i) !=None:
            dcg+=raitings[i]["weight"]/math.log(k+1,2)
            k+=1
    return dcg
	
        


def maximum_discounted_cumulative_gain(user_id, test_graph_users_items):
    dcg = 0.
    raitings = test_graph_users_items['graph'][user_id]
    k = 1
    r = []
    for i in raitings.keys():
        r.append(raitings[i]["weight"])
    r.sort(reverse=True)
    for i in r:
        dcg+=i/math.log(k+1,2)
        k+=1
    return dcg














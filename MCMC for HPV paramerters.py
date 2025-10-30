import pandas as pd
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import beta
import math
from tqdm import tqdm
import time
from multiprocessing import Lock

random.seed(599) 
np.random.seed(599)

real_rate1=
real_rate2=
real_rate3=

gender_ratio=
# Define likelihood function
def likelihood(median_rate1, median_rate2,median_rate3):
    return (median_rate1 - real_rate1) ** 2 + (median_rate2 - real_rate2) ** 2 + (median_rate3 - real_rate3) ** 2

# Define MCMC parameter distributions
alpha1 = 
beta_param1 = 
alpha2 = 
beta_param2 = 
alpha3 =
beta_param3 = 
alpha4 = 
beta_param4 = 
iterations =   # iterations for MCMC
beta_results_acc = []
beta_results_rej = []


# Save results
best_parameters = None
best_likelihood = np.inf

# MCMC iteration process
for i in range(iterations):

    print (f"这是第{i}次迭代")
    # Generate new infection parameters using Beta distributions and adjust according to constraints
    new_prob1 = np.random.beta(alpha1, beta_param1)
    new_prob2 = np.random.beta(alpha2, beta_param2) 
    infection_probabilities = [new_prob1, new_prob2]
    infection_prob_msm_to_female = np.random.beta(alpha3, beta_param3) 
    infection_prob_male_to_female =  infection_prob_msm_to_female
    infection_prob_female_to_msm_or_male = np.random.beta(alpha3, beta_param3)
    clear_rate = np.random.beta(alpha4, beta_param4)  
    print(f"msm_prob1: {new_prob1}")
    print(f"msm_prob2:{new_prob2}")
    print(f"msm_female_prob:{infection_prob_msm_to_female}")
    print(f"msm_male_female_prob:{infection_prob_male_to_female}")
    print(f"msm_female_male_prob:{infection_prob_female_to_male}")
   
    # Define a single MCMC iteration simulation function
    def simulate(data111):
        age_dist_df = data111

        start_time = time.time()

        rate1_list = [] # List of MSM infection rates
        rate2_list = [] # List of heterosexual women infection rates
        rate3_list = [] # List of heterosexual men infection rates
        # Number of simulated nodes
        total_msm_nodes = 
        bisexual_proportion =   # proportion of bisexual MSM
        bisexual_msm_count = int(total_msm_nodes * bisexual_proportion)
        general_msm_count = total_msm_nodes - bisexual_msm_count

        # Create node array
        msm_nodes = np.zeros(total_msm_nodes, dtype=[
            ('node_id', int),
            ('age', int),
            ('is_bisexual', bool),
            ('heterosexual_partners', float),
            ('same_sex_partners', float)
        ])
        msm_nodes['node_id'] = np.arange(1, total_msm_nodes + 1)

        # Assign bisexual identity
        msm_nodes['is_bisexual'][:bisexual_msm_count] = True
        msm_nodes['is_bisexual'][bisexual_msm_count:] = False

        # Custom age distribution (proportion per year)
        age_distribution = {}

        for age in range(18, 20):
            age_distribution[age] = 

        for age in range(20, 30):
            age_distribution[age] = 

        for age in range(30, 40):
            age_distribution[age] = 

        for age in range(40, 50):
            age_distribution[age] = 

        for age in range(50, 71):
            age_distribution[age] = 

        # Compute counts per age (rounded)
        age_counts = {age: round(total_msm_nodes * prop) for age, prop in age_distribution.items()}

        # Correct total population
        current_total = sum(age_counts.values())
        difference = total_msm_nodes - current_total

        # If there's a discrepancy, adjust counts from largest to smallest
        if difference != 0:
            sorted_ages = sorted(age_counts.items(), key=lambda x: -x[1])  # descending order
            for i in range(abs(difference)):
                age = sorted_ages[i % len(sorted_ages)][0]
                age_counts[age] += 1 if difference > 0 else -1

        # Generate assigned ages list
        assigned_ages = []
        for age, count in age_counts.items():
            assigned_ages.extend([age] * count)

        # Shuffle to avoid binding to bisexual ordering
        np.random.shuffle(assigned_ages)

        # Assign ages
        msm_nodes['age'] = assigned_ages

        # Set maximum number of heterosexual partners
        max_partners = 11

        # Function: generate number of heterosexual partners based on age and probability distribution
        def generate_heterosexual_partners(age):
            if age <= 29:
                alpha = 
                beta = 
            elif age <= 39:
                alpha =
                beta = 
            else:
                alpha = 
                beta = 
            
            # Generate discrete distribution samples
            partners = np.arange(0, max_partners + 1)
            probabilities = np.array([alpha * x ** beta if x > 0 else 0 for x in partners])
            
            # Normalize probability distribution
            probabilities /= probabilities.sum()
            
            # Sample from discrete distribution
            return np.random.choice(partners, p=probabilities)

        # Assign heterosexual partner counts for bisexual MSM nodes
        for i in range(bisexual_msm_count):
            age = msm_nodes['age'][i]
            msm_nodes['heterosexual_partners'][i] = generate_heterosexual_partners(age)


        # Function: generate number of same-sex partners based on age and probability distribution
        def generate_same_sex_partners(age):
            if age <= 29:
                alpha = 
                beta = 
                def probability_distribution(x):
                    if x == 0:
                        return 0
                    return alpha * x ** beta
            elif age <= 39:
                alpha = 
                def probability_distribution(x):
                    return alpha * np.exp(-alpha * x)
            else:
                alpha = 
                def probability_distribution(x):
                    return alpha * np.exp(-alpha * x)
            
            # Generate discrete distribution samples
            max_partners = 21  # set a reasonable maximum number of same-sex partners
            partners = np.arange(1, max_partners + 1)
            probabilities = np.array([probability_distribution(x) for x in partners])
            
            # Handle possible zero probabilities and normalize
            probabilities[probabilities < 0] = 0
            total_prob = probabilities.sum()
            if total_prob == 0:
                probabilities = np.ones_like(probabilities) / len(probabilities)  # uniform if all zeros
            
            # Normalize probability distribution
            probabilities /= probabilities.sum()
            
            # Sample from discrete distribution
            return np.random.choice(partners, p=probabilities)

        # Assign same-sex partner counts to all nodes
        for i in range(total_msm_nodes):
            age = msm_nodes['age'][i]
            msm_nodes['same_sex_partners'][i] = generate_same_sex_partners(age)


        # Adjust total same-sex partners to be even
        same_sex_partners_total = msm_nodes['same_sex_partners'].sum()

        # If the total is odd, increase the last node's same-sex partner count by one
        if same_sex_partners_total % 2 != 0:
            msm_nodes['same_sex_partners'][-1] += 1
                
        # Print part of the allocation for verification (commented out)
        # print(pd.DataFrame(msm_nodes).head(50))

        # Convert node data to DataFrame for saving to Excel
        df = pd.DataFrame(msm_nodes)



        # Retrieve node information
        nodes = df['node_id'].tolist()
        ages = df['age'].tolist()
        same_sex_partners = df['same_sex_partners'].tolist()
        heterosexual_partners = df['heterosexual_partners'].tolist()
        bisexual = df['is_bisexual'].tolist()

        # Initialize network
        network = nx.Graph()

        # Add age attribute to MSM nodes
        for i, node in enumerate(nodes):
            network.add_node(node, age=ages[i])
        # Add bisexual attribute to MSM nodes
        for i, node in enumerate(nodes):
            network.add_node(node, bisexual=bisexual[i])

        # Define function to calculate edge weight
        def calculate_weight(i, j, msm_nodes):
            partners_i = msm_nodes[i]['same_sex_partners']
            partners_j = msm_nodes[j]['same_sex_partners']
            age_i = msm_nodes[i]['age']
            age_j = msm_nodes[j]['age']

            if partners_i <= 5 and partners_j <= 5:
                weight = abs(partners_i - partners_j) + abs(age_i - age_j)
            elif partners_i > 5 and partners_j > 5:
                weight = abs(age_i - age_j)
            else:
                weight = 100 + abs(age_i - age_j)

            return weight



        def grasp_connect(G, msm_nodes):
            # Initialize list of all nodes that haven't completed their connections
            unconnected_nodes = list(range(len(msm_nodes)))

            # Termination: all nodes reach same_sex_partners or only one node remains
            while len(unconnected_nodes) > 1:
                # Find node(s) with maximum same_sex_partners among unconnected nodes
                max_partners = -1
                nodes_with_max_partners = []
                for i in unconnected_nodes:
                    node = msm_nodes[i]
                    if node['same_sex_partners'] > max_partners:
                        max_partners = node['same_sex_partners']
                        nodes_with_max_partners = [i]
                    elif node['same_sex_partners'] == max_partners:
                        nodes_with_max_partners.append(i)

                # Randomly select one node i among those with maximum same_sex_partners
                if nodes_with_max_partners:  # ensure list is not empty
                    i = random.choice(nodes_with_max_partners)
                    node_i = msm_nodes[i]

                # Find the node with the minimum weight to connect to i, excluding already connected nodes
                best_node = None
                best_weight = float('inf')

                for j in unconnected_nodes:
                    if i != j and not G.has_edge(node_i['node_id'], msm_nodes[j]['node_id']):  # exclude self and existing edges
                        weight = calculate_weight(i, j, msm_nodes)
                        if weight < best_weight:
                            best_weight = weight
                            best_node = j

                # If a best connection is found, add the edge
                if best_node is not None:
                    G.add_edge(node_i['node_id'], msm_nodes[best_node]['node_id'], weight=best_weight)
                    
                    # Update connected partners count for both nodes
                    G.nodes[node_i['node_id']]['connected_partners'] += 1
                    G.nodes[msm_nodes[best_node]['node_id']]['connected_partners'] += 1
                    
                    # Remove node i if it reached its maximum connections
                    if G.nodes[node_i['node_id']]['connected_partners'] >= G.nodes[node_i['node_id']]['same_sex_partners']:
                        unconnected_nodes.remove(i)
                    
                    # Remove best_node if it also reached its maximum connections
                    if G.nodes[msm_nodes[best_node]['node_id']]['connected_partners'] >= G.nodes[msm_nodes[best_node]['node_id']]['same_sex_partners']:
                        unconnected_nodes.remove(best_node)
            else:
                if i in unconnected_nodes:
                    unconnected_nodes.remove(i)          
        # Create an empty network
        network = nx.Graph()

        # Add initial attributes for all nodes (age, type, connected partners count)
        for node in msm_nodes:
            network.add_node(node['node_id'], age=node['age'], same_sex_partners=node['same_sex_partners'], connected_partners=0)

        # Run the GRASP algorithm
        grasp_connect(network, msm_nodes)       
       
        # Add female nodes for each MSM node
        female_counter = max(nodes) + 1  # Ensure female node IDs do not conflict with MSM node IDs
        female_nodes = []
        female_ages = []  # Create a list to store female node ages


        for i, node in enumerate(nodes):
            num_female_partners = heterosexual_partners[i]  # Use heterosexual_partners to add corresponding number of female nodes for each MSM node
            
            for _ in range(round(num_female_partners)):
                female_node = female_counter
                female_nodes.append(female_node)
                female_ages.append(ages[i])  # Assign MSM node's age to female node
                network.add_node(female_node, age=ages[i])  # Add female node and assign age attribute
                network.add_edge(node, female_node)  # Connect MSM node and female node
                female_counter += 1


        # Define male_partners probability distribution for female nodes
        female_partner_probabilities = [   , , , ]

        # Add assigned male_partners attribute to each female node in the network
        for i, female_node in enumerate(female_nodes):
            male_partners = np.random.choice([1, 2, 3, 4], size=1, p=female_partner_probabilities)
            network.nodes[female_node]['male_partners'] =male_partners.item()

        
        # Calculate the number of male nodes
        num_msm_with_heterosexual_partners = sum(1 for partners in heterosexual_partners if partners > 0)
        num_female_nodes = len(female_nodes)
        num_male_nodes = int(num_female_nodes * gender_ratio - num_msm_with_heterosexual_partners)

        

        # Assume the CSV file has two columns: 'age' and 'density'
        age_bins = age_dist_df['Age'].values
        density_values = age_dist_df['Density'].values

        # Ensure age range is between 18 and 70
        valid_indices = (age_bins >= 18) & (age_bins <= 70)
        age_bins = age_bins[valid_indices]
        density_values = density_values[valid_indices]

        # Normalize density values so their sum equals 1
        density_values /= np.sum(density_values)

        # Create a function to sample ages from the distribution
        def sample_age_from_distribution(n, age_bins, density_values):
            return np.random.choice(age_bins, size=1, p=density_values)

        # Assign ages to male nodes
        # male_ages = sample_age_from_distribution(num_male_nodes, age_bins, density_values).astype(int)
        # Define female_partners probability distribution for male nodes
        male_partner_probabilities = [   ,    ,    ,    ]

        # Add male nodes to the network and assign age and partner counts
        male_counter = max(female_nodes) + 1
        male_nodes = []
        for i, male_node in enumerate(range(male_counter, male_counter + num_male_nodes)):
            male_ages = sample_age_from_distribution(1, age_bins, density_values).astype(int)[0]
            female_partners = np.random.choice([1, 2, 3, 4], size= 1, p=male_partner_probabilities)
            network.add_node(male_node, age=male_ages.item(), female_partners = female_partners.item())  # Add male node and assign age attribute
            
            male_nodes.append(male_node)

        total_partners = sum(network.nodes[node]['male_partners'] for node in female_nodes)
       
        # Define function to calculate weight between female and male nodes
        def calculate_weight_female_male(i, j, network):
            female_age = network.nodes[i]['age']
            male_age = network.nodes[j]['age']            

            female_partners = network.nodes[i]['male_partners']
            male_partners = network.nodes[j]['female_partners']            

            weight = abs(female_age - male_age) + abs(female_partners - male_partners)
            return weight
        # Implement GRASP algorithm: connect female nodes with male nodes
        def grasp_connect_female_male(network, female_nodes, male_nodes):
            # Initialize female and male nodes that have not reached their target number of edges
            unconnected_female_nodes = female_nodes[:]
            unconnected_male_nodes = male_nodes[:]
            
            # Loop until either female or male nodes are exhausted
            while len(unconnected_female_nodes) > 0 and len(unconnected_male_nodes) > 0:
                # Randomly select a female node from the unconnected list
                i = random.choice(unconnected_female_nodes)
                best_male_node = None
                best_weight = float('inf')
                
                # Iterate over unconnected male nodes to find the one with minimum weight
                for j in unconnected_male_nodes:
                    if not network.has_edge(i, j):  # Check there is no existing connection between i and j
                        weight = calculate_weight_female_male(i, j, network)
                        if weight < best_weight:
                            best_weight = weight
                            best_male_node = j
                
                # If a minimal weight male node is found, create an edge
                if best_male_node is not None:
                    network.add_edge(i, best_male_node, weight=best_weight)
                    
                    # Update connected partners count for female node
                    if 'connected_partners' in network.nodes[i]:
                        network.nodes[i]['connected_partners'] += 1
                    else:
                        network.nodes[i]['connected_partners'] = 1
                    
                    # Update connected partners count for male node
                    if 'connected_partners' in network.nodes[best_male_node]:
                        network.nodes[best_male_node]['connected_partners'] += 1
                    else:
                        network.nodes[best_male_node]['connected_partners'] = 1
                    
                    # Remove female node if it reaches its partner limit
                    if network.nodes[i]['connected_partners'] >= network.nodes[i]['male_partners']:
                        unconnected_female_nodes.remove(i)
                    
                    # Remove male node if it reaches its partner limit
                    if network.nodes[best_male_node]['connected_partners'] >= network.nodes[best_male_node]['female_partners']:
                        unconnected_male_nodes.remove(best_male_node)
                else:
                    if i in unconnected_male_nodes:
                        unconnected_male_nodes.remove(i) 
            
           

        # Run GRASP algorithm to connect female and male nodes
        grasp_connect_female_male(network, female_nodes, male_nodes)
    
        # Retrieve all nodes and their attributes from the network
        nodes = list(network.nodes(data=True))
       
        # Initialize all nodes with type and infected attributes
        for node in network.nodes():
            network.nodes[node]['type'] = 'unknown'
            network.nodes[node]['infected'] = False
            # network.nodes[node]['same_sex_partners'] = same_sex_partners
            
        # Assign type attribute to nodes
        for node, attributes in network.nodes(data=True):
            if 'same_sex_partners' in attributes:
                network.nodes[node]['type'] = 'msm'
            elif 'male_partners' in attributes:
                network.nodes[node]['type'] = 'female'
            elif 'female_partners' in attributes:
                network.nodes[node]['type'] = 'male'

        
        # Initial infected nodes setup
        msm_nodes = [node for node, attributes in network.nodes(data=True) if attributes['type'] == 'msm']
        selected_msm_nodes = random.sample(msm_nodes, min(5, len(msm_nodes)))

        # Set infected attribute to True for selected MSM nodes
        for node in selected_msm_nodes:
            network.nodes[node]['infected'] = True

        # Initialize bisexual attribute to False for all MSM nodes
        for node, attributes in network.nodes(data=True):
            if attributes.get('type') == 'msm':
                network.nodes[node]['bisexual'] = False

        # Identify MSM nodes connected to female nodes and set bisexual to True
        for edge in network.edges():
            source, target = edge
            source_type = network.nodes[source].get('type')
            target_type = network.nodes[target].get('type')
            
            if source_type == 'msm' and target_type == 'female':
                network.nodes[source]['bisexual'] = True
            elif source_type == 'female' and target_type == 'msm':
                network.nodes[target]['bisexual'] = True
            
        # Simulate infection transmission for 365 days, transmitting every 5 days
        days = int(365 // 5)
        for day in range(days):
            # Iterate over all infected nodes
            infected_nodes = [node for node, attributes in network.nodes(data=True) if attributes['infected']]
            if not infected_nodes:
                break 
            else:
                for infected_node in infected_nodes:
                    # Randomly select one neighbor edge from the infected node's edges
                    neighbors = list(network.neighbors(infected_node))
                    if neighbors:
                        neighbor = random.choice(neighbors)
                        neighbor_attributes = network.nodes[neighbor]
                        # Determine infection probability based on node types
                        if network.nodes[infected_node]['type'] == 'msm' and neighbor_attributes['type'] == 'msm':
                            prob = random.choice(infection_probabilities)
                            #prob = infection_probabilities
                        elif network.nodes[infected_node]['type'] == 'msm' and neighbor_attributes['type'] == 'female':
                            prob = infection_prob_msm_to_female
                        elif network.nodes[infected_node]['type'] == 'female' and (neighbor_attributes['type'] == 'male' or neighbor_attributes['type'] == 'msm'):
                            prob = infection_prob_female_to_msm_or_male
                        elif network.nodes[infected_node]['type'] == 'male' and neighbor_attributes['type'] == 'female':
                            prob = infection_prob_male_to_female
                        else:
                            prob = 0  # Conditions for infection not met
                        
                        # Determine if infection occurs
                        if random.random() <= prob:
                            network.nodes[neighbor]['infected'] = True
                     # Record infection time; embed clearance module
                    infection_time = network.nodes[infected_node].get('infection_time', 0)
                    infection_time += 5  # Time per transmission cycle
                    network.nodes[infected_node]['infection_time'] = infection_time
                # Find nodes with infection_time equal to 225
                nodes_with_225_days = [node for node in infected_nodes if network.nodes[node]['infection_time'] == 225]
                # Calculate number of eligible infected nodes and select nodes to recover (63%)
                if nodes_with_225_days:
                    num_to_recover = int(len(nodes_with_225_days) * clear_rate)  # 63% proportion
                    recovered_nodes = random.sample(nodes_with_225_days, num_to_recover)
                    # Return selected nodes to susceptible state
                    for recovered_node in recovered_nodes:
                        network.nodes[recovered_node]['infected'] = False
                        network.nodes[recovered_node]['infection_time'] = 0  # Reset infection time after recovery      
        # Implement GRASP algorithm: connect female nodes with male nodes (variant)
        def grasp_connect_female_male1(network, female_nodes, male_nodes):
            # Extract node identifiers
            female_node_ids = [node['node_id'] for node in female_nodes]
            male_node_ids = [node['node_id'] for node in male_nodes]

            # Initialize female and male nodes that have not reached their target number of edges
            unconnected_female_nodes = female_node_ids[:]
            unconnected_male_nodes = male_node_ids[:]
            
            # Loop until either female or male nodes are exhausted
            while len(unconnected_female_nodes) > 0 and len(unconnected_male_nodes) > 0:
                # Randomly select a node from unconnected male nodes
                i = random.choice(unconnected_male_nodes)
                best_male_node = None
                best_weight = float('inf')
                # Check whether the current male node already has edges to all unconnected female nodes
                all_connected = True
                for j in unconnected_female_nodes:
                    if not network.has_edge(i, j):  # If there exists an unconnected female node
                        all_connected = False
                        break
                
                # If the current male node is connected to all unconnected female nodes, remove this male node
                if all_connected:
                    unconnected_male_nodes.remove(i)
                    continue  # Skip the rest of this iteration and continue to the next
                # Iterate over unconnected female nodes to find the one with minimum weight to connect to the male node
                for j in unconnected_female_nodes:
                    if not network.has_edge(i, j):  # Ensure no existing connection between i and j
                        weight = calculate_weight_female_male(j, i, network)
                        if weight < best_weight:
                            best_weight = weight
                            best_male_node = j
                
                # If a minimal-weight female node is found, create an edge
                if best_male_node is not None:
                    network.add_edge(i, best_male_node, weight=best_weight)
                    
                    # Update connected partners count for female node
                    if 'connected_partners' in network.nodes[i]:
                        network.nodes[i]['connected_partners'] += 1
                    else:
                        network.nodes[i]['connected_partners'] = 1
                    
                    # Update connected partners count for male node
                    if 'connected_partners' in network.nodes[best_male_node]:
                        network.nodes[best_male_node]['connected_partners'] += 1
                    else:
                        network.nodes[best_male_node]['connected_partners'] = 1
                    
                    # If female node has reached its connection quota, remove it
                    if network.nodes[i]['connected_partners'] >= network.nodes[i]['female_partners']:
                        unconnected_male_nodes.remove(i)
                    
                    # If male node has reached its connection quota, remove it
                    if network.nodes[best_male_node]['connected_partners'] >= network.nodes[best_male_node]['male_partners']:
                        unconnected_female_nodes.remove(best_male_node)
                else:
                    if i in unconnected_male_nodes:
                        unconnected_male_nodes.remove(i)             
           
       # Initialize an empty list to store yearly data
        data = []        
        # Number of simulation years
        for i in range(200):
            # print(f"Year {i+2}")
            # Step 1: Increase age by 1 for all nodes in the network
            for node in network.nodes():
                if 'age' in network.nodes[node]:
                    network.nodes[node]['age'] += 1        
            
            nodes_to_remove = {}
            # Find MSM nodes that meet removal criteria (age > 70 or specific ages to remove)
            for node, attributes in network.nodes(data=True):               
                    if attributes['age'] > 70:
                        nodes_to_remove[node] = {'type': attributes.get('type', 'unknown'), 'bisexual': attributes.get('bisexual', False)}

            # Randomly remove some MSM nodes by age            
            age_removal_counts = {
                age_remove:   ,
                age_remove:   ,
                age_remove:   
            }

            for target_age, count in age_removal_counts.items():
                # Get all MSM nodes with age == target_age
                age_matched_nodes = [
                    node for node, attr in network.nodes(data=True)
                    if attr.get('type') == 'msm' and attr.get('age') == target_age
                ]
                if len(age_matched_nodes) < count:
                    print(f"Warning: 仅找到 {len(age_matched_nodes)} 个年龄为 {target_age} 岁的MSM节点，少于{count}个，将全部移除。")
                    sampled_nodes = age_matched_nodes
                else:
                    sampled_nodes = random.sample(age_matched_nodes, count)

                for node in sampled_nodes:
                    nodes_to_remove[node] = {
                        'type': network.nodes[node].get('type', 'unknown'),
                        'bisexual': network.nodes[node].get('bisexual', False)
                    }

            # Remove all selected nodes
            for node in nodes_to_remove:
                network.remove_node(node)


            # Step 3: Count removed node types
            removed_types = {}
            for node_info in nodes_to_remove.values():
                node_type = node_info['type']
                if node_type == 'msm':
                    bisexual_status = 'bisexual=true' if node_info['bisexual'] else 'bisexual=false'
                    removed_types[bisexual_status] = removed_types.get(bisexual_status, 0) + 1
                else:
                    removed_types[node_type] = removed_types.get(node_type, 0) + 1

            # Step 4: Add new nodes
            new_msm_nodes = []

            # Add MSM nodes with bisexual=True
            for _ in range(removed_types.get('bisexual=true', 0)):
                new_node = max(network.nodes()) + 1
                network.add_node(new_node, type='msm', bisexual=True, age=18, infected=False)
                new_msm_nodes.append(new_node)

            # Add MSM nodes with bisexual=False
            for _ in range(removed_types.get('bisexual=false', 0)):
                new_node = max(network.nodes()) + 1
                network.add_node(new_node, type='msm', bisexual=False, age=18, infected=False)
                new_msm_nodes.append(new_node)

            
            # Randomly select some of the new MSM nodes and assign age = 20 (to maintain MSM age distribution)
            if len(new_msm_nodes) >=   :
                selected_for_age_20 = random.sample(new_msm_nodes,   )
            else:               
                selected_for_age_20 = new_msm_nodes

            for node in selected_for_age_20:
                network.nodes[node]['age'] = 20
            
            # Add female nodes
            for _ in range(removed_types.get('female', 0)):
                new_node = max(network.nodes()) + 1
                network.add_node(new_node, type='female', age=18, infected=False)

            # Add male nodes
            for _ in range(removed_types.get('male', 0)):
                new_node = max(network.nodes()) + 1
                network.add_node(new_node, type='male', age=18, infected=False)

            
            # Assign infection status to newly added 18-year-old MSM nodes           
            bisexual_true_nodes = [node for node, data in network.nodes(data=True) if data['type'] == 'msm' and data['bisexual']and data['age'] == 18]
            random.shuffle(bisexual_true_nodes)
            num_infected_bisexual_true = int(0.2896 * len(bisexual_true_nodes))
            for node in bisexual_true_nodes[:num_infected_bisexual_true]:
                network.nodes[node]['infected'] = True

            bisexual_false_nodes = [node for node, data in network.nodes(data=True) if data['type'] == 'msm' and not data['bisexual']and data['age'] == 18]
            random.shuffle(bisexual_false_nodes)
            num_infected_bisexual_false = int(0.2896 * len(bisexual_false_nodes))
            for node in bisexual_false_nodes[:num_infected_bisexual_false]:
                network.nodes[node]['infected'] = True

            # Set remaining new nodes as uninfected
            for node in bisexual_true_nodes[num_infected_bisexual_true:]:
                network.nodes[node]['infected'] = False

            for node in bisexual_false_nodes[num_infected_bisexual_false:]:
                network.nodes[node]['infected'] = False
            # Clear all edges in the graph
            network.clear_edges()
            # Assign same_sex_partners counts to all MSM individuals
            for node in network.nodes(data=True):
                if node[1]['type'] == 'msm':
                    age = node[1]['age']
                    node[1]['same_sex_partners'] = generate_same_sex_partners(age)
            # Calculate the sum of same_sex_partners across all MSM nodes
            total_same_sex_partners = sum(node[1]['same_sex_partners'] for node in network.nodes(data=True) if node[1]['type'] == 'msm')

            # If the sum is odd, increment the last MSM node's same_sex_partners by one
            if total_same_sex_partners % 2 != 0:
                last_msm_node = max(node[0] for node in network.nodes(data=True) if node[1]['type'] == 'msm')
                network.nodes[last_msm_node]['same_sex_partners'] += 1
            total_same_sex_partners = sum(node[1]['same_sex_partners'] for node in network.nodes(data=True) if node[1]['type'] == 'msm')
                   
            # Assign heterosexual partner counts to MSM nodes with bisexual=True
            for node in network.nodes(data=True):
                if node[1]['type'] == 'msm' and node[1]['bisexual']:  # Assumes your node attributes contain a 'bisexual' key
                    age = node[1]['age']
                    node[1]['female_partners'] = generate_heterosexual_partners(age)

            # Define male nodes' female_partners probability distribution
            male_partner_probabilities = [0.223, 0.139, 0.145, 0.493]

            # Define female nodes' male_partners probability distribution
            female_partner_probabilities = [0.412, 0.181, 0.135, 0.272]

            # Assign female_partners to nodes with type=male
            for node in network.nodes(data=True):
                if node[1]['type'] == 'male':
                    node[1]['female_partners'] = np.random.choice([1, 2, 3, 4], p=male_partner_probabilities)

            # Assign male_partners to nodes with type=female
            for node in network.nodes(data=True):
                if node[1]['type'] == 'female':
                    node[1]['male_partners'] = np.random.choice([1, 2, 3, 4], p=female_partner_probabilities)
            
            # Ensure msm_nodes is a list where each element is a dict containing attributes for each MSM node
            msm_nodes = []
            # Define a marker for missing values
            MISSING_VALUE = 'missing'
            # Iterate over all nodes in the network
            for node_id in network.nodes():
                node_data = network.nodes[node_id]
                # Check if the node is of MSM type
                if node_data.get('type') == 'msm':
                    # Extract MSM node information, mark missing values as MISSING_VALUE
                    msm_info = {
                        'node_id': node_id,
                        'age': node_data.get('age', MISSING_VALUE),  # Mark as MISSING_VALUE if no age info
                        'same_sex_partners': node_data.get('same_sex_partners', MISSING_VALUE),  # Mark as MISSING_VALUE if absent
                        'bisexual': node_data.get('bisexual', MISSING_VALUE),  # Mark as MISSING_VALUE if no bisexual attribute
                        'heterosexual_partners': node_data.get('heterosexual_partners', MISSING_VALUE),  # Mark as MISSING_VALUE if absent
                        'infected': node_data.get('infected', MISSING_VALUE),
                        'type': node_data.get('type', MISSING_VALUE),}
                    # Append MSM node info to msm_nodes list
                    msm_nodes.append(msm_info)
            
            # Get all female nodes and their attributes
            female_nodes = [
                {'node_id': node, 'age': attr['age'], 'male_partners': attr.get('male_partners', 0),'infected':attr['infected'],'type':attr['type']}
                for node, attr in network.nodes(data=True) if attr['type'] == 'female'    ]

            # Get all male nodes and their attributes
            male_nodes = [
                {'node_id': node, 'age': attr['age'], 'female_partners': attr.get('female_partners', 0),'infected':attr['infected'],'type':attr['type']}
                for node, attr in network.nodes(data=True)
                if attr['type'] == 'male'  or ( attr['type'] == 'msm' and attr.get('bisexual', False))   ]                   
                                      
            # Compute the sum of female_partners using a list comprehension
            total_female_partners = sum(male['female_partners'] for male in male_nodes)
            # print(total_female_partners)
            total_male_partners = sum(female['male_partners'] for female in female_nodes)
            # print(total_male_partners)

            network = nx.Graph()
            # Add initial attributes (age, type, connected partners count) for all nodes
            for node in msm_nodes:
                network.add_node(node['node_id'], age=node['age'], same_sex_partners=node['same_sex_partners'], connected_partners=0,
                                bisexual=node['bisexual'],infected=node['infected'],type=node['type'])
            for node in female_nodes:
                network.add_node(node['node_id'], age=node['age'], male_partners=node['male_partners'], connected_partners=0,
                                infected=node['infected'],type=node['type'])
            for node in male_nodes:
                network.add_node(node['node_id'], age=node['age'], female_partners=node['female_partners'], connected_partners=0,
                                infected=node['infected'],type=node['type'])
            # Run GRASP algorithm to establish MSM-to-MSM connections
            grasp_connect(network, msm_nodes)
            
            # Call the female-male connection function
            grasp_connect_female_male1(network, female_nodes, male_nodes)
            
            # Simulate infection transmission for 365 days, transmitting every 5 days
            days = int(365 // 5)
            for day in range(days):
                # Iterate over all infected nodes
                infected_nodes = [node for node, attributes in network.nodes(data=True) if attributes['infected']]
                # Check if infected_nodes is empty
                if not infected_nodes:
                    break
                else:
                    for infected_node in infected_nodes:
                        # Randomly select one neighbor edge from the infected node's edges
                        neighbors = list(network.neighbors(infected_node))
                        if neighbors:
                            neighbor = random.choice(neighbors)
                            neighbor_attributes = network.nodes[neighbor]
                            # Determine infection probability based on node types
                            if network.nodes[infected_node]['type'] == 'msm' and neighbor_attributes['type'] == 'msm':
                                prob = random.choice(infection_probabilities)
                                #prob = infection_probabilities
                            elif network.nodes[infected_node]['type'] == 'msm' and neighbor_attributes['type'] == 'female':
                                prob = infection_prob_msm_to_female
                            elif network.nodes[infected_node]['type'] == 'female' and (neighbor_attributes['type'] == 'male' or neighbor_attributes['type'] == 'msm'):
                                prob = infection_prob_female_to_msm_or_male
                            elif network.nodes[infected_node]['type'] == 'male' and neighbor_attributes['type'] == 'female':
                                prob = infection_prob_male_to_female
                            else:
                                prob = 0  # Conditions for infection not met
                            
                            # Determine whether infection occurs
                            if random.random() <= prob:
                                network.nodes[neighbor]['infected'] = True
                        # Record infection time; embed clearance module
                        infection_time = network.nodes[infected_node].get('infection_time', 0)
                        infection_time += 5  # Time per transmission cycle
                        network.nodes[infected_node]['infection_time'] = infection_time
                    # Find nodes with infection_time equal to 225
                    nodes_with_225_days = [node for node in infected_nodes if network.nodes[node]['infection_time'] == 225]
                    # Calculate number of eligible infected nodes and select nodes to recover (63%)
                    if nodes_with_225_days:
                        num_to_recover = int(len(nodes_with_225_days) * clear_rate)  # 63% proportion
                        recovered_nodes = random.sample(nodes_with_225_days, num_to_recover)
                        # Return selected nodes to susceptible state
                        for recovered_node in recovered_nodes:
                            network.nodes[recovered_node]['infected'] = False
                            network.nodes[recovered_node]['infection_time'] = 0  # Reset infection time after recovery
            # Final number of infected nodes
            infected_msm_count = sum(1 for node, attributes in network.nodes(data=True) if attributes['infected'] and attributes.get('type') == 'msm')
            msm_count=sum (1 for node, attributes in network.nodes(data=True) if attributes.get('type') == 'msm')
            msm_rate=infected_msm_count/msm_count
            
            # Count infected female nodes
            infected_female_count = sum(1 for node, attributes in network.nodes(data=True) if attributes['infected'] and attributes.get('type')=='female')
            female_count = sum(1 for node, attributes in network.nodes(data=True) if attributes.get('type')=='female')
            female_rate = infected_female_count / female_count
            # Count infected male nodes
            infected_male_count = sum(1 for node, attributes in network.nodes(data=True) if attributes['infected'] and attributes.get('type')=='male')
            male_count = sum(1 for node, attributes in network.nodes(data=True) if attributes.get('type')=='male')
            male_rate = infected_male_count / male_count
           
            rate1_list.append(msm_rate)  # Append result to list
            rate2_list.append(female_rate)
            rate3_list.append(male_rate)
            
        
        # print("out of loop!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        end_time = time.time()
        # print(f"operate time is {end_time - start_time}")
    
        return rate1_list, rate2_list, rate3_list

    # rate1_list, rate2_list = simulate(1)
    
    from multiprocessing import Pool
    import numpy as np
    import concurrent.futures

    def main():
        
        # Read age distribution data
        age_dist_file_path = 'age_density_distribution.csv'
        age_dist_df = pd.read_csv(age_dist_file_path)
        num_simulations = 100
        with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
            # Submit all simulation tasks
            futures = [executor.submit(simulate, age_dist_df) for _ in range(num_simulations)]
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
            
        # Unpack results from each simulation
        group1_results, group2_results,group3_results = zip(*results)
        
        # Merge results into arrays
        group1_results = np.array(group1_results)
        group2_results = np.array(group2_results)
        group3_results = np.array(group3_results)
        # Merge results into a list
        group1_result = np.median(group1_results, axis=0)
        group2_result = np.median(group2_results, axis=0)
        group3_result = np.median(group3_results, axis=0)
        
        return group1_result, group2_result,group3_result

    if __name__ == "__main__":
        group1_result, group2_result,group3_result = main()       
                # Use data from the last 300 years (from year 200 to 500) to calculate median
        stabilized_infection_rate_1 = np.median(group1_result[150:]) 
        stabilized_infection_rate_2 = np.median(group2_result[150:])
        stabilized_infection_rate_3 = np.median(group3_result[150:])
        print(f"Current MSM stabilized infection rate: {stabilized_infection_rate_1}")
        print(f"Current female stabilized infection rate: {stabilized_infection_rate_2}")
        print(f"Current male stabilized infection rate: {stabilized_infection_rate_3}")
        # Calculate likelihood
        current_likelihood = likelihood(stabilized_infection_rate_1, stabilized_infection_rate_2, stabilized_infection_rate_3)
        print(f"Current SS value: {current_likelihood}")
    
    if current_likelihood <= best_likelihood:
        # Always accept better parameters
        best_likelihood = current_likelihood
        best_infected_rate=(stabilized_infection_rate_1, stabilized_infection_rate_2,stabilized_infection_rate_3)
        best_parameters = (new_prob1, new_prob2,infection_prob_msm_to_female, infection_prob_male_to_female, infection_prob_female_to_msm_or_male,clear_rate)
        beta_results_acc.append([best_likelihood] + list(best_parameters)+[stabilized_infection_rate_1]+[stabilized_infection_rate_2]+[stabilized_infection_rate_3])
    else:
        r = random.random()
        if best_likelihood/current_likelihood > r:
            parameters= (new_prob1, new_prob2,infection_prob_msm_to_female, infection_prob_male_to_female, infection_prob_female_to_msm_or_male,clear_rate)
            beta_results_rej.append([current_likelihood] + list(parameters)+[stabilized_infection_rate_1]+[stabilized_infection_rate_2]+[stabilized_infection_rate_3])
    print("Current best Beta:", " ({:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f})".format(*best_parameters))
    print("Current best SS value:", best_likelihood)
    print("Current best infection rates:", best_infected_rate)

    
# After all simulations are completed, save results to Excel files
df_beta_acc = pd.DataFrame(beta_results_acc, columns=['Best_Likelihood', 'Prob1', 'Prob2', 'MSM_to_Female', 'Male_to_Female', 'Female_to_MSM_or_Male','clear_rate','stabilized_rate_1','stabilized_rate_2','stabilized_rate_3'])
df_beta_acc.to_excel('simulation_1000_beta_acc.xlsx', index=False)
df_beta_rej = pd.DataFrame(beta_results_rej, columns=['Best_Likelihood', 'Prob1', 'Prob2', 'MSM_to_Female', 'Male_to_Female', 'Female_to_MSM_or_Male','clear_rate','stabilized_rate_1','stabilized_rate_2','stabilized_rate_3'])
df_beta_rej.to_excel('simulation_1000_beta_rej.xlsx', index=False)

# Output the best results
print("Best parameters:", best_parameters)
print("Best SS value:", best_likelihood)
print("Best simulated infection rates:", best_infected_rate)

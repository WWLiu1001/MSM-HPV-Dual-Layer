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


hpv_vac= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   #Setting hpv vaccination coverage
gender_ratio = 

for hpv_vac_set in hpv_vac:
    random.seed(599)  
    np.random.seed(599) 
    
    MSM_nodes_num =   #Setting msm simulation nodes number
    hpv_msm_num = int(MSM_nodes_num*hpv_vac_set)     #Setting hpv vaccination msm number at initialization
    percent_18_msm =        #The HPV infection rate among 18-year-old MSM
    hpv_vac_female =        #Initial number of females vaccinated and effectively protected
    hpv_vac_female_18 =    #HPV vaccination rate among 18-year-old females
    hpv_num = hpv_msm_num            # Initial number of people vaccinated against HPV

    


    #  Define infection transmission probabilities_mcmc
    # Random selection probability of MSM to MSM transmission
    infection_probabilities = [ ,  ]  # Random selection probability of MSM to MSM transmission
    infection_prob_msm_to_female =        # Probability of MSM to female transmission
    infection_prob_male_to_female =  infection_prob_msm_to_female   # Probability of male to female transmission
    infection_prob_female_to_msm_or_male =    # Probability of female to MSM or male transmission
    clear_rate =   
        
    def simulate(data111):
        bisexual_msm_data, general_msm_data, age_dist_df = data111

        # Store the results of each simulation
        rate1_list = []
        rate2_list = []
        rate3_list = []
        rate4_list = []
   
        # Only retain the data of individuals aged between 18 and 70.
        bisexual_msm_data = bisexual_msm_data[(bisexual_msm_data['age'] >= 18) & (bisexual_msm_data['age'] <= 70)]
        general_msm_data = general_msm_data[(general_msm_data['age'] >= 18) & (general_msm_data['age'] <= 70)]
       
        # Number of simulated nodes
        total_msm_nodes = 
        bisexual_proportion =   # The Proportion of Bisexual MSM
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

        # BisexuaL MSM nodes allocation
        msm_nodes['is_bisexual'][:bisexual_msm_count] = True
        msm_nodes['is_bisexual'][bisexual_msm_count:] = False

        # Define the age distribution of MSM (percentage per year)
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

        # Calculate the number of individuals to be allocated each year 
        age_counts = {age: round(total_msm_nodes * prop) for age, prop in age_distribution.items()}

        # Correct total number of people
        current_total = sum(age_counts.values())
        difference = total_msm_nodes - current_total

        #  If there is any deviation, adjust the number of people in the order from largest to smallest.
        if difference != 0:
            sorted_ages = sorted(age_counts.items(), key=lambda x: -x[1])  
            for i in range(abs(difference)):
                age = sorted_ages[i % len(sorted_ages)][0]
                age_counts[age] += 1 if difference > 0 else -1

        #  Generate age list 
        assigned_ages = []
        for age, count in age_counts.items():
            assigned_ages.extend([age] * count)

        # Disregard the order and avoid associating it with "bisexual"
        np.random.shuffle(assigned_ages)

        #  Allocation of age
        msm_nodes['age'] = assigned_ages

        # Set the maximum number of heterosexual partners
        max_partners = 11

        # Generates the number of heterosexual partners based on the given age and probability distribution
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
            
            # Generate samples from a discrete distribution par
            partners = np.arange(0, max_partners + 1)
            probabilities = np.array([alpha * x ** beta if x > 0 else 0 for x in partners])
            
            # Normalized probability distribution probabilities 
            probabilities /= probabilities.sum()
            
            # Generate samples from a discrete distribution return
            return np.random.choice(partners, p=probabilities)        # Assign heterosexual partners for bisexual MSM nodes
        for i in range(bisexual_msm_count):
            age = msm_nodes['age'][i]
            msm_nodes['heterosexual_partners'][i] = generate_heterosexual_partners(age)


        # Generate number of same-sex partners based on age and probability distribution
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
            max_partners = 21  # Set a reasonable maximum number of same-sex partners
            partners = np.arange(1, max_partners + 1)
            probabilities = np.array([probability_distribution(x) for x in partners])
            
            # Handle possible zero probabilities and normalize
            probabilities[probabilities < 0] = 0
            total_prob = probabilities.sum()
            if total_prob == 0:
                probabilities = np.ones_like(probabilities) / len(probabilities)  # Use uniform distribution if all probabilities are zero
            
            # Normalize probability distribution
            probabilities /= probabilities.sum()
            
            # Generate samples from the discrete distribution
            return np.random.choice(partners, p=probabilities)

        # Assign same-sex partner numbers to all nodes
        for i in range(total_msm_nodes):
            age = msm_nodes['age'][i]
            msm_nodes['same_sex_partners'][i] = generate_same_sex_partners(age)

        # Adjust total number of same-sex partners to be even
        same_sex_partners_total = msm_nodes['same_sex_partners'].sum()

        # If the total is odd, increase the last node’s same-sex partner count by one
        if same_sex_partners_total % 2 != 0:
            msm_nodes['same_sex_partners'][-1] += 1                
        
        # Convert node data to DataFrame for saving to Excel
        df = pd.DataFrame(msm_nodes)

        # Retrieve node information
        nodes = df['node_id'].tolist()
        ages = df['age'].tolist()
        same_sex_partners = df['same_sex_partners'].tolist()
        heterosexual_partners = df['heterosexual_partners'].tolist()
        bisexual = df['is_bisexual'].tolist()

        # Initialize the network
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
            # Initialize list of unconnected nodes
            unconnected_nodes = list(range(len(msm_nodes)))

            # Termination condition: all nodes reach same_sex_partners or only one node remains
            while len(unconnected_nodes) > 1:
                # Find nodes with the maximum same_sex_partners among unconnected nodes
                max_partners = -1
                nodes_with_max_partners = []
                for i in unconnected_nodes:
                    node = msm_nodes[i]
                    if node['same_sex_partners'] > max_partners:
                        max_partners = node['same_sex_partners']
                        nodes_with_max_partners = [i]
                    elif node['same_sex_partners'] == max_partners:
                        nodes_with_max_partners.append(i)

                # Randomly select one node i from nodes with maximum same_sex_partners
                if nodes_with_max_partners:  # Ensure list is not empty
                    i = random.choice(nodes_with_max_partners)
                    node_i = msm_nodes[i]

                # Find node with minimum weight to connect with node i, excluding already connected nodes
                best_node = None
                best_weight = float('inf')

                for j in unconnected_nodes:
                    if i != j and not G.has_edge(node_i['node_id'], msm_nodes[j]['node_id']):  # Exclude self and existing edges
                        weight = calculate_weight(i, j, msm_nodes)
                        if weight < best_weight:
                            best_weight = weight
                            best_node = j

                # If a suitable connection is found, create an edge
                if best_node is not None:
                    G.add_edge(node_i['node_id'], msm_nodes[best_node]['node_id'], weight=best_weight)
                    
                    # Update connected partners count for both nodes
                    G.nodes[node_i['node_id']]['connected_partners'] += 1
                    G.nodes[msm_nodes[best_node]['node_id']]['connected_partners'] += 1
                    
                    # Remove node i if it reaches its maximum connections
                    if G.nodes[node_i['node_id']]['connected_partners'] >= G.nodes[node_i['node_id']]['same_sex_partners']:
                        unconnected_nodes.remove(i)
                    
                    # Remove best_node if it also reaches its maximum connections
                    if G.nodes[msm_nodes[best_node]['node_id']]['connected_partners'] >= G.nodes[msm_nodes[best_node]['node_id']]['same_sex_partners']:
                        unconnected_nodes.remove(best_node)
            else:
                if i in unconnected_nodes:
                    unconnected_nodes.remove(i)         

        # Create an empty network
        network = nx.Graph()

        # Add initial attributes (age, same_sex_partners, connected_partners) for all nodes
        for node in msm_nodes:
            network.add_node(node['node_id'], age=node['age'], same_sex_partners=node['same_sex_partners'], connected_partners=0)

        # Run GRASP algorithm
        grasp_connect(network, msm_nodes)        
       
        # Add female nodes for each MSM node
        female_counter = max(nodes) + 1  # Ensure female node IDs do not overlap with MSM node IDs
        female_nodes = []
        female_ages = []  # List to store female node ages


        for i, node in enumerate(nodes):
            num_female_partners = heterosexual_partners[i]  # Add corresponding number of female nodes for each MSM node
            
            for _ in range(round(num_female_partners)):
                female_node = female_counter
                female_nodes.append(female_node)
                female_ages.append(ages[i])  # Assign MSM node’s age to its female partner
                network.add_node(female_node, age=ages[i])  # Add female node with age attribute
                network.add_edge(node, female_node)  # Connect MSM node and female node
                female_counter += 1


        # Define female nodes’ male_partners probability distribution
        female_partner_probabilities = [   ,    ,   ,    ]        # Add assigned male_partners attribute to each female node in the network
        for i, female_node in enumerate(female_nodes):
            male_partners = np.random.choice([1, 2, 3, 4], size=1, p=female_partner_probabilities)
            network.nodes[female_node]['male_partners'] = male_partners.item()

        
        # Calculate the number of male nodes
        num_msm_with_heterosexual_partners = sum(1 for partners in heterosexual_partners if partners > 0)
        num_female_nodes = len(female_nodes)
        num_male_nodes = int(num_female_nodes * gender_ratio - num_msm_with_heterosexual_partners)
        
        age_bins = age_dist_df['Age'].values
        density_values = age_dist_df['Density'].values

        # Ensure age range is between 18 and 70
        valid_indices = (age_bins >= 18) & (age_bins <= 70)
        age_bins = age_bins[valid_indices]
        density_values = density_values[valid_indices]

        # Normalize density values to make their sum equal to 1
        density_values /= np.sum(density_values)

        # Create a function to sample ages based on distribution
        def sample_age_from_distribution(n, age_bins, density_values):
            return np.random.choice(age_bins, size=1, p=density_values)

        
        # Define female_partners probability distribution for male nodes
        male_partner_probabilities = [    ,    ,    ,    ]

        # Add male nodes to the network and assign age and partner counts
        male_counter = max(female_nodes) + 1
        male_nodes = []
        for i, male_node in enumerate(range(male_counter, male_counter + num_male_nodes)):
            male_ages = sample_age_from_distribution(1, age_bins, density_values).astype(int)[0]
            female_partners = np.random.choice([1, 2, 3, 4], size=1, p=male_partner_probabilities)
            network.add_node(male_node, age=male_ages.item(), female_partners=female_partners.item())  # Add male node with age attribute
            
            male_nodes.append(male_node)

        total_partners = sum(network.nodes[node]['male_partners'] for node in female_nodes)
        


        # Define weight calculation function between female and male nodes
        def calculate_weight_female_male(i, j, network):
            female_age = network.nodes[i]['age']
            male_age = network.nodes[j]['age']            

            female_partners = network.nodes[i]['male_partners']
            male_partners = network.nodes[j]['female_partners']            

            weight = abs(female_age - male_age) + abs(female_partners - male_partners)
            return weight

        # Implement GRASP algorithm to connect female and male nodes
        def grasp_connect_female_male(network, female_nodes, male_nodes):
            # Initialize unconnected female and male nodes
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
                    if not network.has_edge(i, j):  # Ensure no existing connection between i and j
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
            network.nodes[node]['ever_infected'] = False
            network.nodes[node]['hpv_vac'] = False
                        
        # Assign type attribute to nodes
        for node, attributes in network.nodes(data=True):
            network.nodes[node]['infected'] = False        
            if 'same_sex_partners' in attributes:
                network.nodes[node]['type'] = 'msm'
            elif 'male_partners' in attributes:
                network.nodes[node]['type'] = 'female'
            elif 'female_partners' in attributes:
                network.nodes[node]['type'] = 'male'

        
        # Set initial infected nodes
        # Set infected attribute to True for selected MSM nodes (based on calculated initial infection rate)
        msm_nodes = [node for node, attributes in network.nodes(data=True) if attributes['type'] == 'msm']
        selected_msm_nodes = random.sample(msm_nodes, min(592, len(msm_nodes)))
        for node in selected_msm_nodes:
            network.nodes[node]['infected'] = True
            network.nodes[node]['ever_infected'] = True        # Set infected attribute to True for selected female nodes (initial female HPV infection rate)
        female_nodes = [node for node, attributes in network.nodes(data=True) if attributes['type'] == 'female']
        female_n = int(num_female_nodes*0.156)
        selected_female_nodes = random.sample(female_nodes, min(female_n, len(female_nodes)))
        for node in selected_female_nodes:
            network.nodes[node]['infected'] = True
            network.nodes[node]['ever_infected'] = True
        female_n1 = int(num_female_nodes*hpv_vac_female)
        selected_female_nodes1 = random.sample(female_nodes, min(female_n1, len(female_nodes)))
        for node in selected_female_nodes1:
            network.nodes[node]['hpv_vac'] = True
            network.nodes[node]['infected'] = False
            

        # Set infected attribute to True for selected male nodes (initial male HPV infection rate)
        male_nodes = [node for node, attributes in network.nodes(data=True) if attributes['type'] == 'male']
        male_n = int(num_male_nodes*0.145)
        selected_male_nodes = random.sample(male_nodes, min(male_n, len(male_nodes)))
        for node in selected_male_nodes:
            network.nodes[node]['infected'] = True
            network.nodes[node]['ever_infected'] = True

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
                        
                        if (network.nodes[infected_node]['type'] == 'msm' and 
                            neighbor_attributes['type'] == 'msm' and
                            neighbor_attributes['hpv_vac'] == False):
                            prob = random.choice(infection_probabilities)                 
                        elif (network.nodes[infected_node]['type'] == 'msm' and neighbor_attributes['type'] == 'female'and
                            neighbor_attributes['hpv_vac'] == False):
                            prob = infection_prob_msm_to_female
                           
                        elif (network.nodes[infected_node]['type'] == 'female' and 
                            (neighbor_attributes['type'] == 'male' or 
                            (neighbor_attributes['type'] == 'msm' and neighbor_attributes['hpv_vac'] == False))):
                            prob = infection_prob_female_to_msm_or_male                    
                        elif (network.nodes[infected_node]['type'] == 'male' and neighbor_attributes['type'] == 'female'and
                            neighbor_attributes['hpv_vac'] == False):
                            prob = infection_prob_male_to_female
                        else:
                            prob = 0  # Conditions for infection not met
                        
                        # Determine if infection occurs
                        if random.random() <= prob:
                            network.nodes[neighbor]['infected'] = True
                            network.nodes[neighbor]['ever_infected'] = True
                        
                    # Record infection time; embed clearance module
                    infection_time = network.nodes[infected_node].get('infection_time', 0)
                    infection_time += 5  # Time per transmission cycle
                    network.nodes[infected_node]['infection_time'] = infection_time
                # Find nodes with infection_time equal to 225
                nodes_with_225_days = [node for node in infected_nodes if network.nodes[node]['infection_time'] == 225]
                # Calculate number of eligible infected nodes and select 63% to recover
                if nodes_with_225_days:
                    num_to_recover = int(len(nodes_with_225_days) * clear_rate)  
                    recovered_nodes = random.sample(nodes_with_225_days, num_to_recover)
                    # Return selected nodes to susceptible state
                    for recovered_node in recovered_nodes:
                        network.nodes[recovered_node]['infected'] = False
                        network.nodes[recovered_node]['infection_time'] = 0  # Reset infection time after recovery
        # Implement GRASP algorithm: connect female nodes with male nodes
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
                    if not network.has_edge(i, j):  # If there is at least one unconnected female node
                        all_connected = False
                        break
                
                # If current male node is connected to all unconnected female nodes, remove this male node
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
        hpv_vac_female_18 =    # HPV vaccination rate for 18-year-old females
        percent_18_msm =        # HPV infection rate among 18-year-old MSM
        hpv_num = hpv_msm_num 
        for i in range(100):
            
            # Increase age by 1 for all nodes in the network
            for node in network.nodes():
                if 'age' in network.nodes[node]:
                    network.nodes[node]['age'] += 1

            # Collect nodes older than 70 and record their type and bisexual attribute
            nodes_to_remove = {}
            for node, attributes in network.nodes(data=True):
                if attributes['age'] > 70:
                    nodes_to_remove[node] = {'type': attributes.get('type', 'unknown'), 'bisexual': attributes.get('bisexual', False)}

            # Randomly remove some MSM nodes by age            
            age_removal_counts = {
                remove_age1 :  ,
                remove_age2:   ,
                remove_age3: 
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
            for node in nodes_to_remove:
                network.remove_node(node)

            # Count how many removed nodes are of type=msm bisexual=true, type=msm bisexual=false, female, and male respectively
            removed_types = {}
            for node_info in nodes_to_remove.values():
                node_type = node_info['type']
                if node_type == 'msm':
                    bisexual_status = 'bisexual=true' if node_info['bisexual'] else 'bisexual=false'
                    removed_types[bisexual_status] = removed_types.get(bisexual_status, 0) + 1
                else:
                    removed_types[node_type] = removed_types.get(node_type, 0) + 1

            # Add new nodes
            new_msm_nodes = []
           

            # Add corresponding numbers of new nodes of type=msm bisexual=true, type=msm bisexual=false, female, and male, and set new nodes' age to 18
            for _ in range(removed_types.get('bisexual=true', 0)):
                new_node = max(network.nodes()) + 1
                network.add_node(new_node, type='msm', bisexual=True, age=18, infected=False,ever_infected = False,hpv_vac=False)
                new_msm_nodes.append(new_node)
            
            for _ in range(removed_types.get('bisexual=false', 0)):
                new_node = max(network.nodes()) + 1
                network.add_node(new_node, type='msm', bisexual=False, age=18, infected=False,ever_infected = False,hpv_vac=False)
                new_msm_nodes.append(new_node)
            # Randomly select some of the new MSM nodes to assign age so as to maintain the MSM age distribution
            if len(new_msm_nodes) >=   :
                selected_for_age_20 = random.sample(new_msm_nodes, )
            else:                
                selected_for_age_20 = new_msm_nodes

            for node in selected_for_age_20:
                network.nodes[node]['age'] = 20

            for _ in range(removed_types.get('female', 0)):
                new_node = max(network.nodes()) + 1
                network.add_node(new_node, type='female', age=18, infected=False,ever_infected = False,hpv_vac=False)

            for _ in range(removed_types.get('male', 0)):
                new_node = max(network.nodes()) + 1
                network.add_node(new_node, type='male', age=18, infected=False,ever_infected = False,hpv_vac=False)
            
            # Assign infected status to newly added 18-year-old MSM nodes
            msm_true_nodes = [node for node, data in network.nodes(data=True) if data['type'] == 'msm' and data['age'] == 18]
            random.shuffle(msm_true_nodes)
            
            num_infected_true = int(percent_18_msm *(1-0.856*hpv_vac_set) * len(msm_true_nodes)) # 85.6% is vaccine efficacy; due to vaccination, HPV infection rate among 18-year-old MSM decreases, so adjust accordingly
            for node in msm_true_nodes[:num_infected_true]:
                network.nodes[node]['infected'] = True
                network.nodes[node]['ever_infected'] = True
            percent_18_msm =  percent_18_msm *(1-0.8560*hpv_vac_set)        # Implement GRASP algorithm: connect female nodes with male nodes
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
                    if not network.has_edge(i, j):  
                        all_connected = False
                        break
                
                # If the current male node is connected to all unconnected female nodes, remove this male node
                if all_connected:
                    unconnected_male_nodes.remove(i)
                    continue  # Skip the remaining part of this iteration and continue to the next
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
                
        hpv_vac_female_18 =    # HPV vaccination rate for 18-year-old females
        percent_18_msm =        # HPV infection rate among 18-year-old MSM
        hpv_num = hpv_msm_num 
        for i in range(100):
            
            # Increase age by 1 for all nodes in the network
            for node in network.nodes():
                if 'age' in network.nodes[node]:
                    network.nodes[node]['age'] += 1

            # Collect nodes older than 70 and record their type and bisexual attribute
            nodes_to_remove = {}
            for node, attributes in network.nodes(data=True):
                if attributes['age'] > 70:
                    nodes_to_remove[node] = {'type': attributes.get('type', 'unknown'), 'bisexual': attributes.get('bisexual', False)}

            # Randomly remove some MSM nodes by age            
            age_removal_counts = {
                remove_age1 :  ,
                remove_age2:   ,
                remove_age3: 
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
            for node in nodes_to_remove:
                network.remove_node(node)

            # Count how many removed nodes are of type=msm bisexual=true, type=msm bisexual=false, female, and male respectively
            removed_types = {}
            for node_info in nodes_to_remove.values():
                node_type = node_info['type']
                if node_type == 'msm':
                    bisexual_status = 'bisexual=true' if node_info['bisexual'] else 'bisexual=false'
                    removed_types[bisexual_status] = removed_types.get(bisexual_status, 0) + 1
                else:
                    removed_types[node_type] = removed_types.get(node_type, 0) + 1

            # Add new nodes
            new_msm_nodes = []
           

            # Add corresponding numbers of new nodes of type=msm bisexual=true, type=msm bisexual=false, female, and male, and set new nodes' age to 18
            for _ in range(removed_types.get('bisexual=true', 0)):
                new_node = max(network.nodes()) + 1
                network.add_node(new_node, type='msm', bisexual=True, age=18, infected=False,ever_infected = False,hpv_vac=False)
                new_msm_nodes.append(new_node)
            
            for _ in range(removed_types.get('bisexual=false', 0)):
                new_node = max(network.nodes()) + 1
                network.add_node(new_node, type='msm', bisexual=False, age=18, infected=False,ever_infected = False,hpv_vac=False)
                new_msm_nodes.append(new_node)
            # Randomly select some of the new MSM nodes to assign age so as to maintain the MSM age distribution
            if len(new_msm_nodes) >=   :
                selected_for_age_20 = random.sample(new_msm_nodes, )
            else:                
                selected_for_age_20 = new_msm_nodes

            for node in selected_for_age_20:
                network.nodes[node]['age'] = 20

            for _ in range(removed_types.get('female', 0)):
                new_node = max(network.nodes()) + 1
                network.add_node(new_node, type='female', age=18, infected=False,ever_infected = False,hpv_vac=False)

            for _ in range(removed_types.get('male', 0)):
                new_node = max(network.nodes()) + 1
                network.add_node(new_node, type='male', age=18, infected=False,ever_infected = False,hpv_vac=False)
            
            # Assign infected status to newly added 18-year-old MSM nodes
            msm_true_nodes = [node for node, data in network.nodes(data=True) if data['type'] == 'msm' and data['age'] == 18]
            random.shuffle(msm_true_nodes)
            
            num_infected_true = int(percent_18_msm *(1-0.856*hpv_vac_set) * len(msm_true_nodes)) # 85.6% is vaccine efficacy; due to vaccination, HPV infection rate among 18-year-old MSM decreases, so adjust accordingly
            for node in msm_true_nodes[:num_infected_true]:
                network.nodes[node]['infected'] = True
                network.nodes[node]['ever_infected'] = True
            percent_18_msm =  percent_18_msm *(1-0.8560*hpv_vac_set)
            
            
            # Choose different i values depending on vaccination timing (i = 0, 4, 9, 14)
            if i == :
                # MSM-target HPV vaccination
                msm_nodes_yanqi = [node for node, data in network.nodes(data=True) if data['type'] == 'msm' and data['age'] < 45]
                random.shuffle(msm_nodes_yanqi)                
                selected_msm_10_nodes = int(hpv_vac_set * len( msm_nodes_yanqi))
                for node in msm_nodes_yanqi[:selected_msm_10_nodes]:                    
                    network.nodes[node]['hpv_vac'] = True
                    network.nodes[node]['infected'] = False                    
                # Count how many nodes in selected_msm_10_nodes have ever_infected == True
                ever_infected_msm_nodes=[ node for node, data in network.nodes(data=True) if data['type'] == 'msm' 
                                         and data['ever_infected'] == 'True' and data['hpv_vac'] == 'True']                 
                hpv_msm_fail1 = int(len(ever_infected_msm_nodes) * 0.2296) # Prior infected individuals have 10% reduced vaccine efficacy; vaccine failure rate = 1-85.6%*0.9=0.2296
                for node in ever_infected_msm_nodes[:hpv_msm_fail1]:                    
                    network.nodes[node]['hpv_vac'] = False  
                 # Count how many nodes in selected_msm_10_nodes have ever_infected == False
                ever_non_infected_msm_nodes=[node for node, data in network.nodes(data=True) if data['type'] == 'msm' 
                                         and data['ever_infected'] == 'False' and data['hpv_vac'] == 'True']                 
                hpv_msm_fail2 = int(len(ever_non_infected_msm_nodes) * 0.1440)# Vaccine efficacy 85.6%; vaccine failure rate = 1-85.6%=0.1440
                for node in ever_non_infected_msm_nodes[:hpv_msm_fail2]:                    
                    network.nodes[node]['hpv_vac'] = False  

            # HPV vaccination for newly added 18-year-old MSM, choose different i values depending on timing (i = 0, 4, 9, 14)
            if i > :
                msm_18_year_nodes = [node for node, data in network.nodes(data=True) if data['type'] == 'msm' and data['age'] == 18]
                num_18_year_msm_hpv = int(hpv_vac_set * len( msm_18_year_nodes)* 0.8560)
                for node in msm_18_year_nodes[:num_18_year_msm_hpv]:
                    network.nodes[node]['hpv_vac'] = True
                    network.nodes[node]['infected'] = False
                # Also vaccinate newly added MSM who are not 18 (to maintain MSM age distribution)
                msm_20_year_nodes = selected_for_age_20[:]             
                random.shuffle(msm_20_year_nodes)
                num_20_year_msm_hpv = int(hpv_vac_set * len(msm_20_year_nodes) * 0.8560)
                for node in msm_20_year_nodes[:num_20_year_msm_hpv]:
                    network.nodes[node]['hpv_vac'] = True
                    network.nodes[node]['infected'] = False   
                hpv_num = hpv_num + int(hpv_vac_set * (len( msm_18_year_nodes)+len(msm_20_year_nodes)))
                
            # HPV vaccination for newly added 18-year-old females            
            female_18_year_nodes = [node for node, data in network.nodes(data=True) if data['type'] == 'female' and data['age'] == 18]
            num_18_year_female_hpv = int(hpv_vac_female_18 * len( female_18_year_nodes))
            for node in female_18_year_nodes[:num_18_year_female_hpv]:
                network.nodes[node]['hpv_vac'] = True
                network.nodes[node]['infected'] = False      
            if hpv_vac_female_18 < 0.9:
                hpv_vac_female_18 = hpv_vac_female_18 + 0.175      # 18-year-old female vaccination increases by 17.5% annually, aiming toward WHO target (2023 girls coverage = 90%)
                      
            # Clear all edges in the graph
            network.clear_edges()
            # Assign same_sex_partners counts to all individuals of type=msm
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
                   
            # Assign hetero_partners counts to nodes with type=msm and bisexual=True
            for node in network.nodes(data=True):
                if node[1]['type'] == 'msm' and node[1]['bisexual']:  # Assumes your node attributes contain a 'bisexual' key
                    age = node[1]['age']
                    node[1]['female_partners'] = generate_heterosexual_partners(age)

            # Define male nodes' female_partners probability distribution
            male_partner_probabilities = [  ,   ,   ,   ]

            # Define female nodes' male_partners probability distribution
            female_partner_probabilities = [  ,   ,   ,   ]

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
                        'age': node_data.get('age', MISSING_VALUE),  
                        'same_sex_partners': node_data.get('same_sex_partners', MISSING_VALUE),  
                        'bisexual': node_data.get('bisexual', MISSING_VALUE), 
                        'heterosexual_partners': node_data.get('heterosexual_partners', MISSING_VALUE),  
                        'infected': node_data.get('infected', MISSING_VALUE),
                        'type': node_data.get('type', MISSING_VALUE),
                        'ever_infected': node_data.get('ever_infected', MISSING_VALUE),
                        'hpv_vac': node_data.get('hpv_vac', MISSING_VALUE),}
                    # Append MSM node info to msm_nodes list
                    msm_nodes.append(msm_info)
            
            # Get all female nodes and their attributes
            female_nodes = [
                {'node_id': node, 'age': attr['age'], 'male_partners': attr.get('male_partners', 0),'infected':attr['infected'],
                'type':attr['type'],'ever_infected':attr['ever_infected'],'hpv_vac':attr['hpv_vac']}
                for node, attr in network.nodes(data=True) if attr['type'] == 'female'    ]

            # Get all male nodes and their attributes
            male_nodes = [
                {'node_id': node, 'age': attr['age'], 'female_partners': attr.get('female_partners', 0),'infected':attr['infected'],
                'type':attr['type']}
                for node, attr in network.nodes(data=True)
                if attr['type'] == 'male'  or ( attr['type'] == 'msm' and attr.get('bisexual', False))   ]      
                                      
            
            # Compute the sum of female_partners using a list comprehension
            total_female_partners = sum(male['female_partners'] for male in male_nodes)            
            total_male_partners = sum(female['male_partners'] for female in female_nodes)
 
            network = nx.Graph()
            # Add initial attributes (age, type, connected partners count) for all nodes
            for node in msm_nodes:
                network.add_node(node['node_id'], age=node['age'], same_sex_partners=node['same_sex_partners'], connected_partners=0,
                                bisexual=node['bisexual'],infected=node['infected'],type=node['type'],
                                ever_infected=node['ever_infected'],hpv_vac=node['hpv_vac'])
            for node in female_nodes:
                network.add_node(node['node_id'], age=node['age'], male_partners=node['male_partners'], connected_partners=0,
                                infected=node['infected'],type=node['type'],ever_infected=node['ever_infected'],hpv_vac=node['hpv_vac'])
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
                            if (network.nodes[infected_node]['type'] == 'msm' and 
                                neighbor_attributes['type'] == 'msm' and
                                neighbor_attributes['hpv_vac'] == False):
                                prob = random.choice(infection_probabilities)                 
                            elif (network.nodes[infected_node]['type'] == 'msm' and neighbor_attributes['type'] == 'female'and
                                neighbor_attributes['hpv_vac'] == False):
                                prob = infection_prob_msm_to_female                                
                            elif (network.nodes[infected_node]['type'] == 'female' and 
                                (neighbor_attributes['type'] == 'male' or 
                                (neighbor_attributes['type'] == 'msm' and neighbor_attributes['hpv_vac'] == False))):
                                prob = infection_prob_female_to_msm_or_male                    
                            elif (network.nodes[infected_node]['type'] == 'male' and neighbor_attributes['type'] == 'female'and
                                neighbor_attributes['hpv_vac'] == False):
                                prob = infection_prob_male_to_female
                            else:
                                prob = 0  # Conditions for infection not met
                            
                            # Determine whether infection occurs
                            if random.random() <= prob:
                                network.nodes[neighbor]['infected'] = True
                                network.nodes[neighbor]['ever_infected'] = True
                            
                        infection_time = network.nodes[infected_node].get('infection_time', 0)
                        infection_time += 5  # Time per transmission cycle
                        network.nodes[infected_node]['infection_time'] = infection_time
                    # Find nodes with infection_time equal to 225
                    nodes_with_225_days = [node for node in infected_nodes if network.nodes[node]['infection_time'] == 225]
                    # Calculate nodes that meet the condition and select nodes to recover accordingly
                    if nodes_with_225_days:
                        num_to_recover = int(len(nodes_with_225_days) * clear_rate)  
                        recovered_nodes = random.sample(nodes_with_225_days, num_to_recover)
                        # Return selected nodes to susceptible state
                        for recovered_node in recovered_nodes:
                            network.nodes[recovered_node]['infected'] = False
                            network.nodes[recovered_node]['infection_time'] = 0  # Reset infection time after recovery
            
            if i > : # Choose different i values depending on vaccination timing (i=0, 4, 9, 14) 
                # Final count of infected nodes
                infected_msm_count = sum(1 for node, attributes in network.nodes(data=True) if attributes['infected'] and attributes.get('type') == 'msm')                
                rate1_list.append(infected_msm_count)                
                # Count infected female nodes
                infected_female_count = sum(1 for node, attributes in network.nodes(data=True) if attributes['infected'] and attributes.get('type')=='female')
                rate2_list.append(infected_female_count)               
                # Count infected male nodes
                infected_male_count = sum(1 for node, attributes in network.nodes(data=True) if attributes['infected'] and attributes.get('type')=='male')
                rate3_list.append(infected_male_count)
                # Count MSM aged 18
                age_18_msm_count = sum(1 for node, attributes in network.nodes(data=True) if attributes.get('age')== 18 and attributes.get('type') == 'msm')
                rate4_list.append(age_18_msm_count)
           
        return rate1_list, rate2_list,rate3_list,rate4_list

    

    from multiprocessing import Pool
    import numpy as np
    import concurrent.futures

    def main():
        # Read bisexual MSM and general MSM age distribution data in the main process
        bisexual_msm_data = pd.read_excel('bisexual_age.xlsx')
        general_msm_data = pd.read_excel('MSM_age.xlsx')
        # Read age distribution data
        age_dist_file_path = 'age_density_distribution.csv'
        age_dist_df = pd.read_csv(age_dist_file_path)
        num_simulations = 100
        with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
            # Submit all simulation tasks
            futures = [executor.submit(simulate, (bisexual_msm_data, general_msm_data, age_dist_df)) for _ in range(num_simulations)]
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
            
        # Unpack results from each simulation into four groups
        group1_results, group2_results,group3_results,group4_results = zip(*results)
        
        # Combine results into arrays
        group1_results = np.array(group1_results)
        group2_results = np.array(group2_results)
        group3_results = np.array(group3_results)
        group4_results = np.array(group4_results)
        # Calculate means and standard deviations
        group1_mean = np.mean(group1_results, axis=0)
        group1_mean_rounded = np.round(group1_mean).astype(int)  # Round to nearest integer
        group1_std  = np.std (group1_results, axis=0)

        group2_mean = np.mean(group2_results, axis=0)
        group2_mean_rounded = np.round(group2_mean).astype(int)  # Round to nearest integer
        group2_std  = np.std (group2_results, axis=0)

        group3_mean = np.mean(group3_results, axis=0)
        group3_mean_rounded = np.round(group3_mean).astype(int)  # Round to nearest integer
        group3_std  = np.std (group3_results, axis=0)

        group4_mean = np.mean(group4_results, axis=0)
        group4_mean_rounded = np.round(group4_mean).astype(int)  # Round to nearest integer
        group4_std  = np.std (group4_results, axis=0)

        # Combine means and standard deviations into DataFrames (last column is std)
        df1 = pd.DataFrame({'mean': group1_mean_rounded, 'std': group1_std})
        df2 = pd.DataFrame({'mean': group2_mean_rounded, 'std': group2_std})
        df3 = pd.DataFrame({'mean': group3_mean_rounded, 'std': group3_std})
        df4 = pd.DataFrame({'mean': group4_mean_rounded, 'std': group4_std})

        # Write to Excel file
        percentage_str = f"{int(hpv_vac_set * 100)}%"  # Generate percentage string, e.g., "10%", "20%"
        excel_file_path = f'vac_30_year_results.xlsx'

        with pd.ExcelWriter(excel_file_path) as writer:
            df1.to_excel(writer, sheet_name='msm_year_age', index=False)
            df2.to_excel(writer, sheet_name='female_year_age', index=False)
            df3.to_excel(writer, sheet_name='male_year_age', index=False)
            df4.to_excel(writer, sheet_name='Sheet4', index=False)

        # Return rounded means
        return group1_mean_rounded, group2_mean_rounded, group3_mean_rounded, group4_mean_rounded

    if __name__ == "__main__":
        group1_result, group2_result,group3_result,group4_result = main()

    
        
      

        
from brian2 import *

import neuronDynamics as nd
from scipy.spatial import KDTree
import networkx as nx
import random
import numpy as np

def create_network(tab, n, size=800, axon_len=200, theta=0):
    '''
    
    tab = dataframe do neuronio desejado
    n = numero de neuronios na rede
    size = tamanho da imagem gerada
    axon_len = tamanho do axon artificial que sera gerado para o neurônio
    theta = angulo de rotação do axônio artificial
    standard = utiliza uma seed padrao para que a rede sempre fique igual, apenas mudando o numero de neurônios contidos
    
    retorna o grafo, a imagem da rede gerada e o dicionário com os dados da rede de neuronios
    
    '''
    
    df = nd.create_tab(tab,axon_len, theta)
   
    img, neurons_dict, img_color, colors = nd.create_net(df, n, size)
    
    D = nd.create_D(neurons_dict) #Cria a matriz 2xN com os pixels que possuem conexao 
    tree = KDTree(D) #Criacao da KD tree para obter os pares proximos com query_pairs()
    pairs = tree.query_pairs(r=15, p=2)

    edges = nd.create_edges(D, neurons_dict, pairs) #Cria as arestas do grafo

    
    network = nx.MultiDiGraph(edges)    
    
    return network, img_color, neurons_dict

def create_network_from_swc(neuron_swc, n, size=800, axon_len=200, theta=0):
    '''
    
    neuron_swc = Arquivo swc do neuronio que deseja criar a rede
    n = numero de neuronios na rede
    size = tamanho da imagem gerada
    axon_len = tamanho do axon artificial que sera gerado para o neurônio
    theta = angulo de rotação do axônio artificial
    standard = utiliza uma seed padrao para que a rede sempre fique igual, apenas mudando o numero de neurônios contidos
    
    retorna o grafo, a imagem da rede gerada e o dicionário com os dados da rede de neuronios
    
    '''
    
    tab = nd.read_swc(neuron_swc)
    
    df = nd.create_tab(tab,axon_len, theta)
   
    img, neurons_dict, img_color, colors = nd.create_net(df, n, size)
    
    D = nd.create_D(neurons_dict) #Cria a matriz 2xN com os pixels que possuem conexao 
    tree = KDTree(D) #Criacao da KD tree para obter os pares proximos com query_pairs()
    pairs = tree.query_pairs(r=15, p=2)

    edges = nd.create_edges(D, neurons_dict, pairs) #Cria as arestas do grafo

    
    network = nx.MultiDiGraph(edges)    
    
    return network, img_color, neurons_dict


def remove_neurons(region, neurons_dict):
    '''
    region: tupla com (x1, x2), sendo x1 o inicio da região a ser removida e x2 o final
    neurons_dict: dicionário com os dados das conectividades da rede
    
    retorna o grafo da rede com os neurônios removidos
    '''
    D = nd.create_D(neurons_dict) #Cria a matriz 2xN com os pixels que possuem conexao 
    tree = KDTree(D) #Criacao da KD tree para obter os pares proximos com query_pairs()
    pairs = tree.query_pairs(r=20, p=2)

    edges = nd.create_edges(D, neurons_dict, pairs) #Cria as arestas do grafo



    edges_raw = sorted(set(edges))
    
    neurons_to_rm = []
    
    
    # Encontrando quais neuronios devem ser removidos
    for k,v in neurons_dict.items():
        if k[1] >= region[0] and k[1] <= region[1]:
            for i in v:
                neurons_to_rm.append(i[0])
            
    
    # Remove duplicatas
    neurons_to_rm = set(neurons_to_rm)
    
    # Encontrando o indice da tupla que contém o neuronio a ser removido da lista de arestas
    to_remove = []
    for i in range(len(edges_raw)):
        for indice in edges_raw[i]:
            if indice in neurons_to_rm:
                to_remove.append(edges_raw[i])
               
    # Eliminando duplicatas
    to_remove = list(set(to_remove))


    edges_removed = edges_raw.copy() # Salvando a lista de arestas originais antes de fazer a remocao
    # Removendo os neuronios
    for tupla in to_remove:
        edges_removed.remove(tupla)
    
        
    network = nx.MultiDiGraph(edges_removed)
    
    print(edges_removed)
    
    return network

def symulate_dynamics(network, input_neurons, input_voltage, dyn_params):
    '''
    network: Grafo com a rede de neurônios
    input_neurons: Vetor com os neuronios que são carregados durante a siulação
    input_voltage: Voltagem de carregamento dos neurônios selecionados
    dyn_params: Dicionário contendo os dados necessários para a simulação ( {'tau':none, 'threshold':none, 'reset_voltage':none, 'resting_voltage':none} )
    
    retorna a evolução das voltagens em cada neurônio ao final da simulação
    '''
    n = network.number_of_nodes()
    
    start_scope()
    El = dyn_params['resting_voltage']
    thau = dyn_params['tau']
    threshold = dyn_params['threshold']
    reset_voltage = dyn_params['reset_voltage']
    
    eqs = '''
    dv/dt = ((El - v) + V)/tau : volt
    V : volt
    tau : second
    '''
    
    
    G = NeuronGroup(n, eqs, threshold=f'v > {threshold}*volt', reset=f'v = {reset_voltage}*volt', method='exact')

    G.V = ([0]*n)*volt
    for neuron in input_neurons:
        G.V[neuron] = input_voltage*volt
        
    G.v = ([El]*n)
    
    G.tau = ([100*n])*ms
    for neuron in input_neurons:
        G.tau[neuron] = 10*ms
        
    S = Synapses(G, G, on_pre=f'v_post += {input_voltage/10}*volt')
    
    edges = network.edges
    
    edges_raw = sorted(set(edges))
    edges_raw_form = list(zip(*edges_raw))

    vector_i_raw = edges_raw_form[0]
    vector_j_raw = edges_raw_form[1]
    
    S.connect(i=vector_i_raw, j=vector_j_raw)
    
    M = StateMonitor(G, 'v', record=True)

    run(25*ms) # Roda a simulacao
    
    return M
    
    
    
    
    
    
    
    
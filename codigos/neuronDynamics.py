import pandas as pd
import numpy as np
import random
import time
from skimage.draw import line
from collections import defaultdict
#oie
'''
read_swc()

Funcao para ler o arquivo SWC (dado um arquivo de um neuronio em formato SWC, um dataframe é retornado com os dados formatados)

arquivo = nome do arquivo SWC
'''
def read_swc(arquivo):
    
    df = pd.read_csv(arquivo, header=None)
    colunas = ['knot','tipo','x', 'y','z', 'raio','pai']
    remove = []
    for i, row in df.iterrows():
        linha = row[0].split()
        if linha[0] == '#' or linha[0] == '#The':
            remove.append(i)
        if linha[0] == 1:
            break
            
    df = pd.read_csv(arquivo, skiprows=remove, header=None, sep=' ')
    df.drop([0],axis=1, inplace=True)
    df.columns = colunas
    df.loc[0, 'pai'] = 1
    
    return df

'''
muda_res()

Alteracao da resolucao da imagem do neurônio

df = dataframe com os dados dos segmentos do neuronio
k = constante de aumento/diminuicao da imagem
'''
def muda_res(df, k):

    
    df['x'] = df['x']/k
    df['y'] = df['y']/k
    df = df.astype({"x": int, "y": int})
    
    return df

'''
draw_axon()

Remove o axonio original e adiciona as coordenadas de um axonio artificial no daframe

df=dataframe, 
L=comprimento do axonio, 
theta=angulo de direcao do axonio a partir do soma
'''
def draw_axon(df, L, theta):
    from skimage.draw import line

    #Salvando os indices originais para coloca-los apos a remocao
    df['index'] = df.index
   
    #removendo o axonio para posteriormente desenhar um artificial
    axons_index = df[df["tipo"]==2].index
    df = df.drop(axons_index)
    
    df.set_index('index')
        
    #graus para radianos
    theta = np.radians(theta)
    
    #Obtencao das posicoes da imagem novamente apos o deslocamento
    xc, yc = df.loc[0, ['x', 'y']].values # centro
    x = int(L*np.cos(theta) + xc) # x final
    y = int(L*np.sin(theta) + yc) # y final
    
    #Obtem as coordenadas referentes ao axonio artificial
    rr, cc = line(yc, xc, y, x)
    
    #Adiciona o axonio artifical ao final do dataframe
    flag = True
    for row, col in zip(rr, cc):
        if flag:
            df.loc[df.index[-1] + 1] = [0, col, row, 0, df.index[-1] + 1]
            flag = False
        else:    
            df.loc[df.index[-1] + 1] = [0, col, row, df.index[-1], df.index[-1] + 1]
        
    return df

'''
create_tab()

Criacao do dataFrame formatado para posteriormente criar a imagem

df= dataframe que foi criado ao ler o arquivo SWC
L = Comprimento do axonio artificial,
theta = angulacao do axonio artificial
'''
def create_tab(df, L=50, theta=0): 
    
    
    df = df[['tipo', 'x', 'y', 'pai']]
    df = df.astype({"tipo": np.uint8, "x": int, "y": int, "pai": int})
    df['pai'] = df['pai'] - 1
    
    #Desenho do axonio artificial
    df = draw_axon(df, L, theta)

    #shift da imagem (menor valor ser o '0')
    shift_x = min(df['x'])
    shift_y = min(df['y'])
    df['x'] = df['x'] - shift_x
    df['y'] = df['y'] - shift_y

    
    #Rotacao de 90 graus
    df['y'] = max(df['y']) - df['y']
    
    #Definicao dos tons de cinza de cada pixel de acordo com seu tipo (soma, axon, dendrite)
    
    df.loc[df.tipo == 1,'tipo']= 1
    df.loc[df.tipo == 2,'tipo']= 0
    df.loc[df.tipo == 3,'tipo']= 1
    df.loc[df.tipo == 4,'tipo']= 1
    df.loc[df.tipo == 5,'tipo']= 1
    df.loc[df.tipo == 6,'tipo']= 1
    df.loc[df.tipo == 7,'tipo']= 1

    
    return df

'''
create_img()

Criacao da imagem recebendo o dataFrame e a resolucao escolhida

df = dataframe
k = constante de aumento/diminuicao da imagem

'''
def create_img(df, k=1):
    
    #definicao da resolucao (k==1 default)
    if (k != 1):
        df['x'] = df['x']/k
        df['y'] = df['y']/k
        df = df.astype({"x": int, "y": int})
    
    #criacao da matriz de zeros
    imagem = np.zeros([max(df['y']) + 1,max(df['x']) + 1], dtype=np.uint8)
    
    #indexacao dos pixels
    for index, row in df.iterrows():
        imagem[row['y'], row['x']] = row['tipo']
    
    # Preenche os buracos deixados na imagem devido a normalizacao
    imagem = draw_lines(df, imagem)
    
    #imagem = imagem - 1
    
    return imagem

'''
draw_lines()

Completa os 'buracos' que ficam na imagem devido a normalizacao dos valores

df = dataframe
img = imagem
'''
def draw_lines(df, img):
    
    
    # Itera pelo dataframe e liga o 'filho' com o 'pai' usando a funcao line do pacote skimage.draw 
    for index, row in df.iterrows():
        x1, y1 = row['x'], row['y']
        
        pai = row['pai']
        linha_pai = df.loc[pai]
        x2, y2 = int(linha_pai['x']), int(linha_pai['y'])
        
        rr, cc = line(y2, x2, y1, x1)
        img[rr, cc] = 1
    
    return img

'''
rot_df()

Faz a rotacao do neuronio em um angulo aleatorio

df = dataframe
'''
def rot_df(df):
    
    #Obtem um valor aleatorio para o angulo de rotacao
    theta = random.uniform(0, 2*np.pi) #angulo aleatorio de rotacao
    
    #matriz de rotacao
    M = np.array([[np.cos(theta), np.sin(theta)],[-(np.sin(theta)), np.cos(theta)]]) 
    
    df_coord = np.array(df[['x', 'y']])
    
    rot_r = M.dot(df_coord.transpose()).transpose().astype('int32')
    
    rot_r -= np.min(rot_r, axis=0)
    
    df['x'] = rot_r[:, 0]
    df['y'] = rot_r[:, 1]
        
    return df

'''
create_net()

Cria uma rede com neuronios iguais, retornando uma imagem da rede e um dict com as coordenadas que possuem conexao

df = dataframe
n = numero de neuronios na rede
tam_img = tamanho da imagem da rede
'''
def create_net(df, n, tam_img):
    
    neurons_dict = defaultdict(list)
    net_img = np.zeros([tam_img,tam_img], dtype=np.uint8)

    for i in range(n):
        df = rot_df(df)
        img = create_img(df)
        
        #Imagem auxiliar com o tipo do pixel (0 = fundo, 1 = dendrito, 2 = axonio)
        img_aux = img.copy()
        for index, row in df.iterrows():
            if row['tipo'] == 2:
                img_aux[row['y'], row['x']] = int(row['tipo'])
        
        #Delocamento em x
        rand_x = random.randint(0, tam_img-len(img[0])) 
        #Delocamento em y
        rand_y = random.randint(0,tam_img-len(img))
        
        coords = np.nonzero(img)
        coords_x = coords[1] + rand_x
        coords_y = coords[0] + rand_y
        net_img[coords_y, coords_x] = net_img[coords_y, coords_x] + img[coords[0], coords[1]]
        
        for coord_x, coord_y in zip(coords_x, coords_y):
            neurons_dict[(coord_y, coord_x)].append(i)
                
    return net_img, neurons_dict


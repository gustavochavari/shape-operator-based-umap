'''
    K-UMAP: um algoritmo iterativo para aprendizado não supervisionado
    de métricas baseado na curvatura local

    Protótipo em Python do método a ser desenvolvido

    Modificado por Gustavo H. Chavari
'''

import os
import json
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csr_matrix
import sklearn.neighbors as sknn
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
from matplotlib import colors
from scipy.optimize import curve_fit
from numpy.linalg import inv
from numpy.linalg import cond
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding as LLE  
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from sklearn import preprocessing
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from tqdm import tqdm


# Para evitar erro de SSL do fetch_openml
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Global configuration variables
MIN_DIST = 0.1
N_LOW_DIMS = 2
LEARNING_RATE = 1
MAX_ITER = 120

#####################################################
# FUNÇÕES AUXILIARES
#####################################################

def prob_high_dim(dist, rho, sigma, dist_row):
    """
     Para cada linha da matriz de distâncias computa as probabilidades no espaço de alta dimensão
     (1D array)
    """
    d = dist[dist_row] - rho[dist_row]
    d[d < 0] = 0
    return np.exp(- d / sigma)


def k(prob):
    """
    Computa n_neighbor = k (escalar) para cada array de probabilidades de alta dimensionalidade
    """
    return np.power(2, np.sum(prob))


def sigma_binary_search(k_of_sigma, fixed_k):
    """
    Resolve a equação perp_of_sigma(sigma) = fixed_perplexity 
    com relação a sigma com uma busca binária
    """
    sigma_lower_limit = 0
    sigma_upper_limit = 1000
    for i in range(20):
        approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
        if k_of_sigma(approx_sigma) < fixed_k:
            sigma_lower_limit = approx_sigma
        else:
            sigma_upper_limit = approx_sigma
        if np.abs(fixed_k - k_of_sigma(approx_sigma)) <= 1e-5:
            break
    return approx_sigma


def f(x, min_dist):
    """
    Função auxiliar para calcular hiperparâmetros
    """
    y = []
    for i in range(len(x)):
        if x[i] <= min_dist:
            y.append(1)
        else:
            y.append(np.exp(- x[i] + min_dist))
    return y


def prob_low_dim(Y, a, b, distance='euclidean'):
    """
    Computa a matriz de probabilidades q_ij no espaço de baixa dimensão
    """
    if distance == 'mahalanobis':
        m = Y.shape[1]
        sigma = np.cov(Y.T)
        # Se necessário, regulariza matriz de convariâncias
        if cond(sigma) > 1/sys.float_info.epsilon:
            sigma += np.diag(0.0001*np.ones(m))
        inv_sigma = inv(sigma)
        distances = cdist(Y, Y, 'mahalanobis', VI=sigma)
    else:
        distances = euclidean_distances(Y, Y)
    # Calcula inverso das distâncias
    inv_distances = np.power(1 + a * np.square(distances)**b, -1)
    return inv_distances

def CE(P, Y, a, b, distance='euclidean'):
    """
    Computa a Entropia Cruzada (CE) entre a matriz de probabilidades no espaço
    de alta dimensionalidade e as e as coordenadas do espaço de baixa dimensão
    """
    Q = prob_low_dim(Y, a, b, distance)
    return - P * np.log(Q + 0.01) - (1 - P) * np.log(1 - Q + 0.01)

def CE_gradient(P, Y, a, b, distance='euclidean'):
    """
    Computa o  gradiente da Entropia Cruzada (CE)
    """
    y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
    if distance == 'mahalanobis':
        m = Y.shape[1]
        sigma = np.cov(Y.T)
        # Se necessário, regulariza matriz de convariâncias
        if cond(sigma) > 1/sys.float_info.epsilon:
            sigma += np.diag(0.0001*np.ones(m))
        inv_sigma = inv(sigma)
        distances = cdist(Y, Y, 'mahalanobis', VI=sigma)
    else:
        distances = euclidean_distances(Y, Y)
    inv_dist = np.power(1 + a * np.square(distances)**b, -1)
    Q = np.dot(1 - P, np.power(0.001 + np.square(distances), -1))
    np.fill_diagonal(Q, 0)
    Q = Q / np.sum(Q, axis = 1, keepdims = True)
    fact = np.expand_dims(a * P * (1e-8 + np.square(distances))**(b-1) - Q, 2)
    gradient = 2 * b * np.sum(fact * y_diff * np.expand_dims(inv_dist, 2), axis = 1)
    return gradient

def find_ab_params(spread=1.0, min_dist=0.1):
    """Fit a, b params for the UMAP low dimensional curve"""
    # Esta é a função que o UMAP tenta aproximar
    func = lambda x, a, b: 1.0 / (1.0 + a * x ** (2 * b))

    # Cria uma curva de exemplo para o fit
    x = np.linspace(0, spread * 3, 300)
    y = np.zeros_like(x)
    y[x < min_dist] = 1.0
    y[x >= min_dist] = np.exp(-(x[x >= min_dist] - min_dist) / spread)
    
    # Realiza o fit
    (a, b), _ = curve_fit(func, x, y)
    return a, b


def umap(dados,target,N_NEIGHBOR):
    """
    Implementa o algoritmo UMAP padrão
    """
    print('************************************')
    print('UMAP com distância Euclidiana')
    print('************************************\n')
    n = dados.shape[0]
    m = dados.shape[1]
    c = len(np.unique(target))  
    # Calcula as distâncias euclidianas ponto a ponto
    dist = np.square(euclidean_distances(dados, dados))
    rho = [sorted(dist[i])[1] for i in range(dist.shape[0])]
    # Computa a matriz no espaço de alta dimensionalidade   
    prob = np.zeros((n,n))
    sigma_array = []
    for dist_row in range(n):
        func = lambda sigma: k(prob_high_dim(dist, rho, sigma, dist_row))
        binary_search_result = sigma_binary_search(func, N_NEIGHBOR)
        prob[dist_row] = prob_high_dim(dist, rho, binary_search_result, dist_row)
        sigma_array.append(binary_search_result)
        if (dist_row + 1) % 500 == 0:
            print("Busca binária do Sigma terminada em {0} de {1} amostras".format(dist_row + 1, n))
    print("\nSigma = " + str(np.mean(sigma_array)))
    print()
    # Duas formas de calcular a matriz de probabilidades
    #P = prob + np.transpose(prob) - np.multiply(prob, np.transpose(prob))
    P = (prob + np.transpose(prob)) / 2        # Igual ao t-SNE (melhor)    
    # Hiperparâmetros
    a = 1.93
    b = 0.79
    print("Hiperparâmetros a = " + str(a) + " and b = " + str(b))    
    # Semente para os números aleatórios
    np.random.seed(12345)
    # Laplacian Eigenmaps para inicialização das coordenadas dos pontos no espaço de baixa dimensão
    model = SpectralEmbedding(n_components=N_LOW_DIMS, n_neighbors=N_NEIGHBOR)
    #model = LLE(n_components=N_LOW_DIMS, n_neighbors=N_NEIGHBOR)
    #model = Isomap(n_components=N_LOW_DIMS, n_neighbors=N_NEIGHBOR)
    y = model.fit_transform(dados)
    # Inicia a minimização da função de perda
    CE_array = []
    print("\nExecutando a descida do gradiente: \n")
    for i in range(MAX_ITER):
        y = y - LEARNING_RATE * CE_gradient(P, y, a, b)
        CE_current = np.sum(CE(P, y, a, b)) / 1e+5  # Constante apenas para normalizar os valores
        CE_array.append(CE_current)
        if i % 10 == 0:
            print("Cross-Entropy = " + str(CE_current) + " depois de " + str(i) + " iterações")
    print()
    return (CE_array, y)


def curvature_estimation(dados, k,method="gaussian"):
    """
    Computes the curvatures of all samples in the training set
    """
    n = dados.shape[0]
    m = dados.shape[1]
    # Se tiver mais de 80 atributos, aplica PCA antes de calcular curvaturas
    if m > 100:
        m = 100
        model = PCA(n_components=m)
        dados = model.fit_transform(dados)    

    # Primeira forma fundamental
    I = np.zeros((m, m))
    Squared = np.zeros((m, m))
    ncol = (m*(m-1))//2
    Cross = np.zeros((m, ncol))

    # Second fundamental form
    II = np.zeros((m, m))
    S = np.zeros((m, m))
    curvatures = np.zeros(n)

    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='connectivity', include_self=False)
    A = knnGraph.toarray()


    print("Computing shape operator for each point...")
    for i in tqdm(range(n), desc="Computing curvatures"):
        ######## Patch P_i
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        ##########################################
        ######## Estima primeira forma fundamental
        ######## Estudar o efeito da centralização, se muda algo
        amostras = dados[indices] - np.mean(dados[indices], axis=0)
        ni = len(indices)
        if ni > 1:
            # Primeira forma fundamental em i
            I = np.cov(amostras.T)
            #I = I + 0.0001*np.eye(I.shape[0])   # Regulariza
        else:
            I = np.eye(m)      # pontos isolados = identidade

        v, w = np.linalg.eig(I)
        ordem = v.argsort()
    
        Wpca = w[:, ordem[::-1]]
        # Estima segunda forma fundamental
        for j in range(0, m):
            Squared[:, j] = Wpca[:, j]**2
        col = 0
        for j in range(0, m):
            for l in range(j, m):
                if j != l:
                    Cross[:, col] = Wpca[:, j]*Wpca[:, l]
                    col += 1
        # Adiciona coluna de 1's, squared e cross
        Wpca = np.column_stack((np.ones(m), Wpca))
        Wpca = np.hstack((Wpca, Squared))
        Wpca = np.hstack((Wpca, Cross))

        Q = Wpca
        # Discard the first m columns of H
        H = Q[:, (m+1):]

        # Segunda forma fundamental
        II = np.dot(H, H.T)
        S = -np.dot(II, I)
        
        if method == "gaussian":
            curvatures[i] = abs(np.linalg.det(S+np.eye(m)/1e-3))        # curvatura Gaussiana
        else:   
           curvatures[i] = np.trace(S)         
    return curvatures


def calculate_entropy(data, n_bins):
    if n_bins >= len(data):
        return 0

    bins = np.linspace(np.min(data), np.max(data), n_bins + 1)
    counts, _ = np.histogram(data, bins=bins)

    probs = counts / np.sum(counts)
    probs = probs[probs > 0]  

    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

# Não é mais utilizado
def normalize_curvatures(curv):
    """
    Normalize curvature values to [0, 1]
    """
    if curv.min() != curv.max():
        return (curv - curv.min()) / (curv.max() - curv.min())
    else:
        return curv / len(curv)

def percentile_rank(K, n_neighbors,dataset_name,plot=True):
    """
    Assign ranks to K based on adaptive binning guided by entropy.
    """
    entropies = []
    bin_range = range(3, n_neighbors - 1)

    for n_bins in bin_range:
        entropy = calculate_entropy(K, n_bins)
        entropies.append(entropy)

    if not entropies:
        n_bins = 5
    else:
        entropies = np.array(entropies)
        if len(entropies) > 2:
            # Encontra o cotovelo
            second_deriv = np.diff(entropies, n=2)
            elbow_idx = np.argmax(second_deriv) + 1 if len(second_deriv) > 0 else 0
            n_bins = list(bin_range)[elbow_idx]
        else:
            n_bins = list(bin_range)[np.argmax(entropies)]

    bins = np.linspace(np.min(K), np.max(K), n_bins)
    ranks = np.digitize(K, bins, right=False)

    if plot:
        plt.figure(figsize=(12, 5))
        
        # Entropia vs n_bins
        plt.subplot(1, 2, 1)
        plt.plot(bin_range, entropies, 'bo-')
        plt.axvline(x=n_bins, color='r', linestyle='--', label=f'Selected n_bins = {n_bins}')
        plt.xlabel('Number of bins')
        plt.ylabel('Entropy')
        plt.title('Entropy vs Number of Bins')
        plt.legend()
        
        # Histograma
        plt.subplot(1, 2, 2)
        unique_ranks = np.unique(ranks)
        plt.hist(ranks, bins=len(unique_ranks), edgecolor='black', align='left')
        plt.xticks(unique_ranks)
        plt.xlabel('Rank')
        plt.ylabel('Count')
        plt.title('Distribution of Ranks')
        
        plt.tight_layout()
        plt.savefig('Rank-Plot_'+dataset_name+'.png')
        plt.close()

    return ranks

def k_umap(dados, target, N_NEIGHBOR,dataset_name):
    """
    Implementa o algoritmo UMAP com curvatura local
    """
    print('*************************************************')
    print('SH-UMAP com curvatura local')
    print('*************************************************\n')
    n = dados.shape[0]
    m = dados.shape[1]
    c = len(np.unique(target))
    MIN_NEIGH = 3

    print("Calculando curvaturas locais")
    curvatures = curvature_estimation(dados, N_NEIGHBOR,"gaussian")

    K = percentile_rank(curvatures,N_NEIGHBOR,dataset_name)

    unique_values, counts = np.unique(K, return_counts=True)

    print("Ranking de curvatura (Rank: Quantidade)")
    for value, count in zip(unique_values, counts):
        print(f"{value}: {count}")

    dist = np.square(euclidean_distances(dados,dados))
    rho = [sorted(dist[i])[1] for i in range(dist.shape[0])]
    # Computa a matriz no espaço de alta dimensionalidade   
    prob = np.zeros((n,n))
    sigma_array = []
    for dist_row in range(n):
        func = lambda sigma: k(prob_high_dim(dist, rho, sigma, dist_row))
        binary_search_result = sigma_binary_search(func, (N_NEIGHBOR - K[dist_row]))
        prob[dist_row] = prob_high_dim(dist, rho, binary_search_result, dist_row)
        sigma_array.append(binary_search_result)
        if (dist_row + 1) % 500 == 0:
            print("Busca binária do Sigma terminada em {0} de {1} amostras".format(dist_row + 1, n))
    print("\nSigma = " + str(np.mean(sigma_array)))
    print()
    # Duas formas de calcular a matriz de probabilidades
    #P = prob + np.transpose(prob) - np.multiply(prob, np.transpose(prob))
    P = (prob + np.transpose(prob)) / 2        # Igual ao t-SNE (em alguns casos parece melhor)    
    # Hiperparâmetros (sugeridos pelos autores do UMAP)
    #a = 1.93
    #b = 0.79
    a, b = find_ab_params(1,MIN_DIST)

    print("Hiperparâmetros a = " + str(a) + " and b = " + str(b))    
    # Semente para os números aleatórios
    np.random.seed(12345)
    # Laplacian Eigenmaps para inicialização das coordenadas dos pontos no espaço de baixa dimensão
    model = SpectralEmbedding(n_components=N_LOW_DIMS, n_neighbors=N_NEIGHBOR)    
    y = model.fit_transform(P)
    # Inicia a minimização da função de perda
    CE_array = []
    print("\nExecutando a descida do gradiente: \n")
    for i in range(MAX_ITER):
        # Substitui a taxa de aprendizado pela curvatura (pontos de alta curvatura, maior taxa de aprendizado)
        y = y - LEARNING_RATE * CE_gradient(P, y, a, b)        
        CE_current = np.sum(CE(P, y, a, b)) / 1e+5  # Constante apenas para normalizar os valores
        CE_array.append(CE_current)
        if i % 10 == 0:
            print("Cross-Entropy = " + str(CE_current) + " depois de " + str(i) + " iterações")
    print()
    return (CE_array, y)


def PlotaDados(dados, labels, metodo, dataset_name):
    """
    Função para plotagem dos dados de saída
    """
    # Número de classes
    nclass = len(np.unique(labels))
    
    if nclass > 11:
        cores = []
        for c, hex in colors.cnames.items():
            cores.append(c)
        #cores = ['black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred']
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'green', 'black', 'orange', 'magenta', 'darkkhaki', 'brown', 'purple', 'cyan', 'salmon']
    plt.figure(1)
    #plt.figure(figsize=(10,8))
    for i in range(nclass):
        indices = np.where(labels==i)[0]
        cor = cores[i]
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, alpha=0.5, marker='.') 
    nome_arquivo = metodo+'_'+dataset_name + '.png'
    plt.title(metodo+' clusters')
    plt.savefig(nome_arquivo)
    plt.close()


def EvaluationMetrics(dados, target, method, N_NEIGHBOR):
    """
    Computes performance evaluation metrics.
    Uses GMM for clustering and provides KNN classification scores
    with 10-fold cross-validation, including standard deviation.
    """
    print(f'\nQuantitative metrics for {method} features\n')

    if isinstance(dados, pd.Series):
        X = dados.to_frame()
    elif isinstance(dados, pd.DataFrame):
        X = dados
    else: 
        X = pd.DataFrame(dados.real)

    y = target

    c = len(np.unique(y))
    
    gmm = GaussianMixture(n_components=c, random_state=42, n_init=10)
    gmm_labels = gmm.fit_predict(X)

    sc = metrics.silhouette_score(X, y, metric='euclidean')
    ch = metrics.calinski_harabasz_score(X, y)
    db = metrics.davies_bouldin_score(X, y)
    
    ri = metrics.rand_score(gmm_labels, y)
    fm = metrics.fowlkes_mallows_score(gmm_labels, y)
    mi = metrics.mutual_info_score(gmm_labels, y)
    vm = metrics.v_measure_score(gmm_labels, y)

    print(f'Silhouette coefficient: {sc:.4f}')
    print(f'Calinski Harabasz: {ch:.4f}')
    print(f'Davies Bouldin: {db:.4f}')
    print(f'Rand index (GMM): {ri:.4f}')
    print(f'Fowlkes Mallows score (GMM): {fm:.4f}')
    print(f'Mutual info score (GMM): {mi:.4f}')
    print(f'V-measure score (GMM): {vm:.4f}')
    print('-----------------------------------\n')

    clustering_results = {
        'Silhouette': float(sc),
        'Calinski_Harabasz': float(ch),
        'Davies_Bouldin': float(db),
        'Rand_index': float(ri),
        'Fowlkes_Mallows': float(fm),
        'Mutual_info': float(mi),
        'V_measure': float(vm)
    }

    print(f'Evaluating KNN with 10-fold Cross-Validation...')
    clf = KNeighborsClassifier(n_neighbors=N_NEIGHBOR)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    balanced_accuracy_scores = []
    kappa_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    jaccard_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index], \
                          y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            balanced_accuracy_scores.append(metrics.balanced_accuracy_score(y_test, y_pred))
            kappa_scores.append(metrics.cohen_kappa_score(y_test, y_pred))
            precision_scores.append(metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0))
            recall_scores.append(metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(metrics.f1_score(y_test, y_pred, average='weighted', zero_division=0))
            jaccard_scores.append(metrics.jaccard_score(y_test, y_pred, average='weighted', zero_division=0))
    
    knn_classification_results = {
        'Balanced_Accuracy_mean': float(np.mean(balanced_accuracy_scores)),
        'Balanced_Accuracy_std': float(np.std(balanced_accuracy_scores)),
        'Kappa_mean': float(np.mean(kappa_scores)),
        'Kappa_std': float(np.std(kappa_scores)),
        'Precision_weighted_mean': float(np.mean(precision_scores)),
        'Precision_weighted_std': float(np.std(precision_scores)),
        'Recall_weighted_mean': float(np.mean(recall_scores)),
        'Recall_weighted_std': float(np.std(recall_scores)),
        'F1_weighted_mean': float(np.mean(f1_scores)),
        'F1_weighted_std': float(np.std(f1_scores)),
        'Jaccard_weighted_mean': float(np.mean(jaccard_scores)),
        'Jaccard_weighted_std': float(np.std(jaccard_scores))
    }
    
    print(f"  Avg. Balanced Accuracy: {knn_classification_results['Balanced_Accuracy_mean']:.4f} (+/- {knn_classification_results['Balanced_Accuracy_std']:.4f})")
    print(f"  Avg. Kappa: {knn_classification_results['Kappa_mean']:.4f} (+/- {knn_classification_results['Kappa_std']:.4f})")
    print(f"  Avg. Precision: {knn_classification_results['Precision_weighted_mean']:.4f} (+/- {knn_classification_results['Precision_weighted_std']:.4f})")
    print(f"  Avg. Recall: {knn_classification_results['Recall_weighted_mean']:.4f} (+/- {knn_classification_results['Recall_weighted_std']:.4f})")
    print(f"  Avg. F1: {knn_classification_results['F1_weighted_mean']:.4f} (+/- {knn_classification_results['F1_weighted_std']:.4f})")
    print(f"  Avg. Jaccard: {knn_classification_results['Jaccard_weighted_mean']:.4f} (+/- {knn_classification_results['Jaccard_weighted_std']:.4f})")
    print()

    merged_results = clustering_results | knn_classification_results

    return merged_results

        
def main():
    # To avoid unnecessary warning messages
    warnings.simplefilter(action='ignore')

    # Leitura dos dados
    #X = skdata.load_iris()     # 15 - all
    #X = skdata.fetch_openml(name='penguins', version=1)         # 15 - all
    X = skdata.fetch_openml(name='mfeat-karhunen', version=1)   # 15 - all
    #X = skdata.fetch_openml(name='Olivetti_Faces', version=1)   # 15 - all
    #X = skdata.fetch_openml(name='AP_Breast_Colon', version=1)  # 15 - all
    #X = skdata.fetch_openml(name='page-blocks', version=1)      # 15 - all
    #X = skdata.fetch_openml(name='Kuzushiji-MNIST', version=1)  # 15 - all
    #X = skdata.fetch_openml(name='optdigits', version=1)        # 15 - all
    #X = skdata.fetch_openml(name='synthetic_control', version=1)   # 15 - all
    #X = skdata.fetch_openml(name='har', version=1)                 # 15 - all
    #X = skdata.fetch_openml(name='schizo', version=1)              # 15 - all
    #X = skdata.fetch_openml(name='11_Tumors', version=1)           # 15 - all
    #X = skdata.fetch_openml(name='one-hundred-plants-margin', version=1)   # 15 - all

    #X = skdata.load_digits()    # sqrt - all
    #X = skdata.fetch_openml(name='mfeat-fourier', version=1)    # sqrt - all
    #X = skdata.fetch_openml(name='mfeat-factors', version=1)    # sqrt - all
    #X = skdata.fetch_openml(name='semeion', version=1)          # sqrt - all
    #X = skdata.fetch_openml(name='micro-mass', version=1)       # sqrt - all
    #X = skdata.fetch_openml(name='MNIST_784', version=1)        # sqrt - all
    #X = skdata.fetch_openml(name='pendigits', version=1)        # sqrt - all
    #X = skdata.fetch_openml(name='satimage', version=1)         # sqrt - all
    #X = skdata.fetch_openml(name='dilbert', version=1)          # sqrt - all - 20%
    #X = skdata.fetch_openml(name='gina_agnostic', version=1)    # sqrt - all
    #X = skdata.fetch_openml(name='vowel', version=1)            # sqrt - all

    #X = skdata.fetch_openml(name='cars', version=1)     # sqrt - classif
    #X = skdata.fetch_openml(name='UMIST_Faces_Cropped', version=1)    # sqrt - classif
    #X = skdata.fetch_openml(name='gas-drift', version=1)   # sqrt - classif
    #X = skdata.fetch_openml(name='gisette', version=1)     # sqrt - classif
    #X = skdata.fetch_openml(name='OVA_Breast', version=1)      # sqrt - classif
    #X = skdata.fetch_openml(name='ionosphere', version=1)      # sqrt - classif
    #X = skdata.fetch_openml(name='LED-display-domain-7digit', version=1)    # sqrt - classif
    #X = skdata.fetch_openml(name='yeast', version=1)       # sqrt - classif
    #X = skdata.fetch_openml(name='Diabetes130US', version=1)   # sqrt - classif

    #X = skdata.fetch_openml(name='USPS', version=2)    # 15 - classif
    #X = skdata.fetch_openml(name='texture', version=1)     # 15 - classif
    #X = skdata.fetch_openml(name='Fashion-MNIST', version=1)   # 15 - classif
    #X = skdata.fetch_openml(name='cnae-9', version=1)  # 15 - classif
    #X = skdata.fetch_openml(name='Prostate', version=1)    # 15 - classif
    #X = skdata.fetch_openml(name='steel-plates-fault', version=1)      # 15 - classif

    #X = skdata.fetch_openml(name='BurkittLymphoma', version=1) # 15 - cluster

    # Tenta obter o nome do dataset
    dataset_name = X.get('details', {}).get('name', 'N/A')

    dados = X['data']
    target = X['target']
    # Matriz esparsa (em alguns datasets com dimensionalidade muito alta)
    if sp.issparse(dados):
        dados = dados.todense()
        dados = np.asarray(dados)
    else:
        if not isinstance(dados, np.ndarray):
            cat_cols = dados.select_dtypes(['category']).columns
            dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
            # Convert to numpy
            dados = dados.to_numpy()
    le = LabelEncoder()
    le.fit(target)
    target = le.transform(target)

    # Redução do conjunto de dados 
    if dados.shape[0] > 10000:
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.2, random_state=42)
    #elif dados.shape[0] > 10000:
    #    dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.1, random_state=42)
    #elif dados.shape[0] >= 4000:
    #    dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)

    # Remove nan's
    dados = np.nan_to_num(dados)

    # Data standardization (to deal with variables having different units/scales)
    dados = preprocessing.scale(dados)

    # Dimensões da matriz de dados
    n = dados.shape[0]
    m = dados.shape[1]
    c = len(np.unique(target))

    # Parâmetros do algoritmo
    #N_NEIGHBOR = int(np.round(np.log(n)))
    N_NEIGHBOR = int(np.round(np.sqrt(n)))

    print('Dataset: ', dataset_name)
    print('N = ', n)
    print('M = ', m)
    print('C = %d' %c)
    print('K = %d' %N_NEIGHBOR)

    # Chama função UMAP
    print('------------------------------------')
    erro_umap, dados_umap = umap(dados, target, N_NEIGHBOR)
    PlotaDados(dados_umap, target, 'UMAP', dataset_name)

    print('------------------------------------')
    erro_k_umap, dados_k_umap = k_umap(dados, target, N_NEIGHBOR,dataset_name)
    PlotaDados(dados_k_umap, target, 'K-UMAP', dataset_name)


    # Plota função de perda
    START=10
    plt.plot(erro_umap[START:], c='red', label='UMAP (Euclidean)', alpha=0.7)
    plt.plot(erro_k_umap[START:], c='blue', label='K-UMAP', alpha=0.7)
    plt.title("Cross-Entropy", fontsize = 12)
    plt.xlabel("ITERATION", fontsize = 12)
    plt.ylabel("CROSS-ENTROPY", fontsize = 12)
    plt.legend()
    plt.savefig('Cross-Entropy_'+dataset_name+'.png')

    print('Métricas de avaliação de qualidade de agrupamento e classificação')
    print('-------------------------------------------------------------------')
    print()


    umap_metrics = EvaluationMetrics(dados_umap, target, 'UMAP (Euclidean)', N_NEIGHBOR)
    k_umap_metrics = EvaluationMetrics(dados_k_umap, target, 'SH-UMAP', N_NEIGHBOR)


    new_result = {
    dataset_name: {
        'UMAP (Euclidean)': dict(umap_metrics),
        'SH-UMAP': dict(k_umap_metrics)
    }
    }
    
    if os.path.exists('resultados_umap.json'):
        with open('resultados_umap.json', 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
        results.update(new_result)
    else:
        results = new_result
    with open('resultados_umap.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
if __name__ == '__main__':
    main()

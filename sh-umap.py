'''
    SH-UMAP: UMAP adaptativo com curvatura local
    Versão otimizada via Stochastic Gradient Descent

    Authors: Gustavo H. Chavari and Alexandre L. M. Levada
    Created: Tuesday, June 10, 2025, 10:55:12
'''
import numba
import os
import json
import sys
import warnings
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csr_matrix
from scipy.optimize import curve_fit
import sklearn.neighbors as sknn
import matplotlib.pyplot as plt
from scipy import stats
import sklearn.datasets as skdata
from sklearn.neighbors import LocalOutlierFactor
from matplotlib import colors
from numpy.linalg import inv
from numpy.linalg import cond
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding as LLE  
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from sklearn import preprocessing
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder
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
N_EPOCHS = 1000
LEARNING_RATE = 1.0
N_NEG_SAMPLES = 5 

# Sigma binary search
MAX_ITER = 120

#####################################################
# FUNÇÕES AUXILIARES
#####################################################

def prob_high_dim(dists_row, rho_i, sigma):
    """
     Para a linha de k-distâncias de um ponto, computa as probabilidades no espaço de alta dimensão.
     (1D array)
    """
    d = dists_row - rho_i
    d = np.maximum(d, 0)

    return np.exp(- d / sigma)


def k(prob):
    """
    Computa n_neighbor = k (smooth) para cada array de probabilidades de alta dimensionalidade
    """
    return np.power(2, np.sum(prob))

def sigma_binary_search(k_of_sigma, fixed_k):
    """
    Resolve a equação perp_of_sigma(sigma) = fixed_perplexity 
    com relação a sigma com uma busca binária
    """
    sigma_lower_limit = 0
    sigma_upper_limit = 1000
    for i in range(64):
        approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2

        if approx_sigma == 0.0: return 0.0
        
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

def find_ab_params(spread=1.0, min_dist=0.1):
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

# MODIFICADO
# A função abaixo implementa a otimização via SGD
# Também utiliza a otimização via NUMBA
@numba.njit(fastmath=True)
def umap_sgd_optimization(
    y_init,
    head,
    tail,
    n_epochs,
    initial_alpha,
    a,
    b,
    n_neg_samples,
    log_loss=True
):
    """
    Otimiza o embedding usando Descida de Gradiente Estocástica com amostragem negativa.
    """
    y = y_init.copy().astype(np.float32)
    alpha = np.float32(initial_alpha)
    a = np.float32(a)
    b = np.float32(b)
    
    n = y.shape[0]
    n_edges = len(head)

    if log_loss:
        loss_history = np.zeros(n_epochs, dtype=np.float32)

    for epoch in range(n_epochs):
        current_alpha = alpha * (1.0 - epoch / float(n_epochs))

        if log_loss:
            attractive_loss = 0.0
            repulsive_loss = 0.0

        edge_indices = np.random.permutation(n_edges)

        for i in range(n_edges):
            edge_idx = edge_indices[i]
            h_idx, t_idx = head[edge_idx], tail[edge_idx]
            
            y_h = y[h_idx]
            y_t = y[t_idx]

            dist_sq = np.sum(np.square(y_h - y_t))

            # Força atrativa
            # q_ij é a similaridade em baixa dimensão
            q_ij = 1.0 / (1.0 + a * (dist_sq ** b))
            
            if log_loss:
                attractive_loss += np.log(q_ij + 1e-6)

            # Gradiente da força atrativa
            if dist_sq > 0.0:
                grad_coeff = -2.0 * a * b * (dist_sq ** (b - 1.0))
                grad_coeff /= (1.0 + a * (dist_sq ** b))
            else:
                grad_coeff = 0.0
            
            grad = grad_coeff * (y_h - y_t)
            
            # Gradient clip para estabilidade (não entendi?)
            grad = np.clip(grad, -4.0, 4.0)
            
            y[h_idx] += grad * current_alpha
            y[t_idx] -= grad * current_alpha

            # Força repulsiva
        
            for _ in range(n_neg_samples):
                neg_idx = np.random.randint(0, n)
                if neg_idx == h_idx:
                    continue

                y_neg = y[neg_idx]
                dist_sq_neg = np.sum(np.square(y_h - y_neg))
                
                # q_ik é a similaridade em baixa dimensão para a amostra negativa
                q_ik = 1.0 / (1.0 + a * (dist_sq_neg ** b))
                
                if log_loss:
                    repulsive_loss += np.log(1.0 - q_ik + 1e-6)

                # Gradiente da força repulsiva
                if dist_sq_neg > 0.0:
                    grad_coeff_neg = 2.0 * b
                    grad_coeff_neg /= (0.001 + dist_sq_neg) * (1.0 + a * (dist_sq_neg ** b))
                else:
                    grad_coeff_neg = 0.0
                
                grad_neg = grad_coeff_neg * (y_h - y_neg)
                
                grad_neg = np.clip(grad_neg, -4.0, 4.0)
                
                y[h_idx] += grad_neg * current_alpha

        if log_loss:
            total_loss = -(attractive_loss + repulsive_loss) / float(n_edges)
            loss_history[epoch] = total_loss

    if log_loss:
        return loss_history, y
    else:

        return np.array([0.0], dtype=np.float32), y

def umap(dados, target, N_NEIGHBOR):
    """
    Implementa o algoritmo UMAP padrão
    """
    print('************************************')
    print('UMAP com distância Euclidiana')
    print('************************************\n')
    n = dados.shape[0]
    
    #### Constructing a local fuzzy simplicial set ####

    # MODIFICADO: 
    # Usar knn para encontrar o grafo 
    # Isso evita a criação de uma matriz de distância n x n completa.
    print("Construindo o grafo k-NN...")
    knn = NearestNeighbors(n_neighbors=N_NEIGHBOR, metric='euclidean')
    knn.fit(dados)

    # dist_knn e indices_knn são matrizes (n, k)
    dist_knn, indices_knn = knn.kneighbors(dados)
    dist_knn = np.square(dist_knn) # UMAP utiliza distâncias quadráticas

    rho = np.array([dist_knn[i][1] if dist_knn.shape[1] > 1 else 0.0 for i in range(dist_knn.shape[0])])
   
    sigma_array = []

    # Inicializa a matriz de probabilidade como uma matriz esparsa.
    prob = lil_matrix((n, n), dtype=np.float32)
    
    print("Calculando similaridades no espaço de alta dimensão...")
    for i in tqdm(range(n), desc="Busca binária do Sigma"):
        # Extrai as distâncias e índices dos k-vizinhos para o ponto i
        dists_i = dist_knn[i]
        
        # A função de probabilidade agora opera apenas nas k distâncias.
        func = lambda sigma: k(prob_high_dim(dists_i, rho[i], sigma))
        binary_search_result = sigma_binary_search(func, N_NEIGHBOR)
        
        row_probs = prob_high_dim(dists_i, rho[i], binary_search_result)
        
        # Preenche a matriz esparsa apenas para os vizinhos.
        prob[i, indices_knn[i]] = row_probs
        
        sigma_array.append(binary_search_result)

    print(f"Sigma médio = {np.mean(sigma_array)}")

    P = (prob + prob.T).tocsr() / 2
    
    P_coo = P.tocoo()
    head = P_coo.row
    tail = P_coo.col

    a, b = find_ab_params(1, MIN_DIST)
    print(f"Hiperparâmetros a = {a} and b = {b}")

    np.random.seed(12345)
    print("Inicializando com Laplacian Eigenmaps...")
    model = SpectralEmbedding(n_components=N_LOW_DIMS, n_neighbors=N_NEIGHBOR)
    y_init = model.fit_transform(P)
    
    print("Otimizando layout...")

    loss, y_final = umap_sgd_optimization(
        y_init,
        head,
        tail,
        N_EPOCHS,
        LEARNING_RATE,
        a,
        b,
        N_NEG_SAMPLES,
        log_loss=True
    )
            
    print("UMAP Finalizado.")
    return (loss, y_final)


def curvature_estimation(dados, k, method="gaussian"):
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

    for i in tqdm(range(n), desc="Computing curvatures"):
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        amostras = dados[indices] - np.mean(dados[indices], axis=0)
        ni = len(indices)
        if ni > 1:
            I = np.cov(amostras.T)
        else:
            I = np.eye(m)
        v, w = np.linalg.eig(I)
        ordem = v.argsort()
        Wpca = w[:, ordem[::-1]]
        for j in range(0, m):
            Squared[:, j] = Wpca[:, j]**2
        col = 0
        for j in range(0, m):
            for l in range(j, m):
                if j != l:
                    Cross[:, col] = Wpca[:, j]*Wpca[:, l]
                    col += 1
        Wpca = np.column_stack((np.ones(m), Wpca))
        Wpca = np.hstack((Wpca, Squared))
        Wpca = np.hstack((Wpca, Cross))

        Q = Wpca
        H = Q[:, (m+1):]
        II = np.dot(H, H.T)

        # Shape Operator
        S = -np.dot(II, I)
        #S += np.eye(m)/1e-6

        if method == "gaussian":
            curvatures[i] = abs(np.linalg.det(S+np.eye(m)/1e-3))
        elif method == "mean":
            curvatures[i] = np.linalg.trace(S)

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

    return ranks
    

def k_umap(dados, target, N_NEIGHBOR,dataset_name):
    """
    Implementa o algoritmo UMAP com curvatura local
    """
    print('*************************************************')
    print('UMAP com curvatura local via operador de forma')
    print('*************************************************\n')
    n = dados.shape[0]
    c = np.unique(target)
    
    print("Calculando curvaturas locais...")


    knn = NearestNeighbors(n_neighbors=N_NEIGHBOR, metric='euclidean')
    knn.fit(dados)
    dist_knn, indices_knn = knn.kneighbors(dados)
    dist_knn = np.square(dist_knn)

    rho = np.array([dist_knn[i][1] if dist_knn.shape[1] > 1 else 0.0 for i in range(dist_knn.shape[0])])

    curvatures = curvature_estimation(dados, N_NEIGHBOR,"gaussian")

    K = percentile_rank(curvatures,N_NEIGHBOR,dataset_name)

    unique_values, counts = np.unique(K, return_counts=True)

    print("Ranking de curvatura (Rank: Quantidade)")
    for value, count in zip(unique_values, counts):
        print(f"{value}: {count}")

    print("Construindo o grafo k-NN...")
    
    prob = lil_matrix((n, n), dtype=np.float32)
    sigma_array = []
    
    print("Calculando similaridades no espaço de alta dimensão...")
    for i in tqdm(range(n), desc="Busca binária do Sigma (com curvatura)"):
        dists_i = dist_knn[i,:N_NEIGHBOR - K[i]]
        
        func = lambda sigma: k(prob_high_dim(dists_i, rho[i], sigma))
        # A curvatura ajusta o número de vizinhos alvo na busca binária.
        target_k = max(3, N_NEIGHBOR - K[i])
        binary_search_result = sigma_binary_search(func, target_k)

        prob[i, indices_knn[i,:N_NEIGHBOR - K[i]]] = prob_high_dim(dists_i, rho[i], binary_search_result)
        sigma_array.append(binary_search_result)

    print(f"Sigma médio = {np.mean(sigma_array)}")
    
    P = (prob + prob.T).tocsr() / 2
    
    P_coo = P.tocoo()
    head = P_coo.row
    tail = P_coo.col

    a, b = find_ab_params(1, MIN_DIST)
    print(f"Hiperparâmetros a = {a} and b = {b}")

    np.random.seed(12345)
    print("Inicializando com Laplacian Eigenmaps...")
    model = SpectralEmbedding(n_components=N_LOW_DIMS, n_neighbors=N_NEIGHBOR)
    y_init = model.fit_transform(P)
    
    print("Otimizando layout...")
    loss, y_final = umap_sgd_optimization(
        y_init,
        head,
        tail,
        N_EPOCHS,
        LEARNING_RATE,
        a,
        b,
        N_NEG_SAMPLES,
        log_loss=True
    )
            
    print("Finished SH-UMAP")
    return (loss, y_final)


def PlotaDados(dados, labels, metodo, dataset_name):
    """
    Função para plotagem dos dados de saída
    """
    nclass = len(np.unique(labels))
    
    if nclass > 11:
        # Gerar cores aleatórias para muitas classes
        cores = list(colors.cnames.keys())
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'green', 'black', 'orange', 'magenta', 'darkkhaki', 'brown', 'purple', 'cyan', 'salmon']
    
    plt.figure(figsize=(8, 6))
    for i in range(nclass):
        indices = np.where(labels==i)[0]
        # Garante que a cor selecionada seja válida
        cor = cores[i % len(cores)]
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, alpha=0.6, marker='.', label=f'Classe {i}') 
    
    nome_arquivo = metodo+'_'+dataset_name + '.png'
    plt.title(f'{metodo} - Dataset: {dataset_name}')
    plt.xlabel('Dimensão 1')
    plt.ylabel('Dimensão 2')
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

    # GMM
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

    print(f'Avaliação via KNN com 10-fold cross-validation...')
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
    warnings.simplefilter(action='ignore')

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

    dataset_name = X.get('details', {}).get('name', 'NoName') + "SHMOD"
    print(f"Carregando dataset: {dataset_name}\n")

    dados = X['data']
    target = X['target']

    if sp.issparse(dados):
        dados = dados.toarray()
    if not isinstance(dados, np.ndarray):
        dados = pd.DataFrame(dados)
        cat_cols = dados.select_dtypes(['category', 'object']).columns
        for col in cat_cols:
            dados[col] = dados[col].astype('category').cat.codes
        dados = dados.to_numpy()

    le = LabelEncoder()
    target = le.fit_transform(target)

    if dados.shape[0] > 15000:
        print(f"Dataset grande. Reduzindo de {dados.shape[0]} para 5000 amostras.")
        dados, _, target, _ = train_test_split(dados, target, train_size=0.2, random_state=42, stratify=target)
    
    dados = np.nan_to_num(dados)
    dados = preprocessing.scale(dados)

    n, m = dados.shape
    c = len(np.unique(target))

    #N_NEIGHBOR = int(np.round(np.log2(n)))
    N_NEIGHBOR = int(np.round(np.sqrt(n)))
    print(f'Datasetname = {dataset_name}')
    print(f'N = {n}')
    print(f'M = {m}')
    print(f'C = {c}')
    print(f'K = {N_NEIGHBOR}\n')

    erro_umap, dados_umap = umap(dados, target, N_NEIGHBOR)
    PlotaDados(dados_umap, target, 'UMAP', dataset_name)
    
    print('\nSH-UMAP')
    print('------------------------------------')
    erro_k_umap, dados_k_umap = k_umap(dados, target, N_NEIGHBOR,dataset_name)
    PlotaDados(dados_k_umap, target, 'SH-UMAP', dataset_name)

    plt.figure()
    START=8
    plt.plot(erro_umap[START:], c='red', label='UMAP (Euclidean)', alpha=0.7)
    plt.plot(erro_k_umap[START:], c='blue', label='SH-UMAP', alpha=0.7)
    plt.title("Convergência da Entropia Cruzada", fontsize=12)
    plt.xlabel("Iteração", fontsize=12)
    plt.ylabel("Entropia Cruzada", fontsize=12)
    plt.legend()
    plt.savefig('Cross-Entropy_'+dataset_name+'.png')
    plt.close()

    print('\nMétricas de avaliação de qualidade de agrupamento e classificação')
    print('-------------------------------------------------------------------')
    
    umap_metrics = EvaluationMetrics(dados_umap, target, 'UMAP (Euclidean)', N_NEIGHBOR)
    k_umap_metrics = EvaluationMetrics(dados_k_umap, target, 'SH-UMAP', N_NEIGHBOR)


    new_result = {
    dataset_name: {
        'UMAP (Euclidean)': dict(umap_metrics),
        'SH-UMAP': dict(k_umap_metrics)
    }
    }
    
    if os.path.exists('novos_resultados_shumap.json'):
        with open('novos_resultados_shumap.json', 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
        results.update(new_result)
    else:
        results = new_result
    with open('novos_resultados_shumap.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
import sys
import warnings
import numpy as np
import pandas as pd
import scipy as sp
import sklearn.neighbors as sknn
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
from matplotlib import colors
from numpy.linalg import inv
from numpy.linalg import cond
from scipy import optimize
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.metrics.pairwise import euclidean_distances
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


def umap(dados):
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


def gs(X, row_vecs=True, norm = True):
    """
    Gram-Schmidt ortogonalization
    """
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T


def Curvature_Estimation(dados, k):
    """
    Computes the curvatures of all samples in the training set
    """
    n = dados.shape[0]
    m = dados.shape[1]
    # Se tiver mais de 80 atributos, aplica PCA antes de calcular curvaturas
    if m > 200:
        m = 50
        model = PCA(n_components=m)
        dados = model.fit_transform(dados)
    elif m > 80:
        m = 30
        model = PCA(n_components=m)
        dados = model.fit_transform(dados)        
    # Primeira e segunda formas fundamentais
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
    # Computes the means and covariance matrices for each patch
    for i in range(n):
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        ######## Estima primeira forma fundamental
        amostras = dados[indices]
        ni = len(indices)
        if ni > 1:
            # Primeira forma fundamental em i
            I = np.cov(amostras.T)
            I = I + 0.0001*np.eye(I.shape[0])   # Regulariza
        else:
            I = np.eye(m)      # pontos isolados = identidade
        # Compute the eigenvectors
        v, w = np.linalg.eig(I)
        # Sort the eigenvalues
        ordem = v.argsort()
        # Select the eigenvectors in decreasing order (in columns)
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
        # Gram-Schmidt ortogonalization
        Q = gs(Wpca)
        # Discard the first m columns of H
        H = Q[:, (m+1):]
        # Segunda forma fundamental
        II = np.dot(H, H.T)
        S = -np.dot(II, I)
        curvatures[i] = np.linalg.det(S)        # curvatura Gaussiana
        #curvatures[i] = np.trace(S)            # curvatura média
    return curvatures


def normalize_curvatures(curv):
    """
    Optional function to normalize the curvatures to the interval [0, 1]
    """
    if curv.min() != curv.max():
        k = (curv - curv.min())/(curv.max() - curv.min())
    else:
        k = curv/len(curv)
    return k
    

def k_umap(dados):
    """
    Implementa o algoritmo UMAP com curvatura local
    """
    print('*************************************************')
    print('UMAP com curvatura local via operador de forma')
    print('*************************************************\n')
    n = dados.shape[0]
    m = dados.shape[1]
    c = len(np.unique(target))
    MIN_NEIGH = 3
    # Calcula as curvaturas locais
    curvaturas = Curvature_Estimation(dados, N_NEIGHBOR)
    #print(curvaturas)
    K = normalize_curvatures(curvaturas)
    #print(min(K))
    #print(max(K))
    intervalos = np.linspace(min(K), max(K), N_NEIGHBOR-MIN_NEIGH)
    quantis = np.quantile(K, intervalos)
    bins = np.array(quantis)
    # Discrete curvature values obtained after quantization (scores)
    K = np.digitize(K, bins)
    #K += 1
    #print(K)
    #print(min(K))
    #print(max(K))
    #input()
    # t = np.quantile(K, 0.5)
    # for i, x in enumerate(K):
        # if x <= t:
            # K[i] = 0.2
        # else:
            # K[i] = 1
    # t1 = np.quantile(K, 0.4)
    # t2 = np.quantile(K, 0.6)
    # for i, x in enumerate(K):
        # if x <= t1:
            # K[i] = 0.5
        # elif x > t1 and x <= t2:
            # K[i] = 1.0
        # else:
            # K[i] = 2.5
    #K = np.tile(K, [2, 1])
    #print(K.shape)
    dist = np.square(euclidean_distances(dados, dados))
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
    a = 1.93
    b = 0.79
    print("Hiperparâmetros a = " + str(a) + " and b = " + str(b))    
    # Semente para os números aleatórios
    np.random.seed(12345)
    # Laplacian Eigenmaps para inicialização das coordenadas dos pontos no espaço de baixa dimensão
    model = SpectralEmbedding(n_components=N_LOW_DIMS, n_neighbors=N_NEIGHBOR)    
    y = model.fit_transform(dados)
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


def PlotaDados(dados, labels, metodo):
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
    nome_arquivo = metodo + '.png'
    plt.title(metodo+' clusters')
    plt.savefig(nome_arquivo)
    plt.show()


def EvaluationMetrics(dados, target, method):
    """
    Computes performance evaluation metrics
    """
    print()
    print('Quantitative metrics for %s features' %(method))
    print()
    
    lista = []
    lista_p = []
    lista_r = []
    lista_f1 = []
    lista_j = []
    lista_roc = []
    lista_l = []
    lista_k = []
    
    # 7 different classifiers
    neigh = KNeighborsClassifier(n_neighbors=7)
    svm = SVC(gamma='auto')
    nb = GaussianNB()
    dt = DecisionTreeClassifier(random_state=42)
    rfc = RandomForestClassifier()
    
    # 50% for training and 50% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados.real, target, train_size=0.5, random_state=42)
    
    # KNN
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    acc = metrics.balanced_accuracy_score(y_pred, y_test)
    lista.append(acc)
    kappa = metrics.cohen_kappa_score(y_pred, y_test)
    lista_k.append(kappa)
    precision = metrics.precision_score(y_pred, y_test, average='weighted')
    lista_p.append(precision)
    recall = metrics.recall_score(y_pred, y_test, average='weighted')
    lista_r.append(recall)
    f1 = metrics.f1_score(y_pred, y_test, average='weighted')
    lista_f1.append(f1)
    jaccard = metrics.jaccard_score(y_pred, y_test, average='weighted')
    lista_j.append(jaccard)
    
    # SMV
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = metrics.balanced_accuracy_score(y_pred, y_test)
    lista.append(acc)
    kappa = metrics.cohen_kappa_score(y_pred, y_test)
    lista_k.append(kappa)
    precision = metrics.precision_score(y_pred, y_test, average='weighted')
    lista_p.append(precision)
    recall = metrics.recall_score(y_pred, y_test, average='weighted')
    lista_r.append(recall)
    f1 = metrics.f1_score(y_pred, y_test, average='weighted')
    lista_f1.append(f1)
    jaccard = metrics.jaccard_score(y_pred, y_test, average='weighted')
    lista_j.append(jaccard)
        
    # Naive Bayes
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    acc = metrics.balanced_accuracy_score(y_pred, y_test)
    lista.append(acc)
    kappa = metrics.cohen_kappa_score(y_pred, y_test)
    lista_k.append(kappa)
    precision = metrics.precision_score(y_pred, y_test, average='weighted')
    lista_p.append(precision)
    recall = metrics.recall_score(y_pred, y_test, average='weighted')
    lista_r.append(recall)
    f1 = metrics.f1_score(y_pred, y_test, average='weighted')
    lista_f1.append(f1)
    jaccard = metrics.jaccard_score(y_pred, y_test, average='weighted')
    lista_j.append(jaccard)
    
    # Decision Tree
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc = metrics.balanced_accuracy_score(y_pred, y_test)
    lista.append(acc)
    kappa = metrics.cohen_kappa_score(y_pred, y_test)
    lista_k.append(kappa)
    precision = metrics.precision_score(y_pred, y_test, average='weighted')
    lista_p.append(precision)
    recall = metrics.recall_score(y_pred, y_test, average='weighted')
    lista_r.append(recall)
    f1 = metrics.f1_score(y_pred, y_test, average='weighted')
    lista_f1.append(f1)
    jaccard = metrics.jaccard_score(y_pred, y_test, average='weighted')
    lista_j.append(jaccard)
        
    # Random Forest Classifier
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    acc = metrics.balanced_accuracy_score(y_pred, y_test)
    lista.append(acc)
    kappa = metrics.cohen_kappa_score(y_pred, y_test)
    lista_k.append(kappa)
    precision = metrics.precision_score(y_pred, y_test, average='weighted')
    lista_p.append(precision)
    recall = metrics.recall_score(y_pred, y_test, average='weighted')
    lista_r.append(recall)
    f1 = metrics.f1_score(y_pred, y_test, average='weighted')
    lista_f1.append(f1)
    jaccard = metrics.jaccard_score(y_pred, y_test, average='weighted')
    lista_j.append(jaccard)
    
    # GMM clustering
    c = len(np.unique(target))
    #labels_ = GaussianMixture(n_components=c, random_state=42).fit_predict(dados.real)
    labels_ = KMeans(n_clusters=c).fit_predict(dados.real)
    
    # Computes the Silhoutte coefficient
    sc = metrics.silhouette_score(dados.real, target, metric='euclidean')
    ch = metrics.calinski_harabasz_score(dados.real, target)
    db = metrics.davies_bouldin_score(dados.real, target)
    fm = metrics.fowlkes_mallows_score(labels_, target)
    ri = metrics.rand_score(labels_, target)
    mi = metrics.mutual_info_score(labels_, target)
    vm = metrics.v_measure_score(labels_, target)
    
    # Computes the average accuracy    
    acc = max(lista)
    precision = max(lista_p)
    recall = max(lista_r)
    f1 = max(lista_f1)
    jac = max(lista_j)
    kap = max(lista_k)    

    print('Silhouette coefficient: ', sc)
    print('Calinski Harabasz: ', ch)
    print('Davies Bouldin: ', db)
    print('Rand index: ', ri)
    print('Fowlkes Mallows score: ', fm)
    print('Mutual info score: ', mi)
    print('V-measure score: ', vm)
    print('-----------------------------------')
    print('Maximum balanced accuracy: ', acc)
    print('Maximum kappa: ', kap)
    print('Maximum precision: ', precision)
    print('Maximum recall: ', recall)
    print('Maximum F1 weighted: ', f1)
    print('Maximum Jaccard: ', jac)
    print()
    
    return [sc, ch, db, ri, fm, mi, vm, acc, kap, precision, recall, f1, jac]

        
######################################################
# INÍCIO DO SCRIPT
######################################################
# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# Leitura dos dados
X = skdata.load_iris()     # 15 - all
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

dados = X['data']
target = X['target']
# Matriz esparsa (em alguns datasets com dimensionalidade muito alta)
if type(dados) == sp.sparse._csr.csr_matrix:
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
if dados.shape[0] > 50000:
    dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.02, random_state=42)
elif dados.shape[0] > 10000:
    dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.1, random_state=42)
elif dados.shape[0] >= 4000:
    dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)

# Remove nan's
dados = np.nan_to_num(dados)

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

# Dimensões da matriz de dados
n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))

# Parâmetros do algoritmo
#N_NEIGHBOR = int(np.round(np.sqrt(n)))
#N_NEIGHBOR = int(np.round(np.log2(n)))
N_NEIGHBOR = 15
MIN_DIST = 0.1
N_LOW_DIMS = 2
LEARNING_RATE = 1
MAX_ITER = 120

print('N = ', n)
print('M = ', m)
print('C = %d' %c)
print('K = %d' %N_NEIGHBOR)
print()

# Chama função UMAP
erro_umap, dados_umap = umap(dados)
# Plota dados de saída
PlotaDados(dados_umap, target, 'UMAP (Euclidean)')
# Chama função UMAP com distância de Mahalanobis
erro_k_umap, dados_k_umap = k_umap(dados)
# Plota dados de saída
PlotaDados(dados_k_umap, target, 'K-UMAP')
# Plota função de perda
START=8
plt.plot(erro_umap[START:], c='red', label='UMAP (Euclidean)', alpha=0.7)
plt.plot(erro_k_umap[START:], c='blue', label='K-UMAP', alpha=0.7)
plt.title("Cross-Entropy", fontsize = 12)
plt.xlabel("ITERATION", fontsize = 12)
plt.ylabel("CROSS-ENTROPY", fontsize = 12)
plt.legend()
plt.savefig('Cross-Entropy.png')
plt.show()

print('Métricas de avaliação de qualidade de agrupamento e classificação')
print('-------------------------------------------------------------------')
print()
# Métricas de avaliação para UMAP Euclidiano
umap_metrics = EvaluationMetrics(dados_umap, target, 'UMAP (Euclidean)')
# Métricas de avaliação para UMAP com distância de Mahalanobis
k_umap_metrics = EvaluationMetrics(dados_k_umap, target, 'K-UMAP')

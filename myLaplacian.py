# Função que implementa o Laplacian Eigenmaps
def myLaplacian(X, k, d, t, lap='padrao'):
    # Gera o grafo KNN
    knnGraph = sknn.kneighbors_graph(X, n_neighbors=k, mode='distance')
    knnGraph.data = np.exp(-(knnGraph.data**2)/t)
    W = knnGraph.toarray()  # Extrai a matriz de adjacência a partir do grafo KNN
    W = np.maximum(W, W.T)  # Para matriz de adjacência ficar simétrica
    # Matriz diagonal D e Laplaciana L
    D = np.diag(W.sum(1))   # soma as linhas
    L = D - W
    if lap == 'normalizada':
        lambdas, alphas = eigh(np.dot(inv(D), L), eigvals=(1, d))   # descarta menor autovalor (zero)
    else:
        lambdas, alphas = eigh(L, eigvals=(1, d))   # descarta menor autovalor (zero)
    return alphas

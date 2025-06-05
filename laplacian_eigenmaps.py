# -*- coding: utf-8 -*-
# created by:  Gustavo Henrique Chavari on 2025-05-01

import numpy as np
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA
import sklearn.neighbors as sknn


class LaplacianEigenmaps():
    """Class for Laplacian Eigenmaps dimensionality reduction"""

    def __init__(self, dim:int = 2, k:int = 3, eps = None, graph:str = 'k-nearest', weights:str = 'heat kernel', 
                 sigma:float = 0.1, laplacian:str = 'unnormalized'):
    
        self.dim = dim
        self.k = k
        self.eps = eps
        self.weights = weights
        self.sigma = sigma
        self.laplacian = laplacian

    def heat_kernel(self, x1, x2):
        return np.exp(-(np.linalg.norm(x1-x2)**2)/self.sigma)
    
    def rbf(self, x1, x2):
        return np.exp(-self.sigma*np.linalg.norm(x1-x2)**2)
    
    def simple(self):
        return 1
        
    def adjacency_matrix(self, X):
        """Construct the adjacency matrix using k-nearest neighbors"""
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(X)
        _, indices = nbrs.kneighbors(X)
        
        W = np.zeros((X.shape[0], X.shape[0]))
        for i in range(W.shape[0]):
            for j in range(W.shape[0]):
                if j in indices[i]:
                    if self.weights == 'heat kernel':
                        W[i][j] = self.heat_kernel(X[i],X[j])
                    elif self.weights == 'rbf':
                        W[i][j] = self.rbf(X[i],X[j])
                    elif self.weights == 'simple':
                        W[i][j] = self.simple()
        
        return W

    
    def fit_transform(self, X):
        """
        Fit the model and transform the data
        
        """
        W = self.adjacency_matrix(X)
        
        D = np.diag(W.sum(axis=1))
        
        L = D - W
        
        if self.laplacian == 'unnormalized':
            # Solve generalized eigenvalue problem: Ly = λDy
            eigvals, eigvecs = eigh(L,D)

        #elif self.laplacian == 'random':
            # Compute random walk normalized Laplacian: L_rw = D^(-1)L = I - D^(-1)W

        elif self.laplacian == 'symmetrized':
            # Compute symmetrically normalized Laplacian: L_sym = D^(-1/2)LD^(-1/2)

            D_sq = np.diag(1.0 / np.sqrt(np.diag(D)))

            L_sym = D_sq@ L @ D_sq

            eigvals, eigvecs = eigh(L_sym)

        
        # Sort eigenvalues and eigenvectors
        idx = np.argsort(eigvals)  # Skip smallest eigenvalue
        self.embedding_ = eigvecs[:, idx[1: 1 + self.dim + 1]]
        
        return self.embedding_
    
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
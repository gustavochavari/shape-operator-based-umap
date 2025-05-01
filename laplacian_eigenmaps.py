# -*- coding: utf-8 -*-
# created by:  Gustavo Henrique Chavari on 2025-05-01

import numpy as np
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.linalg import eigh


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
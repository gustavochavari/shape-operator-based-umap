def estimate_curvature_delaunay(data, k=15):
    """
    Computes the local Gaussian curvature for each point in a dataset using
    a method based on Delaunay triangulation of local 2D projections.

    This method avoids high-dimensional eigenvalue decomposition by approximating
    the manifold locally with a 2D tangent plane found via PCA.

    Args:
        data (np.ndarray): The input dataset of shape (n_samples, n_features).
        k (int): The number of nearest neighbors to define the local patch
                 for each point.

    Returns:
        np.ndarray: An array of shape (n_samples,) containing the estimated
                    curvature for each point.
    """
    n_samples = data.shape[0]
    curvatures = np.zeros(n_samples)

    # 1. Find the k nearest neighbors for all points at once
    nn = NearestNeighbors(n_neighbors=k ) # 
    nn.fit(data)
    # distances, indices = nn.kneighbors(data)
    # O `indices` conterá o próprio ponto em indices[:, 0]
    indices = nn.kneighbors(data, return_distance=False)

    # 2. Loop through each point to calculate its local curvature
    for i in range(n_samples):
        # 2a. Define the local patch
        patch_indices = indices[i]
        local_patch = data[patch_indices, :]

        # 2b. Find the best-fit 2D plane using PCA
        if local_patch.shape[0] < 3: # Need at least 3 points for a 2D plane
             curvatures[i] = 0 # Cannot form a triangle, assume flat
             continue
        
        pca = PCA(n_components=2)
        projected_patch = pca.fit_transform(local_patch)

        # The point `i` is now the first point in the projected patch
        # (since it's the closest to itself)
        center_point_idx = 0
        
        try:
            # 2c. Perform 2D Delaunay triangulation
            tri = Delaunay(projected_patch)
        except (ValueError, RuntimeError):
            # If points are collinear, triangulation fails. Assume zero curvature.
            curvatures[i] = 0
            continue
            
        # 2d. Find all triangles attached to the center point
        # A simplex is a triangle, tri.simplices gives vertex indices
        incident_simplices = tri.simplices[np.any(tri.simplices == center_point_idx, axis=1)]

        if incident_simplices.shape[0] < 1:
            curvatures[i] = 0
            continue
            
        total_angle = 0
        
        # 2e. Calculate the sum of angles around the center point
        for simplex in incident_simplices:
            # Get the indices of the other two vertices in the triangle
            v1_idx, v2_idx = [s for s in simplex if s != center_point_idx]

            # Get the 2D coordinates
            p_center = projected_patch[center_point_idx]
            p1 = projected_patch[v1_idx]
            p2 = projected_patch[v2_idx]

            # Create vectors from the center point
            vec1 = p1 - p_center
            vec2 = p2 - p_center

            # Calculate the angle between the two vectors using the dot product
            # cos(theta) = (v1 . v2) / (||v1|| * ||v2||)
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            
            # Avoid division by zero for identical points
            if norm_product == 0:
                continue

            cosine_angle = dot_product / norm_product
            # Clip to handle potential floating point inaccuracies
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            
            angle = np.arccos(cosine_angle)
            total_angle += angle
            
        # 2f. The angle defect is the curvature
        curvature = 2 * np.pi - total_angle
        curvatures[i] = curvature

    return curvatures



def SphereCurvature(dados,k,d=None):
    n = dados.shape[0]
    m = dados.shape[1]
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    A = knnGraph.toarray()
    
    R = []
    for i in range(n):
        vizinhos = A[i, :] - np.mean(A[i, :],axis=0)
        indices = vizinhos.nonzero()[0]
        r_i = fit_nsphere(dados[indices])
        R.append(1/r_i**2)

    
    return R

def fit_nsphere(X):
    sphere = geometry.n_sphere.NSphere.best_fit(X)
    
    return sphere.radius
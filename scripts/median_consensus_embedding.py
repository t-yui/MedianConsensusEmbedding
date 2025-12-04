import numpy as np
from sklearn.manifold import MDS
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances



def geometric_median_matrices(X_list, eps=1e-15, tol=1e-5, max_iter=500):
    z = np.mean(np.array(X_list), axis=0)
    for _ in range(max_iter):
        distances = np.array([np.linalg.norm(X - z, "fro") for X in X_list])
        weights = 1 / (distances + eps)
        new_z = np.sum([w * X for w, X in zip(weights, X_list)], axis=0) / np.sum(
            weights
        )
        if np.linalg.norm(new_z - z, "fro") < tol:
            return new_z
        z = new_z
    return z


def normalize_embedding(Y):
    Y = Y - Y.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(Y)
    return Y if norm == 0 else Y / norm


def MCE(base_embeddings, n_components=2, rs_mds=1):
    m = len(base_embeddings)

    # convert each embedding to distance and calculate geometric median
    distance_matrices = []
    for i in range(m):
        base_embedding_tmp = normalize_embedding(base_embeddings[i])
        X = pairwise_distances(base_embedding_tmp, metric="euclidean")
        distance_matrices.append(X)
    consensus_x = geometric_median_matrices(distance_matrices)

    # mds embedding
    mds = MDS(
        n_components=n_components, dissimilarity="precomputed", random_state=rs_mds
    )
    consensus_y = mds.fit_transform(consensus_x)
    return (consensus_y, consensus_x)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    np.random.seed(100)
    m = 10

    data = load_digits().data

    # generate each embedding
    base_embeddings = []
    for _ in range(m):
        tsne = TSNE(
            n_components=2,
            random_state=np.random.randint(0, 2**32 - 1),
            init="random",
        )
        embedding = tsne.fit_transform(data)
        base_embeddings.append(embedding)

    y, x = MCE(base_embeddings)

    plt.figure(figsize=(6, 6))
    plt.scatter(y[:, 0], y[:, 1], c=load_digits().target)
    plt.tick_params(
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
        bottom=False,
        left=False,
        right=False,
        top=False,
    )
    plt.xlabel("Dimension 1", fontsize=18)
    plt.ylabel("Dimension 2", fontsize=18)
    plt.show()

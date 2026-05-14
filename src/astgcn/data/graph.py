import pandas as pd
import numpy as np

def get_distance_matrix(file_path):
    """
    从 CSV 文件中读取距离矩阵，CSV 文件应该包含三列：from, to, cost，分别表示节点 i 到节点 j 的距离
    :param file_path: CSV 文件的路径
    """
    
    distance = pd.read_csv(file_path)
    
    cnt = [0] * len(distance) # 统计每个节点的连接数，初始为0，注意此时 cnt 长度不是节点数量，cnt 长度是边的数量
    for _, row in distance.iterrows():
        i = int(row["from"])
        j = int(row["to"])
        cnt[i] += 1
        cnt[j] += 1
    num_nodes = 0
    for i in range(len(cnt)):
        if cnt[i] > 0:
            num_nodes = i + 1
            
    distance_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for _, row in distance.iterrows():
        i = int(row["from"])
        j = int(row["to"])
        cost = float(row["cost"])

        distance_matrix[i, j] = cost
        distance_matrix[j, i] = cost  # 按无向图处理，双向联通

    return distance_matrix

def get_adjacency_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """
    根据距离矩阵构造邻接矩阵。

    distance_matrix[i, j] > 0 表示节点 i 和节点 j 相连
    adjacency_matrix[i, j] = 1 表示存在边
    """
    adjacency_matrix = (distance_matrix > 0).astype(np.float32)
    return adjacency_matrix

def get_normalized_laplacian(adj_matrix: np.ndarray) -> np.ndarray:
    """
    构造归一化拉普拉斯矩阵。

    L = I - D^{-1/2} A D^{-1/2}
    其中 D 是度矩阵，D[i, i] = sum_j A[i, j]
    A 是邻接矩阵，I 是单位矩阵

    :param adj_matrix: 邻接矩阵，形状为 (num_nodes, num_nodes)
    :return: 归一化拉普拉斯矩阵，形状为 (num_nodes, num_nodes)
    """
    if adj_matrix.ndim != 2:
        raise ValueError("邻接矩阵必须是二维矩阵")

    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("邻接矩阵必须是方阵")

    num_nodes = adj_matrix.shape[0]

    degree = np.sum(adj_matrix, axis=1) # 计算每个节点的度数，即每行的和

    d_inv_sqrt = np.zeros_like(degree, dtype=np.float32) 
    nonzero_mask = degree > 0
    d_inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(degree[nonzero_mask]) 

    d_inv_sqrt_matrix = np.diag(d_inv_sqrt) # 构造 D^{-1/2} 的对角矩阵

    identity = np.eye(num_nodes, dtype=np.float32)

    laplacian = identity - d_inv_sqrt_matrix @ adj_matrix @ d_inv_sqrt_matrix

    return laplacian.astype(np.float32)

def get_scaled_laplacian(laplacian: np.ndarray) -> np.ndarray:
    """
    构造缩放拉普拉斯矩阵。保证数据都在[-1, 1]

    L_tilde = 2L / lambda_max - I

    :param laplacian: 归一化拉普拉斯矩阵，形状为 (num_nodes, num_nodes)
    :return: 缩放拉普拉斯矩阵，形状为 (num_nodes, num_nodes)
    """
    if laplacian.ndim != 2:
        raise ValueError("拉普拉斯矩阵必须是二维矩阵")

    if laplacian.shape[0] != laplacian.shape[1]:
        raise ValueError("拉普拉斯矩阵必须是方阵")

    num_nodes = laplacian.shape[0]

    lambda_max = np.linalg.eigvals(laplacian).real.max()

    if lambda_max == 0:
        lambda_max = 1.0

    identity = np.eye(num_nodes, dtype=np.float32)

    scaled_laplacian = (2.0 * laplacian / lambda_max) - identity

    return scaled_laplacian.astype(np.float32)

def get_chebyshev_polynomials(
    scaled_laplacian: np.ndarray,
    k_order: int,
) -> np.ndarray:
    """
    生成 Chebyshev 多项式矩阵。

    T_0 = I
    T_1 = L_tilde
    T_k = 2 * L_tilde * T_{k-1} - T_{k-2}

    :param scaled_laplacian: 缩放拉普拉斯矩阵，形状为 (num_nodes, num_nodes)
    :param k_order: Chebyshev 阶数 K
    :return: 形状为 (k_order, num_nodes, num_nodes)
    """
    if scaled_laplacian.ndim != 2:
        raise ValueError("缩放拉普拉斯矩阵必须是二维矩阵")

    if scaled_laplacian.shape[0] != scaled_laplacian.shape[1]:
        raise ValueError("缩放拉普拉斯矩阵必须是方阵")

    if k_order <= 0:
        raise ValueError("k_order 必须是正整数")

    num_nodes = scaled_laplacian.shape[0]

    cheb_polynomials = []

    # T_0 = I
    cheb_polynomials.append(np.eye(num_nodes, dtype=np.float32))

    if k_order == 1:
        return np.stack(cheb_polynomials, axis=0)

    # T_1 = L_tilde
    cheb_polynomials.append(scaled_laplacian.astype(np.float32))

    # T_k = 2LT_{k-1} - T_{k-2}
    for k in range(2, k_order):
        t_k = (
            2 * scaled_laplacian @ cheb_polynomials[k - 1]
            - cheb_polynomials[k - 2]
        )
        cheb_polynomials.append(t_k.astype(np.float32))

    return np.stack(cheb_polynomials, axis=0)

def build_graph_data(file_path, k_order):
    """
    构建图数据。

    :param file_path: 距离矩阵文件路径
    :param k_order: Chebyshev 阶数 K
    :return: Chebyshev 多项式矩阵
    """
    distance_matrix = get_distance_matrix(file_path)
    adjacency_matrix = get_adjacency_matrix(distance_matrix)
    laplacian = get_normalized_laplacian(adjacency_matrix)
    scaled_laplacian = get_scaled_laplacian(laplacian)
    chebyshev_polynomials = get_chebyshev_polynomials(scaled_laplacian, k_order)

    return {
        "distance_matrix": distance_matrix,
        "adjacency_matrix": adjacency_matrix,
        "normalized_laplacian": laplacian,
        "scaled_laplacian": scaled_laplacian,
        "chebyshev_polynomials": chebyshev_polynomials,
    }


if __name__ == "__main__":
    distance_matrix = get_distance_matrix("data/raw/PEMS04/distance.csv")
    adjacency_matrix = get_adjacency_matrix(distance_matrix)
    laplacian = get_normalized_laplacian(adjacency_matrix)
    scaled_laplacian = get_scaled_laplacian(laplacian)
    cheb = get_chebyshev_polynomials(scaled_laplacian, k_order=3)
    
    print("Distance Matrix:\n", distance_matrix)
    print("Adjacency Matrix:\n", adjacency_matrix)
    print("Normalized Laplacian:\n", laplacian)
    print("Scaled Laplacian:\n", scaled_laplacian)
    print("Chebyshev Polynomials:\n", cheb)    

    print(cheb.shape)
    print(cheb.dtype)
    print(np.isnan(cheb).any())
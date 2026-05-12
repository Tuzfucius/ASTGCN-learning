from astgcn.data.graph import build_graph_data

graph_data = build_graph_data("data\\raw\\PEMS04\\distance.csv", k_order=3)
# print("距离矩阵：", graph_data["distance_matrix"])
# print("邻接矩阵：", graph_data["adjacency_matrix"])
# print("归一化拉普拉斯矩阵：", graph_data["normalized_laplacian"])
# print("缩放拉普拉斯矩阵：", graph_data["scaled_laplacian"])
# print("Chebyshev 多项式矩阵：", graph_data["chebyshev_polynomials"])
print("距离矩阵：", graph_data["distance_matrix"].shape)
print("邻接矩阵：", graph_data["adjacency_matrix"].shape)
print("归一化拉普拉斯矩阵：", graph_data["normalized_laplacian"].shape)
print("缩放拉普拉斯矩阵：", graph_data["scaled_laplacian"].shape)
print("Chebyshev 多项式矩阵：", graph_data["chebyshev_polynomials"].shape)
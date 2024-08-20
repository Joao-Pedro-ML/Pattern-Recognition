import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parâmetros
a = 6
mean1 = np.array([0, 0, 0])
cov1 = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.01]])

means2 = np.array([
    [a, 0, 0],
    [a/2, a/2, 0],
    [0, a, 0],
    [-a/2, a/2, 0],
    [-a, 0, 0],
    [-a/2, -a/2, 0],
    [0, -a, 0],
    [a/2, -a/2, 0]
])
cov2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.01]])

# Gerar os dados
data_class1 = np.random.multivariate_normal(mean1, cov1, 100)
data_class2 = []
for mean in means2:
    data_class2.append(np.random.multivariate_normal(mean, cov2, 100))
data_class2 = np.vstack(data_class2)

# Combinar e criar os rótulos
data = np.vstack([data_class1, data_class2])
labels = np.hstack([np.zeros(100), np.ones(800)])

# Embaralhar os dados
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Calcular as matrizes de dispersão intra-classe (Sw) e entre-classe (Sb)
mean_total = np.mean(data, axis=0)
Sw = np.zeros((3, 3))
Sb = np.zeros((3, 3))

for c in [0, 1]:
    if c == 0:
        data_c = data[labels == 0]
    else:
        data_c = data[labels == 1]
    
    mean_c = np.mean(data_c, axis=0)
    Sw += np.cov(data_c, rowvar=False) * data_c.shape[0]
    n_c = data_c.shape[0]
    mean_diff = (mean_c - mean_total).reshape(3, 1)
    Sb += n_c * mean_diff.dot(mean_diff.T)

# Calcular os autovalores e autovetores de Sw^-1Sb
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

# Ordenar os autovalores e autovetores
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Projetar os dados nos autovetores correspondentes aos dois maiores autovalores
W = eigenvectors[:, :2]  # Seleciona os dois autovetores correspondentes aos maiores autovalores
lda_data = data.dot(W)

# Visualizar os dados projetados
plt.figure()
plt.scatter(lda_data[:, 0], lda_data[:, 1], c=labels, cmap='viridis', marker='o')
plt.title('LDA Projected Data using Eigenvectors of Sw^-1Sb')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(*plt.scatter(lda_data[:, 0], lda_data[:, 1], c=labels, cmap='viridis').legend_elements(), title="Classes")
plt.show()

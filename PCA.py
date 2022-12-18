import subprocess

import numpy as np
from matplotlib import pyplot as plt


# !chmod + x. / pcadata_colab.exe   # In case you can't read the file due to permission issues

# Process data to convert string data into a 100x10 matrix
def process_data():
    data = subprocess.run(["./pcadata_colab.exe", "23333"], stdout=subprocess.PIPE).stdout.decode("utf-8")
    data = data.replace("[", "").replace("]", "")
    data = data.replace('\n', ',')
    data = data.split(',')
    data.pop()  # Because we get an empty character at the end of the array after performing splitting
    data = np.array(data, dtype='float64')
    data = data.reshape(100, 10)
    return data


def find_covariance_matrix(data):
    data_mean = data - np.mean(data, axis=0)  # Standardization
    data = data_mean / np.std(data_mean, axis=0)
    C = (data.T @ data) / data.shape[0]
    return C


def find_eigen_vs(data, cov_m):
    w, V = np.linalg.eig(cov_m)
    trans_data = data @ V.T  # Transforming data
    return w, V, trans_data


if __name__ == "__main__":
    data = process_data()
    cov_m = find_covariance_matrix(data)
    eig_val, eig_vec, transf_data = find_eigen_vs(data, cov_m)
    print("Covariance matrix : ")
    print(cov_m)
    print("Top two eigen values are : ")
    print(eig_val[0:2])
    print("Top two eigen vectors are :")
    print(eig_vec[0:2])
    print("TRANSFORMED DATA _________")
    print(transf_data[:, [0, 1]])
    print("Plotting Data ")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.scatter(transf_data[:, 0], transf_data[:, 1])

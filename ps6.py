import numpy as np
import scipy
import pandas as pd
from sklearn.metrics import accuracy_score

# 0a: read data
df = pd.read_csv('input/iris_dataset.csv', header=None)

# convert to np array
iris = df.values

# define X and y of iris
iris_X = iris[:, :4]
iris_Y = iris[:, 4]

# 0b. Split shuffled data
np.random.shuffle(iris)
iris_X = iris[:125, :]

iris_X_train = iris[:125, :4]
iris_X_test = iris[125:, :4]

iris_Y_train = iris[:125, 4]
iris_Y_test = iris[125:, 4]

# 0c. Split X_train into subsets
train_1 = iris_X[iris_X[:, 4] == 1]
train_2 = iris_X[iris_X[:, 4] == 2]
train_3 = iris_X[iris_X[:, 4] == 3]

# Store new Training Data
X_train_1 = train_1[:, :4]
X_train_2 = train_2[:, :4]
X_train_3 = train_3[:, :4]

Y_train_1 = train_1[:, 4]
Y_train_2 = train_2[:, 4]
Y_train_3 = train_3[:, 4]

# print size
print("Size of X_train_1:", X_train_1.shape)
print("Size of X_train_2: ", X_train_2.shape)
print("Size of X_train_3: ", X_train_3.shape)

# 1a. Compute mean and sd for each feature per subset
mean1_0 = np.mean(X_train_1[:, 0])
mean1_1 = np.mean(X_train_1[:, 1])
mean1_2 = np.mean(X_train_1[:, 2])
mean1_3 = np.mean(X_train_1[:, 3])

mean2_0 = np.mean(X_train_2[:, 0])
mean2_1 = np.mean(X_train_2[:, 1])
mean2_2 = np.mean(X_train_2[:, 2])
mean2_3 = np.mean(X_train_2[:, 3])

mean3_0 = np.mean(X_train_3[:, 0])
mean3_1 = np.mean(X_train_3[:, 1])
mean3_2 = np.mean(X_train_3[:, 2])
mean3_3 = np.mean(X_train_3[:, 3])

sd1_0 = np.std(X_train_1[:, 0])
sd1_1 = np.std(X_train_1[:, 1])
sd1_2 = np.std(X_train_1[:, 2])
sd1_3 = np.std(X_train_1[:, 3])

sd2_0 = np.std(X_train_2[:, 0])
sd2_1 = np.std(X_train_2[:, 1])
sd2_2 = np.std(X_train_2[:, 2])
sd2_3 = np.std(X_train_2[:, 3])

sd3_0 = np.std(X_train_3[:, 0])
sd3_1 = np.std(X_train_3[:, 1])
sd3_2 = np.std(X_train_3[:, 2])
sd3_3 = np.std(X_train_3[:, 3])

# Print feature mean and std for each class
print("Class 1: Mean and Standard Deviation")
print(f"Feat. 1: {mean1_0:.3f}, {sd1_0:.3f}")
print(f"Feat. 2: {mean1_1:.3f}, {sd1_1:.3f}")
print(f"Feat. 3: {mean1_2:.3f}, {sd1_2:.3f}")
print(f"Feat. 4: {mean1_3:.3f}, {sd1_3:.3f}")
print("Class 2: Mean and Standard Deviation")
print(f"Feat. 1: {mean2_0:.3f}, {sd2_0:.3f}")
print(f"Feat. 2: {mean2_1:.3f}, {sd2_1:.3f}")
print(f"Feat. 3: {mean2_2:.3f}, {sd2_2:.3f}")
print(f"Feat. 4: {mean2_3:.3f}, {sd2_3:.3f}")
print("Class 3: Mean and Standard Deviation")
print(f"Feat. 1: {mean3_0:.3f}, {sd3_0:.3f}")
print(f"Feat. 2: {mean3_1:.3f}, {sd3_1:.3f}")
print(f"Feat. 3: {mean3_2:.3f}, {sd3_2:.3f}")
print(f"Feat. 4: {mean3_3:.3f}, {sd3_3:.3f}")

results = []
# 4b. i. calculate p(x_j|w_1) and ii. sum the logarithm of that value
for i in range(len(iris_X_test)):
    ln_1_x, ln_2_x, ln_3_x = 0, 0, 0

    for j in range(4):
        x_test_j = iris_X_test[i][j]

        # class 1
        mean = locals()[f'mean1_{j}']
        sd = locals()[f'sd1_{j}']
        p_x_w1 = 1 / (np.sqrt(2 * np.pi) * sd) * np.exp(-(x_test_j - mean) ** 2 / (2 * sd ** 2))
        ln_1_x += np.log(p_x_w1)

        # class 2
        mean = locals()[f'mean2_{j}']
        sd = locals()[f'sd2_{j}']
        p_x_w2 = 1 / (np.sqrt(2 * np.pi) * sd) * np.exp(-(x_test_j - mean) ** 2 / (2 * sd ** 2))
        ln_2_x += np.log(p_x_w2)

        # class 3
        mean = locals()[f'mean3_{j}']
        sd = locals()[f'sd3_{j}']
        p_x_w3 = 1 / (np.sqrt(2 * np.pi) * sd) * np.exp(-(x_test_j - mean) ** 2 / (2 * sd ** 2))
        ln_3_x += np.log(p_x_w3)

    # add log posterior probability
    ln_1_x += np.log(1.0/3.0)
    ln_2_x += np.log(1.0/3.0)
    ln_3_x += np.log(1.0/3.0)

    # classify on highest prob
    if (ln_1_x > ln_2_x and ln_1_x > ln_3_x):
        results.append(1)
    elif ln_2_x > ln_1_x and ln_2_x > ln_3_x:
        results.append(2)
    else:
        results.append(3)

# compute accuracy
accuracy = accuracy_score(iris_Y_test, results)

print("Accuracy on test set: ", accuracy)

# 2a. Get estimate of covariance matrix
sigma_1 = np.cov(X_train_1.T)
sigma_2 = np.cov(X_train_2.T)
sigma_3 = np.cov(X_train_3.T)

# Print Size and Snapshots
print("Covariance Size")
print("Sigma 1:", sigma_1.shape)
print("Sigma 2:", sigma_2.shape)
print("Sigma 3:", sigma_3.shape)

print("Covariance Matrix Sigma 1")
print(sigma_1)
print("Covariance Matrix Sigma 2")
print(sigma_2)
print("Covariance Matrix Sigma 3")
print(sigma_3)

# Find mean vectors for each class
mean_1 = np.array([[mean1_0], [mean1_1], [mean1_2], [mean1_3]])
mean_2 = np.array([[mean2_0], [mean2_1], [mean2_2], [mean2_3]])
mean_3 = np.array([[mean3_0], [mean3_1], [mean3_2], [mean3_3]])

# print
print("Size of Mean Vector 1")
print(mean_1.shape)
print("Size of Mean Vector 2")
print(mean_2.shape)
print("Size of Mean Vector 3")
print(mean_3.shape)

print("Mean Vector 1")
print(mean_1)
print("Mean Vector 2")
print(mean_2)
print("Mean Vector 3")
print(mean_3)

# Compute C1, C2, and C3
C1 = -(sigma_1.shape[0]/2) * np.log(2 * np.pi) - (0.5 * np.log(np.linalg.det(sigma_1)))
C2 = -(sigma_2.shape[0]/2) * np.log(2 * np.pi) - (0.5 * np.log(np.linalg.det(sigma_2)))
C3 = -(sigma_3.shape[0]/2) * np.log(2 * np.pi) - (0.5 * np.log(np.linalg.det(sigma_3)))

sigma_1_inv = np.linalg.inv(sigma_1)
sigma_2_inv = np.linalg.inv(sigma_2)
sigma_3_inv = np.linalg.inv(sigma_3)

# Compute g1, g2, and g3
g1 = np.array([-0.5 * (x.reshape(-1, 1) - mean_1).T @ sigma_1_inv @ (x.reshape(-1, 1) - mean_1) + np.log(1.0/3.0) + C1 for x in iris_X_test])
g2 = np.array([-0.5 * (x.reshape(-1, 1) - mean_2).T @ sigma_2_inv @ (x.reshape(-1, 1) - mean_2) + np.log(1.0/3.0) + C2 for x in iris_X_test])
g3 = np.array([-0.5 * (x.reshape(-1, 1) - mean_3).T @ sigma_3_inv @ (x.reshape(-1, 1) - mean_3) + np.log(1.0/3.0) + C3 for x in iris_X_test])


g = np.column_stack((g1, g2, g3))

# Predict using accuracy score
predicted = np.argmax(g, axis=1) + 1
accuracy = accuracy_score(iris_Y_test, predicted)
print("Accuracy of MLE:", accuracy)

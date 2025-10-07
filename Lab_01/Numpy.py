import numpy as np

vector1 = np.random.uniform(50, 150, 10)   
vector2 = np.random.uniform(1.5, 2.5, 10)  

matrix = np.column_stack((vector1, vector2))

max_index = np.argmax(matrix[:, 1])
corresponding_value = matrix[max_index, 0]

print("\nMatrix:\n", matrix)
print("\nMaximum value in second column:", matrix[max_index, 1])
print("\nCorresponding value in first column:", corresponding_value)

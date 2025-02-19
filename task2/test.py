import torch
import numpy as np
from collections import deque


def main():
    """
    tensor1 = torch.tensor([[0.1, 0.5, 0.2, 0.1, 0.05, 0.05],
                            [0.2, 0.2, 0.4, 0.1, 0.05, 0.05],
                            [0.0, 0.3, 0.3, 0.3, 0.05, 0.05],
                            [0.25, 0.25, 0.25, 0.1, 0.1, 0.05]])

    tensor2 = torch.tensor([[0.2, 0.4, 0.2, 0.1, 0.05, 0.05],
                            [0.3, 0.1, 0.4, 0.1, 0.05, 0.05],
                            [0.1, 0.2, 0.3, 0.3, 0.05, 0.05],
                            [0.2, 0.3, 0.2, 0.1, 0.15, 0.05]])

    tensor3 = torch.tensor([[0.3, 0.3, 0.1, 0.1, 0.1, 0.1],
                            [0.4, 0.1, 0.3, 0.1, 0.05, 0.05],
                            [0.15, 0.25, 0.2, 0.25, 0.05, 0.1],
                            [0.2, 0.25, 0.25, 0.1, 0.1, 0.1]])
    
    tensors = [tensor1, tensor2, tensor3]

    # Stack the tensors into a single tensor of shape (3, N, 6)
    stacked_tensors = torch.stack(tensors)

    # Compute the mean along the first dimension (averaging the distributions)
    average_probabilities1 = torch.mean(stacked_tensors, dim=0)
    average_probabilities2 = torch.mean(stacked_tensors, dim=1)

    print(average_probabilities1)
    print(average_probabilities2)
    """
    """
    num_rows = 4  # Number of rows in the matrix
    matrix_deque = deque(maxlen=num_rows)
    for i in range(6):  # Suppose we have 6 columns
        # Create a new column (as a NumPy array or a list)
        new_column = torch.tensor([i for _ in range(num_rows)])  # Example column
        print(new_column.shape)

        # Append the new column to the deque
        matrix_deque.append(new_column)

    # Convert the deque to a NumPy array for easy manipulation
    print(matrix_deque)
    final_matrix = torch.transpose(torch.stack(list(matrix_deque)), 0, 1)  # Transpose to get the correct shape

    print(final_matrix)
    print(final_matrix.size())
    print(final_matrix.ndim)
    """
    x = torch.randint(low=0, high=10, size=(3,2,2))
    print(x)
    means = torch.mean(x.float(), dim=2)
    print(means)
  

if __name__ == "__main__":
    main()

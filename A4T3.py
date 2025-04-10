import numpy as np
import scipy.special

def scaled_dot_product_attention(Q, K, V):
    # Step 1: Compute the dot product of Q and Kᵀ
    matmul_qk = np.dot(Q, K.T)
    
    # Step 2: Scale the result by dividing by √d, where d is the dimension of K
    d_k = K.shape[1]  # Get the dimension of the key (the number of columns in K)
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)
    
    # Step 3: Apply softmax to get attention weights
    attention_weights = scipy.special.softmax(scaled_attention_logits, axis=-1)
    
    # Step 4: Multiply the attention weights by V to get the final output
    output = np.dot(attention_weights, V)
    
    return attention_weights, output

# Test input matrices
Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# Apply scaled dot-product attention
attention_weights, output = scaled_dot_product_attention(Q, K, V)

# Display results
print("Attention Weights Matrix (after softmax):")
print(attention_weights)

print("\nFinal Output Matrix:")
print(output)

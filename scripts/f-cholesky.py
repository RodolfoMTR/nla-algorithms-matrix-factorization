import numpy as np
import pandas as pd
np.set_printoptions(formatter={'float': lambda x: f"{x:.4e}"})
def cholesky(A):
    n = A.shape[0]

    # Verificar simetria
    if not np.allclose(A, A.T):
        raise ValueError('La matriz no es simetrica')

    # Verificar definida positiva
    if np.min(np.linalg.eigvals(A)) <= 0:
        raise ValueError('La matriz no es positiva definida')

    H = np.zeros((n,n))

    for k in range(n):
        for i in range(k):
            suma1 = sum(H[i,j]*H[k,j] for j in range(i))
            H[k,i] = (A[k,i] - suma1) / H[i,i]

        suma2 = sum(H[k,j]**2 for j in range(k))
        H[k,k] = np.sqrt(A[k,k] - suma2)

    return H
if __name__ == "__main__":
    # Un ejemplo pequeno para probar que el script funciona solo
    A = np.array([[4,1,1],
              [1,3,0],
              [1,0,2]], float)
    H=cholesky(A)
    print('A\n',A)
    print('matriz H\n',H)
    print('Matriz H.T\n',H.T)
    print('A=H*H.T\n',H@H.T)

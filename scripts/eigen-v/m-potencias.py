import numpy as np
import pandas as pd
np.set_printoptions(formatter={'float': lambda x: f"{x:.4e}"})

#en tp se indica si se quiere el eige-value maximo o minimo con 'max' o 'min'
def metodo_potencias(A,x0,tol,tp='max'):
  err=np.inf
  if tp=='min':
    if np.min(np.linalg.eig(A)[0])<=0:
      raise ValueError("La matriz no es positiva definida")
    A=np.linalg.inv(A)
    while err>tol:
      xnp=A@x0
      xn=xnp/np.max(xnp)
      err= np.linalg.norm(xn-x0,2)
      x0=xn
  else:
    while err>tol:
      xnp=A@x0
      xn=xnp/np.max(xnp)
      err= np.linalg.norm(xn-x0,2)
      x0=xn
  return np.max(xnp)
if __name__ == "__main__":
    # Un ejemplo pequeno para probar que el script funciona solo
    A = np.array([
        [2, 1, 0],
        [1, 3, 1],
        [0, 1, 4]
    ], dtype=float)
    x0=np.array([[1],[1],[1]])
    tol=1e-4
    eig1=metodo_potencias(A,x0,tol)
    eig2=metodo_potencias(A,x0,tol,'min')
    print('eigen-value maximo:\n',eig1)
    print('eigen-value minimo:\n',1/eig2)
    print('--------------------')
    print('eigen-values con numpy:\n',np.linalg.eig(A)[0])

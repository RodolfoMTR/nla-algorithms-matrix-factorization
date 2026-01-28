import numpy as np
import pandas as pd
np.set_printoptions(formatter={'float': lambda x: f"{x:.4e}"})

def HouseHolder(x,n):
  if np.sign(x[[0]])==0:
    u=x+np.linalg.norm(x,2)*np.eye(n)[:,[0]]
    H=np.eye(n)-2*u@u.T/(u.T@u)
  else:
    u=x+np.sign(x[[0]])*np.linalg.norm(x,2)*np.eye(n)[:,[0]]
    H=np.eye(n)-2*u@u.T/(u.T@u)
  return H
def fact_QR(A):
  n=A.shape[0]
  Q=np.eye(n)
  for i in range(n-1):
    x=A[i:,[i]]
    H0=HouseHolder(x,n-i)
    H=np.eye(n)
    H[i:,i:]=H0
    Q=Q@H
  R=Q.T@A
  return Q,R

if __name__ == "__main__":
    # Un ejemplo pequeno para probar que el script funciona solo
    A=np.array([[-1,4,2,0],[0,3,1,-2],[3,1,5,1],[-2,0,1,4]])
    [Q,R]=fact_QR(A)
    print('Matriz original\n',A)
    print('Q\n',Q)
    print('----------------')
    print('R\n',R)
    print('----------------')
    print('Matriz Q*R\n',Q@R)


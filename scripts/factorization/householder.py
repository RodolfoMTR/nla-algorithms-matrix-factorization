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
if __name__ == "__main__":
    # Un ejemplo pequeno para probar que el script funciona solo
    x=np.array([[-1],[0],[3],[-2]])
    n=4
    H=HouseHolder(x,n)
    print('x\n',x)
    print('------------')
    print('H\n',H)
    print('------------')
    print('H*x\n',H@x)

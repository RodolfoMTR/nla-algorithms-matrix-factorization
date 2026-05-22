def forward_substitution(A, b):
    n = b.shape[0]
    y = np.zeros_like(b,dtype='float')
    y[0] = b[0] / A[0, 0]
    for i in range(n):
      suma=0
      for j in range(i):
        suma=suma+A[i,j]*y[j]
      y[i]=(b[i]-suma)/A[i,i]
    return y
def backward_substitution(A, b):
    n = b.shape[0]
    x = np.zeros_like(b,dtype='float')
    x[n-1] = b[n-1] / A[n-1, n-1]
    for i in range(n-2, -1, -1):
      suma=0
      for j in range(i+1,n):
        suma=suma+A[i,j]*x[j]
      x[i]=(b[i]-suma)/A[i,i]
    return x

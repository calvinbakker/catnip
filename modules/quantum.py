import numpy as np

def expi(x):
    return np.e**(1j*2*np.pi*x)

def sin(x):
    return np.sin(2*np.pi*x)

def cos(x):
    return np.cos(2*np.pi*x)

def unitary_matrix(a,b,c):
    a = a.astype(np.cdouble)
    b = b.astype(np.cdouble)
    c = c.astype(np.cdouble)
    return np.array([
        [expi(b)*cos(a),-expi(-c)*sin(a)],
        [expi(c)*sin(a), expi(-b)*cos(a)]
    ])

def check_unitary(U):
    l = U.shape[0]
    det = np.abs(np.linalg.det(U))
    if det!=0:
        U = U/np.sqrt(det)
        if np.sum(np.around((np.conj(U).T)@U,2)==np.identity(l))==l**2:
            return "😉"
        else:
            raise ValueError("Not unitary, matrix-product problem.")
    else:
        raise ValueError("Not unitary, determinant problem.")

    
def vector_entanglement(v):
    w = v/np.sqrt(np.conj(v)@v)
    R = np.outer(w,w)
    i = np.identity(4)
    R2 = np.array([i[0],i[2]])@(np.array([i[0],i[2]])@R).T+np.array([i[1],i[3]])@(np.array([i[1],i[3]])@R).T

    _,s,_ = np.linalg.svd(R2)
    
    s = s[s>0]
    s = s/np.sum(s)
    ee = -np.sum(s*np.log2(s))
    return ee

def quantum_CA(g):
    U1 = unitary_matrix(g[0],g[1],g[2])
    U2 = unitary_matrix(g[3],g[4],g[5])*expi(g[6])
    O = np.zeros((2,2))
    C = np.block([
        [U1,O],
        [O,U2]])
    return C
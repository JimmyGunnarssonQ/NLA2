import numpy.linalg as nl
import numpy as np 
import numpy.random as nr
import scipy.linalg as sl
class ortho:
    def __init__(self,matrix):
        self.matrix = matrix
        self.mval, self.nval = self.matrix.shape

    def gramschmidt(self):
        a0 = self.matrix[:,0].tolist()
        a0/=nl.norm(a0)
        Q = [a0]
        for i in range(1,self.nval):
            a = self.matrix[:,i].tolist()
            newvec = (a - sum(np.multiply(u, np.dot(u,a)) for u in Q)).tolist()
            newvec/=nl.norm(newvec)
            Q.append(newvec)
        Q = np.array(Q).transpose()

        return Q
    
    def devi(self):
        GSmat = self.gramschmidt()
        numident = np.matmul(GSmat.transpose(), GSmat) 

        twonl.norm = abs(nl.norm(GSmat, 2) - 1)
        ident = nl.norm(numident- np.eye(self.nval), 2)
        eigenvalues,_ = nl.eig(numident)
        determinant = nl.det(numident)

        return f"2-nl.norm deviation: {twonl.norm} \n2-nl.norm deviation from identity: {ident} \nrange of eigenvalues: ({min(eigenvalues)}, {max(eigenvalues)}) \ndeterminant: {determinant}"
    
    def householder(self):
        
        for i in range(self.nval):
            x = self.matrix[:,i]
            print(x)
            newvec = [sign(x[0])*nl.norm(x)]
            for j in range(x.shape[0]-1):
                newvec.append(0)
            v = newvec + x 


def sign(x):
    if x<0:
        s=-1 
    else:
        s=1
    return s 
val = 3
matin = nr.rand(val+2,val)
mat = ortho(matin)

#print(matin)
#q,_ = sl.qr(matin, mode = "economic")
#print(q)
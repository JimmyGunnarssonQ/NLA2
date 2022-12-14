import numpy.linalg as nl
import numpy as np 
import numpy.random as nr
import scipy.linalg as sl
class ortho:
    def __init__(self,matrix):
        self.matrix = matrix
        self.mval, self.nval = self.matrix.shape

    def gramschmidt(self):
        '''Gram Schmidt method, generates an orthogonal matrix Q '''
        a0 = self.matrix[:,0].tolist()
        a0/=nl.norm(a0)
        Q = [a0] #initial vector 


        for i in range(1,self.nval):
            a = self.matrix[:,i].tolist() #pegging value 
            newvec = (a - sum(np.multiply(u, np.dot(u,a)) for u in Q)).tolist() #gram Schmidt method 
            newvec/=nl.norm(newvec) #normalisation 
            Q.append(newvec) #appending the new column vector 
        Q = np.array(Q).transpose() #transpose is needed because we work with row vector internally 

        return Q
    
    def devi(self):
        ''' test run for deviations'''
        GSmat = self.gramschmidt()
        numident = np.matmul(GSmat.transpose(), GSmat) 

        twonl.norm = abs(nl.norm(GSmat, 2) - 1)
        ident = nl.norm(numident- np.eye(self.nval), 2)
        eigenvalues,_ = nl.eig(numident)
        determinant = nl.det(numident)

        return f"2-nl.norm deviation: {twonl.norm} \n2-nl.norm deviation from identity: {ident} \nrange of eigenvalues: ({min(eigenvalues)}, {max(eigenvalues)}) \ndeterminant: {determinant}"
    
    def householder(self):
        '''Householder method, used to generate an upper triangular matrix R '''
        dummymatrix = self.matrix #dummy matrix to be used for Householder reflections
        R = self.matrix #we fill it up with Householder reflected submatricies 


        for i in range(self.nval):
            a = dummymatrix[:,0] #vector to be modified for reflection
            mpart, npart = dummymatrix.shape
            v = a- np.array([nl.norm(a)] + (mpart-1)*[0]) #vector used for reflection axis
            if nl.norm(v)<1e-15:
                pass 
            else:
                v/=nl.norm(v) #normalisation 


                Q = np.eye(mpart) - np.multiply(np.outer(v,v),2) #reflection matrix 
                dummymatrix = np.matmul(Q, dummymatrix) #performing the Household reflection
                R[i:,i:] = dummymatrix #updating values
                dummymatrix = dummymatrix[1:, 1:] #new itteration matrix submatrix with dim m-1 x n-1 


        return R[:self.nval,] #reduced form 

    def givens(self):
        '''Given's rotations'''
        dummymatrix = self.matrix #matrix to be modified 
        for i in range(self.mval):
            for j in range(i+1,self.nval):
                rotmat = np.eye(self.mval, self.mval) 
                a = dummymatrix[i,i]
                b = dummymatrix[j,i]
                r = np.sqrt(a**2 + b**2)
                if abs(r)<= 1e-15: #to not overflow, values already 0
                    pass 
                elif b == 0: #special cases 
                    cos = a/r 
                    sin = 0 
                elif a == 0: #ditto
                    cos = 1 
                    sin = b/r 
                else: #if we do not have special cases 
                    cos = a/r 
                    sin = -b/r 

                    rotmat[i,i] = cos 
                    rotmat[j,j] = cos 
                    rotmat[i,j] = -sin 
                    rotmat[j,i] = sin #setting up Given's rotation matrix 


                    dummymatrix = rotmat@dummymatrix #for new itt run 
        R = dummymatrix[:self.nval,] #reduced form 
        return R 




'''Some testing values for verifications '''
val = 3
matin = nr.rand(val+2,val)
mat = ortho(matin)
Q = mat.gramschmidt()
R = mat.householder()
Rg = mat.givens()

print(Rg)
''' may be used to compare our home cooked Q,R'''
q,r = nl.qr(matin)
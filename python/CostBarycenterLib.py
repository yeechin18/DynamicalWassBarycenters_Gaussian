# -*- coding: utf-8 -*-
import scipy.linalg as spLin
import numpy as np
import torch
from MatrixSquareRoot import *


class WassersteinPSD(): #in practice (in dwb.py) an identity matrix of p.dim size is passed to an instantiation of this class.
    # Compute the cost with respect to the Wasserstein distance
    def __init__(self, baryS0, baryIter=2):
        self.baryIter = baryIter
        self.baryS0 = baryS0 #is this the reference covariance matrix? 
        
    def distance(self, A, B):
        return WassersteinPSD.dist(A,B) #EQUATION2 see below

    def dist(A,B):
        if (torch.is_tensor(A)):
            sqrtA = MatrixSquareRoot.apply(A)
            tmp = torch.matmul(torch.matmul(sqrtA, B), sqrtA)
            return torch.sqrt(torch.trace(A+B-2*MatrixSquareRoot.apply(tmp)))
        else:
            dim = int(np.sqrt(A.size))
            A = np.reshape(A, (dim,dim))
            B = np.reshape(B, (dim,dim))
            sqrtA = spLin.sqrtm(A)
            tmp = MatrixMultiply([sqrtA, B, sqrtA])
            return torch.sqrt(np.trace(A+B-2*spLin.sqrtm(tmp))) #EQUATION2 second term concerning only covariances in Wasserstein-2 distance between gaussian distributions with covariance matrix A,B
        
    def barycenter(self, covs, weights): # calculates the covariance of the barycentre... or something
        Sn_rt = []
        Sn_rt.append(MatrixSquareRoot.apply(self.baryS0))
        for i in range(self.baryIter):
            Sn_rt.append(MatrixSquareRoot.apply(
                                    torch.mm(torch.mm(torch.inverse(Sn_rt[i]), 
                                    torch.matrix_power(torch.einsum('a,abc->bc',weights, 
                                    MatrixSquareRootT.apply(torch.matmul(torch.matmul(Sn_rt[i], covs), Sn_rt[i]))), 2)), 
                                    torch.inverse(Sn_rt[i]))))
        return torch.mm(Sn_rt[-1],Sn_rt[-1])


class Euclidean():
    def distance(self, A, B):
        return Euclidean.dist(A,B) #EQUATION2
        
    def dist(A, B):
        if (torch.is_tensor(A)):
            return torch.norm(A-B, p='fro')
        else:
            return np.norm(A-B,ord='fro') #EQUATION2 - this would be used to calculate the first term concerning only means in wasserstein2 distance between two gaussians

        
    def barycenter(self, covs, weights):
        if (len(covs.size())==2):
            return torch.einsum('a,ab->b',weights, covs)
        elif (len(covs.size())==3):
            return torch.einsum('a,abc->bc',weights, covs)
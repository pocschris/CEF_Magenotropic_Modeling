# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:24:52 2022

@author: Chris
"""

import numpy as np
from numpy.linalg import matrix_power as powm
from numpy.linalg import eig
from tensorflow.linalg import expm
from scipy.optimize import fsolve
from scipy.interpolate import interp1d


# Numerical Physical constants that are globals
muB = 5.78828e-2  # [meV/T];
kB  = 8.617e-2  # [meV/K];
muBcgs = 9.274e-21 # cgs muB
NA = 6.022e23 # Avogadro
C0 = 2.0416


def generate_J_mat(J):
    dim = int(2*J+1)
    m = np.linspace(J,-J,dim)
    v = np.sqrt(J*(J+1) - m[:-1]*(m[:-1]-1))
    
    # Using the basis {J_z,J_+,J_-}
    mat = np.zeros((3,dim,dim))
    mat[0,:,:] = np.diag(m)
    mat[1,:-1,1:] = np.diag(v)
    mat[2,1:,:-1] = np.diag(v)
    return mat

def generate_stev_D3d(J_mat):
    dim = J_mat.shape[1]
    J = (dim-1)/2
    X = J*(J+1)
    
    iden = np.identity(dim)
    Jz = J_mat[0]
    Jp = J_mat[1]
    Jm = J_mat[2]
    A = powm(Jp,3) + powm(Jm,3)
    
    # using the basis {O20, O40, O60, O43, O63, O66}
    mat = np.zeros((6,dim,dim))
    mat[0] = 3* Jz @ Jz - X*iden
    mat[1] = 35*powm(Jz,4) - (30*X - 25)*Jz@Jz + (3*X*X - 6*X)*iden
    mat[2] = (1/4)*( A @ Jz + Jz @ A )
    mat[3] = 231*powm(Jz,6) - (315*X-735)*powm(Jz,4) + (105*X*X - 525*X +294)*powm(Jz,2) - (5*X*X*X + 40*X*X -60*X)*iden
    mat[4] = (1/4)*( A @ (11*powm(Jz,3) - (3*X + 59)*Jz ) + (11*powm(Jz,3) -(3*X + 59)*Jz) @ A )
    mat[5] = (1/2)*(powm(Jp,6) + powm(Jm,6))
    return mat

def tr(X):
    return np.real(np.trace(X,axis1=1,axis2=2))

class model_CEF:
    def __init__(self, params, exch, J=7/2, gJ=8/7, z=6):
        self.scaling = params[0]
        self.exch = exch
        self.Bmn  = params[1:7]
        self.gJ = gJ
        self.z = z
        self.dim = int(2*J+1)
        self.J_mat =  generate_J_mat(J)
        self.O_mat = generate_stev_D3d(self.J_mat)
        self.HCEF = self.compute_HCEF()
        self.E0 = min(eig(self.HCEF)[0])
        
        
    def compute_HCEF(self):
        HCEF = np.zeros((self.dim,self.dim));
        for i in range(self.Bmn.shape[0]):
            HCEF += self.Bmn[i]*self.O_mat[i]
        return HCEF
    
    
    def compute_HMF(self,H,M):
        # Size of H & M should be (N,3)
        # outputs a dim by dim matrix at each H, output has size (N,dim,dim)
        dim = self.dim
        N = H.shape[0]
        iden = np.identity(dim)
        
        Jx = (self.J_mat[1] + self.J_mat[2])/2
        Jy = (self.J_mat[1] - self.J_mat[2])/(2j)
        Jz = self.J_mat[0]
        
        HMF = np.zeros((N,dim,dim))
        HMF[:] = self.HCEF
        
        temp1 = np.sum((M @ self.exch) * M, 1);
        E1 = (-self.z/2)*temp1[:,None,None]*iden[None,:,:]
        
        temp2 = -(muB*self.gJ*H - (self.z/2) * M @ (self.exch + self.exch.T) )
        E2 = temp2[:,0][:,None,None]*Jx[None,:,:] 
        + temp2[:,1][:,None,None]*Jy[None,:,:]
        + temp2[:,2][:,None,None]*Jz[None,:,:]
        
        HMF += E1 + E2
        return HMF


    def compute_MvH(self,H,T):
        # Size of H should be (N,3)
        # Specify a single temperature T in [K]
        B = 1./(kB*T)
        dim = self.dim
        N = H.shape[0]
        iden = np.identity(dim)
        
        E0 = self.E0
        Jx = (self.J_mat[1] + self.J_mat[2])/2
        Jy = (self.J_mat[1] - self.J_mat[2])/(2j)
        Jz = self.J_mat[0]

        ef = lambda M: expm(-B*(self.compute_HMF(H,M) - E0*iden ))      
        f0 = lambda M: (tr(ef(M) @ Jx)/tr(ef(M))) - M[:,0]
        f1 = lambda M: (tr(ef(M) @ Jy)/tr(ef(M))) - M[:,1]
        f2 = lambda M: (tr(ef(M) @ Jz)/tr(ef(M))) - M[:,2]
        f3 = lambda Mg: np.concatenate([f0(np.reshape(Mg,(N,3),'F')),
                                        f1(np.reshape(Mg,(N,3),'F')),
                                        f2(np.reshape(Mg,(N,3),'F'))])
        
        Mg = np.zeros(N*3) 
        Mg = fsolve(f3,Mg)

        return np.reshape(Mg,(N,3),'F')
    
    def compute_kvH(self,H1d,T,Th,Ph=0,dTh=.01) :
        # Size of H should be (N,3)
        # Specify a single temperature T in [K]
        B = 1./(kB*T)
        dim = self.dim
        N = H1d.shape[0]
        iden = np.identity(dim)
        M = np.zeros((N,3))
        H = np.zeros((N,3))
        F = np.zeros((N,3))
        
        it=0
        for th in [Th-dTh,Th,Th+dTh]:
            H[:,0] = H1d*np.cos(Ph)*np.sin(th)
            H[:,1] = H1d*np.sin(Ph)*np.sin(th)
            H[:,2] = H1d*np.cos(th)
            
            M = self.compute_MvH(H,T)
            HMF = self.compute_HMF(H,M)
            Z = tr(expm(-B*(HMF - self.E0*iden )))
            F[:,it] = -kB*T*np.log(Z) + self.E0
            it +=1
            
        return (F[:,0] -2*F[:,1] + F[:,2])/(dTh*dTh)
        
 
class model_CEF_HighSymm(model_CEF):
    
    def __init__(self, params, exch, J=7/2, gJ=8/7, z=6):
        super().__init__(params, exch, J, gJ, z)
     
    def compute_MabvH(self,H1d,T):
        B = 1./(kB*T)
        dim = self.dim
        gJ = self.gJ
        A1 = self.z*kB/(muB*gJ);
        E0 = self.E0
        Jx = (self.J_mat[1] + self.J_mat[2])/2
        
        iden = np.identity(dim) 
        
        HpZ = self.HCEF - gJ*muB*H1d[:,None,None]*Jx[None,:] - E0*iden
        ef   = expm(-B*HpZ)
        mx   = tr(ef @ Jx)/tr(ef)
        
        X1 = H1d + self.exch[0,0]*A1*mx
        f_int = interp1d(np.concatenate([[0],X1]),
                         np.concatenate([[0],gJ*muB*mx]))
        return f_int(H1d)
    
    
    def compute_McvH(self,H1d,T):
        B = 1./(kB*T)
        dim = self.dim
        gJ = self.gJ
        A1 = self.z*kB/(muB*gJ);
        E0 = self.E0
        Jz = self.J_mat[0]
        
        iden = np.identity(dim) 
        
        HpZ = self.HCEF - gJ*muB*H1d[:,None,None]*Jz[None,:] - E0*iden
        ef   = expm(-B*HpZ)
        mz   = tr(ef @ Jz)/tr(ef)
        
        X1 = H1d + self.exch[0,0]*A1*mz
        f_int = interp1d(np.concatenate([[0],X1]),
                         np.concatenate([[0],gJ*muB*mz]))
        return f_int(H1d)
    
    
    def compute_XTabvH(self,H1d,T,hs=.01):    
        B = 1./(kB*T)
        dim = self.dim
        gJ = self.gJ
        A1 = self.z*kB/(muB*gJ);
        E0 = self.E0
        Jx = (self.J_mat[1] + self.J_mat[2])/2
        Jz = self.J_mat[0]
        
        iden = np.identity(dim) 
        
        HpZ = self.HCEF - gJ*muB*H1d[:,None,None]*Jz[None,:] - E0*iden
        ef   = expm(-B*HpZ)
        mz   = tr(ef @ Jz)/tr(ef)
        mxT0 = tr(ef @ Jx)/tr(ef)
        
        HpZ -= gJ*muB*hs*Jx
        ef   = expm(-B*HpZ)
        mxT1 = tr(ef @ Jx)/tr(ef)
        
        X1 = H1d + self.exch[2,2]*A1*mz
        chiT = (mxT1-mxT0)/(hs + self.exch[0,0]*A1*(mxT1-mxT0))
        f_int = interp1d(np.concatenate([[0],X1]),
                         np.concatenate([chiT[0],chiT]))
        return f_int(H1d)
    
         
    def compute_XTcvH(self,H1d,T,hs=.01):    
        B = 1./(kB*T)
        dim = self.dim
        gJ = self.gJ
        A1 = self.z*kB/(muB*gJ);
        E0 = self.E0
        Jx = (self.J_mat[1] + self.J_mat[2])/2
        Jz = self.J_mat[0]
        
        iden = np.identity(dim) 
        
        HpZ = self.HCEF - gJ*muB*H1d[:,None,None]*Jx[None,:] - E0*iden
        ef   = expm(-B*HpZ)
        mx   = tr(ef @ Jx)/tr(ef)
        mzT0 = tr(ef @ Jz)/tr(ef)
        
        HpZ -= gJ*muB*hs*Jz
        ef   = expm(-B*HpZ)
        mzT1 = tr(ef @ Jz)/tr(ef)
        
        X1 = H1d + self.exch[0,0]*A1*mx
        chiT = (mzT1-mzT0)/(hs + self.exch[2,2]*A1*(mzT1-mzT0))
        f_int = interp1d(np.concatenate([[0],X1]),
                         np.concatenate([chiT[0],chiT]))
        return f_int(H1d)
    
        
    def compute_kabvH(self,H1d,T,hs=.01):    
        B = 1./(kB*T)
        dim = self.dim
        gJ = self.gJ
        A1 = self.z*kB/(muB*gJ);
        E0 = self.E0
        Jx = (self.J_mat[1] + self.J_mat[2])/2
        Jz = self.J_mat[0]
        
        iden = np.identity(dim) 
        
        HpZ = self.HCEF - gJ*muB*H1d[:,None,None]*Jx[None,:] - E0*iden
        ef   = expm(-B*HpZ)
        mx   = tr(ef @ Jx)/tr(ef)
        mzT0 = tr(ef @ Jz)/tr(ef)
        
        HpZ -= gJ*muB*hs*Jz
        ef   = expm(-B*HpZ)
        mzT1 = tr(ef @ Jz)/tr(ef)
        
        X1 = H1d + self.exch[0,0]*A1*mx
        X2 = np.power(X1,2)
        chiT = (mzT1-mzT0)/(hs + self.exch[2,2]*A1*(mzT1-mzT0))
        kvm  = gJ*muB*(X1*mx - X2*chiT) 
        f_int = interp1d(np.concatenate([[0],X1]),
                         np.concatenate([[0],kvm]))
        return f_int(H1d)
    
    
    def compute_kcvH(self,H1d,T,hs=.01):    
        B = 1./(kB*T)
        dim = self.dim
        gJ = self.gJ
        A1 = self.z*kB/(muB*gJ);
        E0 = self.E0
        Jx = (self.J_mat[1] + self.J_mat[2])/2
        Jz = self.J_mat[0]
        
        iden = np.identity(dim) 
        
        HpZ = self.HCEF - gJ*muB*H1d[:,None,None]*Jz[None,:] - E0*iden
        ef   = expm(-B*HpZ)
        mz   = tr(ef @ Jz)/tr(ef)
        mxT0 = tr(ef @ Jx)/tr(ef)
        
        HpZ -= gJ*muB*hs*Jx
        ef   = expm(-B*HpZ)
        mxT1 = tr(ef @ Jx)/tr(ef)
        
        X1 = H1d + self.exch[2,2]*A1*mz
        X2 = np.power(X1,2)
        chiT = (mxT1-mxT0)/(hs + self.exch[0,0]*A1*(mxT1-mxT0))
        kvm  = gJ*muB*(X1*mz - X2*chiT) 
        f_int = interp1d(np.concatenate([[0],X1]),
                         np.concatenate([[0],kvm]))
        return f_int(H1d)
        
    
    





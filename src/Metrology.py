'''
Author: yesunhuang yesunhuang@uchicago.edu
Date: 2022-10-15 15:45:23
LastEditors: yesunhuang yesunhuang@uchicago.edu
LastEditTime: 2022-10-15 23:58:26
FilePath: \Quantum-Measurement-and-Metrology\src\Metrology.py
Description: Implement the program for the class quantum measurement

Copyright (c) 2022 by yesunhuang yesunhuang@uchicago.edu, All Rights Reserved. 
'''

#import everything
from typing import List, Tuple
import numpy as np
import qutip as qt

def convert_to_vector(H2matrix:qt.Qobj)->np.ndarray:
    '''
    description: convert the 2x2 hermitian matrix into vector\n
    param {qt.Oobj} H2matrix 2xw hermitian matrix\n
    return {np.ndarray} the vector in the 0.5 pauli basis\n
    '''
    basis=[qt.qeye(2),qt.sigmax(),\
                qt.sigmay(),qt.sigmaz()]
    objVector=[qt.expect(H2matrix,base) for base in basis]
    return np.asarray(objVector,dtype=complex)

def stack_operator(basis:List[qt.Qobj],
                    former:List[qt.Qobj])->List[qt.Qobj]:  
    '''
    description: tensor the former operators with basis to get new operators\n
    param {List} basis the basis operators\n
    param {List} former the former operators\n
    return {List} the stacked operators
    '''
    newOps=[]
    for formerOp in former:
        for base in basis:
            newOps.append(qt.tensor(formerOp,base))
    return newOps

def qubits_process_tomography(rhoIn:List[qt.Qobj],
                                rhoOut:List[qt.Qobj],
                                basis:List[qt.Qobj]=None)\
                                ->Tuple[np.ndarray,List[qt.Qobj]]:
    '''
    description: process tomography of two qubits\n
    param {List} rhoIn The input states\n
    param {List} rhoOut The output states\n
    param {List} basis The tomography basis, defaut for pauli basis\n
    return {Tuple} the results (chi,E)
    '''
    d=rhoIn[0].shape[0]
    n=int(np.log2(d))
    if basis==None:
        pBasis=[qt.qeye(2),qt.sigmax(),\
                qt.sigmay(),qt.sigmaz]
        basis=[]
        for _ in range(0,n-1):
            basis=stack_operator(pBasis,basis)
    W=[]
    for base_i in basis:
        for base_j in basis:
            Wk=[]
            for rho_k in rhoIn:
                rs=base_i*rho_k*base_j.dag()
                #rs [dxd]
                #rs.flatten [1xd^2]
                Wk.append(rs.full().flatten())
            #Wk[d^2xd^2]
            #Wk.faltten [1xd^4]
            W.append(np.asarray(Wk).flatten())
    #W[d^2xd^2]
    W=np.asarray(W)
    #rho_i.faltten [1xd^2]
    rhoOutM=[rho_i.full().flatten() for rho_i in rhoOut]
    #rhoOutM [d^2xd^2]
    #rhoOutM.reshape [1,d^4]
    rhoOutM=np.asarray(rhoOutM).reshape((1,d**4))
    chi=np.linalg.solve(W.transpose(),rhoOutM.transpose())
    chi=chi.reshape((d**2,d**2))
    #calculate the eigenvectors and eigen values
    d,M_d=np.linalg.eigh(chi)
    D=np.diag(d)
    #print(D)
    #print(M_d)
    M=M_d@np.sqrt(D)
    #print(M)
    #basisM [dxdxd^2]
    basisM=np.asarray([base.full() for base in basis]).transpose((1,2,0))
    #print(basisM.shape)
    #E [d^2xdxd]
    #E=(basisM@M).transpose((2,0,1))
    #print(M[:,0])
    E=basisM@M
    #print(E[:,:,0])
    E=E.transpose((2,0,1))
    #print(E[0])
    Eops=[qt.Qobj(op) for op in E]
    return chi, Eops

def apply_process_E(E:List[qt.Qobj],rho:qt.Qobj):
    '''
    description: apply the diagonal tomography operator\n
    param {List} E diagonal operators\n
    param {qt.Qobj} rho the density matrix\n
    return {qt.Qobj} the density matrix after the process
    '''
    rho_new=0
    for op in E:
        rho_new=rho_new+op*rho*op.dag()
    return rho_new

def apply_process_Chi(chi:np.ndarray,base:List[qt.Qobj],rho:qt.Qobj):
    '''
    description: apply the diagonal tomography operator\n
    param {chi} chi the chi matrix\n
    param {List} base base operators\n
    param {qt.Qobj} rho the density matrix\n
    return {qt.Qobj} the density matrix after the process
    '''
    rho_new=0
    for i in range(0,chi.shape[0]):
        for j in range(0,chi.shape[1]):
            rho_new=rho_new+chi[i,j]*base[i]*rho*base[j].dag()
    return rho_new
        

        


    

    

    
        
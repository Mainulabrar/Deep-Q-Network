# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 23:02:04 2019
treatment planning optimization with fixed parameters 
@author: writed by Chenyang Shen, modified by Chao Wang
"""
INPUT_SIZE = 100  # DVH interval number


import numpy as np
#import time
from numpy import zeros, sqrt, sort
from numpy import linalg as LA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from scipy.sparse import csr_matrix
import skcuda.linalg as linalg
import skcuda.misc as misc
import math 
import time

def MinimizeDoseOAR_dvh(MPTV, MBLA, MREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC, xVec,gamma,pdose,maxiter):
    beta=2
    lambdaBLA = lambdaBLA/lambdaPTV
    lambdaREC = lambdaREC/lambdaPTV
    #the current MPTV is stored in csc format, so here we transfer it to csr format. The csc format will be used as MPTVT for dot product in cuda later
    MPTVcsr=csr_matrix(MPTV)
    MBLAcsr=csr_matrix(MBLA)
    MRECcsr=csr_matrix(MREC)
    #test
    #print(MPTVcsr.data.shape[0])
    #for i in range(10):
    #    print("MPTV.indices, MPTV.indptr",MPTV.indices[i],MPTV.indptr[i],MPTVcsr.indices[i],MPTVcsr.indptr[i])
    #print(MPTVcsr.indptr.shape[0])
    print(MPTVcsr.shape[0])
    #print(MPTVcsr.indptr[0],MPTVcsr.indptr[1])
    print(MPTVcsr.dtype)

    mod=SourceModule("""
    #include <thrust/sort.h>
    #include <thrust/binary_search.h>
    #include <thrust/device_vector.h>
    #include <thrust/execution_policy.h>

    struct sparseStru{
        int nrows;
        float *data;
        int *indices, *indptr, *nrowS;
    };
    struct xVecAll{
 //       float *new,*old,*xVecPTV1,*xVecPTV2, *xVecBLA, *xVecREC, *xVecR, *xVecP, *xVecAP;
       int nrows;
    };
    struct dosePTV{
        float *d, *y, *dy_sorted, *gamma;
        int *dy_indices, posi;
    };
    struct doseOAR{
        float *d;
        int *d_indices, posi;
    };
    struct para{
        float lamPTV,lamBLA,lamREC,vPTV,vBLA,vREC,tPTV,tBLA,tREC;
    };
    extern "C"{
    __global__ void dotsortDose(float *data,int *indices, int *indptr, int nrows,float *xVec,float *d,int flag, float *d_sorted)
    {
    /****** d=M.dot(xVec) and d_sorted=sort(d)with d_indices ordered accordingly ****/
    /** flag =0, do the sorting; flag=1, no sorting; flag=2, only sorting. **/
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < nrows)
        {
            const int row_start = indptr[row];
            const int row_end = indptr[row + 1];
            //test struct copy to GPU
           /* if (row==0)
            {
                printf("%d, %d \\n", row_start, row_end);
            }*/
            float sum =0;
            //int i=0;
            for (int element = row_start; element < row_end; element++)
            {
                sum += data[element] * xVec[indices[element]];
                // test if data correctly copyed to GPU
                /*if (row==0 & i<5)
                {
                    printf("position=%d, M_data=%f, M_indptr=%d \\n", element,data[element],indptr[element]);
                    i+=1;
                }*/
            }
            d[row] = sum;
            if (flag==0)
            {
                d_sorted[row]=sum;
            }
        }
        __syncthreads();

        if (row== 0)
        {
            if (flag==0)
            {
               // thrust::stable_sort_by_key(thrust::device, d_sorted, d_sorted+nrows,d_indices);
                thrust::stable_sort(thrust::device, d_sorted, d_sorted+nrows);
              //  printf("d_sorted_0=%f, d_indices=%d, flag=%d\\n", d_sorted[0],d_indices[0], flag);
            }
         /*   else if (flag==1)
            {
                for (int i=0;i<1;i++)
                    printf("hello=====================, no sort, d=%f, ind=%d,flag=%d\\n", d[i],i,flag);
            }*/
        }
        __syncthreads();
    }
    __global__ void scaleDose(int nrows, float *xVec, int posi, float *d_sorted,float pdose, int flag, int nrows2, float *d, float *y,float *xVec_old)
    {
    /** scale the dose to D95=1, update xVec. If flag=1, update d, d_sorted, y as well***/
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ float factor;
        if (threadIdx.x==0)
            factor=pdose/d_sorted[posi];
        __syncthreads();
        if (row<nrows)
        {
            xVec[row]*=factor;
            xVec_old[row]=xVec[row];
        }

        //if (row==0)
          //  printf("before scale: d_sorted_0=%f, d_0=%f, d_95=%f, scaling_facotr=%f, flag=%d\\n",d_sorted[0], d[0], d_sorted[posi], factor,flag);
        if (flag==1)
        {
            if (row<nrows2)
            {
                d[row]*=factor;
                d_sorted[row]*=factor;
                y[row]=d[row];
            }
            __syncthreads();
            /*if (row==20)
            {
                printf("after scale:factor=%f, d_lens=%d, posi=%d, d_sorted_0=%f, d_0=%f, d_95=%f, flag=%d\\n",factor,nrows2, posi, d_sorted[0], d[0], d_sorted[posi], flag);
                printf("hello=================\\n");
            }*/
        }
        __syncthreads();

    }
    __global__ void nrowOverDose(int nrows, int *d_indices, int *rowShort, float *d,float posD)
    {
    /** to get how many rows are over given dose criteria. Total count will be returned, with corresponding row indices be sorted from small to large and returned. **/
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row<nrows)
        {
            if (posD<d[row])
            {
              //  printf("row, %d\\n", row);

               d_indices[row]=1;
               atomicAdd(rowShort, 1);
            }
            else
                d_indices[row]=0;
        }
        __syncthreads();
       /* if (row==0)
        {
            printf("total rows =%d, rows exceeding dose criteria=%d\\n", nrows,*nrowS);
            for (int i=0;i<*nrowS&i<100;i++)
            {
                if (d_indices[i]!=0)
                     printf("d_indices_over=%d,i=%d\\n", d_indices[i],i);
            }
        }*/

    }

    __global__ void overDose(float *r,float *r2,int flag, int nrows,int ncols, int *ncol2,int *d_indices,float *data,int *indices, int *indptr,float *y, float *d,float lam,float t)
    {
    /** to obtain the total over dose. Flag=1, r=r+MPTV_overT.dot(pdose*t-D_over)*lam/nrows;flag=2, r=r+MPTVT.dot(y-d)*lam/ncols; flag=0, temp=MPTV_over*p_over**/
        int row = blockIdx.x * blockDim.x + threadIdx.x;
       // if (row==0)
         //   printf("flag=%d,ncols=%d, ncol2=%d,d_indices_start=%d, d_indices_end=%d,lam=%f\\n", flag,ncols, *ncol2, d_indices[ncols-(*ncol2)],d_indices[ncols-1],lam);

        if (flag==0)
        {
            if (row<nrows)
            {
                if (d_indices[row]>0)
                {
                    const int row_start=indptr[row];
                    const int row_end=indptr[row+1];
                    float sum=0;
                    for (int element = row_start; element < row_end; element++)
                        sum += data[element] * y[indices[element]];
                    r[row]=r2[row]+sum*lam;
                 }

                else
                    r[row]=0;
            }
        }
        else if (flag==1|flag==2)
        {
            if (row<nrows)
            {
                const int row_start = indptr[row];
                const int row_end = indptr[row + 1];
                float sum =0;
                int i=0;
                for (int element = row_start; element < row_end; element++)
                {
                    if (flag==1)
                    {
                       if(d_indices[indices[element]]>0)
                       {
                            sum += data[element] * (t*y[indices[element]]-d[indices[element]]);

       /*                     if (row==0&i<5)
                            {
                                printf("element=%d,indices[element]=%d,data=%f,y=%f,d=%f \\n", element,indices[element],data[element],t*y[indices[element]],d[indices[element]]);
                                i+=1;
                            }*/
                        }
                    }
                    else if (flag==2)
                    {
                        sum += data[element] * (y[indices[element]]-d[indices[element]]);
                    }
                }
                if (flag==1)
                    r[row]=r2[row]+sum*lam/(*ncol2);
                else if (flag==2)
                    r[row]=r2[row]+sum*lam/ncols;
            }
        }
        __syncthreads();
    }

    __global__ void adding(int nrows, float *x, float *y, float *z, float t,int flag,float *xx)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
       // if (row==0)
      //      printf("x=%f,y=%f,z=%f,t=%f,flag=%d\\n",x[0],y[0],z[0],t,flag);
        if (row<nrows)
        {
            x[row]=y[row]+t*z[row];
            if (flag==1)
                x[row]=x[row]>=0?x[row]:0;
            if (flag==2)
            {
                xx[row]=x[row];
            }
        }
        __syncthreads();

        if (flag == 2)
        {
            if (row==0)
            {
                thrust::stable_sort(thrust::device, xx, xx+nrows);
            }
        }
       /* if (row==0)
        {
            printf("after:x=%f,y=%f,z=%f,t=%f,flag=%d\\n",x[0],y[0],z[0],t,flag);
        }*/
        __syncthreads();
    }
    __global__ void copying(int nrows, float *x, float *y)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row<nrows)
        {
            x[row]=y[row];
        }
        __syncthreads();
    }
    __global__ void ydose(int nrows, float *y, float d1,float d2)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
       // if (row==0)
         //   printf("nrows=%d,nrow2=%d, ind[0]=%d,d_indices_start=%d, d_indices_end=%d,d1=%f,d2=%f\\n", nrows,nrow2, ind[0], ind[nrow2],ind[nrows-1],d1,d2);
        if (row<nrows)
        {
            if (y[row]>=d1 &y[row]<d2)
                y[row]=d2;
        }
        __syncthreads();
    }

    }
        """,no_extern_c=True)

 ## transfer data from python to GPU with arrays and structures
    class sparseStru:
        def __init__(self,array):
            self.data = gpuarray.to_gpu(array.data.astype(np.float32))
            self.indices = gpuarray.to_gpu(array.indices.astype(np.int32))
            self.indptr =gpuarray.to_gpu(array.indptr.astype(np.int32))
            self.nrows=np.int32(array.indptr.shape[0]-1) # this is to accomodate both csr and csc formats
    class xVecAll:
        def __init__(self,xVec):
            self.nrows=np.int32(xVec.shape)
            self.new = gpuarray.to_gpu(xVec.astype(np.float32))
            self.old = gpuarray.to_gpu(xVec.astype(np.float32))
            self.PTV1 =gpuarray.to_gpu(np.empty_like(xVec).astype(np.float32))
            self.PTV2 =gpuarray.to_gpu(np.empty_like(xVec).astype(np.float32))
            self.BLA = gpuarray.to_gpu(np.empty_like(xVec).astype(np.float32))
            self.REC =gpuarray.to_gpu(np.empty_like(xVec).astype(np.float32))
            self.R = gpuarray.to_gpu(np.empty_like(xVec).astype(np.float32))
            self.P =gpuarray.to_gpu(np.empty_like(xVec).astype(np.float32))
            self.AP =gpuarray.to_gpu(np.empty_like(xVec).astype(np.float32))
    class dosePTV:
        def __init__(self,gamma):
            self.d = gpuarray.to_gpu(np.empty_like(gamma).astype(np.float32))
            self.y =gpuarray.to_gpu(np.empty_like(gamma).astype(np.float32))
            self.dy_sorted =gpuarray.to_gpu(np.empty_like(gamma).astype(np.float32))
            self.dy_indices = gpuarray.to_gpu(np.empty_like(gamma).astype(np.int32))
            self.y_sorted =gpuarray.to_gpu(np.empty_like(gamma).astype(np.float32))
            self.y_indices = gpuarray.to_gpu(np.empty_like(gamma).astype(np.int32))
            self.gamma =gpuarray.to_gpu(gamma.astype(np.float32))
            self.posi=np.int32(np.round(0.05*np.float32(gamma.shape)))
    class doseOAR:
        def __init__(self,OAR):
            self.d = gpuarray.to_gpu(np.zeros(OAR.shape[0]).astype(np.float32))
            self.d_sorted = gpuarray.to_gpu(np.zeros(OAR.shape[0]).astype(np.float32))
            self.d_indices =gpuarray.to_gpu(np.zeros(OAR.shape[0]).astype(np.int32))
            self.posi=np.int32(np.round(0.05*np.float32(OAR.shape[0])))
    class para:
        def __init__(self,lambdaPTV,lambdaBLA,lambdaREC,VPTV,VBLA,VREC,tPTV,tBLA,tREC):
            self.lamPTV=np.float32(lambdaPTV)
            self.lamBLA=np.float32(lambdaBLA)
            self.lamREC=np.float32(lambdaREC)
            self.vPTV=np.float32(VPTV)
            self.vBLA=np.float32(VBLA)
            self.vREC=np.float32(VREC)
            self.tPTV=np.float32(tPTV)
            self.tBLA=np.float32(tBLA)
            self.tREC=np.float32(tREC)
            self.beta=np.float32(beta)

    # start here to initiate all GPU memories
    MPTV_gpu = sparseStru(MPTVcsr)
    MPTVT_gpu= sparseStru(MPTV)
    MBLA_gpu = sparseStru(MBLAcsr)
    MBLAT_gpu= sparseStru(MBLA)
    MREC_gpu = sparseStru(MRECcsr)
    MRECT_gpu= sparseStru(MREC)
    xVec_gpu=xVecAll(xVec)
    #test
    #for i in range(MPTVcsr.indptr[0],MPTVcsr.indptr[1]):
     #   print(MPTVcsr.data[i],xVec[MPTVcsr.indices[i]])
    PTV_gpu=dosePTV(gamma)
    BLA_gpu=doseOAR(MBLA)
    REC_gpu=doseOAR(MREC)

    #container to store temporaty dose
    num=math.ceil(max(MPTV.shape[0],MBLA.shape[0],MREC.shape[0]))
    tempD_gpu=gpuarray.to_gpu(np.zeros(num).astype(np.float32))
    tempDzero_gpu=gpuarray.to_gpu(np.zeros(num).astype(np.float32))
    pdoseA_gpu=gpuarray.to_gpu((np.ones(num)*pdose).astype(np.float32))
    #iter, converge pointers
    iter=0
    iter_gpu = np.int32(iter)
    converge=0
    converge_gpu = np.int32(converge)
    # all other constant parameters
    para_gpu=para(lambdaPTV,lambdaBLA,lambdaREC,VPTV,VBLA,VREC,tPTV,tBLA,tREC)
    pdose_gpu=np.float32(pdose)

    # initialize threads and blocks
    ThreadNum=int(1024)
    BlockNumPTV=math.ceil(MPTV.shape[0]/ThreadNum)
    BlockNumBLA=math.ceil(MBLA.shape[0]/ThreadNum)
    BlockNumREC=math.ceil(MREC.shape[0]/ThreadNum)

    #dose scaling
    BlockNum1=math.ceil(max(MPTV.shape[0],MPTV.shape[1])/ThreadNum)
    BlockNum0=math.ceil(MPTV.shape[1]/ThreadNum)
    # define a value to scale the over dose values, since the r is too small represented in float32 format
    Sca=1000;

    # load GPU kernels
    func_dose = mod.get_function("dotsortDose")
    func_sPTV=mod.get_function("scaleDose")
    func_nrowOver=mod.get_function("nrowOverDose")
    func_overDose=mod.get_function("overDose")
    func_adding=mod.get_function("adding")
    func_copying=mod.get_function("copying")
    func_ydose=mod.get_function("ydose")
    stream1=cuda.Stream()
    stream2=cuda.Stream()
    stream3=cuda.Stream()
    # run kernels, start the GPU computation
 #   print("nrows= ", MPTV_gpu.nrows,"ThreadNum= ", ThreadNum, "BlockNumPTV=", BlockNumPTV)
    func_dose(MPTV_gpu.data,MPTV_gpu.indices, MPTV_gpu.indptr, MPTV_gpu.nrows,xVec_gpu.new,PTV_gpu.d,np.int32(0),PTV_gpu.dy_sorted,block = (ThreadNum, 1, 1), grid=(BlockNumPTV, 1))
  #  print("PTVdose[0]=", PTV_gpu.d[0], "PTVdose_sorted[0]= ", PTV_gpu.dy_sorted[0])
    func_sPTV(xVec_gpu.nrows, xVec_gpu.new,PTV_gpu.posi,PTV_gpu.dy_sorted,np.float32(pdose),np.int32(1), MPTV_gpu.nrows,PTV_gpu.d,PTV_gpu.y,xVec_gpu.old,block = (ThreadNum, 1, 1), grid=(BlockNum1, 1))

    start_time = time.time()
    #start the iteration
    for iter in range(maxiter):
        print("iteration====================================================================",iter)
        func_copying(MPTVT_gpu.nrows,xVec_gpu.old,xVec_gpu.new,block =(ThreadNum,1,1),grid=(BlockNum0,1))
        # DBLA=MBLA.dot(xVec) DBLA_sorted=sort(DBLA)
        func_dose(MBLA_gpu.data,MBLA_gpu.indices, MBLA_gpu.indptr, MBLA_gpu.nrows,xVec_gpu.new,BLA_gpu.d,np.int32(0),BLA_gpu.d_sorted,block = (ThreadNum, 1, 1), grid=(BlockNumBLA, 1))
           # DBLA=MREC.dot(xVec) DREC_sorted=sort(DREC)
        func_dose(MREC_gpu.data,MREC_gpu.indices, MREC_gpu.indptr, MREC_gpu.nrows,xVec_gpu.new,REC_gpu.d,np.int32(0),REC_gpu.d_sorted,block = (ThreadNum, 1, 1), grid=(BlockNumREC, 1))
       # r=-MPTVT.dot(gamma)
        rowShort=gpuarray.to_gpu(np.array(0).astype(np.int32))
        func_overDose(xVec_gpu.R,tempDzero_gpu,np.int32(2),MPTVT_gpu.nrows,MPTV_gpu.nrows,rowShort,PTV_gpu.dy_indices,MPTVT_gpu.data,MPTVT_gpu.indices,MPTVT_gpu.indptr,PTV_gpu.gamma,tempDzero_gpu,np.float32(-Sca),np.float32(1),block=(ThreadNum,1,1), grid=(BlockNum0,1))
       # r=r+MPTVT.dot((y-DPTV)*beta/y.shape[0])
    #__global__ void overDose(float *r,int flag, int nrows,int ncols, int *ncol2,int *d_indices,float *data,int *indices, int *indptr,float *y, float *d,float lam,float t)
        func_overDose(xVec_gpu.R,xVec_gpu.R,np.int32(2),MPTVT_gpu.nrows,MPTV_gpu.nrows,rowShort,PTV_gpu.dy_indices,MPTVT_gpu.data,MPTVT_gpu.indices,MPTVT_gpu.indptr,PTV_gpu.y,PTV_gpu.d,np.float32(beta*Sca),np.float32(1),block=(ThreadNum,1,1), grid=(BlockNum0,1))
 #       print("PTV_dmax", PTV_gpu.dy_sorted[-1].get(),pdose*tPTV)
        if PTV_gpu.dy_sorted[-1].get()>pdose*tPTV:
            posi = int(round((1 - VPTV) * MPTV.shape[0]))-1
            if posi < 0:
                posi = 0
            DPTVV = max(PTV_gpu.dy_sorted[posi].get(), pdose * tPTV)
            rowShortP=gpuarray.to_gpu(np.array(0).astype(np.int32))
            func_nrowOver(MPTV_gpu.nrows,PTV_gpu.dy_indices,rowShortP,PTV_gpu.d,np.float32(DPTVV), block = (ThreadNum, 1, 1), grid=(BlockNumPTV, 1))

            print("rowShortP",rowShortP)
            #r=r+lambdaPTV*MPTV1T.dot(pdose*tPTV*np.ones((DPTVs.shape)) -DPTVs) / DPTVs.shape
            func_overDose(xVec_gpu.R,xVec_gpu.R,np.int32(1),MPTVT_gpu.nrows,MPTV_gpu.nrows,rowShortP,PTV_gpu.dy_indices,MPTVT_gpu.data, MPTVT_gpu.indices, MPTVT_gpu.indptr,pdoseA_gpu,PTV_gpu.d,np.float32(Sca*lambdaPTV),para_gpu.tPTV,block=(ThreadNum,1,1),grid=(BlockNum0,1))

        #  print("para",para_gpu.lamPTV,pdose_gpu,para_gpu.tPTV)

  #      print("BLA_dmax",BLA_gpu.d_sorted[-1].get(),pdose*tBLA)
        if BLA_gpu.d_sorted[-1].get()>pdose*tBLA:
            posi = int(round((1 - VBLA) * MBLA.shape[0]))-1
            if posi<0:
                posi=0
            DBLAV = max(BLA_gpu.d_sorted[posi].get(),pdose * tBLA)
            rowShortB=gpuarray.to_gpu(np.array(0).astype(np.int32))
            func_nrowOver(MBLA_gpu.nrows,BLA_gpu.d_indices,rowShortB, BLA_gpu.d,np.float32(DBLAV), block = (ThreadNum,1,1), grid=(BlockNumBLA, 1))
            print("rowShortB",rowShortB)
           # r = r+lambdaBLA * MBLA1T.dot(pdose*tBLA*np.ones((DBLAs.shape,))-DBLAs) / DBLAs.shape
            func_overDose(xVec_gpu.R,xVec_gpu.R,np.int32(1),MBLAT_gpu.nrows,MBLA_gpu.nrows,rowShortB,BLA_gpu.d_indices,MBLAT_gpu.data, MBLAT_gpu.indices, MBLAT_gpu.indptr, pdoseA_gpu,BLA_gpu.d,np.float32(lambdaBLA*Sca),para_gpu.tBLA,block=(ThreadNum,1,1),grid=(BlockNum0,1))

   #     print("REC_dmax",REC_gpu.d_sorted[-1].get(),pdose*tBLA)
        if REC_gpu.d_sorted[-1].get()>pdose*tREC:
            posi = int(round((1 - VREC) * MREC.shape[0]))-1
            if posi<0:
             posi=0
            DRECV = max(REC_gpu.d_sorted[posi].get(),pdose * tREC)
            rowShortR=gpuarray.to_gpu(np.array(0).astype(np.int32))
            func_nrowOver(MREC_gpu.nrows,REC_gpu.d_indices,rowShortR,REC_gpu.d,np.float32(DRECV),block = (ThreadNum, 1, 1), grid=(BlockNumREC, 1))

            print("rowShortR",rowShortR)
            #r = r+lambdaREC * MREC1T.dot(pdose*tREC*np.ones((DRECs.shape))-DRECs) / DRECs.shape
            func_overDose(xVec_gpu.R,xVec_gpu.R,np.int32(1),MRECT_gpu.nrows,MREC_gpu.nrows,rowShortR,REC_gpu.d_indices,MRECT_gpu.data,MRECT_gpu.indices,MRECT_gpu.indptr,pdoseA_gpu,REC_gpu.d,np.float32(lambdaREC*Sca),para_gpu.tREC,block = (ThreadNum,1,1), grid=(BlockNum0,1))

        func_copying(MPTVT_gpu.nrows,xVec_gpu.P,xVec_gpu.R,block =(ThreadNum,1,1),grid=(BlockNum0,1))
        xVecR=xVec_gpu.R.get()
        rsold=np.inner(xVecR,xVecR)/Sca/Sca
        #rsold=linalg.dot(xVec_gpu.R.get(), xVec_gpu.R.get())/Sca/Sca
    #    for i in range(5):
     #       print("xVec_gpu.R[i]=",xVec_gpu.R[i],rsold)# this contains the scale factor "Sca"
        print("rsold=",rsold, "iter=", iter, "=========================")# this not

        if rsold>1e-10:
            for iloop in range (3):
                #tempD=MPTV.dot(p)
                func_dose(MPTV_gpu.data,MPTV_gpu.indices, MPTV_gpu.indptr, MPTV_gpu.nrows,xVec_gpu.P,tempD_gpu,PTV_gpu.dy_indices,np.int32(1),PTV_gpu.dy_sorted,block = (ThreadNum, 1, 1), grid=(BlockNumPTV, 1))
                #Ap=beta*MPTVT.dot(tempD)/y.shape[0]
                func_overDose(xVec_gpu.AP,tempDzero_gpu,np.int32(2),MPTVT_gpu.nrows,MPTV_gpu.nrows,rowShort,PTV_gpu.dy_indices,MPTVT_gpu.data,MPTVT_gpu.indices,MPTVT_gpu.indptr,tempD_gpu,tempDzero_gpu,np.float32(beta),np.float32(1),block =(ThreadNum,1,1),grid=(BlockNum0,1))
                if PTV_gpu.dy_sorted[-1].get()>pdose*tPTV:
                 #   temp = MPTV1.dot(p)
        #            print("BlockNum_overDose",BlockNumPO)
                    func_overDose(tempD_gpu,tempDzero_gpu,np.int32(0),MPTV_gpu.nrows,MPTV_gpu.nrows,rowShortP,PTV_gpu.dy_indices,MPTV_gpu.data,MPTV_gpu.indices,MPTV_gpu.indptr,xVec_gpu.P,tempDzero_gpu,np.float32(1),np.float32(1),block =(ThreadNum,1,1),grid=(BlockNumPTV,1))
        #            Ap = Ap+MPTV1T.dot(temp)* lambdaPTV / MPTV1.shape[0]
                    func_overDose(xVec_gpu.AP,xVec_gpu.AP,np.int32(1),MPTVT_gpu.nrows,MPTV_gpu.nrows,rowShortP,PTV_gpu.dy_indices,MPTVT_gpu.data,MPTVT_gpu.indices,MPTVT_gpu.indptr,tempD_gpu,tempDzero_gpu,np.float32(lambdaPTV),np.float32(1),block =(ThreadNum,1,1),grid=(BlockNum0,1))

                if BLA_gpu.d_sorted[-1].get()>pdose*tBLA:
         #           print("start Ap for BLA")
         #           temp = MBLA1.dot(p)
          #          print("BlockNum_overDoseBLA",BlockNumBO)
                    func_overDose(tempD_gpu,tempDzero_gpu,np.int32(0),MBLA_gpu.nrows,MBLA_gpu.nrows,rowShortB,BLA_gpu.d_indices,MBLA_gpu.data,MBLA_gpu.indices,MBLA_gpu.indptr,xVec_gpu.P,tempDzero_gpu,np.float32(1),np.float32(1),block =(ThreadNum,1,1),grid=(BlockNumBLA,1))
          #          Ap = Ap+MBLA1T.dot(temp)* lambdaBLA / MBLA1.shape[0]
                    func_overDose(xVec_gpu.AP,xVec_gpu.AP,np.int32(1),MBLAT_gpu.nrows,MBLA_gpu.nrows,rowShortB,BLA_gpu.d_indices,MBLAT_gpu.data,MBLAT_gpu.indices,MBLAT_gpu.indptr,tempD_gpu,tempDzero_gpu,np.float32(lambdaBLA),np.float32(1),block =(ThreadNum,1,1),grid=(BlockNum0,1))

                if REC_gpu.d_sorted[-1].get()>pdose*tREC:
                #    print("start Ap for REC")
           #         temp = MREC1.dot(p)
                    func_overDose(tempD_gpu,tempDzero_gpu,np.int32(0),MREC_gpu.nrows,MREC_gpu.nrows,rowShortR,REC_gpu.d_indices,MREC_gpu.data,MREC_gpu.indices,MREC_gpu.indptr,xVec_gpu.P,tempDzero_gpu,np.float32(1),np.float32(1),block =(ThreadNum,1,1),grid=(BlockNumREC,1))
                    #Ap = Ap+MREC1T.dot(temp)* lambdaREC / MREC1.shape[0]
                    func_overDose(xVec_gpu.AP,xVec_gpu.AP,np.int32(1),MRECT_gpu.nrows,MREC_gpu.nrows,rowShortR,REC_gpu.d_indices,MRECT_gpu.data,MRECT_gpu.indices,MRECT_gpu.indptr,tempD_gpu,tempDzero_gpu,np.float32(lambdaREC),np.float32(1),block =(ThreadNum,1,1),grid=(BlockNum0,1))
                P=xVec_gpu.P.get()
                AP=xVec_gpu.AP.get()
                pAp=np.inner(P,AP)/Sca/Sca
                #pAp = linalg.dot(xVec_gpu.P, xVec_gpu.AP)/Sca/Sca
      #          for i in range(5):
       #             print("xVec_gpu.AP[i]=",xVec_gpu.AP[i])
                #pAp = linalg.dot(xVec_gpu.P, xVec_gpu.AP)/Sca/Sca
                alpha = rsold / pAp
               # xVec = xVec + alpha * p
               # xVec[xVec<0]=0
                func_adding(MPTVT_gpu.nrows,xVec_gpu.new,xVec_gpu.new,xVec_gpu.P,np.float32(alpha/Sca),np.int32(1),block =(ThreadNum,1,1),grid=(BlockNum0,1))
               # r = r - alpha * Ap
                func_adding(MPTVT_gpu.nrows,xVec_gpu.R,xVec_gpu.R,xVec_gpu.AP,np.float32(-alpha),np.int32(0),block =(ThreadNum,1,1),grid=(BlockNum0,1))
                xVecR=xVec_gpu.R.get()
                rsnew=np.inner(xVecR,xVecR)/Sca/Sca
                #rsnew=linalg.dot(xVec_gpu.R, xVec_gpu.R)/Sca/Sca
                print("rnew=",rsnew,"iloop",iloop)
                if math.sqrt(rsnew) < 1e-5:
                    break
                #p = r + (rsnew / rsold) * p
                func_adding(MPTVT_gpu.nrows,xVec_gpu.P,xVec_gpu.R,xVec_gpu.P,np.float32(rsnew/rsold),np.int32(0),block =(ThreadNum,1,1),grid=(BlockNum0,1))
                rsold = rsnew

        func_dose(MPTV_gpu.data,MPTV_gpu.indices, MPTV_gpu.indptr, MPTV_gpu.nrows,xVec_gpu.new,PTV_gpu.d,np.int32(0),PTV_gpu.dy_sorted,block = (ThreadNum, 1, 1), grid=(BlockNumPTV, 1))
        func_adding(MPTV_gpu.nrows,PTV_gpu.y,PTV_gpu.d,PTV_gpu.gamma,np.float32(MPTV.shape[0]/beta),np.int32(2),PTV_gpu.y_sorted,block =(ThreadNum,1,1),grid=(BlockNumPTV,1))
        posi=int(0.05*float(MPTV.shape[0]))
        D95=PTV_gpu.y_sorted[posi].get()
        print("posi",posi,"D95_y",D95,"nrows",MPTV_gpu.nrows)
        if D95<pdose:
             #y[(y>=D95)&(y<pdose)]=pdose
            func_ydose(MPTV_gpu.nrows, PTV_gpu.y,np.float32(D95),np.float32(pdose),block =(ThreadNum,1,1),grid=(BlockNumPTV,1))
        func_adding(MPTV_gpu.nrows,PTV_gpu.gamma,PTV_gpu.gamma,PTV_gpu.d,np.float32(beta/MPTV.shape[0]),np.int32(0),block =(ThreadNum,1,1),grid=(BlockNumPTV,1))
        func_adding(MPTV_gpu.nrows,PTV_gpu.gamma,PTV_gpu.gamma,PTV_gpu.y,np.float32(-beta/MPTV.shape[0]),np.int32(0),block =(ThreadNum,1,1),grid=(BlockNumPTV,1))

        xVec=xVec_gpu.new.get()
        xVec_old=xVec_gpu.old.get()
        print( "LA.norm(xVec- xVec_old)", LA.norm(xVec- xVec_old),"LA.norm(xVec)",LA.norm(xVec),"LA.norm(xVec_old)",LA.norm(xVec_old))
        if LA.norm(xVec- xVec_old)/LA.norm(xVec)<5e-3:
            break;

    func_dose(MPTV_gpu.data,MPTV_gpu.indices, MPTV_gpu.indptr, MPTV_gpu.nrows,xVec_gpu.new,PTV_gpu.d,PTV_gpu.dy_indices,np.int32(0),PTV_gpu.dy_sorted,block = (ThreadNum, 1, 1), grid=(BlockNumPTV, 1))
    func_sPTV(xVec_gpu.nrows, xVec_gpu.new,PTV_gpu.posi,PTV_gpu.dy_sorted,np.float32(pdose),np.int32(0), MPTV_gpu.nrows,PTV_gpu.d,PTV_gpu.y,xVec_gpu.old,block = (ThreadNum, 1, 1), grid=(BlockNum1, 1))
    converge=1
    if iter == maxiter - 1:
        converge = 0
    print("optimization time: {} seconds for {} iterrations".format((time.time() - start_time), iter))

    #copy back to CPU
    xVec=xVec_gpu.new.get()
    gamma=PTV_gpu.gamma.get()

    return xVec, iter, converge, gamma



def runOpt_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter):
    # run optimization and generate DVH curves 
    xVec, iter, converge, gamma = MinimizeDoseOAR_dvh(MPTV, MBLA, MREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC, xVec, gamma,pdose,maxiter)
#    j = 0
    DPTV = MPTV.dot(xVec)
    DBLA = MBLA.dot(xVec)
    DREC = MREC.dot(xVec)
    DPTV = np.sort(DPTV)
    DPTV = np.flipud(DPTV)
    DBLA = np.sort(DBLA)
    DBLA = np.flipud(DBLA)
    DREC = np.sort(DREC)
    DREC = np.flipud(DREC)

    ## Plot DVH curve for optimized plan
    edge_ptv = np.zeros((INPUT_SIZE+1,))
    edge_ptv[1:INPUT_SIZE+1] = np.linspace(pdose,pdose*1.15, INPUT_SIZE)
    (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
    y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)

    edge_bladder = np.zeros((INPUT_SIZE+1,))
    edge_bladder[1:INPUT_SIZE+1] = np.linspace(0.6*pdose, 1.1*pdose, INPUT_SIZE)
    (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
    y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)

    edge_rectum = np.zeros((INPUT_SIZE+1,))
    edge_rectum[1:INPUT_SIZE+1] = np.linspace(0.6*pdose, 1.1*pdose, INPUT_SIZE)
    (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
    y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)

    Y = zeros((INPUT_SIZE,3))
    Y[:, 0] = y_ptv
    Y[:, 1] = y_bladder
    Y[:, 2] = y_rectum

    Y = np.reshape(Y,(100*3,),order = 'F')


    return Y, iter, xVec, gamma
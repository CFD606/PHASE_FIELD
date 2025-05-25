#ifndef CONVENIENCE_FUNCITON_H
#define CONVENIENCE_FUNCITON_H

#include"cuda_memory_and_transfer.h"


__device__ int index_4d (int i, int j,int k,int l)
{
//20200320	int ans=(nx+4)*((ny+4)*((nz/cpu+4)*l+k)+j)+i;
	int ans = i + (nx + 4)*(j + (ny + 4)*(k + (nz/cpu + 4)*l));
	return ans;
}

__device__ int index_3d (int i, int j, int k)
{
//20200320	int ans=(nx+4)*((ny+4)*k+j)+i;
	int ans = i + (nx + 4)*(j + k*(ny+4));
	return ans;
}

__global__ void array_do( double *phi_d, double *phi)
{
	// c[2,2,2] = c_d[0,0,0]

	// ii:0~nx-1, jj:0~ny-1, kk:0~nz/cpu-1 
	int iindex;
	int ii = threadIdx.x + blockIdx.x * blockDim.x;
	int jj = threadIdx.y + blockIdx.y * blockDim.y;
	int kk = threadIdx.z + blockIdx.z * blockDim.z;
	if (ii < nx && jj < ny && kk < nz/cpu) {
		iindex	= nx * (kk * ny + jj) + ii;
	}

	// i:2~nx+1, j:2~ny+1. k:2~nz/cpu+1
	int index;
	int i = threadIdx.x + blockIdx.x * blockDim.x+2;
	int j = threadIdx.y + blockIdx.y * blockDim.y+2;
	int k = threadIdx.z + blockIdx.z * blockDim.z+2;
//20200323	if (i <= nx+1 && j <= ny+1 && k <= nz/cpu+1) {
	if (i < nx + 2 && j < ny + 2 && k < nz/cpu + 2){
		index = index_3d(i,j,k);
	}
	
	phi[index] = phi_d[iindex];
}

__global__ void array_undo( double *phi_d, double *phi)
{
	// c_d[0,0,0] = c[2,2,2] 

	// ii:0~nx-1, jj:0~ny-1, kk:0~nz/cpu-1 
	int iindex;
	int ii = threadIdx.x + blockIdx.x * blockDim.x;
	int jj = threadIdx.y + blockIdx.y * blockDim.y;
	int kk = threadIdx.z + blockIdx.z * blockDim.z;
	if (ii < nx && jj < ny && kk < nz/cpu) {
		iindex	= nx * (kk * ny + jj) + ii;
	}

	// i:2~nx+1, j:2~ny+1. k:2~nz/cpu+1
	int index;
	int i = threadIdx.x + blockIdx.x * blockDim.x+2;
	int j = threadIdx.y + blockIdx.y * blockDim.y+2;
	int k = threadIdx.z + blockIdx.z * blockDim.z+2;
//20200323	if (i <= nx+1 && j <= ny+1 && k <= nz/cpu+1) {
	if (i < nx + 2 && j < ny + 2 && k < nz/cpu + 2){
		index = index_3d(i,j,k);
	}

	phi_d[iindex] = phi[index];
}

__global__ void transfer2xz( double *xz_d,double *c)
{
	int ii=threadIdx.x;
	int kk= blockIdx.x;
	int iindex	=nx*(kk)+ii;
	int i=threadIdx.x+2;
	int j=ny/2+2;
	int k= blockIdx.x+2;
	int index=index_3d(i,j,k);
	xz_d[iindex]=c[index];
}
__global__ void transfer2xz_sym( double *xz_d,double *c)
{
	int ii=threadIdx.x;
	int kk= blockIdx.x;
	int iindex	=nx*(kk)+ii;
	int i=threadIdx.x+2;
	int j=2;
	int k= blockIdx.x+2;
	int index=index_3d(i,j,k);
	xz_d[iindex]=c[index];
}

__global__ void init_u_vel( double *u,double *c)
{
	// i:0~nx+3, j:0~ny+3, k:0~nz/cpu+3
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	
	int idx = index_3d(i, j, k);
  	int icent = (double)(nx+4)/2.0;
	
	if ( initial_type==8 ){
		if(i>icent){
    		u[idx] = - c[idx] * u_coll_7/2.0;
  		}else{
    		u[idx] =   c[idx] * u_coll_7/2.0;
  		}
	}else{
		u[idx] = c[idx] * u_coll_9;
	}
	

}




void adjust_array_side_to_mid(){
	array_do <<<bpgBuff , tpbBuff>>>( c_d,c );
	array_do <<<bpgBuff , tpbBuff>>>( m_d,m );
	array_do <<<bpgBuff , tpbBuff>>>( p_d,p );
	array_do <<<bpgBuff , tpbBuff>>>( u_d,u );
	array_do <<<bpgBuff , tpbBuff>>>( v_d,v );
	array_do <<<bpgBuff , tpbBuff>>>( w_d,w );
	array_do <<<bpgBuff , tpbBuff>>>( a_d,a );
	CHECK_CUDA(cudaDeviceSynchronize());
}

void adjust_array_mid_to_side(){
	array_undo <<<bpgBuff , tpbBuff>>>( c_d,c );
	array_undo <<<bpgBuff , tpbBuff>>>( m_d,m );
	array_undo <<<bpgBuff , tpbBuff>>>( p_d,p );
	array_undo <<<bpgBuff , tpbBuff>>>( u_d,u );
	array_undo <<<bpgBuff , tpbBuff>>>( v_d,v );
	array_undo <<<bpgBuff , tpbBuff>>>( w_d,w );
	array_undo <<<bpgBuff , tpbBuff>>>( a_d,a );
	array_undo <<<bpgBuff , tpbBuff>>>( DE_d,DE );
	CHECK_CUDA(cudaDeviceSynchronize());
}



#endif
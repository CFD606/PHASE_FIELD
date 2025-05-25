#ifndef BOUNDARY_H
#define BOUNDARY_H




__global__ void boundary_zm2( double *phi, double *t_phi)
{
	// t_c[,,2] = c[,,2] 
	// t_c[,,3] = c[,,3]
	// t_c[,,4] = c[,,nz/cpu] 
	// t_c[,,5] = c[,,nz/cpu+1]

	// i:0~nx+3, j:0~ny+3, k:2,3,nz/cpu,nz/cpu+1
	int k, l, index, index_t;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
//20200323	
//	if (i >=0 && i <= nx+3 && j >= 0 && j <= ny+3) {
	if (i < nx + 4 && j < ny + 4){
		for (l = 0; l < 2; l++) {
			k = 2;
			index	= index_3d(i,j,k+l);
			index_t	= index_3d(i,j,2+l);
			t_phi[index_t] = phi[index];
			
			k = nz/cpu;
			index	= index_3d(i,j,k+l);
			index_t	= index_3d(i,j,4+l);
			t_phi[index_t] = phi[index];
		}
	}
}

__global__ void boundary_zm2_undo( double *phi, double *t_phi)
{
	// c[,,0] = t_c[,,0]
	// c[,,1] = t_c[,,1]
	// c[,,nz/cpu+2] = t_c[,,6]
	// c[,,nz/cpu+3] = t_c[,,7]

	// i:0~nx+3, j:0~ny+3, k:0,1,nz/cpu+2,nz/cpu+3
	int k, l, index, index_t;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >=0 && i <= nx+3 && j >= 0 && j <= ny+3) {
		for (l = 0; l < 2; l++) {
			k = 0;
			index	= index_3d(i,j,k+l);
			index_t	= index_3d(i,j,0+l);
			phi[index] = t_phi[index_t];
			
			k = nz / cpu + 2;
			index	= index_3d(i,j,k+l);
			index_t	= index_3d(i,j,6+l);
			
			phi[index] = t_phi[index_t];
		}
	}
}


__global__ void boundary_zm1( double *phi, double *t_phi)
{
	// t_c[,,1] = c[,,2] 
	// t_c[,,2] = c[,,nz/cpu+1]

	// i:0~nx+3, j:0~ny+3, k:2,nz/cpu+1
	int k, index, index_t;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (i >=0 && i <= nx+3 && j >= 0 && j <= ny+3) {
		k = 2;
		index	= index_3d(i,j,k);
		index_t	= index_3d(i,j,1);
		t_phi[index_t] = phi[index];

		k = nz / cpu + 1;
		index	= index_3d(i,j,k);
		index_t	= index_3d(i,j,2);
		t_phi[index_t] = phi[index];
	}
}

__global__ void boundary_zm1_undo( double *phi, double *t_phi)
{
	// c[,,1] = t_c[,,0]
	// c[,,nz/cpu+2] = t_c[,,3]
	// i:0~nx+3, j:0~ny+3, k:1,nz/cpu+2

	int k, index, index_t;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (i >=0 && i <= nx+3 && j >= 0 && j <= ny+3) {
		k = 1;
		index	= index_3d(i,j,k);
		index_t	= index_3d(i,j,0);
		phi[index] = t_phi[index_t];

		k = nz / cpu + 2;
		index	= index_3d(i,j,k);
		index_t	= index_3d(i,j,3);
		phi[index] = t_phi[index_t];
	}
}


__global__ void boundary_ym( double *phi)
{
	// c[,0,] = c[,ny,]
	// c[,1,] = c[,ny+1,]
	// c[,ny+2,] = c[,2,]
	// c[,ny+3,] = c[,3,]
	// i:0~nx+3, j:0,1,ny+2,ny+3, k:0~nz/cpu+3

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= 0 && i <= nx+3 && k >= 0 && k <= nz/cpu+3) {
		int distance = (ny) * (nx+4);
		for (int j = 0;j <= 1; j++) {
			int index  = index_3d(i,j,k);
			phi[index] = phi[index + distance];
		}
		for (int j = ny + 2; j <= ny + 3; j++) {
			int index  = index_3d(i,j,k);
			phi[index] = phi[index - distance];
		}
	}
}

__global__ void boundary_xm( double *phi)
{
	// c[0,,] = c[nx,,]
	// c[1,,] = c[nx+1,,]
	// c[nx+2,,] = c[,2,,]
	// c[nx+3,,] = c[3,,]
	// i:0,1,nx+2,nx+3, j:0~ny+3, k:0~nz/cpu+3

	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (j >= 0 && j <= ny+3 && k >= 0 && k <= nz/cpu+3) {
		int distance = nx;
		for (int i = 0; i <= 1; i++) {
			int index  = index_3d(i,j,k);
			phi[index] = phi[index + distance];
		}
		for (int i = nx + 2; i <= nx + 3; i++) {
			int index  = index_3d(i,j,k);
			phi[index] = phi[index - distance];
		}
	}
}

__global__ void boundary_ym_in( double *phi)
{
	// c[,0,] = c[,ny,]
	// c[,1,] = c[,ny+1,]
	// c[,ny+2,] = c[,2,]
	// c[,ny+3,] = c[,3,]
	// i:0~nx+3, j:0,1,ny+2,ny+3, k:4~nz/cpu-1

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (i >= 0 && i <= nx+3 && k >= 4 && k <= nz/cpu-1) {
		int distance = (ny) * (nx+4);
		for(int j = 0;j <= 1; j++) {
			int index  = index_3d(i,j,k);
			phi[index] = phi[index + distance];
		}
		for(int j = ny + 2; j <= ny+3; j++) {
			int index  = index_3d(i,j,k);
			phi[index] = phi[index - distance];
		}
	}
}

__global__ void boundary_xm_in( double *phi)
{
	// c[0,,] = c[nx,,]
	// c[1,,] = c[nx+1,,]
	// c[nx+2,,] = c[,2,,]
	// c[nx+3,,] = c[3,,]
	// i:0,1,nx+2,nx+3, j:0~ny+3, k:4~nz/cpu-1

	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;

	if (j >= 0 && j <= ny+3 && k >= 4 && k <= nz/cpu-1) {
		int distance = nx;
		for(int i = 0; i <= 1; i++) {
			int index  = index_3d(i,j,k);
			phi[index] = phi[index + distance];
		}
		for(int i = nx + 2; i <= nx + 3; i++) {
			int index  = index_3d(i,j,k);
			phi[index] = phi[index - distance];
		}
	}
}
 __global__ void boundary_ym_bc( double *phi)
{
	// c[,0,] = c[,ny,]
	// c[,1,] = c[,ny+1,]
	// c[,ny+2,] = c[,2,]
	// c[,ny+3,] = c[,3,]
	// i:0~nx+3, j:0,1,ny+2,ny+3, k:2,3,nz/cpu,nz/cpu+1

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (i >= 0 && i <= nx+3) {
		int kk[4] = {2, 3, nz/cpu, nz/cpu+1};
		int distance = (ny) * (nx+4);
		for (int t = 0;t < 4; t++) {
			int k = kk[t];
			for (int j = 0;j <= 1; j++) {
				int index  = index_3d(i,j,k);
				phi[index] = phi[index + distance];
			}
			for (int j = ny + 2; j <= ny + 3; j++) {
				int index  = index_3d(i,j,k);
				phi[index] = phi[index - distance];
			}
		}
	}
}

__global__ void boundary_xm_bc( double *phi)
{
	// c[0,,] = c[nx,,]
	// c[1,,] = c[nx+1,,]
	// c[nx+2,,] = c[,2,,]
	// c[nx+3,,] = c[3,,]
	// i:0,1,nx+2,nx+3, j:0~ny+3, k:2,3,nz/cpu,nz/cpu+1

	int j = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (j >= 0 && j <= ny+3) {
		int kk[4] = {2, 3, nz/cpu, nz/cpu+1};
		int distance = nx;
		for (int t = 0; t < 4; t++) {
			int k = kk[t];
			for (int i = 0;i <= 1; i++) {
				int index  = index_3d(i,j,k);				
				phi[index] = phi[index + distance];
			}
			for (int i = nx + 2; i <= nx + 3; i++) {
				int index = index_3d(i,j,k);
				phi[index] = phi[index - distance];
			}
		}
	}
} 

__global__ void boundary_xd_in( double *g,double *h)
{
	// g[1,,,] = g[nx+1,,,]
	// g[nx+2,,,] = g[2,,,]
	// i:1,nx+2, j:1~ny+2, k:3~nz/cpu

	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;

	if (j >= 1 && j <= ny+2 && k >= 3 && k <= nz/cpu) {
		int i, index_l;
		int distance = nx;
		for (int l = 0;l < q; l++) {
			i = 1;
			index_l = index_4d(i,j,k,l);
			g[index_l] = g[index_l + distance];
			h[index_l] = h[index_l + distance];
			i = nx + 2;
			index_l = index_4d(i,j,k,l);
			g[index_l] = g[index_l - distance];
			h[index_l] = h[index_l - distance];
		}
	}
}

__global__ void boundary_xd_bc( double *g,double *h)
{
	// g[1,,,] = g[nx+1,,,]
	// g[nx+2,,,] = g[2,,,]
	// i:1,nx+2, j:1~ny+2, k:2,nz/cpu+1

	int j = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (j >= 1 && j <= ny+2) {
		int i, k, index_l;
		int distance = nx;
		for (int l = 0;l < q; l++) {
			i = 1;
			k = 2;
			index_l = index_4d(i,j,k,l);
			g[index_l] = g[index_l + distance];
			h[index_l] = h[index_l + distance];

			i = 1;
			k = nz / cpu + 1;
			index_l = index_4d(i,j,k,l);
			g[index_l] = g[index_l + distance];
			h[index_l] = h[index_l + distance];

			i = nx + 2;
			k = 2;
			index_l = index_4d(i,j,k,l);
			g[index_l] = g[index_l - distance];
			h[index_l] = h[index_l - distance];

			i = nx + 2;
			k = nz / cpu + 1;
			index_l = index_4d(i,j,k,l);
			g[index_l] = g[index_l - distance];
			h[index_l] = h[index_l - distance];
		}
	}
}

__global__ void boundary_yd_in( double *g,double *h)
{
	// g[,1,,] = g[,ny+1,,]
	// g[,nx+2,,] = g[,2,,]
	// i:1~nx+2, j:1,ny+2, k:3~nz/cpu

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (i >= 1 && i <= nx+2 && k >= 3 && k <= nz/cpu) {
		int j, index_l;
		int distance = (ny) * (nx+4);
		for (int l = 0; l < q; l++) {		
			j = 1;
			index_l = index_4d(i,j,k,l);
			g[index_l] = g[index_l + distance];
			h[index_l] = h[index_l + distance];
			j = ny + 2;
			index_l = index_4d(i,j,k,l);
			g[index_l] = g[index_l - distance];
			h[index_l] = h[index_l - distance];
		}
	}
}

__global__ void boundary_yd_bc( double *g,double *h)
{
	// g[,1,,] = g[,ny+1,,]
	// g[,nx+2,,] = g[,2,,]
	// i:1~nx+2, j:1,ny+2, k:2,nz/cpu+1

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (i >= 1 && i <= nx+2) {
		int j, k, index_l;
		int distance = (ny) * (nx+4);
		for (int l = 0;l < q; l++) {
		
		j = 1;
		k = 2;
		index_l = index_4d(i,j,k,l);
		g[index_l] = g[index_l + distance];
		h[index_l] = h[index_l + distance];

		j = 1;
		k = nz / cpu + 1;
		index_l = index_4d(i,j,k,l);
		g[index_l] = g[index_l + distance];
		h[index_l] = h[index_l + distance];

		j = ny + 2;
		k = 2;
		index_l = index_4d(i,j,k,l);
		g[index_l] = g[index_l - distance];
		h[index_l] = h[index_l - distance];

		j = ny + 2;
		k = nz / cpu + 1;
		index_l = index_4d(i,j,k,l);
		g[index_l] = g[index_l - distance];
		h[index_l] = h[index_l - distance];
		}
	}
}

__global__ void boundary_zd( double *phi,double *t_phi)
{
	// t_g[,,1,] = g[,,2,]
	// t_g[,,2,] = g[,,nz/cpu+1,]
	// i:1~nx+2, j:1~ny+2, k:2,nz/cpu+1 

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (i >= 1 && i <= nx+2 && j >= 1 && j <= ny+2) {
		if(q==19){
			int k, index_l, index_l_t;
			int l_bot[5] = {6, 12, 14, 16, 17}; 
			int l_top[5] = {5, 11, 13, 15, 18};
			for (int l = 0; l < 5; l++) {
				k = 2;
				index_l   = index_4d(i,j,k,l_bot[l]);
				//index_l_t = ((nx + 4) * (1 * (ny + 4) + j) + i) * 5 + l;
				index_l_t = index_3d(i,j,5+l);
				t_phi[index_l_t] = phi[index_l];

				k = nz / cpu + 1;
				index_l   = index_4d(i,j,k,l_top[l]);
				//index_l_t = ((nx + 4) * (2 * (ny + 4) + j) + i) * 5 + l;
				index_l_t = index_3d(i,j,10+l);
				t_phi[index_l_t] = phi[index_l];
			}
		}else if(q==27){
			int k, index_l, index_l_t;
			int l_bot[9] = {6, 12, 14, 16, 17, 20, 22, 24, 26}; 
			int l_top[9] = {5, 11, 13, 15, 18, 19, 21, 23, 25};
			for (int l = 0; l < 9; l++) {
				k = 2;
				index_l   = index_4d(i,j,k,l_bot[l]);
				//index_l_t = ((nx + 4) * (1 * (ny + 4) + j) + i) * 9 + l;
				index_l_t = index_3d(i,j,9+l);
				t_phi[index_l_t] = phi[index_l];

				k = nz / cpu + 1;
				index_l   = index_4d(i,j,k,l_top[l]);
				//index_l_t = ((nx + 4) * (2 * (ny + 4) + j) + i) * 9 + l;
				index_l_t = index_3d(i,j,18+l);
				t_phi[index_l_t] = phi[index_l];
			}
		}
	}
}
__global__ void boundary_zd_undo( double *phi,double *t_phi)
{
	// g[,,1,] = t_g[,,0,]
	// g[,,nz/cpu+2,] = t_g[,,3,]
	// i:1~nx+2, j:1~ny+2, k:1,nz/cpu+2

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (i >= 1 && i <= nx+2 && j >= 1 && j <= ny+2) {
		if(q==19){
			int k, index_l, index_l_t;
			int l_bot[5] = {6, 12, 14, 16, 17};
			int l_top[5] = {5, 11, 13, 15, 18};
			for (int l = 0;l < 5; l++) {
				k = 1;
				index_l   = index_4d(i,j,k,l_top[l]);
				//index_l_t = ((nx + 4) * (0 * (ny + 4) + j) + i) * 5 + l;
				index_l_t = index_3d(i,j,0+l);
				phi[index_l] = t_phi[index_l_t];

				k = nz / cpu + 2;
				index_l   = index_4d(i,j,k,l_bot[l]);
				//index_l_t = ((nx + 4) * (3 * (ny + 4) + j) + i) * 5 + l;
				index_l_t = index_3d(i,j,15+l);
				phi[index_l] = t_phi[index_l_t];
			}
		}else if(q==27){
			int k, index_l, index_l_t;
			int l_bot[9] = {6, 12, 14, 16, 17, 20, 22, 24, 26}; 
			int l_top[9] = {5, 11, 13, 15, 18, 19, 21, 23, 25};
			for (int l = 0;l < 9; l++) {
				k = 1;
				index_l   = index_4d(i,j,k,l_top[l]);
				//index_l_t = ((nx + 4) * (0 * (ny + 4) + j) + i) * 9 + l;
				index_l_t = index_3d(i,j,0+l);
				phi[index_l] = t_phi[index_l_t];
				
				k = nz / cpu + 2;
				index_l   = index_4d(i,j,k,l_bot[l]);
				//index_l_t = ((nx + 4) * (3 * (ny + 4) + j) + i) * 9 + l;
				index_l_t = index_3d(i,j,27+l);
				phi[index_l] = t_phi[index_l_t];
			}
		}
	}
}

/***********xyz_periodic**********/

/*macro C*/

void boundary_C_bc_xy_z_transfer(){
	boundary_ym_bc<<< bpgx,tpbx>>>(c);
	boundary_xm_bc<<< bpgy,tpby >>>(c);
	boundary_zm2<<<  bpgxy,tpbxy>>>(c,t_c);
}
void boundary_C_in(cudaStream_t stream){

	boundary_ym_in<<< bpgxz , tpbxz   ,0,stream>>>(c);
	boundary_xm_in<<< bpgyz , tpbyz   ,0,stream>>>(c);	
}
void boundary_C_z_transfer_back(cudaStream_t stream){
	boundary_zm2_undo<<< bpgxy,tpbxy,0,stream>>>(c,t_c);
}
/*macro chemical*/

void boundary_chemi_bc_xy_z_transfer(){
	boundary_ym_bc<<< bpgx,tpbx>>>(m);
	boundary_xm_bc<<< bpgy,tpby >>>(m);
	boundary_zm2<<<  bpgxy,tpbxy>>>(m,t_m);
}
void boundary_chemi_in(cudaStream_t stream){
	boundary_ym_in<<< bpgxz , tpbxz   ,0,stream>>>(m);
	boundary_xm_in<<< bpgyz , tpbyz   ,0,stream>>>(m);

}
void boundary_chemi_z_transfer_back(cudaStream_t stream){
	boundary_zm2_undo<<< bpgxy,tpbxy,0,stream>>>(m,t_m);
}
/*macro velo and P_hydro*/
void boundary_others_bc_xy_z_transfer(){


	boundary_ym_bc<<< bpgx,tpbx>>>(u);
	boundary_ym_bc<<< bpgx,tpbx>>>(v);
	boundary_ym_bc<<< bpgx,tpbx>>>(w);
	boundary_ym_bc<<< bpgx,tpbx>>>(p);


	boundary_xm_bc<<< bpgy,tpby >>>(u);
	boundary_xm_bc<<< bpgy,tpby >>>(v);
	boundary_xm_bc<<< bpgy,tpby >>>(w);
	boundary_xm_bc<<< bpgy,tpby >>>(p);

	boundary_zm2<<<  bpgxy,tpbxy>>>(p,t_p);
	boundary_zm1<<<  bpgxy,tpbxy>>>(u,t_u);
	boundary_zm1<<<  bpgxy,tpbxy>>>(v,t_v);
	boundary_zm1<<<  bpgxy,tpbxy>>>(w,t_w);		
}
void boundary_others_in(cudaStream_t stream){

	boundary_ym_in<<< bpgxz , tpbxz   ,0,stream>>>(p);
	boundary_ym_in<<< bpgxz , tpbxz   ,0,stream>>>(u);
	boundary_ym_in<<< bpgxz , tpbxz   ,0,stream>>>(v);
	boundary_ym_in<<< bpgxz , tpbxz   ,0,stream>>>(w);
	
	boundary_xm_in<<< bpgyz , tpbyz   ,0,stream>>>(p);
	boundary_xm_in<<< bpgyz , tpbyz   ,0,stream>>>(u);
	boundary_xm_in<<< bpgyz , tpbyz   ,0,stream>>>(v);
	boundary_xm_in<<< bpgyz , tpbyz   ,0,stream>>>(w);


}
void boundary_others_z_transfer_back(cudaStream_t stream){
	boundary_zm2_undo<<< bpgxy,tpbxy,0,stream>>>(p,t_p);
	boundary_zm1_undo<<< bpgxy,tpbxy,0,stream>>>(u,t_u);
	boundary_zm1_undo<<< bpgxy,tpbxy,0,stream>>>(v,t_v);
	boundary_zm1_undo<<< bpgxy,tpbxy,0,stream>>>(w,t_w);
}
/*distribution function*/

void boundary_distri_bc_xy_z_transfer(double*g,double*h,cudaStream_t stream){
	boundary_yd_bc<<< bpgx,tpbx,0,stream>>>(g,h);
	boundary_xd_bc<<< bpgy,tpby ,0,stream>>>(g,h);
	boundary_zd<<<  bpgxy,tpbxy,0,stream>>>(g,t_g);
	boundary_zd<<<  bpgxy,tpbxy,0,stream>>>(h,t_h);
}
void boundary_distri_in(double*g,double*h,cudaStream_t stream){
	boundary_yd_in<<< bpgxz , tpbxz   ,0,stream>>>(g,h);
	boundary_xd_in<<< bpgyz , tpbyz   ,0,stream>>>(g,h);
}
void boundary_distri_z_transfer_back(double*g,double*h,cudaStream_t stream){
	boundary_zd_undo<<< bpgxy,tpbxy,0,stream>>>(g,t_g);
	boundary_zd_undo<<< bpgxy,tpbxy,0,stream>>>(h,t_h);	
}


/****************************wall boundary******************************/

// =============
// z direction
// =============

// marco
__global__ void wall_zm2_undo( double *phi, double *t_phi,int myid, int lastp)
{
	// i:0~nx+3, j:0~ny+3, k:0,1,nz/cpu+2,nz/cpu+3
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k, l, index, index_t, index_in;
	
	if (i >=0 && i <= nx+3 && j >= 0 && j <= ny+3) {
		for (l = 0; l < 2; l++) {
			k = 0;
			index	= index_3d(i,j,k+l);
			index_t	= index_3d(i,j,0+l);
			index_in = index_3d(i,j,3-l);
			if (myid == 0){
				// c[,,0]= c[,,3]
				// c[,,1]= c[,,2]
        		phi[index] = phi[index_in];
			}else{
				// c[,,0] = t_c[,,0]
				// c[,,1] = t_c[,,1]
				// c[,,nz/cpu+2] = t_c[,,6]
				// c[,,nz/cpu+3] = t_c[,,7]

				phi[index] = t_phi[index_t];
			}
			
			k = nz / cpu + 2;
			index	= index_3d(i,j,k+l);
			index_t	= index_3d(i,j,6+l);
			index_in = index_3d(i,j,k-l-1);
			if (myid == lastp){
				// c[,,nz/cpu+2] = c[,,nz/cpu+1]
				// c[,,nz/cpu+3] = c[,,nz/cpu]
       			phi[index] = phi[index_in];
			}else{
				phi[index] = t_phi[index_t];
			}
		}
	}
}
__global__ void wall_zm1_undo_u( double *phi, double *t_phi,int myid, int lastp)
{
	// i:0~nx+3, j:0~ny+3, k:1,nz/cpu+2
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k, index, index_t, index_in;
	
	if (i >=0 && i <= nx+3 && j >= 0 && j <= ny+3) {
		k = 1;
		index	= index_3d(i,j,k);
		index_t	= index_3d(i,j,0);
		index_in = index_3d(i,j,2);
		if (myid == 0){
			// c[,,1]= -c[,,2]
			phi[index] =(-2*u_0) - phi[index_in] ;//0.0 - phi[index_in];
		}
		else{
			// c[,,1] = t_c[,,0]
			// c[,,nz/cpu+2] = t_c[,,3]
			phi[index] = t_phi[index_t];//0.0 - t_phi[index_t];
		}

		k = nz / cpu + 2;
		index	= index_3d(i,j,k);
		index_t	= index_3d(i,j,3);
		index_in = index_3d(i,j,nz/cpu+1);
		if (myid == lastp){
			// c[,,nz/cpu+2]= -c[,,nz/cpu+1]
			phi[index] =2*u_0 - phi[index_in] ;//0.0 - phi[index_in];
      
		}
		else{
			// c[,,1] = t_c[,,0]
			// c[,,nz/cpu+2] = t_c[,,3]
			phi[index] = t_phi[index_t];//0.0 - t_phi[index_t];
		}
	}
}
__global__ void wall_zm1_undo_vw( double *phi, double *t_phi,int myid, int lastp)
{
	// i:0~nx+3, j:0~ny+3, k:1,nz/cpu+2
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k, index, index_t, index_in;
	
	if (i >=0 && i <= nx+3 && j >= 0 && j <= ny+3) {
		k = 1;
		index	= index_3d(i,j,k);
		index_t	= index_3d(i,j,0);
		index_in = index_3d(i,j,2);
		if (myid == 0){
			// c[,,1]= -c[,,2]
			phi[index] = - phi[index_in];//0.0 - phi[index_in];
		}
		else{
			// c[,,1] = t_c[,,0]
			// c[,,nz/cpu+2] = t_c[,,3]
			phi[index] = t_phi[index_t];//0.0 - t_phi[index_t];
		}

		k = nz / cpu + 2;
		index	= index_3d(i,j,k);
		index_t	= index_3d(i,j,3);
		index_in = index_3d(i,j,nz/cpu+1);
		if (myid == lastp){
			// c[,,nz/cpu+2]= -c[,,nz/cpu+1]
			phi[index] = - phi[index_in];//0.0 - phi[index_in];
      
		}
		else{
			// c[,,1] = t_c[,,0]
			// c[,,nz/cpu+2] = t_c[,,3]
			phi[index] = t_phi[index_t];//0.0 - t_phi[index_t];
		}
	}
}
// distribution
__global__ void halfway_zd_g( double *g, double *t_g, double *g_t,int myid, int lastp)
{
	// g[,,1,] = t_g[,,0,]
	// g[,,nz/cpu+2,] = t_g[,,3,]
	
	// i:1~nx+2, j:1~ny+2, k:1,nz/cpu+2
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k, l, index_l, edt, index_l_t;
	int l_bot[5] = {6, 12, 14, 16, 17};
	int l_top[5] = {5, 11, 13, 15, 18};
	double temp1;
	if (i >= 1 && i <= nx+2 && j >= 1 && j <= ny+2) {
		k = 1;
		
		if(myid == 0){
			for(l=0; l<19; l++){
				edt 		= et_d[l];
				index_l 	= index_4d(i,j,k,l);
				temp1 		= g[index_l-edt];
				g[index_l] 	= temp1;
			}
			
			double g_s_x = g[index_4d(i,j,k,1)]+g[index_4d(i,j,k,7)]+g[index_4d(i,j,k,9)]-g[index_4d(i,j,k,2)]-g[index_4d(i,j,k,8)]-g[index_4d(i,j,k,10)];
			double g_s_y = g[index_4d(i,j,k,3)]+g[index_4d(i,j,k,7)]+g[index_4d(i,j,k,10)]-g[index_4d(i,j,k,4)]-g[index_4d(i,j,k,8)]-g[index_4d(i,j,k,9)];
			g[index_4d(i,j,k,5 )]=g[index_4d(i,j,k,6 )];
			g[index_4d(i,j,k,11)]=g[index_4d(i,j,k,12)] + (0.0 - u_0)/18.0;
			g[index_4d(i,j,k,13)]=g[index_4d(i,j,k,14)] + (0.0 + u_0)/18.0;
			g[index_4d(i,j,k,15)]=g[index_4d(i,j,k,16)];
			g[index_4d(i,j,k,18)]=g[index_4d(i,j,k,17)];


			// for(l=2; l<19; l=l+2){
			// 	// index_l 	= index_4d(i,j,k,l);
			// 	index_ll 	= index_4d(i,j,k,l-1);
			// 	temp1 		= g[index_4d(i,j,k,l)];
			// 	temp2 		= g[index_ll];
			// 	g[index_4d(i,j,k,l)] 	= temp2;
			// 	g[index_ll] = temp1;
			// }
		}else{
			for(int l = 0; l < 5; l ++){
     			index_l   	 = index_4d(i,j,k,l_top[l]);
				//index_l_t 	 = ((nx + 4) * (0 * (ny + 4) + j) + i) * 5 + l;
				index_l_t = index_3d(i,j,0+l);				
				g[index_l] 	 = t_g[index_l_t];
      			g_t[index_l] = g[index_l];
      		}
		}
		
		k = nz / cpu + 2;
		
		if(myid == lastp){
			for(l=0; l<19; l++){
				edt = et_d[l];
				index_l = index_4d(i,j,k,l);
				temp1 = g[index_l-edt];
				g[index_l] = temp1;
			}

			double g_s_x = g[index_4d(i,j,k,1)]+g[index_4d(i,j,k,7)]+g[index_4d(i,j,k,9)]-g[index_4d(i,j,k,2)]-g[index_4d(i,j,k,8)]-g[index_4d(i,j,k,10)];
			double g_s_y = g[index_4d(i,j,k,3)]+g[index_4d(i,j,k,7)]+g[index_4d(i,j,k,10)]-g[index_4d(i,j,k,4)]-g[index_4d(i,j,k,8)]-g[index_4d(i,j,k,9)];
			g[index_4d(i,j,k,6 )]=g[index_4d(i,j,k,5 )];
			g[index_4d(i,j,k,12)]=g[index_4d(i,j,k,11)] -  (0.0 + u_0)/18.0;
			g[index_4d(i,j,k,14)]=g[index_4d(i,j,k,13)] -  (0.0 - u_0)/18.0;
			g[index_4d(i,j,k,16)]=g[index_4d(i,j,k,15)] ;
			g[index_4d(i,j,k,17)]=g[index_4d(i,j,k,18)] ;
			
			// for(l=2; l<19; l=l+2){
			// 	index_l = index_4d(i,j,k,l);
			// 	index_ll = index_4d(i,j,k,l-1);
			// 	temp1 = g[index_l];
			// 	temp2 = g[index_ll];
			// 	g[index_l] = temp2;
			// 	g[index_ll] = temp1;
			// }
		}else{
			for(int l = 0 ; l<5 ; l++){
				index_l   = index_4d(i,j,k,l_bot[l]);
				//index_l_t = ((nx + 4) * (3 * (ny + 4) + j) + i) * 5 + l;
				index_l_t = index_3d(i,j,15+l);				
				g[index_l] = t_g[index_l_t];
				g_t[index_l] = g[index_l];
      		}
		}
	}
}

__global__ void halfway_zd_h( double *h, double *t_h, double *h_t,int myid, int lastp)
{
	// h[,,1,] = t_h[,,0,]
	// h[,,nz/cpu+2,] = t_h[,,3,]
	
	// i:1~nx+2, j:1~ny+2, k:1,nz/cpu+2
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k, l, index_l, edt, index_l_t;
	int l_bot[5] = {6, 12, 14, 16, 17};
	int l_top[5] = {5, 11, 13, 15, 18};
	double temp1;


	if (i >= 1 && i <= nx+2 && j >= 1 && j <= ny+2) {
		k = 1;
		int index3d = index_3d(i,j,k);	
		if(myid == 0){
			for(l=0; l<19; l++){
				edt 		= et_d[l];
				index_l 	= index_4d(i,j,k,l);
				temp1 		= h[index_l-edt];
				h[index_l] 	= temp1;
			}

			h[index_4d(i,j,k,5 )]=h[index_4d(i,j,k,6 )];
			h[index_4d(i,j,k,11)]=h[index_4d(i,j,k,12)];
			h[index_4d(i,j,k,13)]=h[index_4d(i,j,k,14)];
			h[index_4d(i,j,k,15)]=h[index_4d(i,j,k,16)] ;
			h[index_4d(i,j,k,18)]=h[index_4d(i,j,k,17)] ;

		}else{
			for(int l = 0; l < 5; l ++){
     			index_l   	 = index_4d(i,j,k,l_top[l]);
				//index_l_t 	 = ((nx + 4) * (0 * (ny + 4) + j) + i) * 5 + l;
				index_l_t = index_3d(i,j,0+l);				
				h[index_l] 	 = t_h[index_l_t];
      			h_t[index_l] = h[index_l];
      		}
		}
		
		k = nz / cpu + 2;
		index3d = index_3d(i,j,k);	
		if(myid == lastp){
			for(l=0; l<19; l++){
				edt = et_d[l];
				index_l = index_4d(i,j,k,l);
				temp1 = h[index_l-edt];
				h[index_l] = temp1;
			}

			h[index_4d(i,j,k,6 )]=h[index_4d(i,j,k,5 )];
			h[index_4d(i,j,k,12)]=h[index_4d(i,j,k,11)];
			h[index_4d(i,j,k,14)]=h[index_4d(i,j,k,13)];
			h[index_4d(i,j,k,16)]=h[index_4d(i,j,k,15)];
			h[index_4d(i,j,k,17)]=h[index_4d(i,j,k,18)];
		}else{
			for(int l = 0 ; l<5 ; l++){
				index_l   = index_4d(i,j,k,l_bot[l]);
				//index_l_t = ((nx + 4) * (3 * (ny + 4) + j) + i) * 5 + l;
				index_l_t = index_3d(i,j,15+l);				
				h[index_l] = t_h[index_l_t];
				h_t[index_l] = h[index_l];
      		}
		}
	}
}

__global__ void boundary_ym_sym( double *phi)
{
	// c[,0,] = c[,4,]
	// c[,1,] = c[,3,]
	// c[,ny+2,] = c[,ny,]
	// c[,ny+3,] = c[,ny-1,]
	// i:0~nx+3, j:0,1,ny+2,ny+3, k:0~nz/cpu+3

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;
	int j, index;

	if (i >= 0 && i <= nx+3 && k >= 0 && k <= nz/cpu+3) {
		int distance = 3 * (nx+4);
		
		j = 0;
		index  = index_3d(i,j,k);
		phi[index] = phi[index + distance + 1*(nx+4)];
		phi[index + 1*(nx+4)] = phi[index + distance];
		
		j = ny+2;
		index  = index_3d(i,j,k);
		phi[index] = phi[index - distance + 1*(nx+4)];
		phi[index + 1*(nx+4)] = phi[index - distance];
		
	}
}

__global__ void boundary_ym_sym_in( double *phi)
{
	// c[,0,] = c[,4,]
	// c[,1,] = c[,3,]
	// c[,ny+2,] = c[,ny,]
	// c[,ny+3,] = c[,ny-1,]
	// i:0~nx+3, j:0,1,ny+2,ny+3, k:4~nz/cpu-1

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;
	int j, index;
	
	if (i >= 0 && i <= nx+3 && k >= 4 && k <= nz/cpu-1) {
		int distance = 3 * (nx+4);
		j = 0;
		index  = index_3d(i,j,k);
		phi[index] = phi[index + distance + 1*(nx+4)];
		phi[index + 1*(nx+4)] = phi[index + distance];
		
		j = ny+2;
		index  = index_3d(i,j,k);
		phi[index] = phi[index - distance + 1*(nx+4)];
		phi[index + 1*(nx+4)] = phi[index - distance];
	}
}

__global__ void boundary_ym_sym_bc( double *phi)
{
	// c[,0,] = c[,4,]
	// c[,1,] = c[,3,]
	// c[,ny+2,] = c[,ny,]
	// c[,ny+3,] = c[,ny-1,]
	// i:0~nx+3, j:0,1,ny+2,ny+3, k:2,3,nz/cpu,nz/cpu+1

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j, index;
	
	if (i >= 0 && i <= nx+3) {
		int kk[4] = {2, 3, nz/cpu, nz/cpu+1};
		int distance = 3 * (nx+4);
		for (int t = 0;t < 4; t++) {
			int k = kk[t];
			j = 0;
			index  = index_3d(i,j,k);
			phi[index] = phi[index + distance + 1*(nx+4)];
			phi[index + 1*(nx+4)] = phi[index + distance];

			j = ny+2;
			index  = index_3d(i,j,k);
			phi[index] = phi[index - distance + 1*(nx+4)];
			phi[index + 1*(nx+4)] = phi[index - distance];
		}
	}
}

__global__ void boundary_yd_sym_in( double *g,double *h)
{
	// g[,ny+2,,sym_l] = g[,ny,,sym_r]
	// g[,1,,sym_r] = g[,3,,sym_l]
	// i:1~nx+2, j:1,ny+2, k:3~nz/cpu

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y + blockIdx.y * blockDim.y;
	int sym_l[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
	int sym_r[19] = {0, 1, 2, 4, 3, 5, 6, 9, 10, 7, 8, 11, 12, 13, 14, 18, 17, 16, 15};
	
	if (i >= 1 && i <= nx+2 && k >= 3 && k <= nz/cpu) {
		int j, index_l_l, index_l_r;
		int distance = 2 * (nx+4);
		for (int l = 0; l < q; l++) {		
			j = 1;
			index_l_l = index_4d(i,j,k,sym_l[l]);
			index_l_r = index_4d(i,j,k,sym_r[l]);
			g[index_l_l] = g[index_l_r + distance];
			h[index_l_l] = h[index_l_r + distance];
			j = ny + 2;
			index_l_l = index_4d(i,j,k,sym_l[l]);
			index_l_r = index_4d(i,j,k,sym_r[l]);
			g[index_l_r] = g[index_l_l - distance];
			h[index_l_r] = h[index_l_l - distance];
		}
	}
}

__global__ void boundary_yd_sym_bc( double *g,double *h)
{
	// g[,ny+2,,sym_l] = g[,ny,,sym_r]
	// g[,1,,sym_r] = g[,3,,sym_l]
	// i:1~nx+2, j:1,ny+2, k:2,nz/cpu+1

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int sym_l[19] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
	int sym_r[19] = {0, 1, 2, 4, 3, 5, 6, 9, 10, 7, 8, 11, 12, 13, 14, 18, 17, 16, 15};
	
	if (i >= 1 && i <= nx+2) {
		int j, k, index_l_l, index_l_r;
		int distance = 2 * (nx+4);
		for (int l = 0;l < q; l++) {
		
		j = 1;
		k = 2;
		index_l_l = index_4d(i,j,k,sym_l[l]);
		index_l_r = index_4d(i,j,k,sym_r[l]);
		g[index_l_l] = g[index_l_r + distance];
		h[index_l_l] = h[index_l_r + distance];

		j = 1;
		k = nz / cpu + 1;
		index_l_l = index_4d(i,j,k,sym_l[l]);
		index_l_r = index_4d(i,j,k,sym_r[l]);
		g[index_l_l] = g[index_l_r + distance];
		h[index_l_l] = h[index_l_r + distance];

		j = ny + 2;
		k = 2;
		index_l_l = index_4d(i,j,k,sym_l[l]);
		index_l_r = index_4d(i,j,k,sym_r[l]);
		g[index_l_r] = g[index_l_l - distance];
		h[index_l_r] = h[index_l_l - distance];

		j = ny + 2;
		k = nz / cpu + 1;
		index_l_l = index_4d(i,j,k,sym_l[l]);
		index_l_r = index_4d(i,j,k,sym_r[l]);
		g[index_l_r] = g[index_l_l - distance];
		h[index_l_r] = h[index_l_l - distance];
		}
	}
}

/***********x_periodic_y_symmetric**********/

/*macro C*/

void boundary_C_bc_x_sym_y_z_transfer(){
	boundary_ym_sym_bc<<< bpgx,tpbx>>>(c);
	boundary_xm_bc<<< bpgy,tpby >>>(c);
	boundary_zm2<<<  bpgxy,tpbxy>>>(c,t_c);
}
void boundary_C_sym_y_in(cudaStream_t stream){

	boundary_ym_sym_in<<< bpgxz , tpbxz   ,0,stream>>>(c);
	boundary_xm_in<<< bpgyz , tpbyz   ,0,stream>>>(c);	
}

/*macro chemical*/

void boundary_chemi_bc_x_sym_y_z_transfer(){
	boundary_ym_sym_bc<<< bpgx,tpbx>>>(m);
	boundary_xm_bc<<< bpgy,tpby >>>(m);
	boundary_zm2<<<  bpgxy,tpbxy>>>(m,t_m);
}
void boundary_chemi_sym_y_in(cudaStream_t stream){
	boundary_ym_sym_in<<< bpgxz , tpbxz   ,0,stream>>>(m);
	boundary_xm_in<<< bpgyz , tpbyz   ,0,stream>>>(m);

}

/*macro velo and P_hydro*/
void boundary_others_bc_x_sym_y_z_transfer(){


	boundary_ym_sym_bc<<< bpgx,tpbx>>>(u);
	boundary_ym_sym_bc<<< bpgx,tpbx>>>(v);
	boundary_ym_sym_bc<<< bpgx,tpbx>>>(w);
	boundary_ym_sym_bc<<< bpgx,tpbx>>>(p);


	boundary_xm_bc<<< bpgy,tpby >>>(u);
	boundary_xm_bc<<< bpgy,tpby >>>(v);
	boundary_xm_bc<<< bpgy,tpby >>>(w);
	boundary_xm_bc<<< bpgy,tpby >>>(p);

	boundary_zm2<<<  bpgxy,tpbxy>>>(p,t_p);
	boundary_zm1<<<  bpgxy,tpbxy>>>(u,t_u);
	boundary_zm1<<<  bpgxy,tpbxy>>>(v,t_v);
	boundary_zm1<<<  bpgxy,tpbxy>>>(w,t_w);		
}
void boundary_others_sym_y_in(cudaStream_t stream){

	boundary_ym_sym_in<<< bpgxz , tpbxz   ,0,stream>>>(p);
	boundary_ym_sym_in<<< bpgxz , tpbxz   ,0,stream>>>(u);
	boundary_ym_sym_in<<< bpgxz , tpbxz   ,0,stream>>>(v);
	boundary_ym_sym_in<<< bpgxz , tpbxz   ,0,stream>>>(w);
	
	boundary_xm_in<<< bpgyz , tpbyz   ,0,stream>>>(p);
	boundary_xm_in<<< bpgyz , tpbyz   ,0,stream>>>(u);
	boundary_xm_in<<< bpgyz , tpbyz   ,0,stream>>>(v);
	boundary_xm_in<<< bpgyz , tpbyz   ,0,stream>>>(w);


}

/*distribution function*/

void boundary_distri_bc_x_sym_y_z_transfer(double*g,double*h,cudaStream_t stream){
	boundary_yd_sym_bc<<< bpgx,tpbx,0,stream>>>(g,h);
	boundary_xd_bc<<< bpgy,tpby ,0,stream>>>(g,h);
	boundary_zd<<<  bpgxy,tpbxy,0,stream>>>(g,t_g);
	boundary_zd<<<  bpgxy,tpbxy,0,stream>>>(h,t_h);
}
void boundary_distri_sym_y_in(double*g,double*h,cudaStream_t stream){
	boundary_yd_sym_in<<< bpgxz , tpbxz   ,0,stream>>>(g,h);
	boundary_xd_in<<< bpgyz , tpbyz   ,0,stream>>>(g,h);
}

/***********z_wall**********/

/*macro C*/

void boundary_C_wall_z_transfer_back(cudaStream_t stream){
	wall_zm2_undo	<<< bpgxy , tpbxy  ,0,stream>>>( c,t_c,myid,lastp );
}

/*macro chemical*/

void boundary_chemi_wall_z_transfer_back(cudaStream_t stream){
	wall_zm2_undo	<<< bpgxy , tpbxy  ,0,stream>>>( m,t_m,myid,lastp );
}

/*macro velo and P_hydro*/

void boundary_others_wall_z_transfer_back(cudaStream_t stream){
	wall_zm2_undo		<<< bpgxy , tpbxy  ,0,stream>>>( p,t_p,myid,lastp );
	wall_zm1_undo_u		<<< bpgxy , tpbxy  ,0,stream>>>( u,t_u,myid,lastp );
	wall_zm1_undo_vw	<<< bpgxy , tpbxy  ,0,stream>>>( v,t_v,myid,lastp );
	wall_zm1_undo_vw	<<< bpgxy , tpbxy  ,0,stream>>>( w,t_w,myid,lastp );
}

/*distribution function*/

void boundary_distri_wall_z_transfer_back(double*g,double*h,cudaStream_t stream){
	halfway_zd_g <<< bpgxy , tpbxy   ,0,stream>>>( g,t_g,g_t, myid, lastp);
    halfway_zd_h <<< bpgxy , tpbxy   ,0,stream>>>( h,t_h,h_t, myid, lastp);
}




#endif
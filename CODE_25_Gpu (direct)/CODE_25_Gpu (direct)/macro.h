#ifndef MACRO_H
#define MACRO_H




__global__ void macro_h_in(double *h,double *h_next,double *c,double*lap_m)
{
	// i:2~nx+1, j:2~ny+1, k:4~nz/cpu-1
	int i = threadIdx.x + blockIdx.x * blockDim.x ;
	int j = threadIdx.y + blockIdx.y * blockDim.y ;
	int k = threadIdx.z + blockIdx.z * blockDim.z ;
	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1 && k >= 4 && k <= nz/cpu-1) {
		int index = index_3d(i,j,k);
		double sum_c = 0.0;
		double lap_mu = lap_m[index] ;	
		for (int l = 0; l < q; l++) {
			int index_l = index_4d(i,j,k,l);
			int edt = et_d[l];
			sum_c = h[index_l - edt]+ sum_c;
			h_next[index_l] = h[index_l - edt];
		}
		if(PH_model == 0){
			c[index] = sum_c + 0.5*dt*mobility*lap_mu;
		}
		else{
			c[index] = sum_c;
		}
	}
	
}

__global__ void macro_h_bc(double *h,double *h_next,double *c,double*lap_m)
{
	// i:2~nx+1, j:2~ny+1, k:2,3,nz/cpu,nz/cpu+1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1) {
		int kk[4] = {2, 3, nz/cpu, nz/cpu+1};
		for (int k = 0; k < 4; k++) {
			int index = index_3d(i,j,kk[k]);
			double lap_mu = lap_m[index] ;
			double sum_c = 0.0;
			for (int l = 0; l < q; l++) {
				int index_l = index_4d(i,j,kk[k],l);
				int edt = et_d[l];
				sum_c = h[index_l - edt] + sum_c;
				h_next[index_l] = h[index_l - edt];
			}
			if(PH_model == 0){
				c[index] = sum_c + 0.5*dt*mobility*lap_mu;
			}
			else{
				c[index] = sum_c;
			}
		}
	}
}

__global__ void macro_g_in( double *g, double *g_next,double *c,double *m,double *p,double *gra_c,double *gra_m,double *u,double *v,double *w)
{
	// i:2~nx+1, j:2~ny+1, k:4~nz/cpu-1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	
	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1 && k >= 4 && k <= nz/cpu-1) {
		int index=index_3d(i,j,k);
		const double cs2_inv=3.0;
		const double cs2=1.0/cs2_inv;
		double cc=c[index];
		double r=cc*rho_l+((double)1.0-cc)*rho_g;
		double dr=rho_l-rho_g;
		
		double gr_rx_c=gra_c[index_4d(i,j,k,0)]*dr;
		double gr_ry_c=gra_c[index_4d(i,j,k,1)]*dr;
		double gr_rz_c=gra_c[index_4d(i,j,k,2)]*dr;
		double gr_mx_c=gra_m[index_4d(i,j,k,0)];
		double gr_my_c=gra_m[index_4d(i,j,k,1)];
		double gr_mz_c=gra_m[index_4d(i,j,k,2)];
		
		
		double sum_u=0.0;
		double sum_v=0.0;
		double sum_w=0.0;
		double sum_p=0.0;
		
		for(int l=0;l<q;l++){
		int index_l =index_4d(i,j,k,l);
		double temp_g=g[index_l-et_d[l]];
		sum_u=ex_d[l]*temp_g+sum_u;
		sum_v=ey_d[l]*temp_g+sum_v;
		sum_w=ez_d[l]*temp_g+sum_w;
		sum_p=      temp_g+sum_p;
		g_next[index_l]=temp_g;
		}
		
		double gra = (r-rho_l)*gra_ac; //20220810, gravity force, x deriction
		double uu=(sum_u*cs2_inv-0.5*dt*(cc*gr_mx_c + gra))/r;	
		double vv=(sum_v*cs2_inv-0.5*dt*cc*gr_my_c)/r;
		double ww=(sum_w*cs2_inv-0.5*dt*cc*gr_mz_c)/r;
		u[index]=uu;
		v[index]=vv;
		w[index]=ww;
		p[index]=sum_p+0.5*dt*(uu*gr_rx_c+vv*gr_ry_c+ww*gr_rz_c)*cs2;
	}
}
	
__global__ void macro_g_bc( double *g, double *g_next,double *c,double *m,double *p,double *gra_c,double *gra_m,double *u,double *v,double *w)
{
	// i:2~nx+1, j:2~ny+1, k:2,3,nz/cpu,nz/cpu+1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int kk[4] = {2, 3, nz/cpu, nz/cpu+1};
	
	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1) {
		const double cs2_inv=3.0;
		const double cs2=1.0/cs2_inv;
		for (int k=0;k<4;k++){
		int index=index_3d(i,j,kk[k]);
		double cc=c[index];
		double r=cc*rho_l+((double)1.0-cc)*rho_g;
		double dr=rho_l-rho_g;
		
	 	double gr_rx_c=gra_c[index_4d(i,j,kk[k],0)]*dr;
		double gr_ry_c=gra_c[index_4d(i,j,kk[k],1)]*dr;
		double gr_rz_c=gra_c[index_4d(i,j,kk[k],2)]*dr;
		double gr_mx_c=gra_m[index_4d(i,j,kk[k],0)];
		double gr_my_c=gra_m[index_4d(i,j,kk[k],1)];
		double gr_mz_c=gra_m[index_4d(i,j,kk[k],2)];

		double sum_u=0.0;
		double sum_v=0.0;
		double sum_w=0.0;
		double sum_p=0.0;
		
		for(int l=0;l<q;l++){
		int index_l =index_4d(i,j,kk[k],l);
		double temp_g=g[index_l-et_d[l]];
		sum_u=ex_d[l]*temp_g+sum_u;
		sum_v=ey_d[l]*temp_g+sum_v;
		sum_w=ez_d[l]*temp_g+sum_w;
		sum_p=      temp_g+sum_p;
		g_next[index_l]=temp_g;
		}

   		double gra = (r-rho_l)*gra_ac; //20220810, gravity force, x deriction
		double uu=(sum_u*cs2_inv-0.5*dt*(cc*gr_mx_c + gra))/r;	
		double vv=(sum_v*cs2_inv-0.5*dt*cc*gr_my_c)/r;
		double ww=(sum_w*cs2_inv-0.5*dt*cc*gr_mz_c)/r;
		u[index]=uu;
		v[index]=vv;
		w[index]=ww;
		p[index]=sum_p+0.5*dt*(uu*gr_rx_c+vv*gr_ry_c+ww*gr_rz_c)*cs2;
		}
	}
}


__global__ void macro_h_halfway_bc(double *h,double *h_next,double *c,double*lap_m,int myid, int lastp)
{
	// i:2~nx+1, j:2~ny+1, k:2,3,nz/cpu,nz/cpu+1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1) {
		int kk[4] = {2, 3, nz/cpu, nz/cpu+1};
		for (int k = 0; k < 4; k++) {
			int index = index_3d(i,j,kk[k]);

			double lap_mu = lap_m[index] ;
			double sum_c = 0.0;
		if(myid == 0 && kk[k] == 2)//bottom halfway
		{
			for(int l=0;l<q;l++){
				int index_l =index_4d(i,j,kk[k],l);
				h_next[index_l]= h[index_l-et_d[l]];
			}

			h_next[index_4d(i,j,kk[k],5 )]=h[index_4d(i,j,kk[k],6 )] ;
			h_next[index_4d(i,j,kk[k],11)]=h[index_4d(i,j,kk[k],12)] ;
			h_next[index_4d(i,j,kk[k],13)]=h[index_4d(i,j,kk[k],14)] ;
			h_next[index_4d(i,j,kk[k],15)]=h[index_4d(i,j,kk[k],16)] ;
			h_next[index_4d(i,j,kk[k],18)]=h[index_4d(i,j,kk[k],17)] ;
			
			for(int l=0;l<q;l++){
				int index_l =index_4d(i,j,kk[k],l);
				sum_c = h_next[index_l] + sum_c;
			}
		}
		else if (myid == lastp && kk[k] == nz/cpu+1)//top halfway
		{
			for(int l=0;l<q;l++){
				int index_l =index_4d(i,j,kk[k],l);
				h_next[index_l]= h[index_l-et_d[l]];
			}
			h_next[index_4d(i,j,kk[k],6 )]=h[index_4d(i,j,kk[k],5 )];
			h_next[index_4d(i,j,kk[k],12)]=h[index_4d(i,j,kk[k],11)];
			h_next[index_4d(i,j,kk[k],14)]=h[index_4d(i,j,kk[k],13)];
			h_next[index_4d(i,j,kk[k],16)]=h[index_4d(i,j,kk[k],15)];
			h_next[index_4d(i,j,kk[k],17)]=h[index_4d(i,j,kk[k],18)];
			for(int l=0;l<q;l++){
				int index_l =index_4d(i,j,kk[k],l);
				sum_c = h_next[index_l] + sum_c;
			}
		}
		else{
			for (int l = 0; l < q; l++) {
				int index_l = index_4d(i,j,kk[k],l);
				int edt = et_d[l];
				sum_c = h[index_l - edt] + sum_c;
				h_next[index_l] = h[index_l - edt];
			}
		}

		if(PH_model == 0){
			c[index] = sum_c + 0.5*dt*mobility*lap_mu;
		}
		else{
			c[index] = sum_c;
		}
		}
	}
}


__global__ void macro_g_halfway_bc( double *g, double *g_next,double *c,double *m,double *p,double *gra_c,double *gra_m,double *u,double *v,double *w,int myid, int lastp)
{
	// i:2~nx+1, j:2~ny+1, k:2,3,nz/cpu,nz/cpu+1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int kk[4] = {2, 3, nz/cpu, nz/cpu+1};
	
	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1) {
		const double cs2_inv=3.0;
		const double cs2=1.0/cs2_inv;
		for (int k=0;k<4;k++){
		int index=index_3d(i,j,kk[k]);
		double cc=c[index];
		double r=cc*rho_l+((double)1.0-cc)*rho_g;
		double dr=rho_l-rho_g;
		
	 	double gr_rx_c=gra_c[index_4d(i,j,kk[k],0)]*dr;
		double gr_ry_c=gra_c[index_4d(i,j,kk[k],1)]*dr;
		double gr_rz_c=gra_c[index_4d(i,j,kk[k],2)]*dr;
		double gr_mx_c=gra_m[index_4d(i,j,kk[k],0)];
		double gr_my_c=gra_m[index_4d(i,j,kk[k],1)];
		double gr_mz_c=gra_m[index_4d(i,j,kk[k],2)];

		double sum_u=0.0;
		double sum_v=0.0;
		double sum_w=0.0;
		double sum_p=0.0;
		if(myid == 0 && kk[k] == 2)//bottom halfway
		{
			for(int l=0;l<q;l++){
				int index_l =index_4d(i,j,kk[k],l);
				g_next[index_l]= g[index_l-et_d[l]];
			}
			g_next[index_4d(i,j,kk[k],5 )]=g[index_4d(i,j,kk[k],6 )];
			g_next[index_4d(i,j,kk[k],11)]=g[index_4d(i,j,kk[k],12)] + (0.0 - u_0)/18.0;
			g_next[index_4d(i,j,kk[k],13)]=g[index_4d(i,j,kk[k],14)] + (0.0 + u_0)/18.0;
			g_next[index_4d(i,j,kk[k],15)]=g[index_4d(i,j,kk[k],16)];
			g_next[index_4d(i,j,kk[k],18)]=g[index_4d(i,j,kk[k],17)];

			for(int l=0;l<q;l++){
			int index_l =index_4d(i,j,kk[k],l);
			double temp_g=g_next[index_l];
			sum_u=ex_d[l]*temp_g+sum_u;
			sum_v=ey_d[l]*temp_g+sum_v;
			sum_w=ez_d[l]*temp_g+sum_w;
			sum_p=      temp_g+sum_p;
			}
		}
		else if (myid == lastp && kk[k] == nz/cpu+1)//top halfway
		{
			for(int l=0;l<q;l++){
				int index_l =index_4d(i,j,kk[k],l);
				g_next[index_l]= g[index_l-et_d[l]];
			}
			g_next[index_4d(i,j,kk[k],6 )]=g[index_4d(i,j,kk[k],5 )];
			g_next[index_4d(i,j,kk[k],12)]=g[index_4d(i,j,kk[k],11)] -  (0.0 + u_0)/18.0;
			g_next[index_4d(i,j,kk[k],14)]=g[index_4d(i,j,kk[k],13)] -  (0.0 - u_0)/18.0;
			g_next[index_4d(i,j,kk[k],16)]=g[index_4d(i,j,kk[k],15)] ;
			g_next[index_4d(i,j,kk[k],17)]=g[index_4d(i,j,kk[k],18)] ;

			for(int l=0;l<q;l++){
			int index_l =index_4d(i,j,kk[k],l);
			double temp_g=g_next[index_l];
			sum_u=ex_d[l]*temp_g+sum_u;
			sum_v=ey_d[l]*temp_g+sum_v;
			sum_w=ez_d[l]*temp_g+sum_w;
			sum_p=      temp_g+sum_p;
			}
		}
		else{
			for(int l=0;l<q;l++){
			int index_l =index_4d(i,j,kk[k],l);
			double temp_g=g[index_l-et_d[l]];
			sum_u=ex_d[l]*temp_g+sum_u;
			sum_v=ey_d[l]*temp_g+sum_v;
			sum_w=ez_d[l]*temp_g+sum_w;
			sum_p=      temp_g+sum_p;
			g_next[index_l]=temp_g;
			}
		}
   		double gra = (r-rho_l)*gra_ac; //20220810, gravity force, x deriction
		double uu=(sum_u*cs2_inv-0.5*dt*(cc*gr_mx_c + gra))/r;	
		double vv=(sum_v*cs2_inv-0.5*dt*cc*gr_my_c)/r;
		double ww=(sum_w*cs2_inv-0.5*dt*cc*gr_mz_c)/r;
		u[index]=uu;
		v[index]=vv;
		w[index]=ww;
		p[index]=sum_p+0.5*dt*(uu*gr_rx_c+vv*gr_ry_c+ww*gr_rz_c)*cs2;
		}
	}
}




#endif
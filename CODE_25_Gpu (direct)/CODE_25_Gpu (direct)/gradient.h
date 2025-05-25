#ifndef SPATIAL_H
#define SPATIAL_H


#include"preparation.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                 gradient                                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ double gra_phi_c(double *phi,double *ep,double cs,int index)
{
	double ans=0.0;
	int l;
	for(l=0;l<q;l++){
	ans=wt_d[l]*ep[l]*(phi[index+et_d[l]]-phi[index-et_d[l]])+ans;
	}
	ans=ans/(2.0*cs*cs*dt);
	return ans;
}

__device__ double gra_phi_m(double *phi,double *ep,double cs,int index)
{
	double ans=0.0;
	int l;
	for(l=0;l<q;l++){
	ans=wt_d[l]*ep[l]*(-phi[index+2*et_d[l]]+5.0*phi[index+et_d[l]]-3.0*phi[index]-phi[index-et_d[l]])+ans;
	}
	ans=ans/(4.0*cs*cs*dt);
	return ans;
	
}

__device__ double grad_phie_c(double *phi,int index,int et)
{
	double ans;
	ans=(phi[index+et]-phi[index-et])*0.5;
	return ans;
}

__device__ double grad_phie_m(double *phi,int index,int et)
{
	double ans;
	ans=(-phi[index+2*et]+5.0*phi[index+et]-3.0*phi[index]-phi[index-et])*0.25;
	return ans;
}

__device__ double lap_phi (double *phi,double cs,int index)
{
	double ans=0.0;
	int l;
	for(l=0;l<q;l++){
	ans=wt_d[l]*(phi[index+et_d[l]]-2.0*phi[index]+phi[index-et_d[l]])+ans;
	}
	ans=ans/(cs*cs*dt);
	return ans;
}

__device__ double laplace_phi (double *phi,double cs2_inv,int index)
{
	double ans=0.0;
	double phi_index=phi[index];
	for(int l=1;l<q;l=l+2){
	ans=2.0*wt_d[l]*(phi[index+et_d[l]]-2.0*phi_index+phi[index-et_d[l]])+ans;
	}
	ans=ans*cs2_inv/dt;
	return ans;
}

__global__ void gradient_cen (	double *gra_phi, double *phi)
{
	// i:2~nx+1, j:2~ny+1, k:2~nz/cpu+1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	
	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1 && k >= 2 && k <= nz/cpu+1) {
		int index = index_3d(i,j,k);
		const double cs2_inv = 3.0;
		double temp   = 0.0;
		double temp_x = 0.0;
		double temp_y = 0.0;
		double temp_z = 0.0;

		for (int l = 1;l < q;l = l + 2) {
			temp = 2.0 * wt_d[l] * (phi[index + et_d[l]] - phi[index - et_d[l]]);
			temp_x = ex_d[l] * temp + temp_x;
			temp_y = ey_d[l] * temp + temp_y;
			temp_z = ez_d[l] * temp + temp_z;
		}
		gra_phi[index_4d(i,j,k,0)] = temp_x * 0.5 * cs2_inv / dt;
		gra_phi[index_4d(i,j,k,1)] = temp_y * 0.5 * cs2_inv / dt;
		gra_phi[index_4d(i,j,k,2)] = temp_z * 0.5 * cs2_inv / dt;
	}
}


__global__ void gradient_cen_n (double *n_d_x,double *n_d_y,double *n_d_z, double *phi)
{
	// i:2~nx+1, j:2~ny+1, k:2~nz/cpu+1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	
	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1 && k >= 2 && k <= nz/cpu+1) {
		int index = index_3d(i,j,k);
		const double cs2_inv = 3.0;
		double gr_cx_c = 0.0, gr_cy_c = 0.0, gr_cz_c = 0.0;
		double temp   = 0.0;
		double temp_x = 0.0;
		double temp_y = 0.0;
		double temp_z = 0.0;

		for (int l = 1;l < q;l = l + 2) {
			temp = 2.0 * wt_d[l] * (phi[index + et_d[l]] - phi[index - et_d[l]]);
			temp_x = ex_d[l] * temp + temp_x;
			temp_y = ey_d[l] * temp + temp_y;
			temp_z = ez_d[l] * temp + temp_z;
		}
		gr_cx_c = temp_x * 0.5 * cs2_inv / dt;
		gr_cy_c = temp_y * 0.5 * cs2_inv / dt;
		gr_cz_c = temp_z * 0.5 * cs2_inv / dt;
		double m_abs = 0.0;
		double sum_m = gr_cx_c * gr_cx_c + gr_cy_c * gr_cy_c + gr_cz_c * gr_cz_c;
		m_abs = sqrt(sum_m);
		n_d_x[index] = gr_cx_c / (m_abs + 10e-12);  // n = (-m)/(|m|+10^(-12))
		n_d_y[index] = gr_cy_c / (m_abs + 10e-12);
		n_d_z[index] = gr_cz_c / (m_abs + 10e-12);
	}
}

__global__ void gradient_mix (	double *gra_phi, double *phi)
{
	// i:2~nx+1, j:2~ny+1, k:2~nz/cpu+1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	
	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1 && k >= 2 && k <= nz/cpu+1) {
		int index = index_3d(i,j,k);
		const double cs2_inv = 3.0;
		double temp   = 0.0;
		double temp_x = 0.0;
		double temp_y = 0.0;
		double temp_z = 0.0;

		for (int l = 1; l < q; l = l + 2) {
			temp   = wt_d[l] * (-phi[index + 2 * et_d[l]] + 6.0 * phi[index + et_d[l]] 
				               - 6.0 * phi[index - et_d[l]] + phi[index - 2 * et_d[l]]);
			temp_x = ex_d[l] * temp + temp_x;
			temp_y = ey_d[l] * temp + temp_y;
			temp_z = ez_d[l] * temp + temp_z;
		}
		gra_phi[index_4d(i,j,k,3)] = temp_x * 0.25 * cs2_inv / dt;
		gra_phi[index_4d(i,j,k,4)] = temp_y * 0.25 * cs2_inv / dt;
		gra_phi[index_4d(i,j,k,5)] = temp_z * 0.25 * cs2_inv / dt;
	}
}

__global__ void laplacian_mu (	double *lap_m, double *m)
{
	// i:1~nx+2, j:1~ny+2, k:1~nz/cpu+2
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	
	if (i >= 1 && i <= nx+2 && j >= 1 && j <= ny+2 && k >= 1 && k <= nz/cpu+2) {
		int index = index_3d(i,j,k);
		double cs2_inv = 3.0;
		lap_m[index] = laplace_phi( m,cs2_inv,index );
	}
}

__global__ void gradient_cen_cm (	double *gra_phi, double *c, double *sf)
{
	// i:2~nx+1, j:2~ny+1, k:2~nz/cpu+1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	
	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1 && k >= 2 && k <= nz/cpu+1) {
		int index = index_3d(i,j,k);
		
		sf[index] = c[index] * gra_phi[index_4d(i,j,k,0)]; //only x direction
	}
}

__global__ void modified_c (double *c, double *modi_c)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i < nx+4 && j < ny+4 && k < nz/cpu+4){
		int index = index_3d(i, j, k);
		modi_c[index] = asin(2.0*c[index]-1.0);
	}
}

__global__ void surface_tension_force (double *c, double *modi_c, double *sf, double kappa, double *ex_d, double *wt_in, int *et_in, double *mu)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i > 1 && i < nx+2 && j > 1 && j < ny+2 && k > 1 && k < nz/cpu+2){
		int index = index_3d(i, j, k);
		double cs = 1.0/sqrt(3.0);
		double cs2_inv = 3.0;
		double cl = c[index];
		if (bulk_free_energy == 1){
			if (SF_Type == 2){// mu grad c
				sf[index] = mu[index] * gra_phi_c(c, ex_d, cs, index);
			} else if (SF_Type == 3){//-c grad mu
				sf[index] = - cl * gra_phi_c(mu, ex_d, cs, index);
			} else if (SF_Type == 4){//kim
				double grad_c_cen = gra_phi_c(c, ex_d, cs, index);
				sf[index] = -kappa * laplace_phi(c, cs2_inv, index) * grad_c_cen;
			}
		} else {
			double grad_modi_c_cen = gra_phi_c(modi_c, ex_d, cs, index);//only x direction
			sf[index] = - kappa * laplace_phi(modi_c, cs2_inv, index) * grad_modi_c_cen * pow(cl * (1.0 - cl), 3.5);
			if (SF_Type == 2){// mu grad c
                sf[index] = mu[index] * gra_phi_c(c, ex_d, cs, index);
            } else if (SF_Type == 3){//-c grad mu
                sf[index] = - cl * gra_phi_c(mu, ex_d, cs, index);
            } else if (SF_Type == 4){//kim
                double grad_c_cen = gra_phi_c(c, ex_d, cs, index);
                sf[index] = -kappa * laplace_phi(c, cs2_inv, index) * grad_c_cen;
            }
		}
	}
}

#endif
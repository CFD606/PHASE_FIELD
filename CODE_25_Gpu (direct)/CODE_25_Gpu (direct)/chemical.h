#ifndef CHEMICAL
#define CHEMICAL

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                chemical mu                                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





__global__ void  chemical( double *c,double *m,double *n_d_x,double *n_d_y,double *n_d_z,double kappa,double beta )
{
	// i:2~nx+1, j:2~ny+1, k:2~nz/cpu+1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;

	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1 && k >= 2 && k <= nz/cpu+1) {
		int index = index_3d(i,j,k);
		const double cs = 1.0 / pow(3.0,0.5);
		double cl = c[index]; //c local

		if (energy_conserve == 1){
			double K = 0.00005, delta = 5.0, A = 0.000000001; // K為彈性係數ㄝ, delta沒有說, A是錨點係數(正數), 以上三個文章都說是常數
			// f_bulk cal
			double n_d_xx = n_d_x[index];
			double n_d_yy = n_d_y[index];
			double n_d_zz = n_d_z[index];
			double gr_nx_x = gra_phi_c(n_d_x,ex_d,cs,index);
			double gr_nx_y = gra_phi_c(n_d_x,ey_d,cs,index);
			double gr_nx_z = gra_phi_c(n_d_x,ez_d,cs,index);
			double gr_ny_x = gra_phi_c(n_d_y,ex_d,cs,index);
			double gr_ny_y = gra_phi_c(n_d_y,ey_d,cs,index);
			double gr_ny_z = gra_phi_c(n_d_y,ez_d,cs,index);
			double gr_nz_x = gra_phi_c(n_d_z,ex_d,cs,index);
			double gr_nz_y = gra_phi_c(n_d_z,ey_d,cs,index);
			double gr_nz_z = gra_phi_c(n_d_z,ez_d,cs,index);
			double gr_n_sum = gr_nx_x*gr_nx_x + gr_ny_y*gr_ny_y + gr_nz_z*gr_nz_z + 2*gr_nx_y*gr_ny_x + 2*gr_nx_z*gr_nz_x + 2*gr_ny_z*gr_nz_y;
			double nn = n_d_xx*n_d_xx + n_d_yy*n_d_yy + n_d_zz*n_d_zz;
			double f_bulk = K * (1.0/2.0 * gr_n_sum + (nn-1.0)*(nn-1.0)/4.0/delta/delta);
			// f_anch cal
			double gr_cx_c = gra_phi_c(c,ex_d,cs,index);
			double gr_cy_c = gra_phi_c(c,ey_d,cs,index);
			double gr_cz_c = gra_phi_c(c,ez_d,cs,index);
			double sum_n_grad = n_d_xx*gr_cx_c + n_d_yy*gr_cy_c + n_d_zz*gr_cz_c;
			double f_anch = A * sum_n_grad * (n_d_x[index+1]-n_d_x[index-1] + n_d_y[index+nx]-n_d_y[index-nx] + n_d_z[index+nx*ny]-n_d_z[index-nx*ny]);

			m[index] = 2.0 * beta * cl * (cl - 1.0) * (2.0 * cl - 1.0) - kappa * lap_phi( c,cs,index ) + 1.0/2.0*f_bulk - f_anch;
		} else {
			if (bulk_free_energy == 1){
				m[index] = 2.0 * beta * cl * (cl - 1.0) * (2.0 * cl - 1.0) - kappa * lap_phi( c,cs,index );
		//		m[index] = -kappa * lap_phi( c,wt,cs,et,index );
			} else {
				m[index] = beta*(1.0 - 2.0*cl)*cl*(1.0 - cl)/fabs((cl + 1E-16)*(1.0 - cl + 1E-16)) - kappa * lap_phi( c,cs,index );
			}
		}
	}
}


__global__ void  chemical_in( double *c,double *m, double *n_d_x,double *n_d_y,double *n_d_z, double kappa,double beta )
{
	// i:2~nx+1, j:2~ny+1, k:4~nz/cpu-1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	
	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1 && k >= 4 && k <= nz/cpu-1) {
		int index = index_3d(i,j,k);
		const double cs = 1.0 / pow(3.0,0.5);
		const double cs2_inv = 3.0;
		double cl = c[index];

		if (energy_conserve == 1){
			double K = 0.00005, delta = 5.0, A = 0.000000001; // K為彈性係數ㄝ, delta沒有說, A是錨點係數(正數), 以上三個文章都說是常數
			// f_bulk cal
			double n_d_xx = n_d_x[index];
			double n_d_yy = n_d_y[index];
			double n_d_zz = n_d_z[index];
			double gr_nx_x = gra_phi_c(n_d_x,ex_d,cs,index);
			double gr_nx_y = gra_phi_c(n_d_x,ey_d,cs,index);
			double gr_nx_z = gra_phi_c(n_d_x,ez_d,cs,index);
			double gr_ny_x = gra_phi_c(n_d_y,ex_d,cs,index);
			double gr_ny_y = gra_phi_c(n_d_y,ey_d,cs,index);
			double gr_ny_z = gra_phi_c(n_d_y,ez_d,cs,index);
			double gr_nz_x = gra_phi_c(n_d_z,ex_d,cs,index);
			double gr_nz_y = gra_phi_c(n_d_z,ey_d,cs,index);
			double gr_nz_z = gra_phi_c(n_d_z,ez_d,cs,index);
			double gr_n_sum = gr_nx_x*gr_nx_x + gr_ny_y*gr_ny_y + gr_nz_z*gr_nz_z + 2*gr_nx_y*gr_ny_x + 2*gr_nx_z*gr_nz_x + 2*gr_ny_z*gr_nz_y;
			double nn = n_d_xx*n_d_xx + n_d_yy*n_d_yy + n_d_zz*n_d_zz;
			double f_bulk = K * (1.0/2.0 * gr_n_sum + (nn-1.0)*(nn-1.0)/4.0/delta/delta);
			// f_anch cal
			double gr_cx_c = gra_phi_c(c,ex_d,cs,index);
			double gr_cy_c = gra_phi_c(c,ey_d,cs,index);
			double gr_cz_c = gra_phi_c(c,ez_d,cs,index);
			double sum_n_grad = n_d_xx*gr_cx_c + n_d_yy*gr_cy_c + n_d_zz*gr_cz_c;
			double f_anch = A * sum_n_grad * (n_d_x[index+1]-n_d_x[index-1] + n_d_y[index+nx]-n_d_y[index-nx] + n_d_z[index+nx*ny]-n_d_z[index-nx*ny]);
			// if( i == nx/2 && j == ny/2 && k == nz/cpu/2){
			// 	printf("f_bulk = %.4e,\tf_anch = %.4e\n", f_bulk, f_anch);
			// }

			m[index] = 2.0 * beta * cl * (cl - 1.0) * (2.0 * cl - 1.0) - kappa * laplace_phi( c,cs2_inv,index ) + 1.0/2.0*f_bulk - f_anch;
		} else {
			if (bulk_free_energy == 1){
				m[index] = 2.0 * beta * cl * (cl - 1.0) * (2.0 * cl - 1.0) - kappa * laplace_phi( c,cs2_inv,index );
		//		m[index] = -kappa * lap_phi( c,wt,cs,et,index );
			} else {
				m[index] = beta*(1.0 - 2.0*cl)*cl*(1.0 - cl)/fabs((cl + 1E-16)*(1.0 - cl + 1E-16)) - kappa * laplace_phi( c,cs2_inv,index );
			}
		}
	}
}

__global__ void  chemical_bc( double *c,double *m, double *n_d_x,double *n_d_y,double *n_d_z, double kappa,double beta )
{
	// i:2~nx+1, j:2~ny+1, k:2,3,nz/cpu,nz/cpu+1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int kk[4]= {2, 3, nz/cpu, nz/cpu+1};
	
	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1) {
		for (int k = 0; k < 4; k++) {
			int index = index_3d(i,j,kk[k]);
			const double cs = 1.0 / pow(3.0,0.5);
			const double cs2_inv = 3.0;
			double cl = c[index];

			if (energy_conserve == 1){
				double K = 0.00005, delta = 5.0, A = 0.000000001; // K為彈性係數, delta沒有說, A是錨點係數(正數), 以上三個文章都說是常數
				// f_bulk cal
				double n_d_xx = n_d_x[index];
				double n_d_yy = n_d_y[index];
				double n_d_zz = n_d_z[index];
				double gr_nx_x = gra_phi_c(n_d_x,ex_d,cs,index);
				double gr_nx_y = gra_phi_c(n_d_x,ey_d,cs,index);
				double gr_nx_z = gra_phi_c(n_d_x,ez_d,cs,index);
				double gr_ny_x = gra_phi_c(n_d_y,ex_d,cs,index);
				double gr_ny_y = gra_phi_c(n_d_y,ey_d,cs,index);
				double gr_ny_z = gra_phi_c(n_d_y,ez_d,cs,index);
				double gr_nz_x = gra_phi_c(n_d_z,ex_d,cs,index);
				double gr_nz_y = gra_phi_c(n_d_z,ey_d,cs,index);
				double gr_nz_z = gra_phi_c(n_d_z,ez_d,cs,index);
				double gr_n_sum = gr_nx_x*gr_nx_x + gr_ny_y*gr_ny_y + gr_nz_z*gr_nz_z + 2*gr_nx_y*gr_ny_x + 2*gr_nx_z*gr_nz_x + 2*gr_ny_z*gr_nz_y;
				double nn = n_d_xx*n_d_xx + n_d_yy*n_d_yy + n_d_zz*n_d_zz;
				double f_bulk = K * (1.0/2.0 * gr_n_sum + (nn-1.0)*(nn-1.0)/4.0/delta/delta);
				// f_anch cal
				double gr_cx_c = gra_phi_c(c,ex_d,cs,index);
				double gr_cy_c = gra_phi_c(c,ey_d,cs,index);
				double gr_cz_c = gra_phi_c(c,ez_d,cs,index);
				double sum_n_grad = n_d_xx*gr_cx_c + n_d_yy*gr_cy_c + n_d_zz*gr_cz_c;
				double f_anch = A * sum_n_grad * (n_d_x[index+1]-n_d_x[index-1] + n_d_y[index+nx]-n_d_y[index-nx] + n_d_z[index+nx*ny]-n_d_z[index-nx*ny]);

				m[index] = 2.0 * beta * cl * (cl - 1.0) * (2.0 * cl - 1.0) - kappa * laplace_phi( c,cs2_inv,index ) + 1.0/2.0*f_bulk - f_anch;
			} else {
				if (bulk_free_energy == 1){	
					m[index] = 2.0 * beta * cl * (cl - 1.0) * (2.0 * cl - 1.0) - kappa * laplace_phi( c,cs2_inv,index );
			//		m[index] = -kappa * lap_phi( c,wt,cs,et,index );
				} else {
					m[index] = beta*(1.0 - 2.0*cl)*cl*(1.0 - cl)/fabs((cl + 1E-16)*(1.0 - cl + 1E-16)) - kappa * laplace_phi( c,cs2_inv,index );
				}
			}
		}
	}
}

#endif
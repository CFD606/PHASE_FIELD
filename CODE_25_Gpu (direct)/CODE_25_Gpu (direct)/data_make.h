#ifndef DATA_MAKE_H
#define DATA_MAKE_H

#include"preparation.h"

double maxvalue(double *phi)
{
	double max = 0.0;
	int i, j, k;
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			for (k = 0; k < nz; k++)
			{
				int index = nx * (k * ny + j) + i;
				if (max < phi[index])
				{
					max = phi[index];
				}
			}
		}
	}
	return max;
}
double minvalue(double *phi)
{
	double min = 100.0;
	int i, j, k;
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			for (k = 0; k < nz; k++)
			{
				int index = nx * (k * ny + j) + i;
				if (min > phi[index])
				{
					min = phi[index];
				}
			}
		}
	}
	return min;
}

/******************* pressure ********************/

__global__ void p_real(double *c, double *p, double *a, double beta, double kappa)
{
	// i:2~nx+1, j:2~ny+1, k:2~nz/cpu+1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;

	if (i >= 2 && i <= nx + 1 && j >= 2 && j <= ny + 1 && k >= 2 && k <= nz / cpu + 1)
	{
		double cs = 1.0 / pow(3.0, 0.5);

		int index = index_3d(i, j, k);
		double th, cu, eo, mo;
		double gr_cx_c = gra_phi_c(c, ex_d, cs, index);
		double gr_cy_c = gra_phi_c(c, ey_d, cs, index);
		double gr_cz_c = gra_phi_c(c, ez_d, cs, index);
		double la_c = lap_phi(c,cs, index);
		eo = beta * pow(c[index] * (c[index] - 1.0), 2.0);
		mo = beta * (4.0 * pow(c[index], 3.0) - 6.0 * pow(c[index], 2.0) + 2.0 * c[index]);
		th = c[index] * mo - eo;
		cu = kappa * (-c[index] * la_c + 0.5 * (gr_cx_c * gr_cx_c + gr_cy_c * gr_cy_c + gr_cz_c * gr_cz_c));
		a[index] = th + cu + p[index];
	}
}
double p_difference(double *c, double *a)
{
	int i, j, k;
	double dp = 0.0;
	double p_in = 0.0;
	double p_ou = 0.0;
	int icent = nx / 2;
	int jcent = ny / 2;
	int kcent = nz / 2;
	i = icent;
	j = jcent;
	k = kcent;
	int index = nx * (ny * (k) + j) + i;
	p_in = a[index];
	p_ou = a[0];
	dp = p_in - p_ou;
	return dp;
}
double pressure_check(double *c, double *a)
{
	int i, j, k, index, sum_num_in = 0, sum_num_out = 0;
	double sum_p_in, sum_p_out;

	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			for (k = 0; k < nz; k++)
			{
				index = i+nx*(j+ny*k);
				if(c[index]>=0.5)
				{
					sum_num_in = sum_num_in + 1;
					sum_p_in = sum_p_in + a[index];
				}
				if(c[index]<0.5)
				{
					sum_num_out = sum_num_out + 1;
					sum_p_out = sum_p_out + a[index];
				}
			}
		}
	}

	return (sum_p_in/sum_num_in)-(sum_p_out/sum_num_out);
}

/******************* length ********************/

double length_x(double *c)
{
	int i, icent, j,k;
	double condition_l;
	double position_max, position_min;
	j = ny/2;
	icent = nx / 2;
	k = nz / 2;
	position_max = (double)icent;
	for (i = icent; i < nx - 1; i++)
	{
		int index = nx*j+(nx*ny)*k + i;
		int index_t = nx*j+(nx*ny)*k + i+1;
		condition_l = (c[index] - 0.5) * (c[index_t] - 0.5);
		if (condition_l <= 0.0)
		{
			position_max = (double)(i + 1.0) - (c[index_t] - 0.5) * 1.0 / (c[index_t] - c[index]);
		}
	}
	position_min = (double)icent;
	for (i = icent; i > 0; i--)
	{
		int index = nx*j+(nx+ny)*k + i;
		int index_t = nx*j+(nx*ny)*k + i-1;
		condition_l = (c[index] - 0.5) * (c[index_t] - 0.5);
		if (condition_l <= 0.0)
		{
			position_min = (double)(i - 1.0) + (c[index_t] - 0.5) * 1.0 / (c[index_t] - c[index]);
		}
	}
	return position_max - position_min;
}
double length_y(double *c)
{
	int i,j,k,jcent;
	double condition_l;
	double position_max, position_min;
	jcent = ny/2;
	i = nx/2;
	k = nz/2;
	position_max = (double)jcent;
	for (j = jcent; j < ny; j++)
	{
		int index = nx*j+(nx*ny)*k + i;
		int index_t = nx*(j+1)+(nx*ny)*k + i;
		condition_l = (c[index] - 0.5) * (c[index_t] - 0.5);
		if (condition_l <= 0.0)
		{
			position_max = (double)(j + 1.0)- (c[index_t] - 0.5) * 1.0 / (c[index_t] - c[index]);
		}
	}
	position_min = (double)jcent;
	for (j = jcent; j > 0; j--)
	{
		int index = nx*j+(nx*ny)*k + i;
		int index_t = nx*(j-1)+(nx*ny)*k + i;
		condition_l = (c[index] - 0.50) * (c[index_t] - 0.50);
		if (condition_l <= 0.0)
		{
			position_min = (double)(j - 1.0)+ (c[index_t] - 0.5) * 1.0 / (c[index_t] - c[index]);
		}
	}
	return position_max - position_min;
}
double length_z(double *c)
{
	int i, k, j,kcent;
	double condition_l;
	double position_max, position_min;
	kcent = nz / 2;
	j = ny/2;
	i = nx / 2;
	position_max = (double)kcent;
	for (k = kcent; k < nz - 1; k++)
	{
		int index = nx*j+(nx*ny)*k + i;
		int index_t = nx*j+(nx*ny)*(k+1) + i;
		condition_l = (c[index] - 0.5) * (c[index_t] - 0.5);
		if (condition_l <= 0.0)
		{
			position_max = (double)(k + 1.0) - (c[index_t] - 0.5) * 1.0 / (c[index_t] - c[index]);
		}
	}
	position_min = (double)kcent;
	for (k = kcent; k > 0; k--)
	{
		int index = nx*j+(nx*ny)*k + i;
		int index_t = nx*j+(nx*ny)*(k-1) + i; 
		condition_l = (c[index] - 0.5) * (c[index_t] - 0.5);
		if (condition_l <= 0.0)
		{
			position_min = (double)(k - 1.0) + (c[index_t] - 0.5) * 1.0 / (c[index_t] - c[index]);
		}
	}
	return position_max - position_min; // position_max;//
}
void length_xyz(double *c, double *lx, double *lz, int step)
{
	lx[step / 2 - 1] = 0.0;
	lz[step / 2 - 1] = 0.0;
	lx[step / 2 - 1] = length_x(c);
	lz[step / 2 - 1] = length_z(c);
}
void diameter_3d_print(){
	char name[5];
	if(step == iprint){
		sprintf(name,"w");
		diameter_3d = fopen("./data/diameter_xyz.dat",name);
		CHECK_FILE(diameter_3d,"./data/diameter_xyz.dat");
		fprintf( diameter_3d, "VARIABLES=\"step\",\"diameter\"\n");
    	fprintf( diameter_3d, "ZONE T=\"mobility:%f,surface_tension:%f\" F=POINT\n",(double)mobility,(double)surface_tension);
    	fprintf( diameter_3d, "I=%d\n", stepall/iprint);
	}
	else{
		sprintf(name,"a");	
		diameter_3d = fopen("./data/diameter_xyz.dat",name);
	}

	fprintf( diameter_3d,"%d\t%e\n",step,cbrt(length_x(c_f_h)*length_z(c_f_h)*length_y(c_f_h))/2.0);
    fclose(diameter_3d);


}

/************************* Energy calculation ***********************************/

double Energy_KE(double *cc, double *u, double *v, double *w)
{
	int i, j, k, index;
	double sum_KE = 0.0, rho;

	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			for (k = 0; k < nz; k++)
			{
				/*
				index = i+nx*(j+ny*k);
				if(cc[index]>=0.5){
					rho = cc[index]*rho_l+((double)1.0-cc[index])*rho_g;
					sum_KE = sum_KE + 0.5*rho*(u[index]*u[index] + v[index]*v[index] + w[index]*w[index]);
				}else{
					sum_KE = sum_KE + 0;
				}
				*/
				index = i + nx * (j + ny * k);
				rho = cc[index] * rho_l + ((double)1.0 - cc[index]) * rho_g;
				sum_KE = sum_KE + 0.5 * rho * (u[index] * u[index] + v[index] * v[index] + w[index] * w[index]);
			}
		}
	}

	return sum_KE;
}

double Energy_KE_gas(double *cc, double *u, double *v, double *w)
{
	int i, j, k, index;
	double sum_KE = 0.0, rho;

	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			for (k = 0; k < nz; k++)
			{
				index = i+nx*(j+ny*k);
				if(cc[index]<=0.1)
				{
					rho = cc[index]*rho_l+((double)1.0-cc[index])*rho_g;
					sum_KE = sum_KE + 0.5*rho*(u[index]*u[index] + v[index]*v[index] + w[index]*w[index]);
				}
				else
				{
					sum_KE = sum_KE + 0;
				}
			}
		}
	}

	return sum_KE;
}

double Energy_KE_liq(double *cc, double *u, double *v, double *w)
{
	int i, j, k, index;
	double sum_KE = 0.0, rho;

	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			for (k = 0; k < nz; k++)
			{
				index = i+nx*(j+ny*k);
				if(cc[index]>=0.9)
				{
					rho = cc[index]*rho_l+((double)1.0-cc[index])*rho_g;
					sum_KE = sum_KE + 0.5*rho*(u[index]*u[index] + v[index]*v[index] + w[index]*w[index]);
				}
				else
				{
					sum_KE = sum_KE + 0;
				}
			}
		}
	}

	return sum_KE;
}

double Energy_KE_diff(double *cc, double *u, double *v, double *w)
{
	int i, j, k, index;
	double sum_KE = 0.0, rho;

	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			for (k = 0; k < nz; k++)
			{
				index = i+nx*(j+ny*k);
				if(cc[index]<0.9 && cc[index]>0.1)
				{
					rho = cc[index]*rho_l+((double)1.0-cc[index])*rho_g;
					sum_KE = sum_KE + 0.5*rho*(u[index]*u[index] + v[index]*v[index] + w[index]*w[index]);
				}
				else
				{
					sum_KE = sum_KE + 0;
				}
			}
		}
	}

	return sum_KE;
}

double Energy_VDR(double *cc, double *u, double *v, double *w)
{
	int i, j, k, index;
	double sum_VDR, rho, mu;
	double dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz;
	double cs = 1 / sqrt(3);
	double nu, tau;

	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			for (k = 0; k < nz; k++)
			{

				index = i + nx * (j + ny * k);

				tau = cc[index] * tau_l + ((double)1.0 - cc[index]) * tau_g;
				nu = tau * dt * cs * cs;
				rho = cc[index] * rho_l + ((double)1.0 - cc[index]) * rho_g;
				mu = nu * rho;
				/*
							if(cc[index]>=0.5){
								// second order
			/*
								if(i==0){
									dudx = (u[index+1] - u[index+nx-1])/(2*dx);
									dvdx = (v[index+1] - v[index+nx-1])/(2*dx);
									dwdx = (w[index+1] - w[index+nx-1])/(2*dx);
								}else if(i==nx){
									dudx = (u[index-nx+1] - u[index-1])/(2*dx);
									dvdx = (v[index-nx+1] - v[index-1])/(2*dx);
									dwdx = (w[index-nx+1] - w[index-1])/(2*dx);
								}else{
									dudx = (u[index+1] - u[index-1])/(2*dx);
									dvdx = (v[index+1] - v[index-1])/(2*dx);
									dwdx = (w[index+1] - w[index-1])/(2*dx);
								}

								if(j==0){
									dvdy = (v[index+nx] - v[index+ny*nx-nx])/(2*dy);
									dudy = (u[index+nx] - u[index+ny*nx-nx])/(2*dy);
									dwdy = (w[index+nx] - w[index+ny*nx-nx])/(2*dy);
								}else if(j==ny){
									dvdy = (v[index-nx*ny+nx] - v[index-nx])/(2*dy);
									dudy = (u[index-nx*ny+nx] - u[index-nx])/(2*dy);
									dwdy = (w[index-nx*ny+nx] - w[index-nx])/(2*dy);
								}else{
									dvdy = (v[index+nx] - v[index-nx])/(2*dy);
									dudy = (u[index+nx] - u[index-nx])/(2*dy);
									dwdy = (w[index+nx] - w[index-nx])/(2*dy);
								}

								if(k==0){
									dwdz = (w[index+nx*ny] - w[index+nx*ny*nz-nx*ny])/(2*dz);
									dudz = (u[index+nx*ny] - u[index+nx*ny*nz-nx*ny])/(2*dz);
									dvdz = (v[index+nx*ny] - v[index+nx*ny*nz-nx*ny])/(2*dz);
								}else if(k==nz){
									dwdz = (w[index-nx*ny*nz+nx*ny] - w[index-nx*ny])/(2*dz);
									dudz = (u[index-nx*ny*nz+nx*ny] - u[index-nx*ny])/(2*dz);
									dvdz = (v[index-nx*ny*nz+nx*ny] - v[index-nx*ny])/(2*dz);
								}else{
									dwdz = (w[index+nx*ny] - w[index-nx*ny])/(2*dz);
									dudz = (u[index+nx*ny] - u[index-nx*ny])/(2*dz);
									dvdz = (v[index+nx*ny] - v[index-nx*ny])/(2*dz);
								}

								// fourth order
								if(i==0){
									dudx = (-u[index+2] + 8*u[index+1] - 8*u[index+nx-1] + u[index+nx-2])/(12*dx);
									dvdx = (-v[index+2] + 8*v[index+1] - 8*v[index+nx-1] + v[index+nx-2])/(12*dx);
									dwdx = (-w[index+2] + 8*w[index+1] - 8*w[index+nx-1] + w[index+nx-2])/(12*dx);
								}else if(i==nx){
									dudx = (-u[index-nx+2] + 8*u[index-nx+1] - 8*u[index-1] + u[index-2])/(12*dx);
									dvdx = (-v[index-nx+2] + 8*v[index-nx+1] - 8*v[index-1] + v[index-2])/(12*dx);
									dwdx = (-w[index-nx+2] + 8*w[index-nx+1] - 8*w[index-1] + w[index-2])/(12*dx);
								}else{
									dudx = (-u[index+2] + 8*u[index+1] - 8*u[index-1] + u[index-2])/(12*dx);
									dvdx = (-v[index+2] + 8*v[index+1] - 8*v[index-1] + v[index-2])/(12*dx);
									dwdx = (-w[index+2] + 8*w[index+1] - 8*w[index-1] + w[index-2])/(12*dx);
								}

								if(j==0){
									dvdy = (-v[index+2*nx] + 8*v[index+nx] - 8*v[index+nx*ny-nx] + v[index+nx*ny-2*nx])/(12*dy);
									dudy = (-u[index+2*nx] + 8*u[index+nx] - 8*u[index+nx*ny-nx] + u[index+nx*ny-2*nx])/(12*dy);
									dwdy = (-w[index+2*nx] + 8*w[index+nx] - 8*w[index+nx*ny-nx] + w[index+nx*ny-2*nx])/(12*dy);
								}else if(j==ny){
									dvdy = (-v[index-nx*ny+2*nx] + 8*v[index-nx*ny+nx] - 8*v[index-nx] + v[index-2*nx])/(12*dy);
									dudy = (-u[index-nx*ny+2*nx] + 8*u[index-nx*ny+nx] - 8*u[index-nx] + u[index-2*nx])/(12*dy);
									dwdy = (-w[index-nx*ny+2*nx] + 8*w[index-nx*ny+nx] - 8*w[index-nx] + w[index-2*nx])/(12*dy);
								}else{
									dvdy = (-v[index+2*nx] + 8*v[index+nx] - 8*v[index-nx] + v[index-2*nx])/(12*dy);
									dudy = (-u[index+2*nx] + 8*u[index+nx] - 8*u[index-nx] + u[index-2*nx])/(12*dy);
									dwdy = (-w[index+2*nx] + 8*w[index+nx] - 8*w[index-nx] + w[index-2*nx])/(12*dy);
								}

								if(k==0){
									dwdz = (-w[index+2*nx*ny] + 8*w[index+nx*ny] - 8*w[index+nx*ny*nz-nx*ny] + w[index+nx*ny*nz-2*nx*ny])/(12*dz);
									dudz = (-u[index+2*nx*ny] + 8*u[index+nx*ny] - 8*u[index+nx*ny*nz-nx*ny] + u[index+nx*ny*nz-2*nx*ny])/(12*dz);
									dvdz = (-v[index+2*nx*ny] + 8*v[index+nx*ny] - 8*v[index+nx*ny*nz-nx*ny] + v[index+nx*ny*nz-2*nx*ny])/(12*dz);
								}else if(k==nz){
									dwdz = (-w[index-nx*ny*nz+2*nx*ny] + 8*w[index-nx*ny*nz+nx*ny] - 8*w[index-nx*ny] + w[index-2*nx*ny])/(12*dz);
									dudz = (-u[index-nx*ny*nz+2*nx*ny] + 8*u[index-nx*ny*nz+nx*ny] - 8*u[index-nx*ny] + u[index-2*nx*ny])/(12*dz);
									dvdz = (-v[index-nx*ny*nz+2*nx*ny] + 8*v[index-nx*ny*nz+nx*ny] - 8*v[index-nx*ny] + v[index-2*nx*ny])/(12*dz);
								}else{
									dwdz = (-w[index+2*nx*ny] + 8*w[index+nx*ny] - 8*w[index-nx*ny] + w[index-2*nx*ny])/(12*dz);
									dudz = (-u[index+2*nx*ny] + 8*u[index+nx*ny] - 8*u[index-nx*ny] + u[index-2*nx*ny])/(12*dz);
									dvdz = (-v[index+2*nx*ny] + 8*v[index+nx*ny] - 8*v[index-nx*ny] + v[index-2*nx*ny])/(12*dz);
								}

								sum_VDR = sum_VDR + mu*(2*dudx*dudx + 2*dvdy*dvdy + 2*dwdz*dwdz + (dvdx + dudy)*(dvdx + dudy) + (dwdy + dvdz)*(dwdy + dvdz) + (dudz + dwdx)*(dudz + dwdx)) - 2*mu*(dudx + dvdy + dwdz)*(dudx + dvdy + dwdz)/3.0;
							}else{
								sum_VDR = sum_VDR + 0;
							}
			*/
				// second order
				
				if (i == 0)
				{
					dudx = (u[index + 1] - u[index + nx - 1]) / (2 * dx);
					dvdx = (v[index + 1] - v[index + nx - 1]) / (2 * dx);
					dwdx = (w[index + 1] - w[index + nx - 1]) / (2 * dx);
				}
				else if (i == nx)
				{
					dudx = (u[index - nx + 1] - u[index - 1]) / (2 * dx);
					dvdx = (v[index - nx + 1] - v[index - 1]) / (2 * dx);
					dwdx = (w[index - nx + 1] - w[index - 1]) / (2 * dx);
				}
				else
				{
					dudx = (u[index + 1] - u[index - 1]) / (2 * dx);
					dvdx = (v[index + 1] - v[index - 1]) / (2 * dx);
					dwdx = (w[index + 1] - w[index - 1]) / (2 * dx);
				}

				if (j == 0)
				{
					dvdy = (v[index + nx] - v[index + ny * nx - nx]) / (2 * dy);
					dudy = (u[index + nx] - u[index + ny * nx - nx]) / (2 * dy);
					dwdy = (w[index + nx] - w[index + ny * nx - nx]) / (2 * dy);
				}
				else if (j == ny)
				{
					dvdy = (v[index - nx * ny + nx] - v[index - nx]) / (2 * dy);
					dudy = (u[index - nx * ny + nx] - u[index - nx]) / (2 * dy);
					dwdy = (w[index - nx * ny + nx] - w[index - nx]) / (2 * dy);
				}
				else
				{
					dvdy = (v[index + nx] - v[index - nx]) / (2 * dy);
					dudy = (u[index + nx] - u[index - nx]) / (2 * dy);
					dwdy = (w[index + nx] - w[index - nx]) / (2 * dy);
				}

				if (k == 0)
				{
					dwdz = (w[index + nx * ny] - w[index + nx * ny * nz - nx * ny]) / (2 * dz);
					dudz = (u[index + nx * ny] - u[index + nx * ny * nz - nx * ny]) / (2 * dz);
					dvdz = (v[index + nx * ny] - v[index + nx * ny * nz - nx * ny]) / (2 * dz);
				}
				else if (k == nz)
				{
					dwdz = (w[index - nx * ny * nz + nx * ny] - w[index - nx * ny]) / (2 * dz);
					dudz = (u[index - nx * ny * nz + nx * ny] - u[index - nx * ny]) / (2 * dz);
					dvdz = (v[index - nx * ny * nz + nx * ny] - v[index - nx * ny]) / (2 * dz);
				}
				else
				{
					dwdz = (w[index + nx * ny] - w[index - nx * ny]) / (2 * dz);
					dudz = (u[index + nx * ny] - u[index - nx * ny]) / (2 * dz);
					dvdz = (v[index + nx * ny] - v[index - nx * ny]) / (2 * dz);
				}
				/*
				//fourth order
				if(i==0){
					dudx = (-u[index+2] + 8*u[index+1] - 8*u[index+nx-1] + u[index+nx-2])/(12*dx);
					dvdx = (-v[index+2] + 8*v[index+1] - 8*v[index+nx-1] + v[index+nx-2])/(12*dx);
					dwdx = (-w[index+2] + 8*w[index+1] - 8*w[index+nx-1] + w[index+nx-2])/(12*dx);
				}else if(i==nx){
					dudx = (-u[index-nx+2] + 8*u[index-nx+1] - 8*u[index-1] + u[index-2])/(12*dx);
					dvdx = (-v[index-nx+2] + 8*v[index-nx+1] - 8*v[index-1] + v[index-2])/(12*dx);
					dwdx = (-w[index-nx+2] + 8*w[index-nx+1] - 8*w[index-1] + w[index-2])/(12*dx);
				}else{
					dudx = (-u[index+2] + 8*u[index+1] - 8*u[index-1] + u[index-2])/(12*dx);
					dvdx = (-v[index+2] + 8*v[index+1] - 8*v[index-1] + v[index-2])/(12*dx);
					dwdx = (-w[index+2] + 8*w[index+1] - 8*w[index-1] + w[index-2])/(12*dx);
				}

				if(j==0){
					dvdy = (-v[index+2*nx] + 8*v[index+nx] - 8*v[index+nx*ny-nx] + v[index+nx*ny-2*nx])/(12*dy);
					dudy = (-u[index+2*nx] + 8*u[index+nx] - 8*u[index+nx*ny-nx] + u[index+nx*ny-2*nx])/(12*dy);
					dwdy = (-w[index+2*nx] + 8*w[index+nx] - 8*w[index+nx*ny-nx] + w[index+nx*ny-2*nx])/(12*dy);
				}else if(j==ny){
					dvdy = (-v[index-nx*ny+2*nx] + 8*v[index-nx*ny+nx] - 8*v[index-nx] + v[index-2*nx])/(12*dy);
					dudy = (-u[index-nx*ny+2*nx] + 8*u[index-nx*ny+nx] - 8*u[index-nx] + u[index-2*nx])/(12*dy);
					dwdy = (-w[index-nx*ny+2*nx] + 8*w[index-nx*ny+nx] - 8*w[index-nx] + w[index-2*nx])/(12*dy);
				}else{
					dvdy = (-v[index+2*nx] + 8*v[index+nx] - 8*v[index-nx] + v[index-2*nx])/(12*dy);
					dudy = (-u[index+2*nx] + 8*u[index+nx] - 8*u[index-nx] + u[index-2*nx])/(12*dy);
					dwdy = (-w[index+2*nx] + 8*w[index+nx] - 8*w[index-nx] + w[index-2*nx])/(12*dy);
				}

				if(k==0){
					dwdz = (-w[index+2*nx*ny] + 8*w[index+nx*ny] - 8*w[index+nx*ny*nz-nx*ny] + w[index+nx*ny*nz-2*nx*ny])/(12*dz);
					dudz = (-u[index+2*nx*ny] + 8*u[index+nx*ny] - 8*u[index+nx*ny*nz-nx*ny] + u[index+nx*ny*nz-2*nx*ny])/(12*dz);
					dvdz = (-v[index+2*nx*ny] + 8*v[index+nx*ny] - 8*v[index+nx*ny*nz-nx*ny] + v[index+nx*ny*nz-2*nx*ny])/(12*dz);
				}else if(k==nz){
					dwdz = (-w[index-nx*ny*nz+2*nx*ny] + 8*w[index-nx*ny*nz+nx*ny] - 8*w[index-nx*ny] + w[index-2*nx*ny])/(12*dz);
					dudz = (-u[index-nx*ny*nz+2*nx*ny] + 8*u[index-nx*ny*nz+nx*ny] - 8*u[index-nx*ny] + u[index-2*nx*ny])/(12*dz);
					dvdz = (-v[index-nx*ny*nz+2*nx*ny] + 8*v[index-nx*ny*nz+nx*ny] - 8*v[index-nx*ny] + v[index-2*nx*ny])/(12*dz);
				}else{
					dwdz = (-w[index+2*nx*ny] + 8*w[index+nx*ny] - 8*w[index-nx*ny] + w[index-2*nx*ny])/(12*dz);
					dudz = (-u[index+2*nx*ny] + 8*u[index+nx*ny] - 8*u[index-nx*ny] + u[index-2*nx*ny])/(12*dz);
					dvdz = (-v[index+2*nx*ny] + 8*v[index+nx*ny] - 8*v[index-nx*ny] + v[index-2*nx*ny])/(12*dz);
				}
				*/
				sum_VDR = sum_VDR + mu * (2 * dudx * dudx + 2 * dvdy * dvdy + 2 * dwdz * dwdz + (dvdx + dudy) * (dvdx + dudy) + (dwdy + dvdz) * (dwdy + dvdz) + (dudz + dwdx) * (dudz + dwdx)) - 2 * mu * (dudx + dvdy + dwdz) * (dudx + dvdy + dwdz) / 3.0;
			}
		}
	}

	return sum_VDR;
}

__global__ void DE_cal(double *cc, double *u, double *v, double *w, double *DE)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i > 1 && i < nx + 2 && j > 1 && j < ny + 2 && k > 1 && k < nz / cpu + 2)
	{
		int index = index_3d(i, j, k);
		/*
				if(cc[index]>=0.5){
					double cs = 1.0/sqrt(3.0);
					double tau = cc[index]*tau_l+((double)1.0-cc[index])*tau_g;
					double nu = tau*dt*cs*cs;
					double rho = cc[index]*rho_l+((double)1.0-cc[index])*rho_g;
					double mu = nu * rho;
					// second order difference
					/*
					double dudx = (u[index+1] - u[index-1])/(2*dx);
					double dvdx = (v[index+1] - v[index-1])/(2*dx);
					double dwdx = (w[index+1] - w[index-1])/(2*dx);

					double dvdy = (v[index+(nx+4)] - v[index-(nx+4)])/(2*dy);
					double dudy = (u[index+(nx+4)] - u[index-(nx+4)])/(2*dy);
					double dwdy = (w[index+(nx+4)] - w[index-(nx+4)])/(2*dy);

					double dwdz = (w[index+(nx+4)*(ny+4)] - w[index-(nx+4)*(ny+4)])/(2*dz);
					double dudz = (u[index+(nx+4)*(ny+4)] - u[index-(nx+4)*(ny+4)])/(2*dz);
					double dvdz = (v[index+(nx+4)*(ny+4)] - v[index-(nx+4)*(ny+4)])/(2*dz);

					// fourth order difference
					double dudx = (-u[index+2] + 8*u[index+1] - 8*u[index-1] + u[index-2])/(12*dx);
					double dvdx = (-v[index+2] + 8*v[index+1] - 8*v[index-1] + v[index-2])/(12*dx);
					double dwdx = (-w[index+2] + 8*w[index+1] - 8*w[index-1] + w[index-2])/(12*dx);

					double dvdy = (-v[index+2*(nx+4)] + 8*v[index+(nx+4)] - 8*v[index-(nx+4)] + v[index-2*(nx+4)])/(12*dy);
					double dudy = (-u[index+2*(nx+4)] + 8*u[index+(nx+4)] - 8*u[index-(nx+4)] + u[index-2*(nx+4)])/(12*dy);
					double dwdy = (-w[index+2*(nx+4)] + 8*w[index+(nx+4)] - 8*w[index-(nx+4)] + w[index-2*(nx+4)])/(12*dy);

					double dwdz = (-w[index+2*(nx+4)*(ny+4)] + 8*w[index+(nx+4)*(ny+4)] - 8*w[index-(nx+4)*(ny+4)] + w[index-2*(nx+4)*(ny+4)])/(12*dz);
					double dudz = (-u[index+2*(nx+4)*(ny+4)] + 8*u[index+(nx+4)*(ny+4)] - 8*u[index-(nx+4)*(ny+4)] + u[index-2*(nx+4)*(ny+4)])/(12*dz);
					double dvdz = (-v[index+2*(nx+4)*(ny+4)] + 8*v[index+(nx+4)*(ny+4)] - 8*v[index-(nx+4)*(ny+4)] + v[index-2*(nx+4)*(ny+4)])/(12*dz);

					DE[index] = DE[index] + mu*(2*dudx*dudx + 2*dvdy*dvdy + 2*dwdz*dwdz + (dvdx + dudy)*(dvdx + dudy) + (dwdy + dvdz)*(dwdy + dvdz) + (dudz + dwdx)*(dudz + dwdx)) - 2*mu*(dudx + dvdy + dwdz)*(dudx + dvdy + dwdz)/3.0;
				}else{
					DE[index] = DE[index] + 0.0;
				}
		*/
		double cs = 1.0 / sqrt(3.0);
		double tau = cc[index] * tau_l + ((double)1.0 - cc[index]) * tau_g;
		double nu = tau * dt * cs * cs;
		double rho = cc[index] * rho_l + ((double)1.0 - cc[index]) * rho_g;
		double mu = nu * rho;
		
		//second order
		
		double dudx = (u[index + 1] - u[index - 1]) / (2 * dx);
		double dvdx = (v[index + 1] - v[index - 1]) / (2 * dx);
		double dwdx = (w[index + 1] - w[index - 1]) / (2 * dx);
		double dvdy = (v[index + (nx + 4)] - v[index - (nx + 4)]) / (2 * dy);
		double dudy = (u[index + (nx + 4)] - u[index - (nx + 4)]) / (2 * dy);
		double dwdy = (w[index + (nx + 4)] - w[index - (nx + 4)]) / (2 * dy);
		double dwdz = (w[index + (nx + 4) * (ny + 4)] - w[index - (nx + 4) * (ny + 4)]) / (2 * dz);
		double dudz = (u[index + (nx + 4) * (ny + 4)] - u[index - (nx + 4) * (ny + 4)]) / (2 * dz);
		double dvdz = (v[index + (nx + 4) * (ny + 4)] - v[index - (nx + 4) * (ny + 4)]) / (2 * dz);
		/*
		//fourth order
		double dudx = (-u[index+2] + 8*u[index+1] - 8*u[index-1] + u[index-2])/(12*dx);
		double dvdx = (-v[index+2] + 8*v[index+1] - 8*v[index-1] + v[index-2])/(12*dx);
		double dwdx = (-w[index+2] + 8*w[index+1] - 8*w[index-1] + w[index-2])/(12*dx);
		double dvdy = (-v[index+2*(nx+4)] + 8*v[index+(nx+4)] - 8*v[index-(nx+4)] + v[index-2*(nx+4)])/(12*dy);
		double dudy = (-u[index+2*(nx+4)] + 8*u[index+(nx+4)] - 8*u[index-(nx+4)] + u[index-2*(nx+4)])/(12*dy);
		double dwdy = (-w[index+2*(nx+4)] + 8*w[index+(nx+4)] - 8*w[index-(nx+4)] + w[index-2*(nx+4)])/(12*dy);
		double dwdz = (-w[index+2*(nx+4)*(ny+4)] + 8*w[index+(nx+4)*(ny+4)] - 8*w[index-(nx+4)*(ny+4)] + w[index-2*(nx+4)*(ny+4)])/(12*dz);
		double dudz = (-u[index+2*(nx+4)*(ny+4)] + 8*u[index+(nx+4)*(ny+4)] - 8*u[index-(nx+4)*(ny+4)] + u[index-2*(nx+4)*(ny+4)])/(12*dz);
		double dvdz = (-v[index+2*(nx+4)*(ny+4)] + 8*v[index+(nx+4)*(ny+4)] - 8*v[index-(nx+4)*(ny+4)] + v[index-2*(nx+4)*(ny+4)])/(12*dz);
		*/
		DE[index] = DE[index] + mu * (2 * dudx * dudx + 2 * dvdy * dvdy + 2 * dwdz * dwdz + (dvdx + dudy) * (dvdx + dudy) + (dwdy + dvdz) * (dwdy + dvdz) + (dudz + dwdx) * (dudz + dwdx)) - 2 * mu * (dudx + dvdy + dwdz) * (dudx + dvdy + dwdz) / 3.0;
	}
}

__global__ void DEgas_cal(double *cc, double *u, double *v, double *w, double *DE)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i > 1 && i < nx + 2 && j > 1 && j < ny + 2 && k > 1 && k < nz / cpu + 2)
	{
		int index = index_3d(i, j, k);

		if (cc[index] <= 0.1)
		{
			double cs = 1.0 / sqrt(3.0);
			double tau = cc[index] * tau_l + ((double)1.0 - cc[index]) * tau_g;
			double nu = tau * dt * cs * cs;
			double rho = cc[index] * rho_l + ((double)1.0 - cc[index]) * rho_g;
			double mu = nu * rho;
			// second order difference

			double dudx = (u[index + 1] - u[index - 1]) / (2 * dx);
			double dvdx = (v[index + 1] - v[index - 1]) / (2 * dx);
			double dwdx = (w[index + 1] - w[index - 1]) / (2 * dx);

			double dvdy = (v[index + (nx + 4)] - v[index - (nx + 4)]) / (2 * dy);
			double dudy = (u[index + (nx + 4)] - u[index - (nx + 4)]) / (2 * dy);
			double dwdy = (w[index + (nx + 4)] - w[index - (nx + 4)]) / (2 * dy);

			double dwdz = (w[index + (nx + 4) * (ny + 4)] - w[index - (nx + 4) * (ny + 4)]) / (2 * dz);
			double dudz = (u[index + (nx + 4) * (ny + 4)] - u[index - (nx + 4) * (ny + 4)]) / (2 * dz);
			double dvdz = (v[index + (nx + 4) * (ny + 4)] - v[index - (nx + 4) * (ny + 4)]) / (2 * dz);

			// fourth order difference
			// double dudx = (-u[index+2] + 8*u[index+1] - 8*u[index-1] + u[index-2])/(12*dx);
			// double dvdx = (-v[index+2] + 8*v[index+1] - 8*v[index-1] + v[index-2])/(12*dx);
			// double dwdx = (-w[index+2] + 8*w[index+1] - 8*w[index-1] + w[index-2])/(12*dx);

			// double dvdy = (-v[index+2*(nx+4)] + 8*v[index+(nx+4)] - 8*v[index-(nx+4)] + v[index-2*(nx+4)])/(12*dy);
			// double dudy = (-u[index+2*(nx+4)] + 8*u[index+(nx+4)] - 8*u[index-(nx+4)] + u[index-2*(nx+4)])/(12*dy);
			// double dwdy = (-w[index+2*(nx+4)] + 8*w[index+(nx+4)] - 8*w[index-(nx+4)] + w[index-2*(nx+4)])/(12*dy);

			// double dwdz = (-w[index+2*(nx+4)*(ny+4)] + 8*w[index+(nx+4)*(ny+4)] - 8*w[index-(nx+4)*(ny+4)] + w[index-2*(nx+4)*(ny+4)])/(12*dz);
			// double dudz = (-u[index+2*(nx+4)*(ny+4)] + 8*u[index+(nx+4)*(ny+4)] - 8*u[index-(nx+4)*(ny+4)] + u[index-2*(nx+4)*(ny+4)])/(12*dz);
			// double dvdz = (-v[index+2*(nx+4)*(ny+4)] + 8*v[index+(nx+4)*(ny+4)] - 8*v[index-(nx+4)*(ny+4)] + v[index-2*(nx+4)*(ny+4)])/(12*dz);

			DE[index] = DE[index] + mu * (2 * dudx * dudx + 2 * dvdy * dvdy + 2 * dwdz * dwdz + (dvdx + dudy) * (dvdx + dudy) + (dwdy + dvdz) * (dwdy + dvdz) + (dudz + dwdx) * (dudz + dwdx)) - 2 * mu * (dudx + dvdy + dwdz) * (dudx + dvdy + dwdz) / 3.0;
		}
		else
		{
			DE[index] = DE[index] + 0.0;
		}
	}
}

__global__ void DEliq_cal(double *cc, double *u, double *v, double *w, double *DE)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i > 1 && i < nx + 2 && j > 1 && j < ny + 2 && k > 1 && k < nz / cpu + 2)
	{
		int index = index_3d(i, j, k);

		if (cc[index] >= 0.9)
		{
			double cs = 1.0 / sqrt(3.0);
			double tau = cc[index] * tau_l + ((double)1.0 - cc[index]) * tau_g;
			double nu = tau * dt * cs * cs;
			double rho = cc[index] * rho_l + ((double)1.0 - cc[index]) * rho_g;
			double mu = nu * rho;
			// second order difference

			double dudx = (u[index + 1] - u[index - 1]) / (2 * dx);
			double dvdx = (v[index + 1] - v[index - 1]) / (2 * dx);
			double dwdx = (w[index + 1] - w[index - 1]) / (2 * dx);

			double dvdy = (v[index + (nx + 4)] - v[index - (nx + 4)]) / (2 * dy);
			double dudy = (u[index + (nx + 4)] - u[index - (nx + 4)]) / (2 * dy);
			double dwdy = (w[index + (nx + 4)] - w[index - (nx + 4)]) / (2 * dy);

			double dwdz = (w[index + (nx + 4) * (ny + 4)] - w[index - (nx + 4) * (ny + 4)]) / (2 * dz);
			double dudz = (u[index + (nx + 4) * (ny + 4)] - u[index - (nx + 4) * (ny + 4)]) / (2 * dz);
			double dvdz = (v[index + (nx + 4) * (ny + 4)] - v[index - (nx + 4) * (ny + 4)]) / (2 * dz);

			// fourth order difference
			// double dudx = (-u[index+2] + 8*u[index+1] - 8*u[index-1] + u[index-2])/(12*dx);
			// double dvdx = (-v[index+2] + 8*v[index+1] - 8*v[index-1] + v[index-2])/(12*dx);
			// double dwdx = (-w[index+2] + 8*w[index+1] - 8*w[index-1] + w[index-2])/(12*dx);

			// double dvdy = (-v[index+2*(nx+4)] + 8*v[index+(nx+4)] - 8*v[index-(nx+4)] + v[index-2*(nx+4)])/(12*dy);
			// double dudy = (-u[index+2*(nx+4)] + 8*u[index+(nx+4)] - 8*u[index-(nx+4)] + u[index-2*(nx+4)])/(12*dy);
			// double dwdy = (-w[index+2*(nx+4)] + 8*w[index+(nx+4)] - 8*w[index-(nx+4)] + w[index-2*(nx+4)])/(12*dy);

			// double dwdz = (-w[index+2*(nx+4)*(ny+4)] + 8*w[index+(nx+4)*(ny+4)] - 8*w[index-(nx+4)*(ny+4)] + w[index-2*(nx+4)*(ny+4)])/(12*dz);
			// double dudz = (-u[index+2*(nx+4)*(ny+4)] + 8*u[index+(nx+4)*(ny+4)] - 8*u[index-(nx+4)*(ny+4)] + u[index-2*(nx+4)*(ny+4)])/(12*dz);
			// double dvdz = (-v[index+2*(nx+4)*(ny+4)] + 8*v[index+(nx+4)*(ny+4)] - 8*v[index-(nx+4)*(ny+4)] + v[index-2*(nx+4)*(ny+4)])/(12*dz);

			DE[index] = DE[index] + mu * (2 * dudx * dudx + 2 * dvdy * dvdy + 2 * dwdz * dwdz + (dvdx + dudy) * (dvdx + dudy) + (dwdy + dvdz) * (dwdy + dvdz) + (dudz + dwdx) * (dudz + dwdx)) - 2 * mu * (dudx + dvdy + dwdz) * (dudx + dvdy + dwdz) / 3.0;
		}
		else
		{
			DE[index] = DE[index] + 0.0;
		}
	}
}

__global__ void DEdiff_cal(double *cc, double *u, double *v, double *w, double *DE)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i > 1 && i < nx + 2 && j > 1 && j < ny + 2 && k > 1 && k < nz / cpu + 2)
	{
		int index = index_3d(i, j, k);

		if (cc[index] > 0.1)
		{
			if (cc[index] < 0.9)
			{
				double cs = 1.0 / sqrt(3.0);
				double tau = cc[index] * tau_l + ((double)1.0 - cc[index]) * tau_g;
				double nu = tau * dt * cs * cs;
				double rho = cc[index] * rho_l + ((double)1.0 - cc[index]) * rho_g;
				double mu = nu * rho;
				// second order difference

				double dudx = (u[index + 1] - u[index - 1]) / (2 * dx);
				double dvdx = (v[index + 1] - v[index - 1]) / (2 * dx);
				double dwdx = (w[index + 1] - w[index - 1]) / (2 * dx);

				double dvdy = (v[index + (nx + 4)] - v[index - (nx + 4)]) / (2 * dy);
				double dudy = (u[index + (nx + 4)] - u[index - (nx + 4)]) / (2 * dy);
				double dwdy = (w[index + (nx + 4)] - w[index - (nx + 4)]) / (2 * dy);

				double dwdz = (w[index + (nx + 4) * (ny + 4)] - w[index - (nx + 4) * (ny + 4)]) / (2 * dz);
				double dudz = (u[index + (nx + 4) * (ny + 4)] - u[index - (nx + 4) * (ny + 4)]) / (2 * dz);
				double dvdz = (v[index + (nx + 4) * (ny + 4)] - v[index - (nx + 4) * (ny + 4)]) / (2 * dz);

				// fourth order difference
				// double dudx = (-u[index+2] + 8*u[index+1] - 8*u[index-1] + u[index-2])/(12*dx);
				// double dvdx = (-v[index+2] + 8*v[index+1] - 8*v[index-1] + v[index-2])/(12*dx);
				// double dwdx = (-w[index+2] + 8*w[index+1] - 8*w[index-1] + w[index-2])/(12*dx);

				// double dvdy = (-v[index+2*(nx+4)] + 8*v[index+(nx+4)] - 8*v[index-(nx+4)] + v[index-2*(nx+4)])/(12*dy);
				// double dudy = (-u[index+2*(nx+4)] + 8*u[index+(nx+4)] - 8*u[index-(nx+4)] + u[index-2*(nx+4)])/(12*dy);
				// double dwdy = (-w[index+2*(nx+4)] + 8*w[index+(nx+4)] - 8*w[index-(nx+4)] + w[index-2*(nx+4)])/(12*dy);

				// double dwdz = (-w[index+2*(nx+4)*(ny+4)] + 8*w[index+(nx+4)*(ny+4)] - 8*w[index-(nx+4)*(ny+4)] + w[index-2*(nx+4)*(ny+4)])/(12*dz);
				// double dudz = (-u[index+2*(nx+4)*(ny+4)] + 8*u[index+(nx+4)*(ny+4)] - 8*u[index-(nx+4)*(ny+4)] + u[index-2*(nx+4)*(ny+4)])/(12*dz);
				// double dvdz = (-v[index+2*(nx+4)*(ny+4)] + 8*v[index+(nx+4)*(ny+4)] - 8*v[index-(nx+4)*(ny+4)] + v[index-2*(nx+4)*(ny+4)])/(12*dz);

				DE[index] = DE[index] + mu * (2 * dudx * dudx + 2 * dvdy * dvdy + 2 * dwdz * dwdz + (dvdx + dudy) * (dvdx + dudy) + (dwdy + dvdz) * (dwdy + dvdz) + (dudz + dwdx) * (dudz + dwdx)) - 2 * mu * (dudx + dvdy + dwdz) * (dudx + dvdy + dwdz) / 3.0;
			}
		}
		else
		{
			DE[index] = DE[index] + 0.0;
		}
	}
}

__global__ void DE_cal_d(double *cc, double *u_in, double *v_in, double *w_in, double *p, double *g, double *m, double *gra_c, double *gra_m,
						 double *DE)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i > 1 && i < nx + 2 && j > 1 && j < ny + 2 && k > 1 && k < nz / cpu + 2)
	{
		int index = index_3d(i, j, k);
		double u=u_in[index];
		double v=v_in[index];
		double w=w_in[index];
		double c=cc[index];

		double cs = 1.0 / sqrt(3.0);
		double tau = c * tau_l + ((double)1.0 - c) * tau_g;
		double nu = tau * dt * cs * cs;
		double rho = c * rho_l + ((double)1.0 - c) * rho_g;
		double mu = nu * rho;
		double udotu=u*u+v*v+w*w;

		double gr_cx_c=gra_c[index_4d(i,j,k,0)];
		double gr_cy_c=gra_c[index_4d(i,j,k,1)];
		double gr_cz_c=gra_c[index_4d(i,j,k,2)];
		double gr_bx_c=gra_m[index_4d(i,j,k,0)];
		double gr_by_c=gra_m[index_4d(i,j,k,1)];
		double gr_bz_c=gra_m[index_4d(i,j,k,2)];

		double eps_xx = 0.0, eps_yy = 0.0, eps_zz = 0.0, eps_xy = 0.0, eps_xz = 0.0, eps_yz = 0.0;

		for(int l = 0;l < q;l++){
			int index_l = index_4d(i, j, k, l);
			double ex=ex_d[l];
			double ey=ey_d[l];
			double ez=ez_d[l];
			double wt=wt_d[l];
			int    et=et_d[l];

			double edotu=ex*u+ey*v+ez*w;
			double uugly=edotu/cs/cs+edotu*edotu*0.5/cs/cs/cs/cs-udotu*0.5/cs/cs;
			double gamma=wt*(1.0+uugly);

			double temp_cc = grad_phie_c( cc,index,et ) - ( u * gr_cx_c + v * gr_cy_c + w * gr_cz_c );  //for h,g
			double temp_bc = grad_phie_c( m ,index,et ) - ( u * gr_bx_c + v * gr_by_c + w * gr_bz_c );  //for h,g

			double gra = (rho-rho_l)*(gra_ac*ex - gra_ac*u);

			double geq_t=wt*(p[index]+rho*cs*cs*uugly);
			double temp_gc = temp_cc*(rho_l-rho_g)*cs*cs*wt*uugly-(temp_bc*c + gra)*gamma;
			//geq_t = geq_t-0.5*dt*temp_gc;
			double g_ori = (0.5*dt*temp_gc + g[index_l] + 1.0/2.0/tau*geq_t) / (1.0+1.0/2.0/tau);

			eps_xx = eps_xx + ex*ex*(g_ori - geq_t);
			eps_yy = eps_yy + ey*ey*(g_ori - geq_t);
			eps_zz = eps_zz + ez*ez*(g_ori - geq_t);
			eps_xy = eps_xy + ex*ey*(g_ori - geq_t);
			eps_xz = eps_xz + ex*ez*(g_ori - geq_t);
			eps_yz = eps_yz + ey*ez*(g_ori - geq_t);
		}

		eps_xx = (-1.0/2.0/rho/cs/cs/cs/cs/tau) * eps_xx;
		eps_yy = (-1.0/2.0/rho/cs/cs/cs/cs/tau) * eps_yy;
		eps_zz = (-1.0/2.0/rho/cs/cs/cs/cs/tau) * eps_zz;
		eps_xy = (-1.0/rho/cs/cs/cs/cs/tau)   * eps_xy;
		eps_xz = (-1.0/rho/cs/cs/cs/cs/tau)   * eps_xz;
		eps_yz = (-1.0/rho/cs/cs/cs/cs/tau)   * eps_yz;

		DE[index] = DE[index] + mu * (2 * eps_xx * eps_xx + 2 * eps_yy * eps_yy + 2 * eps_zz * eps_zz + eps_xy * eps_xy + eps_yz * eps_yz + eps_xz * eps_xz) - 2 * mu * (eps_xx + eps_yy + eps_zz) * (eps_xx + eps_yy + eps_zz) / 3.0;

	}
}

/************************* droplet ***********************************/
// 20230306 2D 質心 
void center_single_droplet(double *c, double *cx, double *cz, int indexx)
{
	int i, k;
	double sumc_x, sumc_z;
	int index;
	double sum_cx, sum_cz;
	for (i = 0; i < nx - 1; i++){
		for (k = 0; k < nz - 1; k++){
			index = (nx) * (k) + i;
			if( c[index] >= 0.5){
				sumc_z += c[index];
				sum_cz += c[index] * k;
			}
		}
	}
	for (k = 0; k < nz - 1; k++){
		for (i = 0; i < nx - 1; i++){
			index = (nx) * (k) + i;
			if( c[index] >= 0.5){
				sumc_x += c[index];
				sum_cx += c[index] * i;
			}
		}
	}
	cx[indexx] = sum_cx/sumc_x;
	cz[indexx] = sum_cz/sumc_z;
}

// 20230306 算橢圓 D number
void D_single_droplet(double *c, double *cx, double *cz, double *D_value, int indexx)
{
	int i, k;
	double B=1000.0, L=0.0, B_temp, L_temp, distance_x, distance_z, x_temp;
	int index;
	for (k = 0; k < nz - 1; k++){
		for (i = 0; i < nx - 1; i++){
			index = (nx) * (k) + i;
			if( c[index] >= 0.5 ){
				distance_x = cx[indexx] - i;
				distance_z = cz[indexx] - k;
				B_temp = sqrt(distance_x*distance_x + distance_z*distance_z);
				L_temp = sqrt(distance_x*distance_x + distance_z*distance_z);
				if(B>B_temp && ((c[index]-0.5)*(c[index-1]-0.5)) < 0){
					x_temp = i - (c[index]-0.5)/(c[index]-c[index-1]);
					distance_x = cx[indexx] - x_temp;
					B = sqrt(distance_x*distance_x + distance_z*distance_z) ;
				}
				if(L<L_temp && ((c[index]-0.5)*(c[index-1]-0.5)) < 0){
					L=L_temp;
				}
			}
		}
	}
	// printf("L=%f\tB=%f\n",L,B);
	D_value[indexx] = (2.0*L-2.0*B)/(2.0*L+2.0*B);
}

// check 三維質心和直接用二維算質心的差別
void center_single_droplet_3D(double *c, double*cx, double*cy, double*cz)
{
	int i, j, k;
	double sumc;
	int index;
	double sum_cx, sum_cy, sum_cz;
	for (i = 0; i < nx - 1; i++){
		for(j = 0; j < ny - 1 ; j++){
			for (k = 0; k < nz - 1; k++){
				index = nx*ny*k + nx*j + i;
				if( c[index] >= 0.5){
					sumc += c[index];
					sum_cz += c[index] * k;
				}
			}
		}
	}
	for (k = 0; k < nz - 1; k++){
		for(j = 0; j < ny - 1 ; j++){
			for (i = 0; i < nx - 1; i++){
				index = nx*ny*k + nx*j + i;
				if( c[index] >= 0.5){
					sum_cx += c[index] * i;
				}
			}
		}
	}
	for (k = 0; k < nz - 1; k++){
		for(i = 0; i < nx - 1 ; i++){
			for (j = 0; j < ny - 1; j++){
				index = nx*ny*k + nx*j + i;
				if( c[index] >= 0.5){
					sum_cy += c[index] * j;
				}
			}
		}
	}
	*cx = sum_cx/sumc;
	*cy = sum_cy/sumc;
	*cz = sum_cz/sumc;
}
// 20230310 算兩顆質心 x的質心比較小的z的質心必須比較大，這個程式才能用，並且碰撞後會合併
void center_two_droplet_coale(double *c, double *cx_1, double *cz_1, double *cx_2,double *cz_2, int indexx)
{
	int i, k;
	double sumc_1=0.0, sumc_2=0.0, x_1[nz-1], x_2[nz-1], z_limit=1000.0, sumc_l1, sumc_l2, sum_cl1_x, sum_cl2_x, sum_cl1_z, sum_cl2_z;
	int index;
	double sum_cx_1=0.0, sum_cx_2=0.0, sum_cz_1=0.0, sum_cz_2=0.0;
	for (k = 0; k < nz - 1; k++){
		x_2[k] = 0;
		x_1[k] = 0;
	}
	for (k = 0; k < nz - 1; k++){
		for (i = 0; i < nx - 1; i++){
			index = (nx) * (k) + i;
			if( c[index] >= 0.5){
				if(c[index-1]<0.5 && x_1[k] == 0){
					x_2[k] = i;
					x_1[k] = x_2[k];
				}
				if(c[index-1]<0.5){
					x_2[k] = i;
				}
			}
		}
	}
	for (k = 0; k < nz - 1; k++){
		for (i = 0; i < nx - 1; i++){
			index = (nx) * (k) + i;
			if( c[index] >= 0.5){
				if(x_1[k]!=x_2[k]){
					z_limit = k;
				}
				if(x_1[k]==x_2[k]){
					sumc_l2 += c[index];
					sum_cl2_x += c[index] * i;
					sum_cl2_z += c[index] * k;
					if(z_limit<=k){
						sumc_l1 += sumc_l2;
						sum_cl1_x += sum_cl2_x;
						sum_cl1_z += sum_cl2_z;
						sumc_l2 = 0;
						sum_cl2_x = 0;
						sum_cl2_z = 0;
					}
				}else{
					if(x_2[k]>i){
						sumc_l1 += c[index];
						sum_cl1_x += c[index] * i;
						sum_cl1_z += c[index] * k;
					}else{
						sumc_l2 += c[index];
						sum_cl2_x += c[index] * i;
						sum_cl2_z += c[index] * k;
					}
				}
			}
		}
		sumc_1 += sumc_l1;
		sumc_2 += sumc_l2;
		sumc_l1 = 0;
		sumc_l2 = 0;
		sum_cx_1 += sum_cl1_x;
		sum_cx_2 += sum_cl2_x;
		sum_cz_1 += sum_cl1_z;
		sum_cz_2 += sum_cl2_z;
		sum_cl1_x = 0;
		sum_cl2_x = 0;
		sum_cl1_z = 0;
		sum_cl2_z = 0;
	}
	cx_1[indexx] = sum_cx_1/sumc_1;
	cx_2[indexx] = sum_cx_2/sumc_2;
	cz_1[indexx] = sum_cz_1/sumc_1;
	cz_2[indexx] = sum_cz_2/sumc_2;
}
// 20230318 算兩顆質心 x的質心比較小的z的質心必須比較大，這個程式才能用，並且預估是碰撞後會分離的
void center_two_droplet_sep(double *c, double *cx_1, double *cz_1, 
	double *cx_2, double *cz_2, int indexx)
{
int i, k, index;
// 先找出所有 c>=0.5 的點的全域邊界 (限定後續掃描範圍)
int min_x = nx, max_x = 0, min_z = nz, max_z = 0;
for(k = 0; k < nz; k++){
for(i = 0; i < nx; i++){
index = k * nx + i;
if(c[index] >= 0.5){
if(i < min_x) min_x = i;
if(i > max_x) max_x = i;
if(k < min_z) min_z = k;
if(k > max_z) max_z = k;
}
}
}
// 若邊界未更新，表示沒有液滴點，回傳 (0,0)
if(min_x == nx || max_z == 0){
cx_1[indexx] = 0; cz_1[indexx] = 0;
cx_2[indexx] = 0; cz_2[indexx] = 0;
return;
}


// -------------------------------
// 1. x方向候選計算與左右判定（逐行掃描）
int leftCandidateCount = 0, rightCandidateCount = 0;
double leftCandidateSum = 0.0, rightCandidateSum = 0.0;
for(k = 0; k < nz; k++){
double cross_positions[nx];
int cross_count = 0;
for(i = min_x; i < max_x; i++){
int idx = k * nx + i;
int idx_t = k * nx + (i + 1);
double cond = (c[idx] - 0.5) * (c[idx_t] - 0.5);
if(cond <= 0.0 && (c[idx_t] - c[idx]) != 0.0){
double candidate = i + (0.5 - c[idx]) / (c[idx_t] - c[idx]);
cross_positions[cross_count++] = candidate;
}
}
if(cross_count >= 4){
double candidate_x = 0.0;
if(cross_count % 2 == 0)
candidate_x = (cross_positions[cross_count/2 - 1] + cross_positions[cross_count/2]) / 2.0;
else
candidate_x = cross_positions[cross_count/2];
// 判定左右：左側區間定義為 candidate_x < nx*0.05；右側區間定義為 candidate_x > nx - nx*0.05
int hasLeft = (candidate_x < nx * 0.05);
int hasRight = (candidate_x > nx - nx * 0.05);
// 如果只有單側出現，則該行候選無效
if((hasLeft && !hasRight) || (!hasLeft && hasRight)){
// 不納入候選
} else {
// 如果左右同時存在，或都不存在（候選在中間），則納入候選
if(hasLeft){
leftCandidateCount++;
leftCandidateSum += candidate_x;
} else if(hasRight){
rightCandidateCount++;
rightCandidateSum += candidate_x;
} else {
leftCandidateCount++;
rightCandidateCount++;
leftCandidateSum += candidate_x;
rightCandidateSum += candidate_x;
}

}
}
}
double separation_x = -1;
if((leftCandidateCount >= 4 && rightCandidateCount >= 4) ||
(leftCandidateCount == 0 && rightCandidateCount == 0)){
separation_x = (leftCandidateSum + rightCandidateSum) / (leftCandidateCount + rightCandidateCount);
}


// -------------------------------
// 2. z方向候選計算（逐列掃描）
// 在此處，我們對每一列 (固定 i) 計算交叉點候選，
// 並檢查該列是否屬於左側或右側（即 i 落在 [0, nx*0.05] 或 [nx - nx*0.05, nx-1]）。
int leftBoundaryFlag = 0, rightBoundaryFlag = 0;
double candidate_sum_z = 0.0;
int candidate_count_z = 0;
for(i = 0; i < nx; i++){
double cross_positions[nz];
int cross_count = 0;
for(k = min_z; k < max_z; k++){
int idx = k * nx + i;
int idx_t = (k + 1) * nx + i;
double cond = (c[idx] - 0.5) * (c[idx_t] - 0.5);
if(cond <= 0.0 && (c[idx_t] - c[idx]) != 0.0){
double candidate = k + (0.5 - c[idx]) / (c[idx_t] - c[idx]);
cross_positions[cross_count++] = candidate;
}
}
if(cross_count >= 4){
double candidate_z = 0.0;
if(cross_count % 2 == 0)
candidate_z = (cross_positions[cross_count/2 - 1] + cross_positions[cross_count/2]) / 2.0;
else
candidate_z = cross_positions[cross_count/2];
candidate_sum_z += candidate_z;
candidate_count_z++;

// 判定：若該列屬於左側或右側，則標記
if(i < nx * 0.05)
leftBoundaryFlag = 1;
if(i > nx - nx * 0.05)
rightBoundaryFlag = 1;
}
}
double separation_z = -1;
if(candidate_count_z > 0){
separation_z = candidate_sum_z / candidate_count_z;
}


// -------------------------------
// 3. 判定條件：如果在 z方向檢查中，
//    若只有一側存在（即 leftBoundaryFlag 為真而 rightBoundaryFlag 為假，或反之），
//    則直接返回 (0,0)；如果兩側都存在或都不存在則正常。
if((leftBoundaryFlag && !rightBoundaryFlag) || (!leftBoundaryFlag && rightBoundaryFlag)){

cx_1[indexx] = 0; cz_1[indexx] = 0;
cx_2[indexx] = 0; cz_2[indexx] = 0;
return;
}

// -------------------------------
// 4. 如果 x 或 z 方向都沒有有效候選 (即 separation_x 或 separation_z < 0)，則視為只有一顆液滴，返回 (0,0)
if(separation_x < 0 && separation_z < 0){
cx_1[indexx] = 0; cz_1[indexx] = 0;
cx_2[indexx] = 0; cz_2[indexx] = 0;
return;
}

// -------------------------------
// 5. 根據候選情況選擇分割方式：
//    - 如果兩方向皆有候選，則採用斜向分割；
//    - 如果只有 x 方向候選，則用垂直分割 (以 i < separation_x 判定)；
//    - 如果只有 z 方向候選，則用水平分割 (以 k < separation_z 判定)。
int use_diagonal = 0, use_x = 0, use_z = 0;
if(separation_x > 0 && separation_z > 0){
use_diagonal = 1;
} else if(separation_x > 0){
use_x = 1;
} else if(separation_z > 0){
use_z = 1;
}

double sum1 = 0.0, sum2 = 0.0;
double sum1_x = 0.0, sum1_z = 0.0;
double sum2_x = 0.0, sum2_z = 0.0;
for(k = min_z; k <= max_z; k++){
for(i = min_x; i <= max_x; i++){
index = k * nx + i;
if(c[index] >= 0.5){
if(use_diagonal){
if((i - k) < (separation_x - separation_z)){
sum1 += c[index];
sum1_x += c[index] * i;
sum1_z += c[index] * k;
} else {
sum2 += c[index];
sum2_x += c[index] * i;
sum2_z += c[index] * k;
}
} else if(use_x){
if(i < separation_x){
sum1 += c[index];
sum1_x += c[index] * i;
sum1_z += c[index] * k;
} else {
sum2 += c[index];
sum2_x += c[index] * i;
sum2_z += c[index] * k;
}
} else if(use_z){
if(k < separation_z){
sum1 += c[index];
sum1_x += c[index] * i;
sum1_z += c[index] * k;
} else {
sum2 += c[index];
sum2_x += c[index] * i;
sum2_z += c[index] * k;
}
}
}
}
}

if(sum1 == 0.0 || sum2 == 0.0){
cx_1[indexx] = 0; cz_1[indexx] = 0;
cx_2[indexx] = 0; cz_2[indexx] = 0;
} else {
cx_1[indexx] = sum1_x / sum1;
cz_1[indexx] = sum1_z / sum1;
cx_2[indexx] = sum2_x / sum2;
cz_2[indexx] = sum2_z / sum2;
}
}



/**************************** pre-property print *************************/

void initial_infor(FILE*data){
    switch (initial_type)
    {
    case 0:
		printf("initial initial_type : stationary bubble\n");
		fprintf(data,"initial initial_type : stationary bubble\n");
        break;
    case 1:
		printf("initial initial_type : two bubble collision\n");
		fprintf(data,"initial initial_type : two bubble collision\n");
        break;
    case 3:
		printf("initial initial_type : two bubble merge\n");
		fprintf(data,"initial initial_type : two bubble merge\n");
        break;
    case 4:
		printf("initial initial_type : muti bubble merge\n");
		fprintf(data,"initial initial_type : muti bubble merge\n");
        break;
    case 5:
		printf("initial initial_type : drop_elliptical\n");
		fprintf(data,"initial initial_type : drop_elliptical\n");
        break;
    case 7:
		printf("initial initial_type : cylinder\n");
		fprintf(data,"initial initial_type : cylinder\n");
        break;
	case 12:
		printf("initial initial_type : single droplet\n");
		fprintf(data,"initial initial_type : single droplet\n");
        break;		
    case 13:
		printf("initial initial_type : shear_drop_begin_y_symmetry\n");
		fprintf(data,"initial initial_type : shear_drop_begin_y_symmetry\n");
        break;
    default:
        printf("Invalid initial_type\n");
		break;
    }

}

void boundary_infor(FILE*data){
	switch (boundary_set)
    {
    case 0:
		fprintf(data,"boundary initial_type : all direction periodic\n");
		printf("boundary initial_type : all direction periodic\n");
        break;
    case 1:
		fprintf(data,"boundary initial_type : z direction: moving wall  others : periodic\n");
		printf("boundary initial_type : z direction: moving wall  others : periodic\n");
        break;
    case 2:
		fprintf(data,"boundary initial_type : z direction: moving wall\t y direction : symmetry\t x direction : periodic\n");
		printf("boundary initial_type : z direction: moving wall\t y direction : symmetry\t x direction : periodic\n");
        break;
    default:
        printf("Invalid initial_type\n");
		break;
    }
}

void free_energy_type(FILE*data){
	switch (bulk_free_energy)
    {
    case 0:
		fprintf(data,"bulk_free_energy : double obstacle\n");
		printf("bulk_free_energy : double double obstacle\n");
        break;
    case 1:
		fprintf(data,"bulk_free_energy : double well\n");
		printf("bulk_free_energy : double well\n");
        break;
    default:
        printf("Invalid initial_type\n");
		break;
    }
}

void force_type(FILE*data){
	switch (Force_method)
    {
    case 0:
		fprintf(data,"Force_method : He et al.\n");
		printf("Force_method : He et al.\n");
        break;
    case 2:
		fprintf(data,"Force_method : Guo et al.\n");
		printf("Force_method : Guo et al.\n");
        break;
    case 1:
		fprintf(data,"Force_method : Buick et al.\n");
		printf("Force_method : Buick et al.\n");
        break;
    default:
        printf("Invalid initial_type\n");
		break;
    }
}

void post_condi_0(){
    printf("+----------------------------------------------------------------+\n");
	printf("laplace pressure Check\n");
	printf("pressure=%e\n" , p_difference(c_f_h,a_f_h) );
	printf("error =%lf%\n",(p_difference(c_f_h,a_f_h)-2.0*surface_tension/radd)/(2.0*surface_tension/radd)*100.0);
	double lengthx = length_x(c_f_h);
	double lengthy = length_y(c_f_h);
	double lengthz = length_z(c_f_h);
	double equivalent_diameter = cbrt(lengthx*lengthy*lengthz);
	printf("x width length : %f \t y width lengh :%f \t z width length : %f\n",lengthx,lengthy,lengthz);
	printf("equivalent_diameter = %.2f\n ",equivalent_diameter/2.0);	
}
void condition_0(){
	printf("===============================================================\n");
	fprintf( information, "stationary bubble \n");
	printf( "stationary bubble \n");	
}
void condition_1(){

	double re = u_coll*2.0*radd*3.0/tau_l;
	double we = u_coll*u_coll*2.0*radd*rho_l/surface_tension;
	printf("===============================================================\n");
	fprintf( information, "Three dimensional droplets collision\n");
	fprintf( information, "u_coll =%f,Re =%lf, We=%lf, B=%f\n",u_coll,re,we,b_coll);
	printf("Three dimensional droplets collision\n");
	printf( "u_coll =%f,Re =%lf, We=%lf, B=%f\n",u_coll,re,we,b_coll);
	printf("===============================================================\n");

}
void condition_2(){}
void condition_3(){

	printf("===============================================================\n");
	fprintf( information, "Three dimensional droplet - Merging\n");
	fprintf( information,"Tau_ci=%lf\n",(double)radd*pow(rho_l*radd/surface_tension,0.5));
	fprintf( information,"Oh=%lf\n",pow(rho_l/(surface_tension*radd),0.5)/3.0*dt);
	printf("Three dimensional droplet - Merging\n");
	printf("Tau_ci=%lf\n",(double)radd*pow(rho_l*radd/surface_tension,0.5));
	printf("Oh=%lf\n",pow(rho_l/(surface_tension*radd),0.5)/3.0*dt);
	printf("===============================================================\n");

}
void condition_4(){

	printf("===============================================================\n");
	fprintf( information, "Three dimensional droplets collision\n");
	fprintf( information, "Mean=%f, Std=%f\n",mean_ini_4, std_ini_4);
	printf("Three dimensional - One bubble theory\n");
	printf("Mean=%f, Std=%f\n",mean_ini_4, std_ini_4);
	printf("===============================================================\n");

}
void condition_5(){}
void condition_6(){}
void condition_7(){

	double mu = tau_l * dt * rho_l/3.0;
	double shear_gamma = 2.0*u_0/nz;
	double Re = shear_gamma*(rho_l)*radd*radd/mu;
	double Ca = mu*shear_gamma*radd/surface_tension;
	double Cn = thick/radd;
	double Pe;
	if(PH_model == 0)
	{
		Pe = 2*u_0*radd*radd*radd/surface_tension/mobi/nz;
	}
	else{
		Pe = u_0*radd*radd/mobi/nz ; 
	}	
	printf("===============================================================\n");
	fprintf( information, "two dimensional shear flow single droplet\n");
	fprintf( information, "Re=%lf, Ca=%lf, Cn=%lf ,Pe=%lf\n",Re,Ca,Cn,Pe);
	printf("two dimensional shear flow single droplet\n");
	printf("Re=%lf\n",Re);
	printf("Ca=%lf\n",Ca);
	printf("Cn=%lf\n",Cn);
	printf("Pe=%lf\n",Pe);
	printf("===============================================================\n");

}
void condition_8(){
	double re = u_coll_7*2.0*radd*3.0/tau_l;
	double we = u_coll_7*u_coll_7*2.0*radd*rho_l/surface_tension;
	printf("===============================================================\n");
	fprintf( information, "Three dimensional droplets collision\n");
	fprintf( information, "u_coll_7 =%f,Re =%lf, We_7=%lf, B=%f\n",u_coll_7,re,we,b_coll_7);
	printf("Three dimensional droplets collision\n");
	printf( "u_coll_7 =%f,Re =%lf, We_7=%lf, B=%f\n",u_coll_7,re,we,b_coll_7);
	printf("===============================================================\n");

}
void condition_9(){

	double re = u_coll_9*2.0*radd*3.0/tau_l;
	double we = u_coll_9*u_coll_9*2.0*radd*rho_l/surface_tension;
	printf("===============================================================\n");
	fprintf( information, "Three dimensional One-droplet-moving\n");
	fprintf( information, "u_coll =%f,Re =%lf, We=%lf, B=%f\n",u_coll_9,re,we,b_coll);
	printf("Three dimensional droplets collision\n");
	printf( "u_coll =%f,Re =%lf, We=%lf, B=%f\n",u_coll_9,re,we,b_coll);
	printf("===============================================================\n");
}
void condition_10(){

	double bo = gra_ac*(rho_l-rho_g)*4*radd*radd/surface_tension;
	double Mo = gra_ac*(rho_l-rho_g)*pow(tau_l,4)/rho_l/rho_l/pow(surface_tension,3)/81;
	printf("===============================================================\n");
	fprintf( information, "Three dimensional one bubble moving by gravity\n");
	fprintf( information, "bo=%lf, Mo=%lf\n",bo,Mo);
	printf("Three dimensional one bubble\n");
	printf( "bo=%lf, Mo=%lf\n",bo,Mo);
	printf("===============================================================\n");

}
void condition_11(){
	double mu = tau_g * dt * rho_g/3.0;
	double shear_gamma = 2.0*u_0/nz;
	double Re = shear_gamma*(rho_g)*radd*radd/mu;
	double Ca = mu*shear_gamma*radd/surface_tension;
	printf("===============================================================\n");
	fprintf( information, "Three dimensional shear flow droplet coalescence\n");
	fprintf( information, "Re=%lf, Ca=%lf\n",Re,Ca);
	printf("Three dimensional shear flow droplet coalescence\n");
	printf("Re=%lf, Ca=%lf\n",Re,Ca);
	printf("===============================================================\n");

}
void condition_12(){

	double mu = tau_l * dt * rho_l/3.0;
	double shear_gamma = 2.0*u_0/nz;
	double Re = shear_gamma*(rho_l)*radd*radd/mu;
	double Ca = mu*shear_gamma*radd/surface_tension;
	printf("===============================================================\n");
	fprintf( information, "Three dimensional shear flow single droplet\n");
	fprintf( information, "Re=%lf, Ca=%lf\n",Re,Ca);
	printf("Three dimensional shear flow single droplet\n");
	printf("Re=%lf\n",Re);
	printf("Ca=%lf\n",Ca);
	printf("===============================================================\n");

}
void condition_13(){

	double mu = tau_l * dt * rho_l/3.0;
	double shear_gamma = 2.0*u_0/nz;
	double Re = shear_gamma*(rho_l)*radd*radd/mu;
	double Ca = mu*shear_gamma*radd/surface_tension;
	double Cn = thick/radd;
	double Pe = 2*u_0*radd*radd*radd/surface_tension/mobi/nz;
	printf("mobi=%lf\n",mobi);
	printf("===============================================================\n");
	fprintf( information, "Three dimensional shear flow droplet coalescence\n");
	fprintf( information,"psi_x = %.3f , psi_y = %.3f , psi_z = %.3f\n",(double)4*radd/nx,(double)2*radd/ny,(double)2*radd/nz);
	fprintf( information, "Re=%lf\tCa=%lf\tCn=%lf\tPe=%lf\n",Re,Ca,Cn,Pe);
	fprintf( information, "mobility=%f, velocity_base=%f, velocity_times:%.2f\n",mobi, uuu , timesss);
	printf("Three dimensional shear flow droplet coalescence\n");
	printf("psi_x = %.3f , psi_y = %.3f , psi_z = %.3f\n",(double)4*radd/nx,(double)2*radd/ny,(double)2*radd/nz);
	printf("Re=%lf\tCa=%lf\tCn=%lf\tPe=%lf\n",Re,Ca,Cn,Pe);
	printf("mobility=%f, velocity_base=%f, velocity_times:%.2f\n",mobi, uuu , timesss);
	printf("===============================================================\n");

}

typedef void (*ConditionFunction)();

// 函數表
ConditionFunction conditionTable[] = {
	condition_0,
    condition_1,
    condition_2,	
    condition_3,
    condition_4,
	condition_5,
    condition_6,
    condition_7,
    condition_8,
    condition_9,
    condition_10,
    condition_11,
    condition_12,
    condition_13
};


/**************************** print data *************************/

void Data2d(int j){
	char name[5];
	if(step == iprint){
		sprintf(name,"w");
	}
	else{
		sprintf(name,"a");	
	}
	data_2d = fopen("./data/data_2d.dat",name);
	CHECK_FILE(data_2d,"./data/data_2d.dat");
	fprintf( data_2d, "VARIABLES=\"X\",\"Z\",\"c\",\"u\",\"v\",\"w\",\"P_real\"\n");
	fprintf( data_2d, "ZONE T=\"STEP=%d\" F=POINT\n",step);
	fprintf( data_2d, "I=%d, J=%d\n", nx,nz);
	
	for(int k=0;k< nz;k++){
		for(int i=0;i< nx;i++){
			int index=nx*(k*ny+j)+i;
			fprintf( data_2d, "%d\t%d\t%e\t%e\t%e\t%e\t%e\t\n",i,k,c_f_h[index],u_f_h[index],v_f_h[index],w_f_h[index],a_f_h[index]);
		}
	}
	fclose(data_2d);
}

void Data3d(){
	char name[5];
	if(step == iprint){
		sprintf(name,"w");
	}
	else{
		sprintf(name,"a");	
	}
	data_3d = fopen("./data/data_3d.dat",name);
	CHECK_FILE(data_3d,"./data/data_3d.dat");
	fprintf( data_3d, "VARIABLES=\"X\",\"Y\",\"Z\",\"c\"\n");
	fprintf( data_3d, "ZONE T=\"STEP=%d\" F=POINT\n",step);
	fprintf( data_3d, "I=%d, J=%d, K=%d\n", nx,ny,nz);
	for(int k=0;k<nz;k++){
		for(int j=0;j<ny;j++){
			for(int i=0;i<nx;i++){
				int index=(nx)*(k*(ny)+j)+i;
				fprintf( data_3d, "%d\t%d\t%d\t%e\t\n",
				i,j,k,c_f_h[index],u_f_h[index],v_f_h[index],w_f_h[index],a_f_h[index]);
			}
		}
	}
	fclose(data_3d);
}

void final_print_2d(int j){

	final_2d = fopen("./data/data_2d_final.dat","w");
	CHECK_FILE(final_2d,"./data/data_2d_final.dat");
	fprintf( final_2d, "VARIABLES=\"X\",\"Z\",\"c\",\"u\",\"v\",\"w\",\"P_real\"\n");
	fprintf( final_2d, "ZONE T=\"STEP=%d\" F=POINT\n",step);
	fprintf( final_2d, "I=%d, J=%d\n", nx,nz);
	
	for(int k=0;k< nz;k++){
		for(int i=0;i< nx;i++){
			int index=nx*(k*ny+j)+i;
			fprintf( final_2d, "%d\t%d\t%e\t%e\t%e\t%e\t%e\t\n",i,k,c_f_h[index],u_f_h[index],v_f_h[index],w_f_h[index],a_f_h[index]);
		}
	}
	fclose(final_2d);

}

void final_print_3d(){

	final_3d = fopen("./data/data_3d_final.dat","w");
	CHECK_FILE(final_3d,"./data/data_3d_final.dat");
	fprintf( final_3d, "VARIABLES=\"X\",\"Y\",\"Z\",\"c\"\n");
	fprintf( final_3d, "ZONE T=\"STEP=%d\" F=POINT\n",step);
	fprintf( final_3d, "I=%d, J=%d, K=%d\n", nx,ny,nz);
	for(int k=0;k<nz;k++){
		for(int j=0;j<ny;j++){
			for(int i=0;i<nx;i++){
				int index=(nx)*(k*(ny)+j)+i;
				fprintf( data_3d, "%d\t%d\t%d\t%e\t\n",
				i,j,k,c_f_h[index],u_f_h[index],v_f_h[index],w_f_h[index],a_f_h[index]);
			}
		}
	}
	fclose(final_3d);
}


void one_dimension_C_profile(){
	char name[5];
	if(step == iprint){
		sprintf(name,"w");
	}
	else{
		sprintf(name,"a");	
	}

	oneDC = fopen("./data/oneDC.dat",name);
	CHECK_FILE(oneDC,"./data/oneDC.dat");
    fprintf( oneDC, "VARIABLES=\"X\",\"c\"\n");
    fprintf( oneDC, "ZONE T=\"STEP=%d\" F=POINT\n",step);
    fprintf( oneDC, "I=%d\n", nx);
    int j=ny/2; int k = nz/2;
    for(int i=0;i<nx;i++){
    	int index=nx*(k*ny+j)+i;
        fprintf( oneDC, "%d\t%e\n",i,c_f_h[index]);
    }
    fclose(oneDC);

}

void mass_ratio_2d(){
	char name[5];
	if(step == iprint){
		sprintf(name,"w");
		twoDC = fopen("./data/2d_mass_ratio.dat",name);
		CHECK_FILE(twoDC,"./data/2d_mass_ratio.dat");
		fprintf( twoDC, "VARIABLES=\"t*\",\"C/gridall\"\n");
    	fprintf( twoDC, "ZONE T=\"\" F=POINT\n");
    	fprintf( twoDC, "I=%d\n", stepall/iprint);
	}
	else{
		sprintf(name,"a");	
		twoDC = fopen("./data/2d_mass_ratio.dat",name);
		CHECK_FILE(twoDC,"./data/2d_mass_ratio.dat");

	}
	
	double shear_gamma = 2.0*u_0/nz; 
	double t_star = shear_gamma * step * 2.0;
	double index_C=0.0;

	for(int k=0;k<nz;k++){
		for(int i=0;i<nx;i++){
			int j=0;
			int index=nx*(k*ny+j)+i;
			if (c_f_h[index]>0.5){
				index_C=index_C+1.0;
			}
		}
	}
	double mass_ratio_y=index_C/nx/nz;
	fprintf(twoDC, "%e\t%e\n",t_star,mass_ratio_y);

	fclose(twoDC);


}

void mass_ratio_3d(){
	char name[5];
	if(step == iprint){
		sprintf(name,"w");
		threeDC = fopen("./data/3d_mass_ratio.dat",name);  
		CHECK_FILE(threeDC,"./data/3d_mass_ratio.dat");
    	fprintf( threeDC, "VARIABLES=\"t*\",\"c\"\n");
    	fprintf( threeDC, "ZONE T=\"\" F=POINT\n");
    	fprintf( threeDC, "I=%d\n", stepall/iprint);
	}
	else{
		sprintf(name,"a");	
		threeDC = fopen("./data/3d_mass_ratio.dat",name);  
		CHECK_FILE(threeDC,"./data/3d_mass_ratio.dat");
	}

	double shear_gamma = 2.0*u_0/nz; 
	double t_star = shear_gamma * step;
	double C_higher=0.0;
	for(int k=0;k<nz;k++){
		for(int j=0;j<ny;j++){
			for(int i=0;i<nx;i++){
				int index=nx*(k*ny+j)+i;
				if (c_f_h[index]>=0.5){
					C_higher++;
				}
			}
		}
	}
	double VolumeRatio=C_higher/nx/ny/nz;
	fprintf(threeDC, "%e\t%e\n",t_star,VolumeRatio);
    fclose(threeDC);



}


/*******************droplet data initial_type = 13 **********************/


void dxdt() {
    static int last_output_index = 0; // 上一次輸出的索引

	int current_output_index = (step /(int)value_get)-1;

    // 初始化檔案
    if (step == iprint) {
        double Re_13, Ca_13;
        DXvsDt_data = fopen("./data/DXvsDt.dat", "w");
        CHECK_FILE(DXvsDt_data, "./data/DXvsDt.dat");

        // 寫入標題
        fprintf(DXvsDt_data, "VARIABLES=\"t*\",\"DX/2R\"\n");

        // 計算流體動力參數
        double mu = tau_l * dt * rho_l / 3.0;
        double shear_gamma = 2.0 * u_0 / nz;
        Re_13 = shear_gamma * rho_l * radd * radd / mu;
        Ca_13 = mu * shear_gamma * radd / surface_tension;

        // 寫入區域資訊
        fprintf(DXvsDt_data, "ZONE T=\"DXvsDt_%.2fu_RE_%.2f_Ca_%.2f\" F=POINT\n", timesss, Re_13, Ca_13);
        fprintf(DXvsDt_data, "I=%d\n", (int)(stepall/value_get)-1);

    }
	else{
        DXvsDt_data = fopen("./data/DXvsDt.dat", "a");
        CHECK_FILE(DXvsDt_data, "./data/DXvsDt.dat");
	}

        for (int i = last_output_index; i <= current_output_index; i++) {
            double shear_gamma = 2.0 * u_0 / nz;
            double t_star = shear_gamma * (i+1) * value_get;
            double DX_2R = abs(cx_2_sep[i] - cx_sep[i]) / (2 * radd);
            fprintf(DXvsDt_data, "%e\t%e\n", t_star, DX_2R);
        }
		
    last_output_index = current_output_index+1;
    fclose(DXvsDt_data);
	
}

void dzdt(){

    static int last_output_index = 0; // 上一次輸出的索引

	int current_output_index = (step /(int)value_get)-1;

    // 初始化檔案
    if (step == iprint) {
        double Re_13, Ca_13;
        DZvsDt_data = fopen("./data/DZvsDt.dat", "w");
        CHECK_FILE(DZvsDt_data, "./data/DZvsDt.dat");

        // 寫入標題
        fprintf(DZvsDt_data, "VARIABLES=\"t*\",\"DZ/2R\"\n");

        // 計算流體動力參數
        double mu = tau_l * dt * rho_l / 3.0;
        double shear_gamma = 2.0 * u_0 / nz;
        Re_13 = shear_gamma * rho_l * radd * radd / mu;
        Ca_13 = mu * shear_gamma * radd / surface_tension;

        // 寫入區域資訊
        fprintf(DZvsDt_data, "ZONE T=\"DZvsDt_%.2fu_RE_%.2f_Ca_%.2f\" F=POINT\n", timesss, Re_13, Ca_13);
        fprintf(DZvsDt_data, "I=%d\n", (int)(stepall/value_get)-1);

    }
	else{
        DZvsDt_data = fopen("./data/DZvsDt.dat", "a");
        CHECK_FILE(DZvsDt_data, "./data/DZvsDt.dat");
	}

        for (int i = last_output_index; i <= current_output_index; i++) {
            double shear_gamma = 2.0 * u_0 / nz;
            double t_star = shear_gamma * (i+1) * value_get;
            double DZ_2R = abs(cz_2_sep[i] - cz_sep[i]) / (2 * radd);
            fprintf(DZvsDt_data, "%e\t%e\n", t_star, DZ_2R);
        }
		
    last_output_index = current_output_index+1;
    fclose(DZvsDt_data);
}

void dxdz(){


    static int last_output_index = 0; // 上一次輸出的索引

	int current_output_index = (step /(int)value_get)-1;

    // 初始化檔案
    if (step == iprint) {
        double Re_13, Ca_13;
        trajectories_data = fopen("./data/DXvsDZ.dat", "w");
        CHECK_FILE(trajectories_data, "./data/DXvsDZ.dat");

        // 寫入標題
        fprintf(trajectories_data, "VARIABLES=\"DX/2R\",\"DZ/2R\"\n");

        // 計算流體動力參數
        double mu = tau_l * dt * rho_l / 3.0;
        double shear_gamma = 2.0 * u_0 / nz;
        Re_13 = shear_gamma * rho_l * radd * radd / mu;
        Ca_13 = mu * shear_gamma * radd / surface_tension;

        // 寫入區域資訊
        fprintf(trajectories_data, "ZONE T=\"DXvsDZ_%.2fu_RE_%.2f_Ca_%.2f\" F=POINT\n", timesss, Re_13, Ca_13);
        fprintf(trajectories_data, "I=%d\n", (int)(stepall/value_get)-1);

    }
	else{
        trajectories_data = fopen("./data/DXvsDZ.dat", "a");
        CHECK_FILE(trajectories_data, "./data/DXvsDZ.dat");
	}

        for (int i = last_output_index; i <= current_output_index; i++) {
			double DX_2R = abs(cx_2_sep[i]-cx_sep[i])/2/radd;
			double DZ_2R = abs(cz_2_sep[i]-cz_sep[i])/2/radd;
			fprintf( trajectories_data, "%e\t%e\n",DX_2R,DZ_2R);
        }
		
    last_output_index = current_output_index+1;
	fclose(trajectories_data);

}

void theta(){
    static int last_output_index = 0; // 上一次輸出的索引
	int current_output_index = (step /(int)value_get)-1;
    // 初始化檔案
    if (step == iprint) {
        double Re_13, Ca_13;
        theta_data = fopen("./data/theta.dat", "w");
        CHECK_FILE(theta_data, "./data/theta.dat");
        // 寫入標題
        fprintf(theta_data, "VARIABLES=\"t*\",\"theta\"\n");

        // 計算流體動力參數
        double mu = tau_l * dt * rho_l / 3.0;
        double shear_gamma = 2.0 * u_0 / nz;
        Re_13 = shear_gamma * rho_l * radd * radd / mu;
        Ca_13 = mu * shear_gamma * radd / surface_tension;

        // 寫入區域資訊
        fprintf(theta_data, "ZONE T=\"theta.dat_%.2fu_RE_%.2f_Ca_%.2f\" F=POINT\n", timesss, Re_13, Ca_13);
        fprintf(theta_data, "I=%d\n", (int)(stepall/value_get)-1);

    }
	else{
        theta_data= fopen("./data/theta.dat", "a");
        CHECK_FILE(theta_data, "./data/theta.dat");
	}

        for (int i = last_output_index; i <= current_output_index; i++) {
            double shear_gamma = 2.0 * u_0 / nz;
            double t_star = shear_gamma * (i+1) * value_get;
			double DX = abs(cx_2_sep[i]-cx_sep[i]);
			double DZ = abs(cz_2_sep[i]-cz_sep[i]);
			double cross = sqrt(DX*DX + DZ*DZ);
			double rad	 = asin(DZ/cross);
			if(cross == 0){
				rad = 0 ;
			}
			double theta = rad*180/3.141592654;
			
			fprintf( theta_data, "%e\t%e\n",t_star,theta);
        }
		
    last_output_index = current_output_index+1;
	fclose(theta_data);




}

void Deformation(){

    static int last_output_index = 0; // 上一次輸出的索引
	int current_output_index = (step /(int)value_get)-1;
    // 初始化檔案
    if (step == iprint) {
        double Re_13, Ca_13;
        D_data = fopen("./data/D_data.dat", "w");
        CHECK_FILE(D_data, "./data/D_data.dat");
        // 寫入標題
		fprintf( D_data, "VARIABLES=\"t*\",\"D\"\n");

        // 計算流體動力參數
        double mu = tau_l * dt * rho_l / 3.0;
        double shear_gamma = 2.0 * u_0 / nz;
        Re_13 = shear_gamma * rho_l * radd * radd / mu;
        Ca_13 = mu * shear_gamma * radd / surface_tension;

        // 寫入區域資訊
        fprintf(D_data, "ZONE T=\"D_data.dat_%.2fu_RE_%.2f_Ca_%.2f\" F=POINT\n", timesss, Re_13, Ca_13);
        fprintf(D_data, "I=%d\n", (int)(stepall/value_get)-1);		

    }
	else{
        D_data= fopen("./data/D_data.dat", "a");
        CHECK_FILE(D_data, "./data/D_data.dat");
	}

        for (int i = last_output_index; i <= current_output_index; i++) {
			double shear_gamma = 2.0*u_0/nz;
            double t_star = shear_gamma * (i+1) * value_get;
			fprintf( D_data, "%e\t%e\n",t_star,D_value[i]);
        }
		
    last_output_index = current_output_index+1;
	fclose(D_data);



}
			

#endif
#ifndef INITIAL_H
#define INITIAL_H


void initial_difun (double *g,double *geq,double *g_t,double *h,double *heq,double *h_t)
{
	int i,j,k,l,index_l;
	for(i=0;i<nx+4;i++){
		for(j=0;j<ny+4;j++){
			for(k=0;k<nz/cpu+4;k++){
				for(l=0;l<q;l++){
//					index_l=((nx+4)*(k/cpu*(ny+4)+j)+i)*q+l;	//20200319
					index_l = i + (nx + 4)*(j + (ny + 4)*(k + (nz/cpu + 4)*l));
					g  [index_l]=0.0;
					h  [index_l]=0.0;
					geq[index_l]=0.0;
					heq[index_l]=0.0;
					g_t[index_l]=0.0;
					h_t[index_l]=0.0;
				}
			}
		}
	}
}


//void initial_macro (double *c,double *m,double *p,double *u,double *v,double *w)
void initial_macro (double *c,double *m,double *p,double *u,double *v,double *w)
{
	int i,j,k,index;
	double icent,jcent,kcent;
	for(i=0;i<nx;i++){
		for(j=0;j<ny;j++){
			for(k=0;k<nz;k++){
			index=nx*(k*ny+j)+i;
			c[index]=0.0;
			m[index]=0.0;
			p[index]=0.0;
			u[index]=0.0;
			v[index]=0.0;
			w[index]=0.0;
			}
		}
	}
//20200319	
/*	if(nx%2==0){
		icent=(double)(nx-1.0)/2.0;
	} else{
		icent=(double)(nx+1.0)/2.0;
	}
	if(ny%2==0){
		jcent=(double)(ny-1.0)/2.0;
	} else{
		jcent=(double)(ny+1.0)/2.0;
	}
	if(nz%2==0){
		kcent=(double)(nz-1.0)/2.0;
	} else{
		kcent=(double)(nz+1.0)/2.0;
	}*/
	icent = (double)(nx)/2.0;
	jcent = (double)(ny)/2.0;
	kcent = (double)(nz)/2.0;
	
	if(initial_type==1){ // collision
		if(equally == 1){
	/*20200319		double icent_r = icent + radd * b_coll;
			double icent_l = icent - radd * b_coll;
			double kcent_t = kcent - (7.0/6.0) * radd;
			double kcent_b = kcent + (7.0/6.0) * radd;*/
			double icent_r = icent + (7.0/6.0) * radd;
			double icent_l = icent - (7.0/6.0) * radd;
			double kcent_t = kcent + b_coll * radd;
			double kcent_b = kcent - b_coll * radd;
			double distance;
	/*		for(i = 0; i < nx; i++){
				for(j = 0; j < ny; j++){
					for(k = ((int)kcent); k < nz; k++){
						rad = sqrt((i - icent_l) * (i - icent_l) + (j - jcent) * (j - jcent) + (k - kcent_b) * (k - kcent_b));
						index = nx * (k * ny + j) + i;
						c[index] = (double)0.5 + (double)0.5 * tanh(2.0 * (radd - rad) / thick);
						w[index] = -c[index] * (u_coll / 2.0) + ((double)1.0 - c[index]) * 0.0;
					}
				}
			}*/

			for(i = (int)icent; i < nx; i++){
				for(j = 0; j < ny; j++){
					for(k = 0; k < nz; k++){
						distance = sqrt((i - icent_r)*(i - icent_r) + (j - jcent)*(j - jcent) + (k - kcent_b)*(k - kcent_b));
						index = nx*(j + k*ny) + i;
						c[index] = 0.5*(1.0 - tanh(2.0*(distance - radd)/thick));
						u[index] = - c[index] * u_coll/2.0;
						p[index] = c[index] * 2 * surface_tension / radd + p_out;//20220917
					}
				}
			}

	/*		for(i = 0; i < nx; i++){
				for(j = 0; j < ny; j++){
					for(k = 0; k < ((int)kcent) ; k++){
						rad = sqrt((i - icent_r) * (i - icent_r) + (j - jcent) * (j -jcent) + (k - kcent_t) * (k - kcent_t));
						index = nx * (k * ny + j) + i;
						c[index] = (double)0.5 + (double)0.5 * tanh(2.0 * (radd - rad) / thick);
						w[index] = c[index] * (u_coll / 2.0) + ((double)1.0 - c[index]) * 0.0;
					}
				}
			}*/
			for(i = 0; i < (int)icent; i++){
				for(j = 0; j < ny; j++){
					for(k = 0; k < nz; k++){
						distance = sqrt((i - icent_l)*(i - icent_l) + (j - jcent)*(j - jcent) + (k - kcent_t)*(k - kcent_t));
						index = nx*(j + k*ny) + i;
						c[index] = 0.5*(1.0 - tanh(2.0*(distance - radd)/thick));
						u[index] = c[index] * u_coll/2.0;
						p[index] = c[index] * 2 * surface_tension / radd + p_out;//20220917
					}
				}
			}
		}else{
			double icent_r = icent + (7.0/6.0) * radd;
			double icent_l = icent - (7.0/6.0) * radd;
			double kcent_t = kcent + b_coll * radd;
			double kcent_b = kcent - b_coll * radd;
			double distance;
			for(i = (int)icent; i < nx; i++){
				for(j = 0; j < ny; j++){
					for(k = 0; k < nz; k++){
						distance = sqrt((i - icent_r)*(i - icent_r) + (j - jcent)*(j - jcent) + (k - kcent_b)*(k - kcent_b));
						index = nx*(j + k*ny) + i;
						c[index] = 0.5*(1.0 - tanh(2.0*(distance - radd)/thick));
						u[index] = c[index] * u_coll*3.0/2.0;
						p[index] = c[index] * 2 * surface_tension / radd + p_out;//20220917
					}
				}
			}
			for(i = 0; i < (int)icent; i++){
				for(j = 0; j < ny; j++){
					for(k = 0; k < nz; k++){
						distance = sqrt((i - icent_l)*(i - icent_l) + (j - jcent)*(j - jcent) + (k - kcent_t)*(k - kcent_t));
						index = nx*(j + k*ny) + i;
						c[index] = 0.5*(1.0 - tanh(2.0*(distance - radd)/thick));
						u[index] = c[index] * u_coll*1.0/2.0;
						p[index] = c[index] * 2 * surface_tension / radd + p_out;//20220917
					}
				}
			}
		}
	}
	else if(initial_type==3){ // two drop merging
	
		double icent_r=icent+radd+thick/2.0;
		double icent_l=icent-radd-thick/2.0;
		double kcent_t=kcent;
		double kcent_b=kcent;
		
		for(i=0;i<=(int)icent;i++){
			for(j=0;j<ny;j++){
				for(k=0;k<nz;k++){
					double rad=sqrt( (i-icent_l)*(i-icent_l)+(j-jcent)*(j-jcent)+(k-kcent_b)*(k-kcent_b));
					index=nx*(k*ny+j)+i;
					c[index]=(double)0.5+(double)0.5*tanh(2.0*(radd-rad)/thick);
				}
			}
		} 
		
	 	for(i=(int)icent;i<nx;i++){
			for(j=0;j<ny;j++){
				for(k=0;k<nz;k++){
					double rad=sqrt( (i-icent_r)*(i-icent_r)+(j-jcent)*(j-jcent)+(k-kcent_t)*(k-kcent_t));
					index=nx*(k*ny+j)+i;
					c[index]=(double)0.5+(double)0.5*tanh(2.0*(radd-rad)/thick);
				}
			}
		} 
	}
	
	else if(initial_type==4){ // one bubble theory (phase separation)
		for(i=0;i<nx*ny*nz;i++){
			double uu   = rand() / (double)RAND_MAX;
			double vv   = rand() / (double)RAND_MAX;
			c[i] = ((double)6+sqrt(-2 * log(uu)) * cos(2 * M_PI * vv) * std_ini_4 + mean_ini_4)/12.0;
		}
	}
	
	else if(initial_type==5){ // drop oscillation
		for(i=0;i<nx;i++){
			for(j=0;j<ny;j++){
				for(k=0;k<nz;k++){
					double rad=sqrt( (i-icent)*(i-icent)+(j-jcent)*(j-jcent)+(k-kcent)*(k-kcent)/ar/ar);
					index=nx*(k*ny+j)+i;
					c[index]=(double)0.5+(double)0.5*tanh(2.0*(radd-rad)/thick);
				}
			}
		}
	}
/*20190320	
	else if(initial_type==6){ // wall test
		for(i=0;i<nx;i++){
			for(j=0;j<ny;j++){
				for(k=0;k<nz;k++){
					double rad=sqrt( (i-icent)*(i-icent)+(j-jcent)*(j-jcent)+(k-kcent)*(k-kcent));
					index=nx*(k*ny+j)+i;
					c[index]=(double)0.5+(double)0.5*tanh(2.0*(radd-rad)/thick);
					w[index] = c[index] * (u_coll / 2.0) + ((double)1.0 - c[index]) * 0.0;
				}
			}
		}
	}*/
	else if(initial_type==7){//cylinder
		double epsilon = thick/sqrt(8.0)/atanh(0.9);
		for(i = 0; i < nx; i++){
			for(j = 0; j < ny; j++){
				for(k = 0; k < nz; k++){
					double distance = sqrt((i - icent)*(i - icent)+(k - kcent)*(k - kcent));
					index = nx*(k*ny+j)+i;
					c[index] = 0.5*(1.0 - tanh((distance - radd)/sqrt(2.0)/epsilon));
					if(k==0){
						u[index] = -u_0;
					}
				}
				u[index] = u_0;
			}
		}
	}
	else if(initial_type==8){ // collision (after XXXXX time step give u_coll)
		/*20200319		double icent_r = icent + radd * b_coll;
		double icent_l = icent - radd * b_coll;
		double kcent_t = kcent - (7.0/6.0) * radd;
		double kcent_b = kcent + (7.0/6.0) * radd;*/
		double icent_r = icent + (7.0/6.0) * radd;
		double icent_l = icent - (7.0/6.0) * radd;
		double kcent_t = kcent + b_coll_7 * radd;
		double kcent_b = kcent - b_coll_7 * radd;
		double distance;
		/*for(i = 0; i < nx; i++){
			for(j = 0; j < ny; j++){
				for(k = ((int)kcent); k < nz; k++){
					rad = sqrt((i - icent_l) * (i - icent_l) + (j - jcent) * (j - jcent) + (k - kcent_b) * (k - kcent_b));
					index = nx * (k * ny + j) + i;
					c[index] = (double)0.5 + (double)0.5 * tanh(2.0 * (radd - rad) / thick);
					w[index] = -c[index] * (u_coll / 2.0) + ((double)1.0 - c[index]) * 0.0;
				 }
			}
		}*/
		for(i = (int)icent; i < nx; i++){
			for(j = 0; j < ny; j++){
				for(k = 0; k < nz; k++){
					distance = sqrt((i - icent_r)*(i - icent_r) + (j - jcent)*(j - jcent) + (k - kcent_b)*(k - kcent_b));
					index = nx*(j + k*ny) + i;
					c[index] = 0.5*(1.0 - tanh(2.0*(distance - radd)/thick));
					//u[index] = - c[index] * u_coll/2.0;
				}
			}
		}
/*		for(i = 0; i < nx; i++){
			for(j = 0; j < ny; j++){
				for(k = 0; k < ((int)kcent) ; k++){
					rad = sqrt((i - icent_r) * (i - icent_r) + (j - jcent) * (j -jcent) + (k - kcent_t) * (k - kcent_t));
					index = nx * (k * ny + j) + i;
					c[index] = (double)0.5 + (double)0.5 * tanh(2.0 * (radd - rad) / thick);
					w[index] = c[index] * (u_coll / 2.0) + ((double)1.0 - c[index]) * 0.0;
				}
			}
		}*/
		for(i = 0; i < (int)icent; i++){
			for(j = 0; j < ny; j++){
				for(k = 0; k < nz; k++){
					distance = sqrt((i - icent_l)*(i - icent_l) + (j - jcent)*(j - jcent) + (k - kcent_t)*(k - kcent_t));
					index = nx*(j + k*ny) + i;
					c[index] = 0.5*(1.0 - tanh(2.0*(distance - radd)/thick));
					//u[index] = c[index] * u_coll/2.0;
				}
			}
		}
	}
	else if(initial_type == 9){ // one droplet move_check energy

		double icent_l = icent - 1 * radd;
		double distance;

		for(i = 0; i < nx; i++){
			for(j = 0; j < ny; j++){
				for(k = 0; k < nz; k++){
					distance = sqrt((i - icent_l)*(i - icent_l) + (j - jcent)*(j - jcent) + (k - kcent)*(k - kcent));
					index = nx*(j + k*ny) + i;
					c[index] = 0.5*(1.0 - tanh(2.0*(distance - radd)/thick));
					u[index] = c[index] * u_coll_9;
					p[index] = c[index] * 2 * surface_tension / radd;//20220917
					//if(c[index]>0.1){
						//u[index] = u_coll_9;
					//}
				}
			}
		}
	}
	else if(initial_type == 10){ // one bubble 

		double icent_l = icent - 5 * radd;
		double distance;

		for(i = 0; i < nx; i++){
			for(j = 0; j < ny; j++){
				for(k = 0; k < nz; k++){
					distance = sqrt((i - icent_l)*(i - icent_l) + (j - jcent)*(j - jcent) + (k - kcent)*(k - kcent));
					index = nx*(j + k*ny) + i;
					c[index] = 0.5*(1.0 + tanh(2.0*(distance - radd)/thick));
					//u[index] = c[index] * u_coll_9;
				}
			}
		}
	}
	else if(initial_type == 11){ // shear flow droplet
		double icent_r = icent + (x_coll_2) * radd;
		double icent_l = icent - (x_coll_2) * radd;
		double kcent_t = kcent + b_coll_2 * radd;
		double kcent_b = kcent - b_coll_2 * radd;
		double distance;
		for(i = (int)icent; i < nx; i++){
			for(j = 0; j < ny; j++){
				for(k = 0; k < nz; k++){
					distance = sqrt((i - icent_r)*(i - icent_r) + (j - jcent)*(j - jcent) + (k - kcent_b)*(k - kcent_b));
					index = nx*(j + k*ny) + i;
					c[index] = 0.5*(1.0 - tanh(2.0*(distance - radd)/thick));
					p[index] = c[index] * 2 * surface_tension / radd + p_out;//20220917
					if(k==0){
						u[index] = -u_0;
					}
				}
				u[index] = u_0;
			}
		}
		for(i = 0; i < (int)icent; i++){
			for(j = 0; j < ny; j++){
				for(k = 0; k < nz; k++){
					distance = sqrt((i - icent_l)*(i - icent_l) + (j - jcent)*(j - jcent) + (k - kcent_t)*(k - kcent_t));
					index = nx*(j + k*ny) + i;
					c[index] = 0.5*(1.0 - tanh(2.0*(distance - radd)/thick));
					p[index] = c[index] * 2 * surface_tension / radd + p_out;//20220917
					if(k==0){
						u[index] = -u_0;
					}
				}
				u[index] = u_0;
			}
		}
	}
	else if(initial_type == 12){ // shear flow single droplet
		for(i=0;i<nx;i++){
			for(j=0;j<ny;j++){
				for(k=0;k<nz;k++){
					double distance = sqrt((i - icent)*(i - icent) + (j - jcent)*(j - jcent) + (k - kcent)*(k - kcent));
					index=nx*(k*ny+j)+i;
					if (bulk_free_energy == 1){
						double epsilon = thick/sqrt(8.0)/atanh(1.0-2.0*0.05);
						c[index] = 0.5*(1.0 - tanh((distance - radd)/sqrt(2.0)/epsilon));
						// p[index] = c[index] * 2 * surface_tension / radd; //20220917
					} else {
						double epsilon = thick/sqrt(2.0)/asin(1.0-2.0*0.05);
						c[index] = 0.5*(1.0 - sin(fmin(fmax(sqrt(2.0)*(distance - radd)/epsilon,-M_PI/2.0),M_PI/2.0)));
						// p[index] = c[index] * 2 * surface_tension / radd; //20220917
					}
					if(k==0){
						u[index] = -u_0;
					}
				}
				u[index] = u_0;
			}
		}
	}
	else if(initial_type == 13){ // shear flow droplet_sym
		double icent_r = icent + (x_coll_2) * radd;
		double icent_l = icent - (x_coll_2) * radd;
		double kcent_t = kcent + b_coll_2 * radd;
		double kcent_b = kcent - b_coll_2 * radd;
		double distance;
		for(i = (int)icent; i < nx; i++){
			for(j = 0; j < ny; j++){
				for(k = 0; k < nz; k++){
					distance = sqrt((i - icent_r)*(i - icent_r) + (j - (0))*(j - (0)) + (k - kcent_b)*(k - kcent_b));
					index = nx*(j + k*ny) + i;
					c[index] = 0.5*(1.0 - tanh(2.0*(distance - radd)/thick));
					p[index] = c[index] * 2 * surface_tension / radd + p_out;//20220917
					if(k==0){
						u[index] = -u_0;
					}
				}
				u[index] = u_0;
			}
		}
		for(i = 0; i < (int)icent; i++){
			for(j = 0; j < ny; j++){
				for(k = 0; k < nz; k++){
					distance = sqrt((i - icent_l)*(i - icent_l) + (j - (0))*(j - (0)) + (k - kcent_t)*(k - kcent_t));
					index = nx*(j + k*ny) + i;
					c[index] = 0.5*(1.0 - tanh(2.0*(distance - radd)/thick));
					p[index] = c[index] * 2 * surface_tension / radd + p_out;//20220917
					if(k==0){
						u[index] = -u_0;
					}
				}
				u[index] = u_0;
			}
		}
	}	
	else{					// stationary droplet
		for(i=0;i<nx;i++){
			for(j=0;j<ny;j++){
				for(k=0;k<nz;k++){
					double distance = sqrt((i - icent)*(i - icent) + (j - jcent)*(j - jcent) + (k - kcent)*(k - kcent));
					index=nx*(k*ny+j)+i;
					if (bulk_free_energy == 1){
						double epsilon = thick/sqrt(8.0)/atanh(1.0-2.0*0.05);
						c[index] = 0.5*(1.0 - tanh((distance - radd)/sqrt(2.0)/epsilon));
						// p[index] = c[index] * 2 * surface_tension / radd; //20220917
					} else {
						double epsilon = thick/sqrt(2.0)/asin(1.0-2.0*0.05);
						c[index] = 0.5*(1.0 - sin(fmin(fmax(sqrt(2.0)*(distance - radd)/epsilon,-M_PI/2.0),M_PI/2.0)));
						// p[index] = c[index] * 2 * surface_tension / radd; //20220917
					}
				}
			}
		}
	}
}

#endif
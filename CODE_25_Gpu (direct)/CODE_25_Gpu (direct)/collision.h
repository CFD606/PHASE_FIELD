#ifndef COLLISION
#define COLLISION


#include"preparation.h"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                 eq collision                                                   //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void eq (	double *g, double *h,double *geq ,double *heq ,double *c ,double *m, double *p,double mobi,
			double *u_in,double *v_in,double *w_in)
{
	// i:2~nx+1, j:2~ny+1, k:2~nz/cpu+1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	
	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1 && k >= 2 && k <= nz/cpu+1) {
		int index=index_3d(i,j,k);
		double cs=1.0/pow(3.0,0.5);
		
		double u=u_in[index];
		double v=v_in[index];
		double w=w_in[index];
		double r=c[index]*rho_l+((double)1.0-c[index])*rho_g;
		double dr=rho_l-rho_g;
		double cc = c[index];
	/********************gradient macro central diffent for equilibrium************************/
		double gr_cx_c=gra_phi_c(c,ex_d,cs,index);
		double gr_cy_c=gra_phi_c(c,ey_d,cs,index);
		double gr_cz_c=gra_phi_c(c,ez_d,cs,index);
		double gr_rx_c=gr_cx_c*dr;
		double gr_ry_c=gr_cy_c*dr;
		double gr_rz_c=gr_cz_c*dr;
		double gr_mx_c=gra_phi_c(m,ex_d,cs,index);
		double gr_my_c=gra_phi_c(m,ey_d,cs,index);
		double gr_mz_c=gra_phi_c(m,ez_d,cs,index);
		double gr_px_c=gra_phi_c(p,ex_d,cs,index);
		double gr_py_c=gra_phi_c(p,ey_d,cs,index);
		double gr_pz_c=gra_phi_c(p,ez_d,cs,index);
	/******************** laplacian ************************/		
		double lap_mu = lap_phi(m,cs,index);
	
	/******************** for Allen Cahn equilibrium ************************/
		double m_abs = 0.0;
		double sum_m = gr_cx_c * gr_cx_c + gr_cy_c * gr_cy_c + gr_cz_c * gr_cz_c;
		m_abs = sqrt(sum_m);
		double nvx, nvy, nvz;
		nvx = gr_cx_c / (m_abs + 10e-12);  // n = (-m)/(|m|+10^(-12))
		nvy = gr_cy_c / (m_abs + 10e-12);
		nvz = gr_cz_c / (m_abs + 10e-12);
		double cs2_inv = 1/cs/cs;
		double thick_inv = 1.0 / thick;
		double temp_theta = mobi * cs2_inv * (4 * cc - 4 * cc * cc) * thick_inv;	


		for(int l=0;l<q;l++){

			double ex=ex_d[l];
			double ey=ey_d[l];
			double ez=ez_d[l];
			double wt=wt_d[l];
			int    et=et_d[l];
			int index_l=index_4d(i,j,k,l);
			double udotu=u*u+v*v+w*w;
			double edotu=ex*u+ey*v+ez*w;
			double uugly=edotu/pow(cs,2.0)+edotu*edotu/(2.0*pow(cs,4.0))-udotu/(2.0*pow(cs,2.0));
			double gamma=wt*(1.0+uugly);

			double geq_t=wt*(p[index]+r*cs*cs*uugly);
			double heq_t=c[index]*gamma;
			
		/********************direction central diffent for equilibrium************************/
			double gr_ce_c=grad_phie_c( c,index,et );
			double gr_re_c=gr_ce_c*dr;//grad_phie_c( r,index,et );
			double gr_me_c=grad_phie_c( m,index,et );
			double gr_pe_c=grad_phie_c( p,index,et );

		/******************** gravity ************************/
			double gra = (r-rho_l)*(gra_ac*ex - gra_ac*u); 

		/********************(e-u) central diffent for equilibrium************************/

			double temp_c_c,temp_r_c,temp_m_c,temp_p_c,temp_g_c,temp_g_c_2,temp_h_c,temp_h_c_2;
			temp_m_c =gr_me_c - ( u * gr_mx_c + v * gr_my_c+ w * gr_mz_c ); 
			temp_r_c =gr_re_c - ( u * gr_rx_c + v * gr_ry_c+ w * gr_rz_c );
			temp_c_c =gr_ce_c - ( u * gr_cx_c + v * gr_cy_c+ w * gr_cz_c );
			temp_p_c =gr_pe_c - ( u * gr_px_c + v * gr_py_c+ w * gr_pz_c );

			if(Force_method == 0){

				temp_g_c =temp_r_c*cs*cs*wt*uugly-(temp_m_c*c[index] + gra)*gamma;
				geq[index_l]=geq_t-0.5*dt*temp_g_c;

				if(PH_model == 0)
				{
					temp_h_c =temp_c_c-(temp_p_c+c[index]*temp_m_c + gra)*c[index]/(r*cs*cs);
					heq[index_l]=heq_t-0.5*dt*temp_h_c*gamma-0.5*dt*mobi*lap_mu*gamma;     
				}
				else{
					heq[index_l] = cc*gamma + wt * temp_theta * (ex * nvx + ey * nvy + ez * nvz);
				}				
			}
			else if(Force_method == 1){

				temp_g_c =temp_r_c*cs*cs*wt;
				temp_g_c_2 =wt*(gr_re_c*cs*cs-c[index]*gr_me_c);
				geq[index_l]=geq_t+0.5*dt*temp_g_c-0.5*dt*temp_g_c_2;
				if(PH_model == 0)
				{
					temp_h_c = gamma*(temp_c_c+mobility*lap_mu-(c[index]/r)*temp_r_c);
					temp_h_c_2 = wt*(c[index]/r*cs*cs)*(gr_re_c*cs*cs-gr_pe_c-c[index]*gr_me_c);
					heq[index_l]=heq_t-0.5*temp_h_c-0.5*dt*temp_h_c_2;   
				}
				else{
					heq[index_l] = cc*gamma + wt * temp_theta * (ex * nvx + ey * nvy + ez * nvz);
				}
			}
			else if(Force_method == 2){

				// for heq  F=gra_r*cs*cs-gra_p-C_gra_mu
				double shi_e_u_F_c_h = temp_r_c*cs*cs-temp_p_c-cc*temp_m_c;
				double shi_e_F_c_h = gr_re_c*cs*cs-gr_pe_c-cc*gr_me_c;
				double shi_c_h = (cc/r)*wt*(shi_e_u_F_c_h/pow(cs,2.0)+(edotu*shi_e_F_c_h)/pow(cs,4.0));
				//for geq   F=gra_r*cs*cs-C_gra_mu
				double shi_e_u_F_c_g = temp_r_c*cs*cs-cc*temp_m_c;
				double shi_e_F_c_g = gr_re_c*cs*cs-cc*gr_me_c;
				double shi_c_g = wt*(shi_e_u_F_c_g/pow(cs,2.0)+(edotu*shi_e_F_c_g)/pow(cs,4.0));		
				// g
				temp_g_c =temp_r_c*cs*cs*wt;
				geq[index_l]=geq_t+0.5*dt*temp_g_c-0.5*dt*shi_c_g*cs*cs;

				// h
				if(PH_model == 0)
				{ 
					temp_h_c = gamma*(temp_c_c+mobility*lap_mu-(c[index]/r)*temp_r_c);
					heq[index_l]=heq_t-0.5*temp_h_c-0.5*dt*shi_c_h;     //2024/0911
				}		
				else{
					heq[index_l] = cc*gamma + wt * temp_theta * (ex * nvx + ey * nvy + ez * nvz);
				}						
			}

		}
	}
}

__global__ void eq0 (	double *g, double *h,double *geq ,double *heq )
{
	// i:2~nx+1, j:2~ny+1, k:2~nz/cpu+1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	
	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1 && k >= 2 && k <= nz/cpu+1) {
		for (int l = 0; l < q; l++) {
			int index_l = index_4d(i,j,k,l);
			g[index_l] = geq[index_l];
			h[index_l] = heq[index_l];
		}
	}
}

__global__ void collision (	double *g, double *h,double *geq ,double *heq ,double *c ,double *m, double *p,
							double *u_in,double *v_in,double *w_in,double mobi)
{
	// i:2~nx+1, j:2~ny+1, k:2~nz/cpu+1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	
	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1 && k >= 2 && k <= nz/cpu+1) {
		int index=index_3d(i,j,k);
		double cs=1.0/pow(3.0,0.5);
		
		double u=u_in[index];
		double v=v_in[index];
		double w=w_in[index];
		double cc=c[index];
		double rr=cc*rho_l+((double)1.0-cc)*rho_g;
		double tt=cc*tau_l+((double)1.0-cc)*tau_g;
		double dr=rho_l-rho_g;

	/********************gradient macro mixing diffent for evolution************************/		
		double gr_cx_m=gra_phi_m( c,ex_d,cs,index );
		double gr_cy_m=gra_phi_m( c,ey_d,cs,index );
		double gr_cz_m=gra_phi_m( c,ez_d,cs,index );
		double gr_rx_m=gr_cx_m*dr;
		double gr_ry_m=gr_cy_m*dr;
		double gr_rz_m=gr_cz_m*dr;
		double gr_mx_m=gra_phi_m( m,ex_d,cs,index );
		double gr_my_m=gra_phi_m( m,ey_d,cs,index );
		double gr_mz_m=gra_phi_m( m,ez_d,cs,index );
		double gr_px_m=gra_phi_m( p,ex_d,cs,index );
		double gr_py_m=gra_phi_m( p,ey_d,cs,index );
		double gr_pz_m=gra_phi_m( p,ez_d,cs,index );
	/******************** laplacian ************************/
		double lap_mu  =lap_phi  ( m,cs,index );


		double udotu=u*u+v*v+w*w;		
		for(int l=0;l<q;l++){
			double ex=ex_d[l];
			double ey=ey_d[l];
			double ez=ez_d[l];
			double wt=wt_d[l];
			int    et=et_d[l];
			int index_l=index_4d(i,j,k,l);
			double edotu=ex*u+ey*v+ez*w;
			double uugly=edotu/pow(cs,2.0)+edotu*edotu/(2.0*pow(cs,4.0))-udotu/(2.0*pow(cs,2.0));
			double gamma=wt*(1.0+uugly);

		/********************direction mixing diffent for evolution************************/
			double gr_ce_m=grad_phie_m( c,index,et );
			double gr_re_m=gr_ce_m*dr;
			double gr_me_m=grad_phie_m( m,index,et );
			double gr_pe_m=grad_phie_m( p,index,et );

		/********************(e-u) mixing diffent for evolution************************/
			double temp_c_m,temp_r_m,temp_m_m,temp_p_m,temp_g_m,temp_g_m_2,temp_h_m,temp_h_m_2;
			temp_m_m = gr_me_m-( u * gr_mx_m + v * gr_my_m + w * gr_mz_m);
			temp_r_m = gr_re_m-( u * gr_rx_m + v * gr_ry_m + w * gr_rz_m);  
			temp_c_m = gr_ce_m-( u * gr_cx_m + v * gr_cy_m + w * gr_cz_m);  
			temp_p_m = gr_pe_m-( u * gr_px_m + v * gr_py_m + w * gr_pz_m);
		/******************** gravity ************************/
			double gra = (rr-rho_l)*(gra_ac*ex - gra_ac*u);

			if(Force_method == 0)
			{

				temp_g_m = temp_r_m*cs*cs*wt*uugly-(temp_m_m*cc + gra)*gamma;
				g[index_l]=g[index_l]-(g[index_l]-geq[index_l])/(tt    +0.5)+dt*temp_g_m;

				if(PH_model == 0){

					temp_h_m = temp_c_m-(temp_p_m+cc*temp_m_m + gra)*cc/(rr*cs*cs);
					temp_h_m = gamma*(0.5*mobi*lap_mu+temp_h_m);
					h[index_l]=h[index_l]-(h[index_l]-heq[index_l])/(tau_h +0.5)+dt*temp_h_m;

				}else{
					h[index_l] = h[index_l] * (1.0 - 1.0 / (tau_h + 0.5)) + heq[index_l] / (tau_h + 0.5) ;
				}
			}
			else if (Force_method == 1)
			{
				temp_g_m = temp_r_m*cs*cs*wt;
				temp_g_m_2 = wt*(gr_re_m*cs*cs-cc*gr_me_m);
				g[index_l]=g[index_l]-(g[index_l]-geq[index_l])/(tt+0.5)-dt*temp_g_m+dt*temp_g_m_2;
				
				if(PH_model == 0){

				temp_h_m = gamma*(temp_c_m+mobility*lap_mu-(cc/rr)*temp_r_m);
				temp_h_m_2 = wt*(cc/rr*cs*cs)*(gr_re_m*cs*cs-gr_pe_m-cc*gr_me_m);
				h[index_l]=h[index_l]-(h[index_l]-heq[index_l])/(tau_h +0.5)+dt*(temp_h_m+temp_h_m_2);
				}else{
					h[index_l] = h[index_l] * (1.0 - 1.0 / (tau_h + 0.5)) + heq[index_l] / (tau_h + 0.5);
				}
			}
			else if(Force_method == 2)
			{
				// for h  F=gra_r*cs*cs-gra_p-C_gra_mu
				double shi_e_u_F_m_h = temp_r_m*cs*cs-temp_p_m-cc*temp_m_m;
				double shi_e_F_m_h = gr_re_m*cs*cs-gr_pe_m-cc*gr_me_m;
				double shi_m_h = (cc/rr)*wt*(shi_e_u_F_m_h/pow(cs,2.0)+(edotu*shi_e_F_m_h)/pow(cs,4.0));
				//for g   F=gra_r*cs*cs-C_gra_mu
				double shi_e_u_F_m_g = temp_r_m*cs*cs-cc*temp_m_m;
				double shi_e_F_m_g = gr_re_m*cs*cs-cc*gr_me_m;
				double shi_m_g = wt*(shi_e_u_F_m_g/pow(cs,2.0)+(edotu*shi_e_F_m_g)/pow(cs,4.0));				
				// g
				temp_g_m = temp_r_m*cs*cs*wt;
				g[index_l]=g[index_l]-(g[index_l]-geq[index_l])/(tt+0.5)-dt*temp_g_m+dt*shi_m_g*cs*cs;
	
				// h
				if(PH_model == 0){
					temp_h_m = gamma*(temp_c_m+mobility*lap_mu-(cc/rr)*temp_r_m);
					h[index_l]=h[index_l]-(h[index_l]-heq[index_l])/(tau_h +0.5)+dt*(temp_h_m+shi_m_h);
				}else{
					h[index_l] = h[index_l] * (1.0 - 1.0 / (tau_h + 0.5)) + heq[index_l] / (tau_h + 0.5);
				}
			}
			
		}
	}
}

__global__ void eq_collision_in (	double *g, double *h, double *c ,double *m, double *p,double *gra_c,double *gra_m,double *gra_p,
									double *lap_m,double *u_in,double *v_in,double *w_in,double mobi)
{
	// i:2~nx+1, j:2~ny+1, k:3~nz/cpu
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	
	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1 && k >= 3 && k <= nz/cpu) {
		const double cs2_inv=3.0;
		const double cs2=1.0/cs2_inv;

		const int index=index_3d(i,j,k);		
		double u=u_in[index];
		double v=v_in[index];
		double w=w_in[index];
		double cc=c[index];
		double rr=cc*rho_l+((double)1.0-cc)*rho_g;
		double tt=cc*tau_l+((double)1.0-cc)*tau_g;
		double rr_inv=1.0/rr;
		double dr=rho_l-rho_g;
	/********************gradient macro central diffent for equilibrium************************/
		double gr_cx_c=gra_c[index_4d(i,j,k,0)];
		double gr_cy_c=gra_c[index_4d(i,j,k,1)];
		double gr_cz_c=gra_c[index_4d(i,j,k,2)];
		double gr_rx_c=gr_cx_c*dr;
		double gr_ry_c=gr_cy_c*dr;
		double gr_rz_c=gr_cz_c*dr;
		double gr_mx_c=gra_m[index_4d(i,j,k,0)];
		double gr_my_c=gra_m[index_4d(i,j,k,1)];
		double gr_mz_c=gra_m[index_4d(i,j,k,2)];
		double gr_px_c=gra_p[index_4d(i,j,k,0)];
		double gr_py_c=gra_p[index_4d(i,j,k,1)];
		double gr_pz_c=gra_p[index_4d(i,j,k,2)];
	/********************gradient macro mixing diffent for evolution************************/		
		double gr_cx_m=gra_c[index_4d(i,j,k,3)];
		double gr_cy_m=gra_c[index_4d(i,j,k,4)];
		double gr_cz_m=gra_c[index_4d(i,j,k,5)];
		double gr_rx_m=gr_cx_m*dr;
		double gr_ry_m=gr_cy_m*dr;
		double gr_rz_m=gr_cz_m*dr;
		double gr_mx_m=gra_m[index_4d(i,j,k,3)];
		double gr_my_m=gra_m[index_4d(i,j,k,4)];
		double gr_mz_m=gra_m[index_4d(i,j,k,5)];
		double gr_px_m=gra_p[index_4d(i,j,k,3)];
		double gr_py_m=gra_p[index_4d(i,j,k,4)];
		double gr_pz_m=gra_p[index_4d(i,j,k,5)];

	/******************** laplacian ************************/		
		double lap_mu  =lap_m[index];

	/******************** for Allen Cahn equilibrium ************************/
		double m_abs = 0.0;
		double sum_m = gr_cx_c * gr_cx_c + gr_cy_c * gr_cy_c + gr_cz_c * gr_cz_c;
		m_abs = sqrt(sum_m);
		double nvx, nvy, nvz;
		nvx = gr_cx_c / (m_abs + 10e-12);  // n = (-m)/(|m|+10^(-12))
		nvy = gr_cy_c / (m_abs + 10e-12);
		nvz = gr_cz_c / (m_abs + 10e-12);

		double thick_inv = 1.0 / thick;
		double temp_theta = mobi * cs2_inv * (4 * cc - 4 * cc * cc) * thick_inv;	 // theta = M*(4c(1-c))/thick/cs^2	


		double udotu=u*u+v*v+w*w;
		for(int l=0;l<q;l++){
			int index_l=index_4d(i,j,k,l);
			double ex=ex_d[l];
			double ey=ey_d[l];
			double ez=ez_d[l];
			double wt=wt_d[l];
			int    et=et_d[l];

			double edotu=ex*u+ey*v+ez*w;
			double uugly=edotu*cs2_inv+edotu*edotu*0.5*cs2_inv*cs2_inv-udotu*0.5*cs2_inv;
			double gamma=wt*(1.0+uugly);
		/********************direction central diffent for evolution************************/
			double gr_ce_c=grad_phie_c( c,index,et );
			double gr_re_c=gr_ce_c*dr;//grad_phie_c( r,index,et );
			double gr_me_c=grad_phie_c( m,index,et );
			double gr_pe_c=grad_phie_c( p,index,et );
		/********************direction mixing diffent for evolution************************/
			double gr_ce_m=grad_phie_m( c,index,et );
			double gr_re_m=gr_ce_m*dr;
			double gr_me_m=grad_phie_m( m,index,et );
			double gr_pe_m=grad_phie_m( p,index,et );
		/********************(e-u) central diffent for evolution************************/
			double temp_c_c = gr_ce_c -( u * gr_cx_c + v * gr_cy_c + w * gr_cz_c );  
			double temp_m_c = gr_me_c -( u * gr_mx_c + v * gr_my_c + w * gr_mz_c );  
			double temp_p_c = gr_pe_c - ( u * gr_px_c + v * gr_py_c + w * gr_pz_c );
			double temp_r_c = gr_re_c - ( u * gr_rx_c + v * gr_ry_c + w * gr_rz_c );
		/********************(e-u) mixing diffent for evolution************************/
			double temp_c_m = gr_ce_m -( u * gr_cx_m + v * gr_cy_m + w * gr_cz_m ); 
			double temp_m_m = gr_me_m -( u * gr_mx_m + v * gr_my_m + w * gr_mz_m );  
			double temp_p_m = gr_pe_m - ( u * gr_px_m + v * gr_py_m + w * gr_pz_m );
			double temp_r_m = gr_re_m - ( u * gr_rx_m + v * gr_ry_m + w * gr_rz_m );
		/******************** gravity ************************/
			double gra = (rr-rho_l)*(gra_ac*ex - gra_ac*u); //20220810, gravity force, x deriction
			
			if(Force_method == 0){
				/*******************g equilibrium*****************/
				double geq_t=wt*(p[index]+rr*cs2*uugly);
				double temp_gc = temp_c_c*dr*cs2*wt*uugly-(temp_m_c*cc + gra)*gamma;
				geq_t = geq_t-0.5*dt*temp_gc;

				/*******************g evolution*****************/
				double temp_gm = temp_c_m*dr*cs2*wt*uugly-(temp_m_m*cc + gra)*gamma;
				g[index_l] = g[index_l]*(1.0-1.0/(tt	+0.5))+geq_t/(tt	+0.5)+dt*temp_gm;

				if(PH_model == 0){
					/*******************h equilibrium*****************/
					double heq_t=cc*gamma;
					double temp_hc = temp_c_c-(temp_p_c+cc*temp_m_c + gra)*cc*cs2_inv*rr_inv;
					heq_t = heq_t-0.5*dt*temp_hc*gamma-0.5*dt*mobi*lap_mu*gamma;
					
					/*******************h evolution*****************/
					double temp_hm = temp_c_m-(temp_p_m+cc*temp_m_m + gra)*cc*cs2_inv*rr_inv;
					temp_hm = gamma*(0.5*mobi*lap_mu+temp_hm);
					h[index_l] = h[index_l]*(1.0-1.0/(tau_h	+0.5))+heq_t/(tau_h	+0.5)+dt*temp_hm;

				}else{
					/*******************h equilibrium*****************/
					double heq_t = cc * gamma + wt * temp_theta * (ex * nvx + ey * nvy + ez * nvz);
					/*******************h evolution*****************/					
					h[index_l] = h[index_l] * (1.0 - 1.0 / (tau_h + 0.5)) + heq_t / (tau_h + 0.5);
				}
			}
			else if (Force_method == 1)
			{
				double geq_t=wt*(p[index]+rr*cs2*uugly);
				double temp_gc_1 = wt*temp_c_c*dr*cs2;
				double temp_gc_2 = wt*(gr_ce_c*dr*cs2-cc*gr_me_c);
				geq_t = geq_t+0.5*dt*temp_gc_1-0.5*dt*temp_gc_2;

				double temp_gm_1 = wt*temp_c_m*dr*cs2;
				double temp_gm_2 = wt*(gr_ce_m*dr*cs2-cc*gr_me_m);

				g[index_l] = g[index_l]*(1.0-1.0/(tt	+0.5))+geq_t/(tt	+0.5)-dt*temp_gm_1+dt*temp_gm_2;

				// h
				if(PH_model == 0){
					double heq_t=cc*gamma;

					double temp_hc_1 =gamma*(temp_c_c+mobility*lap_mu-(cc/rr)*temp_c_c*dr);
					double temp_hc_2 = wt*(cc/rr*cs2)*(gr_ce_c*dr*cs2-gr_pe_c-cc*gr_me_c);
					heq_t = heq_t-0.5*dt*(temp_hc_1+temp_hc_2);

					double temp_hm_1 = gamma*(temp_c_m+mobility*lap_mu-(cc/rr)*temp_c_m*dr);
					double temp_hm_2 = wt*(cc/rr*cs2)*(gr_ce_m*dr*cs2-gr_pe_m-cc*gr_me_m);
					h[index_l] = h[index_l]*(1.0-1.0/(tau_h	+0.5))+heq_t/(tau_h	+0.5)+dt*(temp_hm_1+temp_hm_2);
				}else{
					double heq_t = cc * gamma + wt * temp_theta * (ex * nvx + ey * nvy + ez * nvz);
					h[index_l] = h[index_l] * (1.0 - 1.0 / (tau_h + 0.5)) + heq_t / (tau_h + 0.5);
				}
				
			}
			else if (Force_method == 2)
			{
				//for h eq 
				double shi_e_u_F_c_h = temp_r_c*cs2-temp_p_c-cc*temp_m_c;
				double shi_e_F_c_h = gr_re_c*cs2-gr_pe_c-cc*gr_me_c;
				double shi_c_h = (cc/rr)*wt*(shi_e_u_F_c_h/cs2+(edotu*shi_e_F_c_h)/cs2*cs2);
				//for h evolution
				double shi_e_u_F_m_h = temp_r_m*cs2-temp_p_m-cc*temp_m_m;
				double shi_e_F_m_h = gr_re_m*cs2-gr_pe_m-cc*gr_me_m;
				double shi_m_h = (cc/rr)*wt*(shi_e_u_F_m_h/cs2+(edotu*shi_e_F_m_h)/cs2*cs2);
				//for g eq
				double shi_e_u_F_c_g = temp_r_c*cs2-cc*temp_m_c;
				double shi_e_F_c_g = gr_re_c*cs2-cc*gr_me_c;
				double shi_c_g = wt*(shi_e_u_F_c_g/cs2+(edotu*shi_e_F_c_g)/cs2*cs2);		
				//for g evolution
				double shi_e_u_F_m_g = temp_r_m*cs2-cc*temp_m_m;
				double shi_e_F_m_g = gr_re_m*cs2-cc*gr_me_m;
				double shi_m_g = wt*(shi_e_u_F_m_g/cs2+(edotu*shi_e_F_m_g)/cs2*cs2);		
				// g
				double geq_t=wt*(p[index]+rr*cs2*uugly);
				double temp_gc_1 = temp_c_c*dr*cs2*wt;
				geq_t = geq_t+0.5*dt*temp_gc_1-0.5*dt*shi_c_g*cs2;
				double temp_gm_1 = temp_c_m*dr*cs2*wt;	
				g[index_l] = g[index_l]*(1.0-1.0/(tt	+0.5))+geq_t/(tt	+0.5)-dt*temp_gm_1+dt*shi_m_g*cs2;

				// h
				if(PH_model == 0){
					double heq_t=cc*gamma;
					double temp_hc_1 =gamma*(temp_c_c+mobility*lap_mu-(cc/rr)*temp_c_c*dr);
					heq_t = heq_t-0.5*dt*(temp_hc_1+shi_c_h);
					double temp_hm_1 = gamma*(temp_c_m+mobility*lap_mu-(cc/rr)*temp_c_m*dr);
					h[index_l] = h[index_l]*(1.0-1.0/(tau_h	+0.5))+heq_t/(tau_h	+0.5)+dt*(temp_hm_1+shi_m_h);
				}else{
					double heq_t = cc * gamma + wt * temp_theta * (ex * nvx + ey * nvy + ez * nvz);
					h[index_l] = h[index_l] * (1.0 - 1.0 / (tau_h + 0.5)) + heq_t / (tau_h + 0.5);
				}			
			}
			
		}
	}
}

__global__ void eq_collision_bc (	double *g, double *h, double *c ,double *m, double *p,double *gra_c,double *gra_m,double *gra_p,
									double *lap_m,double *u_in,double *v_in,double *w_in,double mobi)
{
	// i:2~nx+1, j:2~ny+1, k:2,nz/cpu+1
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (i >= 2 && i <= nx+1 && j >= 2 && j <= ny+1) {
		const double cs2_inv=3.0;
		const double cs2=1.0/cs2_inv;
		
		for(int k=2;k<nz/cpu+2;k=k+nz/cpu-1){
			
		const int index=index_3d(i,j,k);		
		double u=u_in[index];
		double v=v_in[index];
		double w=w_in[index];
		double cc=c[index];
		double rr=cc*rho_l+((double)1.0-cc)*rho_g;
		double tt=cc*tau_l+((double)1.0-cc)*tau_g;
		double rr_inv=1.0/rr;
		double dr=rho_l-rho_g;
	/********************gradient macro central diffent for equilibrium************************/
		double gr_cx_c=gra_c[index_4d(i,j,k,0)];
		double gr_cy_c=gra_c[index_4d(i,j,k,1)];
		double gr_cz_c=gra_c[index_4d(i,j,k,2)];
		double gr_rx_c=gr_cx_c*dr;
		double gr_ry_c=gr_cy_c*dr;
		double gr_rz_c=gr_cz_c*dr;
		double gr_mx_c=gra_m[index_4d(i,j,k,0)];
		double gr_my_c=gra_m[index_4d(i,j,k,1)];
		double gr_mz_c=gra_m[index_4d(i,j,k,2)];
		double gr_px_c=gra_p[index_4d(i,j,k,0)];
		double gr_py_c=gra_p[index_4d(i,j,k,1)];
		double gr_pz_c=gra_p[index_4d(i,j,k,2)];
	/********************gradient macro mixing diffent for evolution************************/		
		double gr_cx_m=gra_c[index_4d(i,j,k,3)];
		double gr_cy_m=gra_c[index_4d(i,j,k,4)];
		double gr_cz_m=gra_c[index_4d(i,j,k,5)];
		double gr_rx_m=gr_cx_m*dr;
		double gr_ry_m=gr_cy_m*dr;
		double gr_rz_m=gr_cz_m*dr;
		double gr_mx_m=gra_m[index_4d(i,j,k,3)];
		double gr_my_m=gra_m[index_4d(i,j,k,4)];
		double gr_mz_m=gra_m[index_4d(i,j,k,5)];
		double gr_px_m=gra_p[index_4d(i,j,k,3)];
		double gr_py_m=gra_p[index_4d(i,j,k,4)];
		double gr_pz_m=gra_p[index_4d(i,j,k,5)];

	/******************** laplacian ************************/		
		double lap_mu  =lap_m[index];

	/******************** for Allen Cahn equilibrium ************************/
		double m_abs = 0.0;
		double sum_m = gr_cx_c * gr_cx_c + gr_cy_c * gr_cy_c + gr_cz_c * gr_cz_c;
		m_abs = sqrt(sum_m);
		double nvx, nvy, nvz;
		nvx = gr_cx_c / (m_abs + 10e-12);  // n = (-m)/(|m|+10^(-12))
		nvy = gr_cy_c / (m_abs + 10e-12);
		nvz = gr_cz_c / (m_abs + 10e-12);

		double thick_inv = 1.0 / thick;
		double temp_theta = mobi * cs2_inv * (4 * cc - 4 * cc * cc) * thick_inv;	 // theta = M*(4c(1-c))/thick/cs^2	


		double udotu=u*u+v*v+w*w;
		for(int l=0;l<q;l++){
			int index_l=index_4d(i,j,k,l);
			double ex=ex_d[l];
			double ey=ey_d[l];
			double ez=ez_d[l];
			double wt=wt_d[l];
			int    et=et_d[l];

			double edotu=ex*u+ey*v+ez*w;
			double uugly=edotu*cs2_inv+edotu*edotu*0.5*cs2_inv*cs2_inv-udotu*0.5*cs2_inv;
			double gamma=wt*(1.0+uugly);
		/********************direction central diffent for evolution************************/
			double gr_ce_c=grad_phie_c( c,index,et );
			double gr_re_c=gr_ce_c*dr;//grad_phie_c( r,index,et );
			double gr_me_c=grad_phie_c( m,index,et );
			double gr_pe_c=grad_phie_c( p,index,et );
		/********************direction mixing diffent for evolution************************/
			double gr_ce_m=grad_phie_m( c,index,et );
			double gr_re_m=gr_ce_m*dr;
			double gr_me_m=grad_phie_m( m,index,et );
			double gr_pe_m=grad_phie_m( p,index,et );
		/********************(e-u) central diffent for evolution************************/
			double temp_c_c = gr_ce_c -( u * gr_cx_c + v * gr_cy_c + w * gr_cz_c );  
			double temp_m_c = gr_me_c -( u * gr_mx_c + v * gr_my_c + w * gr_mz_c );  
			double temp_p_c = gr_pe_c - ( u * gr_px_c + v * gr_py_c + w * gr_pz_c );
			double temp_r_c = gr_re_c - ( u * gr_rx_c + v * gr_ry_c + w * gr_rz_c );
		/********************(e-u) mixing diffent for evolution************************/
			double temp_c_m = gr_ce_m -( u * gr_cx_m + v * gr_cy_m + w * gr_cz_m ); 
			double temp_m_m = gr_me_m -( u * gr_mx_m + v * gr_my_m + w * gr_mz_m );  
			double temp_p_m = gr_pe_m - ( u * gr_px_m + v * gr_py_m + w * gr_pz_m );
			double temp_r_m = gr_re_m - ( u * gr_rx_m + v * gr_ry_m + w * gr_rz_m );
		/******************** gravity ************************/
			double gra = (rr-rho_l)*(gra_ac*ex - gra_ac*u); //20220810, gravity force, x deriction
			
			if(Force_method == 0){
				/*******************g equilibrium*****************/
				double geq_t=wt*(p[index]+rr*cs2*uugly);
				double temp_gc = temp_c_c*dr*cs2*wt*uugly-(temp_m_c*cc + gra)*gamma;
				geq_t = geq_t-0.5*dt*temp_gc;

				/*******************g evolution*****************/
				double temp_gm = temp_c_m*dr*cs2*wt*uugly-(temp_m_m*cc + gra)*gamma;
				g[index_l] = g[index_l]*(1.0-1.0/(tt	+0.5))+geq_t/(tt	+0.5)+dt*temp_gm;

				if(PH_model == 0){
					/*******************h equilibrium*****************/
					double heq_t=cc*gamma;
					double temp_hc = temp_c_c-(temp_p_c+cc*temp_m_c + gra)*cc*cs2_inv*rr_inv;
					heq_t = heq_t-0.5*dt*temp_hc*gamma-0.5*dt*mobi*lap_mu*gamma;
					
					/*******************h evolution*****************/
					double temp_hm = temp_c_m-(temp_p_m+cc*temp_m_m + gra)*cc*cs2_inv*rr_inv;
					temp_hm = gamma*(0.5*mobi*lap_mu+temp_hm);
					h[index_l] = h[index_l]*(1.0-1.0/(tau_h	+0.5))+heq_t/(tau_h	+0.5)+dt*temp_hm;

				}else{
					/*******************h equilibrium*****************/
					double heq_t = cc * gamma + wt * temp_theta * (ex * nvx + ey * nvy + ez * nvz);
					/*******************h evolution*****************/					
					h[index_l] = h[index_l] * (1.0 - 1.0 / (tau_h + 0.5)) + heq_t / (tau_h + 0.5);
				}
			}
			else if (Force_method == 1)
			{
				double geq_t=wt*(p[index]+rr*cs2*uugly);
				double temp_gc_1 = wt*temp_c_c*dr*cs2;
				double temp_gc_2 = wt*(gr_ce_c*dr*cs2-cc*gr_me_c);
				geq_t = geq_t+0.5*dt*temp_gc_1-0.5*dt*temp_gc_2;

				double temp_gm_1 = wt*temp_c_m*dr*cs2;
				double temp_gm_2 = wt*(gr_ce_m*dr*cs2-cc*gr_me_m);

				g[index_l] = g[index_l]*(1.0-1.0/(tt	+0.5))+geq_t/(tt	+0.5)-dt*temp_gm_1+dt*temp_gm_2;

				// h
				if(PH_model == 0){
					double heq_t=cc*gamma;

					double temp_hc_1 =gamma*(temp_c_c+mobility*lap_mu-(cc/rr)*temp_c_c*dr);
					double temp_hc_2 = wt*(cc/rr*cs2)*(gr_ce_c*dr*cs2-gr_pe_c-cc*gr_me_c);
					heq_t = heq_t-0.5*dt*(temp_hc_1+temp_hc_2);

					double temp_hm_1 = gamma*(temp_c_m+mobility*lap_mu-(cc/rr)*temp_c_m*dr);
					double temp_hm_2 = wt*(cc/rr*cs2)*(gr_ce_m*dr*cs2-gr_pe_m-cc*gr_me_m);
					h[index_l] = h[index_l]*(1.0-1.0/(tau_h	+0.5))+heq_t/(tau_h	+0.5)+dt*(temp_hm_1+temp_hm_2);
				}else{
					double heq_t = cc * gamma + wt * temp_theta * (ex * nvx + ey * nvy + ez * nvz);
					h[index_l] = h[index_l] * (1.0 - 1.0 / (tau_h + 0.5)) + heq_t / (tau_h + 0.5);
				}
				
			}
			else if (Force_method == 2)
			{
				//for h eq 
				double shi_e_u_F_c_h = temp_r_c*cs2-temp_p_c-cc*temp_m_c;
				double shi_e_F_c_h = gr_re_c*cs2-gr_pe_c-cc*gr_me_c;
				double shi_c_h = (cc/rr)*wt*(shi_e_u_F_c_h/cs2+(edotu*shi_e_F_c_h)/cs2*cs2);
				//for h evolution
				double shi_e_u_F_m_h = temp_r_m*cs2-temp_p_m-cc*temp_m_m;
				double shi_e_F_m_h = gr_re_m*cs2-gr_pe_m-cc*gr_me_m;
				double shi_m_h = (cc/rr)*wt*(shi_e_u_F_m_h/cs2+(edotu*shi_e_F_m_h)/cs2*cs2);
				//for g eq
				double shi_e_u_F_c_g = temp_r_c*cs2-cc*temp_m_c;
				double shi_e_F_c_g = gr_re_c*cs2-cc*gr_me_c;
				double shi_c_g = wt*(shi_e_u_F_c_g/cs2+(edotu*shi_e_F_c_g)/cs2*cs2);		
				//for g evolution
				double shi_e_u_F_m_g = temp_r_m*cs2-cc*temp_m_m;
				double shi_e_F_m_g = gr_re_m*cs2-cc*gr_me_m;
				double shi_m_g = wt*(shi_e_u_F_m_g/cs2+(edotu*shi_e_F_m_g)/cs2*cs2);		
				// g
				double geq_t=wt*(p[index]+rr*cs2*uugly);
				double temp_gc_1 = temp_c_c*dr*cs2*wt;
				geq_t = geq_t+0.5*dt*temp_gc_1-0.5*dt*shi_c_g*cs2;
				double temp_gm_1 = temp_c_m*dr*cs2*wt;	
				g[index_l] = g[index_l]*(1.0-1.0/(tt	+0.5))+geq_t/(tt	+0.5)-dt*temp_gm_1+dt*shi_m_g*cs2;

				// h
				if(PH_model == 0){
					double heq_t=cc*gamma;
					double temp_hc_1 =gamma*(temp_c_c+mobility*lap_mu-(cc/rr)*temp_c_c*dr);
					heq_t = heq_t-0.5*dt*(temp_hc_1+shi_c_h);
					double temp_hm_1 = gamma*(temp_c_m+mobility*lap_mu-(cc/rr)*temp_c_m*dr);
					h[index_l] = h[index_l]*(1.0-1.0/(tau_h	+0.5))+heq_t/(tau_h	+0.5)+dt*(temp_hm_1+shi_m_h);
				}else{
					double heq_t = cc * gamma + wt * temp_theta * (ex * nvx + ey * nvy + ez * nvz);
					h[index_l] = h[index_l] * (1.0 - 1.0 / (tau_h + 0.5)) + heq_t / (tau_h + 0.5);
				}			
			}
			
		}
		}
	}
}


#endif

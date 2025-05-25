#ifndef PREPARATION_H
#define PREPARATION_H 

//constant parameter//
#define  dx         1.0
#define  dy         1.0
#define  dz         1.0
#define  dt         1.0
#define  q          19

__constant__ double ex_d[q], ey_d[q], ez_d[q], wt_d[q];
__constant__ int    et_d[q];


void parameter (double *beta,double *zeta,double *mobi,double *kappa,double *ex,double *ey,double *ez,double *wt,int *et)
{
	if (bulk_free_energy == 1){
		*zeta = thick*dx;
		//double epsilon = (*zeta)/2.0/sqrt(2.0)/atanh(1.0-2.0*0.05);
		//*kappa = epsilon * surface_tension * 3.0 * sqrt(2.0);
		//*beta = (*kappa) / epsilon /epsilon;
		//*mobi = 0.0;
		*kappa = 1.5*(*zeta)*surface_tension;
		*beta  = 12*surface_tension/(*zeta);
		if(PH_model == 0){
			//*mobi = 0.02/(*beta);
			*mobi = mobility;
		}else{
			*mobi  = tau_h * dt / 3;  // mobility = tau_h * cs2 * dt
		}
	} else {
		double epsilon = (*zeta)/sqrt(2.0)/asin(1.0-2.0*0.05);
		*kappa = epsilon * surface_tension * 4.0 * sqrt(2.0)/M_PI;
		*beta  = (*kappa) / epsilon / epsilon;
		*mobi = mobility ;
	}
	//ex ey ez
	if(q == 19){
		ex[ 0]= 0.0;	ey[ 0]= 0.0;	ez[ 0]= 0.0;
		ex[ 1]= 1.0;	ey[ 1]= 0.0;	ez[ 1]= 0.0;
		ex[ 2]=-1.0;	ey[ 2]= 0.0;	ez[ 2]= 0.0;
		ex[ 3]= 0.0;	ey[ 3]= 1.0;	ez[ 3]= 0.0;
		ex[ 4]= 0.0;	ey[ 4]=-1.0;	ez[ 4]= 0.0;
		ex[ 5]= 0.0;	ey[ 5]= 0.0;	ez[ 5]= 1.0;
		ex[ 6]= 0.0;	ey[ 6]= 0.0;	ez[ 6]=-1.0;
		ex[ 7]= 1.0;	ey[ 7]= 1.0;	ez[ 7]= 0.0;
		ex[ 8]=-1.0;	ey[ 8]=-1.0;	ez[ 8]= 0.0;
		ex[ 9]= 1.0;	ey[ 9]=-1.0;	ez[ 9]= 0.0;
		ex[10]=-1.0;	ey[10]= 1.0;	ez[10]= 0.0;
		ex[11]= 1.0;	ey[11]= 0.0;	ez[11]= 1.0;
		ex[12]=-1.0;	ey[12]= 0.0;	ez[12]=-1.0;
		ex[13]=-1.0;	ey[13]= 0.0;	ez[13]= 1.0;
		ex[14]= 1.0;	ey[14]= 0.0;	ez[14]=-1.0;
		ex[15]= 0.0;	ey[15]= 1.0;	ez[15]= 1.0;
		ex[16]= 0.0;	ey[16]=-1.0;	ez[16]=-1.0;
		ex[17]= 0.0;	ey[17]= 1.0;	ez[17]=-1.0;
		ex[18]= 0.0;	ey[18]=-1.0;	ez[18]= 1.0;
		//////////////////////////////////////////////
		wt[ 0]=1.0/ 3.0;
		wt[ 1]=1.0/18.0;
		wt[ 2]=1.0/18.0;
		wt[ 3]=1.0/18.0;
		wt[ 4]=1.0/18.0;
		wt[ 5]=1.0/18.0;
		wt[ 6]=1.0/18.0;
		wt[ 7]=1.0/36.0;
		wt[ 8]=1.0/36.0;
		wt[ 9]=1.0/36.0;
		wt[10]=1.0/36.0;
		wt[11]=1.0/36.0;
		wt[12]=1.0/36.0;
		wt[13]=1.0/36.0;
		wt[14]=1.0/36.0;
		wt[15]=1.0/36.0;
		wt[16]=1.0/36.0;
		wt[17]=1.0/36.0;
		wt[18]=1.0/36.0;
	}else if(q==27){
		ex[ 0]= 0.0;	ey[ 0]= 0.0;	ez[ 0]= 0.0;
		ex[ 1]= 1.0;	ey[ 1]= 0.0;	ez[ 1]= 0.0;
		ex[ 2]=-1.0;	ey[ 2]= 0.0;	ez[ 2]= 0.0;
		ex[ 3]= 0.0;	ey[ 3]= 1.0;	ez[ 3]= 0.0;
		ex[ 4]= 0.0;	ey[ 4]=-1.0;	ez[ 4]= 0.0;
		ex[ 5]= 0.0;	ey[ 5]= 0.0;	ez[ 5]= 1.0;
		ex[ 6]= 0.0;	ey[ 6]= 0.0;	ez[ 6]=-1.0;
		ex[ 7]= 1.0;	ey[ 7]= 1.0;	ez[ 7]= 0.0;
		ex[ 8]=-1.0;	ey[ 8]=-1.0;	ez[ 8]= 0.0;
		ex[ 9]= 1.0;	ey[ 9]=-1.0;	ez[ 9]= 0.0;
		ex[10]=-1.0;	ey[10]= 1.0;	ez[10]= 0.0;
		ex[11]= 1.0;	ey[11]= 0.0;	ez[11]= 1.0;
		ex[12]=-1.0;	ey[12]= 0.0;	ez[12]=-1.0;
		ex[13]=-1.0;	ey[13]= 0.0;	ez[13]= 1.0;
		ex[14]= 1.0;	ey[14]= 0.0;	ez[14]=-1.0;
		ex[15]= 0.0;	ey[15]= 1.0;	ez[15]= 1.0;
		ex[16]= 0.0;	ey[16]=-1.0;	ez[16]=-1.0;
		ex[17]= 0.0;	ey[17]= 1.0;	ez[17]=-1.0;
		ex[18]= 0.0;	ey[18]=-1.0;	ez[18]= 1.0;
		ex[19]= 1.0;	ey[19]= 1.0;	ez[19]= 1.0;
		ex[20]= 1.0;	ey[20]= 1.0;	ez[20]=-1.0;
		ex[21]= 1.0;	ey[21]=-1.0;	ez[21]= 1.0;
		ex[22]= 1.0;	ey[22]=-1.0;	ez[22]=-1.0;
		ex[23]=-1.0;	ey[23]= 1.0;	ez[23]= 1.0;
		ex[24]=-1.0;	ey[24]= 1.0;	ez[24]=-1.0;
		ex[25]=-1.0;	ey[25]=-1.0;	ez[25]= 1.0;
		ex[26]=-1.0;	ey[26]=-1.0;	ez[26]=-1.0;
		//wt
		wt[ 0]=8.0/ 27.0;
		wt[ 1]=2.0/27.0;
		wt[ 2]=2.0/27.0;
		wt[ 3]=2.0/27.0;
		wt[ 4]=2.0/27.0;
		wt[ 5]=2.0/27.0;
		wt[ 6]=2.0/27.0;
		wt[ 7]=1.0/54.0;
		wt[ 8]=1.0/54.0;
		wt[ 9]=1.0/54.0;
		wt[10]=1.0/54.0;
		wt[11]=1.0/54.0;
		wt[12]=1.0/54.0;
		wt[13]=1.0/54.0;
		wt[14]=1.0/54.0;
		wt[15]=1.0/54.0;
		wt[16]=1.0/54.0;
		wt[17]=1.0/54.0;
		wt[18]=1.0/54.0;
		wt[19]=1.0/216.0;
		wt[20]=1.0/216.0;
		wt[21]=1.0/216.0;
		wt[22]=1.0/216.0;
		wt[23]=1.0/216.0;
		wt[24]=1.0/216.0;
		wt[25]=1.0/216.0;
		wt[26]=1.0/216.0;
	}
	
	int l;
	for(l = 0; l < q; l++){
		et[l]=(int)ex[l]+(nx+4)*((int)ey[l]+(ny+4)*ez[l]);
	}
}


#endif
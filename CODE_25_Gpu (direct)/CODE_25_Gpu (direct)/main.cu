#include <time.h>
#include <math.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

//define matrix(會切割的)
double *c_d_h, *c_f_h, *c_d, *c; // dicom & final on host/ orifinal & transfered on device
double *m_d_h, *m_f_h, *m_d, *m;
double *n_dir_x, *n_dir_y, *n_dir_z;			 // 用來計算 grad c/|grad c| 20230102
double *p_d_h, *p_f_h, *p_d, *p;
double *u_d_h, *u_f_h, *u_d, *u;
double *v_d_h, *v_f_h, *v_d, *v;
double *w_d_h, *w_f_h, *w_d, *w;
double *a_d_h, *a_f_h, *a_d, *a; //pressure

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////surface tension force
double *sf_d_h, *sf_f_h, *sf_d, *sf;
double *DE_d_h, *DE_f_h, *DE_d, *DE; //20220808
double KE, VDR, DE_sum, DEgas_sum, DEliq_sum, DEdiff_sum; //20220907
double KEgas_sum, KEliq_sum, KEdiff_sum; //20220913
double *modi_c;
double p_check; //20220919
double pgra;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double *xz_d_h, *xz_f_h, *xz_d;	

//define matrix(不會切割的)
int    *et_h;
double *ex_h, *ey_h, *ez_h, *wt_h;
double *h_h, *heq_h, *h_t_h;
double *g_h, *geq_h, *g_t_h;
double *h, *heq, *h_t;
double *g, *geq, *g_t;
//gradient matrix
double *gra_c_h, *gra_c;
double *gra_m_h, *gra_m;
double *gra_p_h, *gra_p;
double *lap_m_h, *lap_m;
//define matrix(邊界交換的小矩陣)
double *t_c_h, *t_c;
double *t_m_h, *t_m;
double *t_p_h, *t_p;
double *t_u_h, *t_u;
double *t_v_h, *t_v;
double *t_w_h, *t_w;
double *t_g_h,*t_g;
double *t_h_h,*t_h;

double *lx,*lz;
double *cx_coa,*cz_coa,*cx_2_coa,*cz_2_coa,*D_value;
double *cx_sep,*cz_sep,*cx_2_sep,*cz_2_sep;
double *boundary_x,*boundary_z;
double beta, zeta, mobi, kappa;

int nproc,myid,iroot;
int lastp, b_nbr, t_nbr, itag;
MPI_Status istat[8];
MPI_Request request_macro_u[10],request_macro_v[10],request_macro_w[10]; // isend irecv 傳值狀態
MPI_Request request_macro_chemical[10],request_macro_c[10],request_macro_p1[10];        
MPI_Request request_distri_g[10] ; // isend irecv 傳值狀態
MPI_Request request_distri_h[10] ;

int tag[24]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23};//isend、irecv用來配對的引數
int step=0;


cudaStream_t  stream0,stream1,stream2;
cudaEvent_t gpu_start, gpu_start_temp, gpu_stop, gpu_stop_temp;


FILE *data_2d;
FILE *data_3d;
FILE*information;
FILE *final_2d;
FILE *final_3d;
FILE *oneDC;
FILE *threeDC;
FILE *twoDC;
FILE *trajectories_data; 
FILE *DXvsDt_data; 
FILE *DZvsDt_data; 
FILE *theta_data; 
FILE *D_data; 
FILE *diameter_3d;

#include "error_check.h"
#include "cuda_memory_and_transfer.h"
#include "parameter.h"
#include "mpi_communication.h"
#include "preparation.h"
#include "initial.h"
#include "function.h"
#include "boundary.h"
#include "gradient.h"
#include "chemical.h"
#include "collision.h"
#include "macro.h"
#include "evolution.h"
#include "post.h"


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                      main                                                      //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
	int n_f	= nx*ny*nz/cpu;
	MPI_Init( &argc, &argv );
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	iroot = 0;
	lastp = nproc-1;
	MPI_Barrier(MPI_COMM_WORLD);
	int num;
	cudaGetDeviceCount(&num);
	cudaSetDevice(myid%num);
	if(myid==0){
		printf("===============================================================\n");
		printf("Checking devices...\n");
		printf("number of precess : %d\n",nproc);
		printf("number of GPU : %d\n",num);
	}
 	mpi_upper_under_setting(myid,nproc,lastp,&t_nbr,&b_nbr);
	Memory_Alloc();
	parameter (&beta,&zeta,&mobi,&kappa,ex_h,ey_h,ez_h,wt_h,et_h);
	Mempy_Symbol_const_memory();
	if(myid == 0){
		initial_macro(c_f_h, m_f_h, p_f_h, u_f_h, v_f_h, w_f_h);
		model_setting_print();
		printf("Initializing...");
		printf("done\n");
		printf("===============================================================\n");
		printf("Iterating...\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
	//scatter
	Scatter_Macro_F_To_D(n_f);
	Mempy_H_To_D_pre_do();
	adjust_array_side_to_mid();

	if(boundary_set==0){
		periodic_pre_do();
	}else if(boundary_set==1){
		preiodic_xy_wall_z_pre_do();
	}else{
		periodic_x_symmetry_y_wall_z_pre_do();
	}
	
	pre_calculation();
	cudaDeviceSynchronize();
	cudaEventRecord(gpu_start_temp,0);
	cudaEventRecord(gpu_start,0);



/************** start evolution *****************/

	for(step=1; step <= stepall; step++){
		
		if(boundary_set == 0){
			
			periodic_xyz_evolution(g,h,g_t,h_t);
			cudaDeviceSynchronize();
			step=step+1;
			periodic_xyz_evolution(g_t,h_t,g,h);
			cudaDeviceSynchronize();

		}else if(boundary_set == 1){

			periodic_xy_wall_z_evolution(g,h,g_t,h_t);
			cudaDeviceSynchronize();
			step=step+1;
			periodic_xy_wall_z_evolution(g_t,h_t,g,h);
			cudaDeviceSynchronize();

		}else{

			periodic_x_symmetry_y_wall_z_evolution(g,h,g_t,h_t);	
			cudaDeviceSynchronize();
			step=step+1;
			periodic_x_symmetry_y_wall_z_evolution(g_t,h_t,g,h);		
			cudaDeviceSynchronize();
		}

		xz_calculation();

		if(step%iprint ==0 ){
		 	value_print_out();
			FILE_out(n_f);
			cuda_time_prediction();
		}
	
	}
/************* end evolution *****************/
	MPI_Barrier(MPI_COMM_WORLD);
	cuda_time_final();
	Final_print_out(n_f);
	CHECK_CUDA(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
	free_memory();
	MPI_Finalize();
	return 0;
	
}

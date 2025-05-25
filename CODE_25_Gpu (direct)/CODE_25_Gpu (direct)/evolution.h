#ifndef EVOLUTION_H
#define EVOLUTION_H

#include <mpi.h>
#include"boundary.h"
#include"parameter.h"
#include"collision.h"
#include"macro.h"
#include"gradient.h"
#include"mpi_communication.h"
#include"preparation.h"
#include "cuda_memory_and_transfer.h"

	int num_trans_m_2=(nx+4)*(ny+4)*2;
	int num_trans_m_1=(nx+4)*(ny+4)*1;
	int num_trans_d=(nx+4)*(ny+4)*5;






/********************pre do**********************/
inline void periodic_pre_do(){

	/********************* phase fied variable **********************/
    boundary_C_bc_xy_z_transfer();
    boundary_C_in(stream1);
	Transfer_c_D_To_H(stream0);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_c();
	CHECK_MPI( MPI_Waitall(4,request_macro_c,MPI_STATUS_IGNORE));
	CHECK_CUDA(cudaDeviceSynchronize());
    boundary_C_z_transfer_back(stream0);
	CHECK_CUDA(cudaDeviceSynchronize());
	/********************* chemical potential **********************/
	gradient_cen_n <<< bpgBuff , tpbBuff >>>(n_dir_x, n_dir_y, n_dir_z, c); //20230102
	chemical     <<< bpgBuff , tpbBuff  >>>( c, m, n_dir_x, n_dir_y, n_dir_z, kappa, beta); //20230102
	boundary_chemi_bc_xy_z_transfer();
	boundary_chemi_in(stream1);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_chemical();
    CHECK_MPI( MPI_Waitall(4,request_macro_chemical,MPI_STATUS_IGNORE) );
	CHECK_CUDA(cudaDeviceSynchronize());	
	boundary_chemi_z_transfer_back(stream2);
	CHECK_CUDA(cudaDeviceSynchronize());
	/********************* velocity and hydro-pressure **********************/
	boundary_others_bc_xy_z_transfer();
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_others();
    CHECK_MPI( MPI_Waitall(4,request_macro_u,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_v,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_w,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_p1,MPI_STATUS_IGNORE) ); 
	CHECK_CUDA(cudaDeviceSynchronize());
	boundary_others_z_transfer_back(stream2);
	boundary_others_in(stream1);
	CHECK_CUDA(cudaStreamSynchronize(stream1));
	MPI_Barrier(MPI_COMM_WORLD);
}

inline void preiodic_xy_wall_z_pre_do(){
	/********************* phase fied variable **********************/
    boundary_C_bc_xy_z_transfer();
    boundary_C_in(stream1);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_c();
	CHECK_MPI( MPI_Waitall(4,request_macro_c,MPI_STATUS_IGNORE));
	CHECK_CUDA(cudaDeviceSynchronize());
	boundary_C_wall_z_transfer_back(stream0);
	CHECK_CUDA(cudaDeviceSynchronize());
	/********************* chemical potential **********************/
	gradient_cen_n <<< bpgBuff , tpbBuff >>>(n_dir_x, n_dir_y, n_dir_z, c);
	chemical     <<< bpgBuff , tpbBuff  >>>( c, m, n_dir_x, n_dir_y, n_dir_z, kappa, beta);
	boundary_chemi_bc_xy_z_transfer();
	boundary_chemi_in(stream1);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_chemical();
    CHECK_MPI( MPI_Waitall(4,request_macro_chemical,MPI_STATUS_IGNORE) );
	CHECK_CUDA(cudaDeviceSynchronize());		
	boundary_chemi_wall_z_transfer_back(stream2);
	/********************* velocity and hydro-pressure **********************/
	boundary_others_bc_xy_z_transfer();
	boundary_others_in(stream1);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_others();
    CHECK_MPI( MPI_Waitall(4,request_macro_u,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_v,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_w,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_p1,MPI_STATUS_IGNORE)); 
	CHECK_CUDA(cudaDeviceSynchronize());
	boundary_others_wall_z_transfer_back(stream2);
}

inline void periodic_x_symmetry_y_wall_z_pre_do(){

	boundary_C_bc_x_sym_y_z_transfer();
	boundary_C_sym_y_in(stream1);
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_c();
	CHECK_MPI( MPI_Waitall(4,request_macro_c,MPI_STATUS_IGNORE));
	CHECK_CUDA(cudaDeviceSynchronize());	
	wall_zm2_undo	<<< bpgxy , tpbxy  >>>( c,t_c,myid,lastp );
	CHECK_CUDA(cudaDeviceSynchronize());	
	/********************* chemical potential **********************/
	gradient_cen_n <<< bpgBuff , tpbBuff >>>(n_dir_x, n_dir_y, n_dir_z, c); 
	chemical     <<< bpgBuff , tpbBuff  >>>( c, m, n_dir_x, n_dir_y, n_dir_z, kappa, beta); 
	boundary_chemi_bc_x_sym_y_z_transfer();
	boundary_chemi_sym_y_in(stream1);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_chemical();
    CHECK_MPI( MPI_Waitall(4,request_macro_chemical,MPI_STATUS_IGNORE) );
	CHECK_CUDA(cudaDeviceSynchronize());
	wall_zm2_undo	<<< bpgxy , tpbxy  >>>( m,t_m,myid,lastp );
	/********************* velocity and hydro-pressure **********************/
	boundary_others_bc_xy_z_transfer();
	boundary_others_in(stream1);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_others();
    CHECK_MPI( MPI_Waitall(4,request_macro_u,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_v,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_w,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_p1,MPI_STATUS_IGNORE)); 
	CHECK_CUDA(cudaDeviceSynchronize());
	wall_zm2_undo	<<< bpgxy , tpbxy  >>>( p,t_p,myid,lastp );
	wall_zm1_undo_u	<<< bpgxy , tpbxy  >>>( u,t_u,myid,lastp );
	wall_zm1_undo_vw	<<< bpgxy , tpbxy  >>>( v,t_v,myid,lastp );
	wall_zm1_undo_vw	<<< bpgxy , tpbxy   >>>( w,t_w,myid,lastp );

}


/********************pre calculation**********************/
inline void pre_calculation(){

	eq         <<< bpgBuff , tpbBuff  >>>( g,h,geq,heq,c,m,p,mobi,u,v,w);
	eq0        <<< bpgBuff , tpbBuff  >>>( g,h,geq,heq );
	collision  <<< bpgBuff , tpbBuff  >>>( g,h,geq,heq,c,m,p,u,v,w,mobi);
	gradient_cen   <<<bpgBuff, tpbBuff,0>>>(gra_c,c);
	gradient_cen   <<<bpgBuff, tpbBuff,0>>>(gra_m,m);
	gradient_cen   <<<bpgBuff, tpbBuff,0>>>(gra_p,p);
	gradient_mix   <<<bpgBuff, tpbBuff,0>>>(gra_c,c);
	gradient_mix   <<<bpgBuff, tpbBuff,0>>>(gra_m,m);
	gradient_mix   <<<bpgBuff, tpbBuff,0>>>(gra_p,p);
	laplacian_mu   <<<bpgBuff, tpbBuff,0>>>(lap_m,m);

}

/********************evolution**********************/
inline void periodic_xyz_evolution(double*g,double*h,double*g_t,double*h_t){

	/********************* distri func **********************/
    eq_collision_bc<<< bpgxy , tpbxy   ,0,stream0>>>( g,h,c,m,p,gra_c,gra_m,gra_p,lap_m,u,v,w,mobi);
    boundary_distri_bc_xy_z_transfer(g,h,stream0);
	eq_collision_in<<< bpgBuff , tpbBuff   ,0,stream1>>>( g,h,c,m,p,gra_c,gra_m,gra_p,lap_m,u,v,w,mobi);
    boundary_distri_in(g,h,stream1);	
	CHECK_CUDA(cudaStreamSynchronize(stream0));
	mpi_transfer_distri();
    CHECK_MPI( MPI_Waitall(4,request_distri_g,MPI_STATUS_IGNORE) );   
    CHECK_MPI( MPI_Waitall(4,request_distri_h,MPI_STATUS_IGNORE) ); 
	CHECK_CUDA(cudaDeviceSynchronize());
    boundary_distri_z_transfer_back(g,h,stream0);
	CHECK_CUDA(cudaDeviceSynchronize());

	/********************* phase fied variable **********************/
	macro_h_bc		<<< bpgxy , tpbxy >>>( h,h_t,c,lap_m );
    boundary_C_bc_xy_z_transfer();
	macro_h_in		<<< bpgBuff , tpbBuff  ,0,stream1>>>( h,h_t,c,lap_m );
    boundary_C_in(stream1);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_c();
	CHECK_MPI( MPI_Waitall(4,request_macro_c,MPI_STATUS_IGNORE));
	CHECK_CUDA(cudaDeviceSynchronize());	
    boundary_C_z_transfer_back(stream0);
	CHECK_CUDA(cudaDeviceSynchronize());
	/********************* chemical potential **********************/
	gradient_cen_n  <<< bpgBuff , tpbBuff >>>(n_dir_x, n_dir_y, n_dir_z, c); //20230102
	chemical_bc		<<< bpgxy , tpbxy  >>>( c, m, n_dir_x, n_dir_y, n_dir_z, kappa, beta ); //20230102
	boundary_chemi_bc_xy_z_transfer();
	chemical_in		<<< bpgBuff , tpbBuff  ,0,stream1>>>( c, m, n_dir_x, n_dir_y, n_dir_z, kappa, beta ); //20230102
	boundary_chemi_in(stream1);
	gradient_cen	<<< bpgBuff , tpbBuff  ,0,stream1>>>(gra_c,c);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_chemical();
    CHECK_MPI( MPI_Waitall(4,request_macro_chemical,MPI_STATUS_IGNORE) );
	CHECK_CUDA(cudaDeviceSynchronize());	
	boundary_chemi_z_transfer_back(stream2);
	CHECK_CUDA(cudaDeviceSynchronize());	
	/********************* velocity and hydro-pressure **********************/

	gradient_cen	<<< bpgBuff , tpbBuff  >>>(gra_m,m);

	macro_g_bc		<<< bpgxy , tpbxy  >>>( g,g_t,c,m,p,gra_c,gra_m,u,v,w);

	boundary_others_bc_xy_z_transfer();

	macro_g_in		<<< bpgBuff , tpbBuff  ,0,stream1>>>( g,g_t,c,m,p,gra_c,gra_m,u,v,w);
	laplacian_mu	<<< bpgBuff , tpbBuff  ,0,stream1>>>(lap_m,m);
	gradient_mix	<<< bpgBuff , tpbBuff  ,0,stream1>>>(gra_m,m);
	gradient_mix	<<< bpgBuff , tpbBuff  ,0,stream1>>>(gra_c,c);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_others();
    CHECK_MPI( MPI_Waitall(4,request_macro_u,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_v,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_w,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_p1,MPI_STATUS_IGNORE) ); 
	CHECK_CUDA(cudaDeviceSynchronize());
	boundary_others_z_transfer_back(stream2);
	boundary_others_in(stream1);
	CHECK_CUDA(cudaStreamSynchronize(stream1));
	gradient_cen   <<< bpgBuff , tpbBuff  ,0>>>(gra_p,p);
	gradient_mix   <<< bpgBuff , tpbBuff  ,0>>>(gra_p,p);
	//DE_cal_d	   <<< bpgBuff , tpbBuff  ,0>>>(c, u, v, w, p, g_t, m, gra_c, gra_m, DE);//202212

}

inline void periodic_xy_wall_z_evolution(double*g,double*h,double*g_t,double*h_t){

	/********************* distri func **********************/
    eq_collision_bc<<< bpgxy , tpbxy   ,0,stream0>>>( g,h,c,m,p,gra_c,gra_m,gra_p,lap_m,u,v,w,mobi);
    boundary_distri_bc_xy_z_transfer(g,h,stream0);
	eq_collision_in<<< bpgBuff , tpbBuff   ,0,stream1>>>( g,h,c,m,p,gra_c,gra_m,gra_p,lap_m,u,v,w,mobi);
    boundary_distri_in(g,h,stream1);	
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_distri();
    CHECK_MPI( MPI_Waitall(4,request_distri_g,MPI_STATUS_IGNORE) );   
    CHECK_MPI( MPI_Waitall(4,request_distri_h,MPI_STATUS_IGNORE) ); 
	CHECK_CUDA(cudaDeviceSynchronize());
	boundary_distri_wall_z_transfer_back(g,h,stream0);
	CHECK_CUDA(cudaDeviceSynchronize());

	/********************* phase fied variable **********************/
	macro_h_bc		<<< bpgxy , tpbxy >>>( h,h_t,c,lap_m);
    boundary_C_bc_xy_z_transfer();
	macro_h_in		<<< bpgBuff , tpbBuff  ,0,stream1>>>( h,h_t,c,lap_m );
    boundary_C_in(stream1);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_c();
	CHECK_MPI( MPI_Waitall(4,request_macro_c,MPI_STATUS_IGNORE));
	CHECK_CUDA(cudaDeviceSynchronize());
    boundary_C_wall_z_transfer_back(stream0);
	CHECK_CUDA(cudaDeviceSynchronize());
	/********************* chemical potential **********************/
	gradient_cen_n  <<< bpgBuff , tpbBuff >>>(n_dir_x, n_dir_y, n_dir_z, c); 
	chemical_bc		<<< bpgxy , tpbxy  >>>( c, m, n_dir_x, n_dir_y, n_dir_z, kappa, beta ); 
	boundary_chemi_bc_xy_z_transfer();
	chemical_in		<<< bpgBuff , tpbBuff  ,0,stream1>>>( c, m, n_dir_x, n_dir_y, n_dir_z, kappa, beta ); 
	boundary_chemi_in(stream1);
	gradient_cen	<<< bpgBuff , tpbBuff  ,0,stream1>>>(gra_c,c);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_chemical();
    CHECK_MPI( MPI_Waitall(4,request_macro_chemical,MPI_STATUS_IGNORE) );
	CHECK_CUDA(cudaDeviceSynchronize());	
	boundary_chemi_wall_z_transfer_back(stream2);
	CHECK_CUDA(cudaDeviceSynchronize());	
	/********************* velocity and hydro-pressure **********************/
	gradient_cen<<< bpgBuff , tpbBuff  >>>(gra_m,m);
	macro_g_bc<<< bpgxy , tpbxy  >>>( g,g_t,c,m,p,gra_c,gra_m,u,v,w);
	boundary_others_bc_xy_z_transfer();
	macro_g_in		<<< bpgBuff , tpbBuff  ,0,stream1>>>( g,g_t,c,m,p,gra_c,gra_m,u,v,w);
	laplacian_mu	<<< bpgBuff , tpbBuff  ,0,stream1>>>(lap_m,m);
	gradient_mix	<<< bpgBuff , tpbBuff  ,0,stream1>>>(gra_m,m);
	gradient_mix	<<< bpgBuff , tpbBuff  ,0,stream1>>>(gra_c,c);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_others();
    CHECK_MPI( MPI_Waitall(4,request_macro_u,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_v,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_w,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_p1,MPI_STATUS_IGNORE) ); 
	CHECK_CUDA(cudaDeviceSynchronize());
	boundary_others_wall_z_transfer_back(stream2);
	boundary_others_in(stream1);
	CHECK_CUDA(cudaStreamSynchronize(stream1));
	gradient_cen   <<< bpgBuff , tpbBuff  ,0>>>(gra_p,p);
	gradient_mix   <<< bpgBuff , tpbBuff  ,0>>>(gra_p,p);
	//DE_cal_d	   <<< bpgBuff , tpbBuff  ,0>>>(c, u, v, w, p, g_t, m, gra_c, gra_m, DE);//202212

}

inline void periodic_x_symmetry_y_wall_z_evolution(double*g,double*h,double*g_t,double*h_t){

	/********************* distri func **********************/
    eq_collision_bc<<< bpgxy , tpbxy   ,0,stream0>>>( g,h,c,m,p,gra_c,gra_m,gra_p,lap_m,u,v,w,mobi);
    boundary_distri_bc_x_sym_y_z_transfer(g,h,stream0);
	eq_collision_in<<< bpgBuff , tpbBuff   ,0,stream1>>>( g,h,c,m,p,gra_c,gra_m,gra_p,lap_m,u,v,w,mobi);
    boundary_distri_sym_y_in(g,h,stream1);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_distri();
    CHECK_MPI( MPI_Waitall(4,request_distri_g,MPI_STATUS_IGNORE) );   
    CHECK_MPI( MPI_Waitall(4,request_distri_h,MPI_STATUS_IGNORE) ); 
	CHECK_CUDA(cudaDeviceSynchronize());
	boundary_distri_wall_z_transfer_back(g,h,stream0);
	CHECK_CUDA(cudaDeviceSynchronize());

	/********************* phase fied variable **********************/
	macro_h_bc		<<< bpgxy , tpbxy >>>( h,h_t,c,lap_m);
    boundary_C_bc_x_sym_y_z_transfer();
	macro_h_in		<<< bpgBuff , tpbBuff  ,0,stream1>>>( h,h_t,c,lap_m );
    boundary_C_sym_y_in(stream1);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_c();
	CHECK_MPI( MPI_Waitall(4,request_macro_c,MPI_STATUS_IGNORE));
	CHECK_CUDA(cudaDeviceSynchronize());
    boundary_C_wall_z_transfer_back(stream0);
	CHECK_CUDA(cudaDeviceSynchronize());
	/********************* chemical potential **********************/
	gradient_cen_n  <<< bpgBuff , tpbBuff >>>(n_dir_x, n_dir_y, n_dir_z, c); 
	chemical_bc		<<< bpgxy , tpbxy  >>>( c, m, n_dir_x, n_dir_y, n_dir_z, kappa, beta ); 
	boundary_chemi_bc_x_sym_y_z_transfer();
	chemical_in		<<< bpgBuff , tpbBuff  ,0,stream1>>>( c, m, n_dir_x, n_dir_y, n_dir_z, kappa, beta ); 
	boundary_chemi_sym_y_in(stream1);
	gradient_cen	<<< bpgBuff , tpbBuff  ,0,stream1>>>(gra_c,c);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_chemical();
    CHECK_MPI( MPI_Waitall(4,request_macro_chemical,MPI_STATUS_IGNORE) );
	CHECK_CUDA(cudaDeviceSynchronize());	
	boundary_chemi_wall_z_transfer_back(stream2);
	CHECK_CUDA(cudaDeviceSynchronize());	
	/********************* velocity and hydro-pressure **********************/
	gradient_cen	<<< bpgBuff , tpbBuff  >>>(gra_m,m);
	macro_g_bc		<<< bpgxy , tpbxy  >>>( g,g_t,c,m,p,gra_c,gra_m,u,v,w);
	boundary_others_bc_x_sym_y_z_transfer();
	macro_g_in		<<< bpgBuff , tpbBuff  ,0,stream1>>>( g,g_t,c,m,p,gra_c,gra_m,u,v,w);
	laplacian_mu	<<< bpgBuff , tpbBuff  ,0,stream1>>>(lap_m,m);
	gradient_mix	<<< bpgBuff , tpbBuff  ,0,stream1>>>(gra_m,m);
	gradient_mix	<<< bpgBuff , tpbBuff  ,0,stream1>>>(gra_c,c);
	CHECK_CUDA(cudaDeviceSynchronize());
	mpi_transfer_macro_others();
    CHECK_MPI( MPI_Waitall(4,request_macro_u,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_v,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_w,MPI_STATUS_IGNORE) );
    CHECK_MPI( MPI_Waitall(4,request_macro_p1,MPI_STATUS_IGNORE) ); 
	CHECK_CUDA(cudaDeviceSynchronize());
	boundary_others_wall_z_transfer_back(stream2);
	boundary_others_sym_y_in(stream1);
	CHECK_CUDA(cudaStreamSynchronize(stream1));
	gradient_cen   <<< bpgBuff , tpbBuff  ,0>>>(gra_p,p);
	gradient_mix   <<< bpgBuff , tpbBuff  ,0>>>(gra_p,p);
	//DE_cal_d	   <<< bpgBuff , tpbBuff  ,0>>>(c, u, v, w, p, g_t, m, gra_c, gra_m, DE);//202212

}


#endif
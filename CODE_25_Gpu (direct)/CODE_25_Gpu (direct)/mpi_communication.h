#ifndef MPI_COM_H
#define MPI_COM_H
#include"mpi.h"
#include"parameter.h"
#include"cuda_memory_and_transfer.h"
#include"error_check.h"

const int tran_difun    =  (nx+4)*(ny+4)*5*size_distri_func_buffer_transfer*2;//只需要交換5個，streaming角度來看只會用到5、個*2是要放交換過來的
const int tran_mac_1    =  (nx+4)*(ny+4)*1*size_u_v_w_buffer_transfer*2 ; //3個巨觀速度、只需要交換兩層、*2是要放交換過來的          
const int tran_mac_2 	=  (nx+4)*(ny+4)*1*size_c_chemical_p1_buffer_transfer*2;//只有一個巨觀數值，要計算gradient要四層、*2要放交換過來的

/*transfer_macro_2*/
const int startb = 0 ;
const int start	 = tran_mac_2*0.25;
const int end	 = tran_mac_2*0.5;
const int endb	 = tran_mac_2*0.75;

/*transfer_macro_1*/
const int startb_1 = 0 ;
const int start_1  = tran_mac_1*0.25;
const int end_1	   = tran_mac_1*0.5;
const int endb_1   = tran_mac_1*0.75;

/*transfer_distribution_function*/
const int startb_d = 0;
const int start_d  = tran_difun *0.25 ;   
const int end_d	   = tran_difun *0.5  ;
const int endb_d   = tran_difun *0.75 ;


void mpi_upper_under_setting(int myid,int nproc,int lastp,int*upper,int*under)
{
	*under = myid - 1; //現在的id-1
	*upper = myid + 1; //現在的id+1
	/*上下邊界傳值*/
	if(boundary_set == 0){
		if(myid == 0){
			*under=lastp;  
		}
		if(myid == lastp){
			*upper = 0;      
		}
	}
	/*上下邊界不傳值*/	
	else{
		if(myid == 0){
			*under=MPI_PROC_NULL;  
		}
		if(myid == lastp){
			*upper =MPI_PROC_NULL;      
		}
	}
}

void Scatter_F_To_D(int n_f,int iroot, int num_arrays, ...) {
    va_list args;
    va_start(args, num_arrays);

    for (int i = 0; i < num_arrays; i++) {
        double* f_h = va_arg(args, double*);
        double* d_h = va_arg(args, double*);

        CHECK_MPI(MPI_Scatter(f_h, n_f, MPI_DOUBLE,
                    d_h, n_f, MPI_DOUBLE,
                    iroot, MPI_COMM_WORLD));
    }

    va_end(args);

}

void Gather_D_To_F(int n_f,int root,int num_arrays,...){
	
	va_list args;
	va_start(args,num_arrays);
    for (int i = 0; i < num_arrays; i++) {
		double*f_h = va_arg(args,double*);
		double*d_h = va_arg(args,double*);
	CHECK_MPI(MPI_Gather( d_h ,n_f, MPI_DOUBLE, f_h ,n_f, MPI_DOUBLE,iroot,MPI_COMM_WORLD));
	}

    va_end(args);	
}


void Scatter_Macro_F_To_D(int n_f){
    Scatter_F_To_D(n_f, iroot,7, c_f_h, c_d_h, m_f_h, m_d_h, p_f_h, p_d_h, u_f_h, u_d_h, v_f_h, v_d_h, w_f_h, w_d_h, a_f_h, a_d_h);
}

void Gather_Macro_D_To_F(int n_f){
    Gather_D_To_F(n_f, iroot,8, c_f_h, c_d_h, m_f_h, m_d_h, p_f_h, p_d_h, u_f_h, u_d_h, v_f_h, v_d_h, w_f_h, w_d_h, a_f_h, a_d_h,DE_f_h,DE_d_h);
}


void mpi_isend_irecv_macro_velo(double*transfer_macro_velo_host,int tag_upper,int tag_under,int recv_from_under,int send_for_under,int send_for_upper,int recv_from_upper
,int upper,int under,MPI_Request request[]){
	//接收來自Rank-1  (under傳來的) 0
	CHECK_MPI( MPI_Irecv(&transfer_macro_velo_host[recv_from_under],(nx+4)*(ny+4)*1*1,MPI_DOUBLE,under,tag_upper,MPI_COMM_WORLD,&request[3]) );
	//傳送給Rank-1 (under要接收) 1/4
	CHECK_MPI( MPI_Isend(&transfer_macro_velo_host[send_for_under],(nx+4)*(ny+4)*1*1,MPI_DOUBLE,under,tag_under,MPI_COMM_WORLD,&request[0])  );
	//傳送給Rank+1 (upper要接收) 2/4
	CHECK_MPI( MPI_Isend(&transfer_macro_velo_host[send_for_upper],(nx+4)*(ny+4)*1*1,MPI_DOUBLE,upper,tag_upper,MPI_COMM_WORLD,&request[1])  );
	//接收來自Rank+1  (upper傳來的) 3/4
	CHECK_MPI( MPI_Irecv(&transfer_macro_velo_host[recv_from_upper],(nx+4)*(ny+4)*1*1,MPI_DOUBLE,upper,tag_under,MPI_COMM_WORLD,&request[2]) );

}	

void mpi_isend_irecv_macro_buffer_4(double*transfer_macro,int tag_upper,int tag_under,int recv_from_under,int send_for_under,int send_for_upper,int recv_from_upper,int upper
,int under,MPI_Request request[]){
	//接收來自Rank-1  (under傳來的) 0
	CHECK_MPI( MPI_Irecv(&transfer_macro[recv_from_under],(nx+4)*(ny+4)*2*1,MPI_DOUBLE,under,tag_upper,MPI_COMM_WORLD,&request[3]) );
	//傳送給Rank-1 (under要接收) 1/4
	CHECK_MPI( MPI_Isend(&transfer_macro[send_for_under],(nx+4)*(ny+4)*2*1,MPI_DOUBLE,under,tag_under,MPI_COMM_WORLD,&request[0]) );
	//傳送給Rank+1 (upper要接收) 2/4
	CHECK_MPI( MPI_Isend(&transfer_macro[send_for_upper],(nx+4)*(ny+4)*2*1,MPI_DOUBLE,upper,tag_upper,MPI_COMM_WORLD,&request[1]) );
	//接收來自Rank+1  (upper傳來的) 3/4
	CHECK_MPI( MPI_Irecv(&transfer_macro[recv_from_upper],(nx+4)*(ny+4)*2*1,MPI_DOUBLE,upper,tag_under,MPI_COMM_WORLD,&request[2]) );

}

void mpi_isend_irecv_distri_fun_g(double*transfer_distri_fun_host,int tag_upper,int tag_under,int recv_from_under,int send_for_under,int send_for_upper,int recv_from_upper,int upper,int under
,MPI_Request request[]){

	//接收來自Rank-1  (under傳來的)	0
	CHECK_MPI( MPI_Irecv(&transfer_distri_fun_host[recv_from_under],(nx+4)*(ny+4)*1*5,MPI_DOUBLE,under,tag_upper,MPI_COMM_WORLD,&request[3]) );
	//傳送給Rank-1 (under要接收) 1/4
	CHECK_MPI( MPI_Isend(&transfer_distri_fun_host[send_for_under],(nx+4)*(ny+4)*1*5,MPI_DOUBLE,under,tag_under,MPI_COMM_WORLD,&request[0]) );
	//傳送給Rank+1 (upper要接收) 2/4
	CHECK_MPI( MPI_Isend(&transfer_distri_fun_host[send_for_upper],(nx+4)*(ny+4)*1*5,MPI_DOUBLE,upper,tag_upper,MPI_COMM_WORLD,&request[1]) );
	//接收來自Rank+1  (upper傳來的) 3/4
	CHECK_MPI( MPI_Irecv(&transfer_distri_fun_host[recv_from_upper],(nx+4)*(ny+4)*1*5,MPI_DOUBLE,upper,tag_under,MPI_COMM_WORLD,&request[2]) );

}

void mpi_isend_irecv_distri_fun_h(double*transfer_distri_fun_host,int tag_upper,int tag_under,int recv_from_under,int send_for_under,int send_for_upper,int recv_from_upper,int upper,int under
,MPI_Request request[]){
	//接收來自Rank-1  (under傳來的)	0
	CHECK_MPI( MPI_Irecv(&transfer_distri_fun_host[recv_from_under],(nx+4)*(ny+4)*1*5,MPI_DOUBLE,under,tag_upper,MPI_COMM_WORLD,&request[3]) );
	//傳送給Rank-1 (under要接收) 1/4
	CHECK_MPI( MPI_Isend(&transfer_distri_fun_host[send_for_under],(nx+4)*(ny+4)*1*5,MPI_DOUBLE,under,tag_under,MPI_COMM_WORLD,&request[0]) );
	//傳送給Rank+1 (upper要接收) 2/4
	CHECK_MPI( MPI_Isend(&transfer_distri_fun_host[send_for_upper],(nx+4)*(ny+4)*1*5,MPI_DOUBLE,upper,tag_upper,MPI_COMM_WORLD,&request[1]) );
	//接收來自Rank+1  (upper傳來的) 3/4
	CHECK_MPI( MPI_Irecv(&transfer_distri_fun_host[recv_from_upper],(nx+4)*(ny+4)*1*5,MPI_DOUBLE,upper,tag_under,MPI_COMM_WORLD,&request[2]) );

	
}




/*********************************************************call mpi function*************************************************************/

/***********macro************/

void mpi_transfer_macro_c(){
	mpi_isend_irecv_macro_buffer_4(t_c,tag[5],tag[6],startb,start,end,endb,t_nbr,b_nbr,request_macro_c);
}
void mpi_transfer_macro_chemical(){
	mpi_isend_irecv_macro_buffer_4(t_m,tag[7],tag[8],startb,start,end,endb,t_nbr,b_nbr,request_macro_chemical);
}
void mpi_transfer_macro_others(){
	mpi_isend_irecv_macro_buffer_4(t_p,tag[9],tag[10],startb,start,end,endb,t_nbr,b_nbr,request_macro_p1);	
	mpi_isend_irecv_macro_velo(t_u,tag[11],tag[12],startb_1,start_1,end_1,endb_1,t_nbr,b_nbr,request_macro_u);
	mpi_isend_irecv_macro_velo(t_v,tag[13],tag[14],startb_1,start_1,end_1,endb_1,t_nbr,b_nbr,request_macro_v);
	mpi_isend_irecv_macro_velo(t_w,tag[15],tag[16],startb_1,start_1,end_1,endb_1,t_nbr,b_nbr,request_macro_w);

}
/***********distri************/

void  mpi_transfer_distri(){
	mpi_isend_irecv_distri_fun_g(t_g,tag[17],tag[18],startb_d,start_d,end_d,endb_d,t_nbr,b_nbr,request_distri_g);
	mpi_isend_irecv_distri_fun_h(t_h,tag[19],tag[20],startb_d,start_d,end_d,endb_d,t_nbr,b_nbr,request_distri_h);	
}





#endif
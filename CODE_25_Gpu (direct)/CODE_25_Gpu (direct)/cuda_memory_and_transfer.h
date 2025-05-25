#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

#include"stdarg.h"
#include"cuda_runtime.h"
#include"cuda.h"
#include"parameter.h"
#include"error_check.h"
#include"preparation.h"

#define size_distri_func_buffer_transfer 2
#define size_u_v_w_buffer_transfer 2
#define size_c_chemical_p1_buffer_transfer 4


dim3 tpbBuff(32, 1,1);
dim3 bpgBuff((nx+4+tpbBuff.x-1)/tpbBuff.x,(ny+4+tpbBuff.y-1)/tpbBuff.y,(nz/cpu+4+tpbBuff.z-1)/tpbBuff.z);
// 2D
dim3 tpbxy(32, 1);
dim3 bpgxy((nx+4+tpbxy.x-1)/tpbxy.x,(ny+4+tpbxy.y-1)/tpbxy.y);
dim3 tpbyz(32, 1);
dim3 bpgyz((ny+4+tpbyz.x-1)/tpbyz.x,(nz/cpu+4+tpbyz.y-1)/tpbyz.y);
dim3 tpbxz(32, 1);
dim3 bpgxz((nx+4+tpbxz.x-1)/tpbxz.x,(nz/cpu+4+tpbxz.y-1)/tpbxz.y);
// 1D
dim3 tpbx(32);
dim3 bpgx((nx+4+tpbx.x-1)/tpbx.x);
dim3 tpby(32);
dim3 bpgy((ny+4+tpby.x-1)/tpby.x);



// final
const size_t size_final_macro         = sizeof(double)*(nx*ny*nz);  
const size_t size_xz_plane_final      = sizeof(double)*nx*nz ;
const size_t size_variable_shear_flow = sizeof(double)*stepall/value_get ;
////////////////////////////////////////////decompose//////////////////////////////////////////////////////////////////////
const size_t size_decompose        =   sizeof(double)*(nx+4)*(ny+4)*(nz/cpu+4);
const size_t size_distri_function  =   sizeof(double)*(nx+4)*(ny+4)*(nz/cpu+4)*q;
const size_t size_gradient         =   sizeof(double)*(nx+4)*(ny+4)*(nz/cpu+4)*6;
const size_t size_xz_plane         =   sizeof(double)*(nx+4)*(nz/cpu+4);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//邊界交換
const size_t size_transfer_distri_function      =  sizeof(double)*(nx+4)*(ny+4)*5*size_distri_func_buffer_transfer*2;//只需要交換5個，streaming角度來看只會用到5、個*2是要放交換過來的
const size_t size_transfer_macro_velo          =  sizeof(double)*(nx+4)*(ny+4)*1*size_u_v_w_buffer_transfer*2 ; //只需要交換兩層、*2是要放交換過來的
const size_t size_transfer_macro_buffer_4       =  sizeof(double)*(nx+4)*(ny+4)*1*size_c_chemical_p1_buffer_transfer*2;  //將c chemical p1 各自產生一個、要算gradient要四層、*2是要放交換過來的 


/***************** Memory Alloc ****************/
void AllocateHostArray_double(const size_t nBytes, const int num_arrays, ...) {
	va_list args;
	va_start( args, num_arrays );

	for( int i = 0; i < num_arrays; i++ ) {
        double **tmp = va_arg(args, double**);
		CHECK_CUDA(cudaMallocHost( (void**)tmp,nBytes));
        if (*tmp != NULL) {
            memset( *tmp, 0.0 ,nBytes );
        }
    }
	va_end( args );
}
void AllocateHostArray_int(const size_t nBytes, const int num_arrays, ...) {
	va_list args;
	va_start( args, num_arrays );

	for( int i = 0; i < num_arrays; i++ ) {
        int **tmp = va_arg(args, int**);
		CHECK_CUDA(cudaMallocHost( (void**)tmp,nBytes));
        if (*tmp != NULL) {
            memset( *tmp,0, nBytes );
        }
    }
	va_end( args );
}
void AllocateDeviceArray_double(const size_t nBytes, const int num_arrays, ...) {
	va_list args;
	va_start( args, num_arrays );

	for( int i = 0; i < num_arrays; i++ ) {
        double **tmp = va_arg(args, double**);
		CHECK_CUDA(cudaMalloc( (void**)tmp,nBytes)); 
        if (*tmp != NULL) {
            CHECK_CUDA(cudaMemset( *tmp, 0.0, nBytes )) ;
        }
    }

	va_end( args );
}
void AllocateDeviceArray_int(const size_t nBytes, const int num_arrays, ...) {
	va_list args;
	va_start( args, num_arrays );

	for( int i = 0; i < num_arrays; i++ ) {
        int **tmp = va_arg(args, int**);
		CHECK_CUDA(cudaMalloc( (void**)tmp,nBytes)); 
        if (*tmp != NULL) {
            CHECK_CUDA(cudaMemset( *tmp, 0, nBytes )) ;
        }
    }

	va_end( args );
}
void FreeHostArray_double(const int num_arrays, ...) {
    va_list args;
    va_start( args, num_arrays );
    for( int z = 0; z < num_arrays; z++ ) {
        double*ptr = va_arg(args, double*);
        if (ptr != NULL) {
            CHECK_CUDA(cudaFreeHost(ptr));
        }
    }
    va_end( args );

}
void FreeHostArray_int(const int num_arrays, ...) {
    va_list args;
    va_start( args, num_arrays );
    for( int z = 0; z < num_arrays; z++ ) {
        int*ptr = va_arg(args, int*);
        if (ptr != NULL) {
            CHECK_CUDA(cudaFreeHost(ptr));
        }
    }
    va_end( args );
}
void FreeDeviceArray_double(const int num_arrays, ...) {
    va_list args;
    va_start( args, num_arrays );

    for( int  z = 0; z < num_arrays; z++ ) {
        double*ptr = va_arg(args, double*);     
        if (ptr != NULL) {
            CHECK_CUDA(cudaFree(ptr));
        }    
    }
    va_end( args );
}
void FreeDeviceArray_int(const int num_arrays, ...) {
    va_list args;
    va_start( args, num_arrays );

    for( int  z = 0; z < num_arrays; z++ ) {
        int*ptr = va_arg(args, int*);     
        if (ptr != NULL) {
            CHECK_CUDA(cudaFree(ptr));
        }    
    }
    va_end( args );
}






void Memory_Alloc(){
    //host
    if(myid == 0){
        /********macro********/      
        AllocateHostArray_double(size_final_macro,9,&c_f_h,&m_f_h,&p_f_h,&a_f_h,&u_f_h,&v_f_h,&w_f_h,&sf_f_h,&DE_f_h);    
        /********cross section********/
        AllocateHostArray_double(size_xz_plane_final,1,&xz_f_h);
        /********shear flow variable********/
        AllocateHostArray_double(size_variable_shear_flow,11,&lx,&lz,&cx_coa,&cz_coa,&cx_2_coa,&cz_2_coa,&cx_sep,&cz_sep,&cx_2_sep,&cz_2_sep,&D_value);
        /**************least square 2D****************/
        AllocateHostArray_double(size_xz_plane_final,2,&boundary_x,&boundary_z);    


    }
    /********macro********/
    AllocateHostArray_double(size_decompose,9,&c_d_h,&m_d_h,&p_d_h,&a_d_h,&u_d_h,&v_d_h,&w_d_h,&sf_d_h,&DE_d_h);   
    /********direction********/ 
    AllocateHostArray_double(sizeof(double)*q,4,&ex_h,&ey_h,&ez_h,&wt_h);
    AllocateHostArray_int(sizeof(int)*q,1,&et_h);
    /********distri function********/
    AllocateHostArray_double(size_distri_function,6,&g_h,&geq_h,&g_t_h,&h_h,&heq_h,&h_t_h);  
    /********gradient********/
    AllocateHostArray_double(size_gradient,3,&gra_c_h,&gra_m_h,&gra_p_h);       
    AllocateHostArray_double(size_decompose,1,&lap_m_h); 
    /********transfer********/
    AllocateHostArray_double(size_transfer_macro_buffer_4,3,&t_c_h,&t_m_h,&t_p_h);  
    AllocateHostArray_double(size_transfer_macro_velo,3,&t_u_h,&t_v_h,&t_w_h);  
    AllocateHostArray_double(size_transfer_distri_function,2,&t_g_h,&t_h_h);  
    /********cross section********/
    AllocateHostArray_double(size_xz_plane,1,&xz_d_h);

    //device
    /********macro********/
    AllocateDeviceArray_double(size_decompose,9,&c_d,&m_d,&p_d,&u_d,&v_d,&w_d,&a_d,&sf_d,&DE_d);
    AllocateDeviceArray_double(size_decompose,9,&c,&m,&p,&u,&v,&w,&a,&sf,&DE);
    /********distri function********/  
    AllocateDeviceArray_double(size_distri_function,6,&h,&g,&heq,&geq,&h_t,&g_t);
    /********transfer********/
    AllocateDeviceArray_double(size_transfer_macro_buffer_4,3,&t_c,&t_m,&t_p);
    AllocateDeviceArray_double(size_transfer_macro_velo,3,&t_u,&t_v,&t_w);
    AllocateDeviceArray_double(size_transfer_distri_function,2,&t_g,&t_h);
    /********gradient********/
    AllocateDeviceArray_double(size_gradient,3,&gra_c,&gra_m,&gra_p);
    AllocateDeviceArray_double(size_decompose,1,&lap_m);
    /********vector********/    
    AllocateDeviceArray_double(size_decompose,3,&n_dir_x,&n_dir_y,&n_dir_z);
    /********modify c********/  
    AllocateDeviceArray_double(size_decompose,1,&modi_c);
    /********cross section********/
    AllocateDeviceArray_double(size_xz_plane,1,&xz_d);

    /******** CUDA Stream and time ********/
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventCreate(&gpu_start_temp);
	cudaEventCreate(&gpu_stop_temp);


}


void free_memory(){

    if (myid == 0){
        /********macro********/    
        FreeHostArray_double(9,c_f_h ,m_f_h, p_f_h, u_f_h, v_f_h, w_f_h, a_f_h, sf_f_h, DE_f_h);
        /********shear flow variable********/    
        FreeHostArray_double(11,lx, lz, cx_coa, cz_coa, cx_2_coa, cz_2_coa, cx_sep, cz_sep, cx_2_sep, cz_2_sep, D_value);
        /********cross section********/
        FreeHostArray_double(1,xz_f_h);
        /**************least square 2D****************/
        FreeHostArray_double(2,boundary_x,boundary_z);
    }

    FreeHostArray_double(9,c_d_h,m_d_h,p_d_h,a_d_h,u_d_h,v_d_h,w_d_h,sf_d_h,DE_d_h);   
    /********direction********/ 
    FreeHostArray_double(4,ex_h,ey_h,ez_h,wt_h);
    FreeHostArray_int(1,et_h);
    /********distri function********/
    FreeHostArray_double(6,g_h,geq_h,g_t_h,h_h,heq_h,h_t_h);  
    /********gradient********/
    FreeHostArray_double(3,gra_c_h,gra_m_h,gra_p_h);       
    FreeHostArray_double(1,lap_m_h); 
    /********transfer********/
    FreeHostArray_double(3,t_c_h,t_m_h,t_p_h);  
    FreeHostArray_double(3,t_u_h,t_v_h,t_w_h);  
    FreeHostArray_double(2,t_g_h,t_h_h);  
    /********cross section********/
    FreeHostArray_double(1,xz_d_h);



    //device
    /********macro********/
    FreeDeviceArray_double(9,c_d,m_d,p_d,u_d,v_d,w_d,a_d,sf_d,DE_d);
    FreeDeviceArray_double(9,c,m,p,u,v,w,a,sf,DE);
    /********distri function********/  
    FreeDeviceArray_double(6,h,g,heq,geq,h_t,g_t);
    /********transfer********/
    FreeDeviceArray_double(3,t_c,t_m,t_p);
    FreeDeviceArray_double(3,t_u,t_v,t_w);
    FreeDeviceArray_double(2,t_g,t_h);
    /********gradient********/
    FreeDeviceArray_double(3,gra_c,gra_m,gra_p);
    FreeDeviceArray_double(1,lap_m);
    /********vector********/    
    FreeDeviceArray_double(3,n_dir_x,n_dir_y,n_dir_z);
    /********modify c********/  
    FreeDeviceArray_double(1,modi_c);
    /********cross section********/
    FreeDeviceArray_double(1,xz_d);


	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);
    cudaEventDestroy(gpu_start_temp);
	cudaEventDestroy(gpu_stop_temp);

}


/*****************transfer information****************/

void Mempy_h_To_d(const size_t nBytes, const int num_arrays, ...) {
    va_list args;
    va_start(args, num_arrays);

    for (int i = 0; i < num_arrays; i++) {
        // 提取兩個參數：device 指針和 host 指針
        double* d_ptr = va_arg(args, double*);
        double* h_ptr = va_arg(args, double*);

        // 進行 cudaMemcpy 拷貝操作
        CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, nBytes, cudaMemcpyHostToDevice));
    }
    va_end(args);
}

void Mempy_d_To_h(const size_t nBytes, const int num_arrays, ...) {
    va_list args;
    va_start(args, num_arrays);

    for (int i = 0; i < num_arrays; i++) {
        // 提取兩個參數：device 指針和 host 指針
        double* d_ptr = va_arg(args, double*);
        double* h_ptr = va_arg(args, double*);
        
        // 進行 cudaMemcpy 拷貝操作
        CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, nBytes, cudaMemcpyDeviceToHost));
    }
    va_end(args);
}

void Mempy_h_To_d_stream(const cudaStream_t stream,const size_t nBytes, const int num_arrays, ...) {
    va_list args;
    va_start(args, num_arrays);

    for (int i = 0; i < num_arrays; i++) {
        // 提取兩個參數：device 指針和 host 指針
        double* d_ptr = va_arg(args, double*);
        double* h_ptr = va_arg(args, double*);

        // 進行 cudaMemcpy 拷貝操作
        CHECK_CUDA(cudaMemcpyAsync(d_ptr, h_ptr,  nBytes, cudaMemcpyHostToDevice,stream));
    }
    va_end(args);
}

void Mempy_d_To_h_stream(const cudaStream_t stream,const size_t nBytes, const int num_arrays, ...) {
    va_list args;
    va_start(args, num_arrays);

    for (int i = 0; i < num_arrays; i++) {
        // 提取兩個參數：device 指針和 host 指針
        double* d_ptr = va_arg(args, double*);
        double* h_ptr = va_arg(args, double*);
        
        // 進行 cudaMemcpy 拷貝操作
        CHECK_CUDA(cudaMemcpyAsync(h_ptr, d_ptr,  nBytes, cudaMemcpyDeviceToHost,stream));
    }
    va_end(args);
}



/***************** const memory ****************/

void Mempy_Symbol_const_memory(){
    cudaMemcpyToSymbol (  ex_d ,  ex_h,   sizeof(double)*q  );
	cudaMemcpyToSymbol (  ey_d ,  ey_h,   sizeof(double)*q  );
	cudaMemcpyToSymbol (  ez_d ,  ez_h,   sizeof(double)*q  );
	cudaMemcpyToSymbol (  wt_d ,  wt_h,   sizeof(double)*q  );
	cudaMemcpyToSymbol (  et_d ,  et_h,   sizeof(int   )*q  );
}

/***************** pre do ****************/

void Mempy_H_To_D_pre_do(){
    Mempy_h_To_d(size_decompose, 7, c_d, c_d_h, m_d, m_d_h, p_d, p_d_h, u_d, u_d_h, v_d, v_d_h, w_d, w_d_h, a_d, a_d_h);
    Mempy_h_To_d(size_distri_function, 6, h, h_h, g, g_h, heq, heq_h, geq, geq_h, h_t, h_t_h, g_t, g_t_h);
    Mempy_h_To_d(size_transfer_macro_buffer_4, 3, t_c, t_c_h, t_m, t_m_h, t_p, t_p_h);
    Mempy_h_To_d(size_u_v_w_buffer_transfer, 3, t_u, t_u_h, t_v, t_v_h, t_w, t_w_h);   
    Mempy_h_To_d(size_transfer_distri_function, 2, t_g, t_g_h, t_h, t_h_h);
    Mempy_h_To_d(size_gradient, 3, gra_c, gra_c_h, gra_m, gra_m_h, gra_p, gra_p_h);
    Mempy_h_To_d(size_decompose, 1, lap_m, lap_m_h);
}

/***************** print out ****************/

void Mempy_D_To_H_macro_print(){

    Mempy_d_To_h(size_decompose, 7, c_d, c_d_h, m_d, m_d_h, p_d, p_d_h, u_d, u_d_h, v_d, v_d_h, w_d, w_d_h, a_d, a_d_h);
    CHECK_CUDA(cudaDeviceSynchronize());
}


/*****************transfer information   d->h ******************/ 

//distri
inline void Transfer_distri_D_To_H(const cudaStream_t stream){

    Mempy_d_To_h_stream(stream,size_transfer_distri_function,2,t_g,t_g_h,t_h,t_h_h);  
}
// macro value
inline void Transfer_c_D_To_H(const cudaStream_t stream){

    Mempy_d_To_h_stream(stream,size_transfer_macro_buffer_4,1,t_c,t_c_h);

}
inline void Transfer_chemical_D_To_H(const cudaStream_t stream){

    Mempy_d_To_h_stream(stream,size_transfer_macro_buffer_4,1,t_m,t_m_h);

}
inline void Transfer_other_D_To_H(const cudaStream_t stream){
    Mempy_d_To_h_stream(stream,size_transfer_macro_velo,3,t_u,t_u_h,t_v,t_v_h,t_w,t_w_h);
    Mempy_d_To_h_stream(stream,size_transfer_macro_buffer_4,1,t_p,t_p_h);
}

/*****************transfer information   h->d ******************/ 

//distri
inline void Transfer_distri_H_To_D(const cudaStream_t stream){

    Mempy_h_To_d_stream(stream,size_transfer_distri_function,2,t_g,t_g_h,t_h,t_h_h);  
}
// macro value
inline void Transfer_c_H_To_D(const cudaStream_t stream){

    Mempy_h_To_d_stream(stream,size_transfer_macro_buffer_4,1,t_c,t_c_h);

}
inline void Transfer_chemical_H_To_D(const cudaStream_t stream){

    Mempy_h_To_d_stream(stream,size_transfer_macro_buffer_4,1,t_m,t_m_h);

}
inline void Transfer_other_H_To_D(const cudaStream_t stream){
    Mempy_h_To_d_stream(stream,size_transfer_macro_velo,3,t_u,t_u_h,t_v,t_v_h,t_w,t_w_h);
    Mempy_h_To_d_stream(stream,size_transfer_macro_buffer_4,1,t_p,t_p_h);
}




#endif
#ifndef POST_H
#define POST_H

#include"data_make.h"
#include"linear_algebra.h"
#include "cuda_memory_and_transfer.h"
/******************first print out********************/

void model_setting_print(){

	information = fopen("information.txt","w");
	if(PH_model==0){
        printf("+----------------------------------------------------------------+\n");
		fprintf( information, "interface model : Cahn-Hilliard\n");
		printf("interface model : Cahn-Hilliard\n");
	}else{
        printf("+----------------------------------------------------------------+\n");
		fprintf( information, "interface model : Allen-Cahn\n");
		printf("interface model : Allen-Cahn\n");
	}
	fprintf(information,"Grid size nx = %d, ny = %d, nz = %d\n",nx,ny,nz);
	fprintf(information, "Radius=%.2f, Thickness=%.2f\n",(double)radd, (double)thick);
	fprintf(information,"kappa = %f, beta = %f\n", kappa, beta);   
	fprintf(information,"Mobility = %f,surface_tension = %f\n",(double)mobi,(double)surface_tension);
    printf("+----------------------------------------------------------------+\n");
    printf("Grid size nx = %d, ny = %d, nz = %d\n",nx,ny,nz);
    printf("+----------------------------------------------------------------+\n");
	printf("Radius = %.2f, Thickness = %.2f\n",(double)radd, (double)thick);
    printf("+----------------------------------------------------------------+\n");
    printf("kappa = %f, beta = %f\n",kappa,beta);  
    printf("+----------------------------------------------------------------+\n");
	printf("Mobility = %f,surface_tension = %f\n",(double)mobi,(double)surface_tension);
	free_energy_type(information);
    printf("+----------------------------------------------------------------+\n");
	initial_infor(information);
    printf("+----------------------------------------------------------------+\n");
	boundary_infor(information);	
	if(PH_model == 0)
	{
    	printf("+----------------------------------------------------------------+\n");
		force_type(information);
	}


 	int numConditions = sizeof(conditionTable) / sizeof(conditionTable[0]);
  	if (initial_type >= 0 && initial_type < numConditions) {
  	    // 呼叫對應函數
  	    conditionTable[initial_type]();
  	} else {
  	    fprintf(stderr, "Invalid initial_type: %d\n", initial_type);
  	}

	fclose(information);

}

/*************middle print out****************/


void cuda_time_prediction(){
	if( myid == iroot){          
		printf("step=%d\n",step);
		cudaEventRecord(gpu_stop_temp,0);
		cudaEventSynchronize(gpu_stop_temp);
		float cudatime_temp;
		cudaEventElapsedTime(&cudatime_temp,gpu_start_temp,gpu_stop_temp);
		cudatime_temp=cudatime_temp/1000.0;//unit sec
		int remain_time=(int)(cudatime_temp/(step)*(stepall-step));
		printf("time remaining: %d hr,%d min,%d sec\n",(int)remain_time/3600,(int)(remain_time%3600)/60,(int)remain_time%60);
    }
}

void FILE_out(int n_f){
	p_real<<<bpgBuff,tpbBuff>>>(c,p,a,beta,kappa);
	adjust_array_mid_to_side();
    Mempy_D_To_H_macro_print();
    Gather_Macro_D_To_F(n_f);  	
	if(myid == 0)
	{
        printf("+----------------------------------------------------------------+\n");
		printf("c max=%lf\n",maxvalue(c_f_h));
		printf("c min=%lf\n",minvalue(c_f_h));
		printf("p_hydro max=%e\n" ,maxvalue(p_f_h));
		printf("p_real max=%e\n" ,maxvalue(a_f_h));
		printf("p_real min=%e\n" ,minvalue(a_f_h));
		printf("u max=%e\n" ,maxvalue(u_f_h));
		printf("v max=%e\n" ,maxvalue(v_f_h));
		printf("w max=%e\n" ,maxvalue(w_f_h));
        printf("+----------------------------------------------------------------+\n");
		printf("Ma u=%e\n" ,maxvalue(u_f_h)*1.7320508);
		printf("Ma v=%e\n" ,maxvalue(v_f_h)*1.7320508);
		printf("Ma w=%e\n" ,maxvalue(w_f_h)*1.7320508);

		//one_dimension_C_profile();
		//mass_ratio_2d();
 		mass_ratio_3d();		

		if(enable_print_out_2d)
		{
			Data2d(y_2d_print_position);
		}

		if(enable_print_out_3d)
		{
			Data3d();
		}

		if(initial_type == 0){
			diameter_3d_print();
            printf("+----------------------------------------------------------------+\n");
			printf("pressure=%e, " ,p_difference(c_f_h,a_f_h));
			printf("error =%lf%\n",(p_difference(c_f_h,a_f_h)-2.0*surface_tension/radd)/(2.0*surface_tension/radd)*100.0);
			printf("x width length : %f \t z width lengh :%f \t y width length : %f\n ",length_x(c_f_h),length_z(c_f_h),length_y(c_f_h));
            
		}
		else if (initial_type == 7 && iprint%value_get == 0)
		{
			printf("D = %e\n",D_value[(int)(step/value_get)-1]);			
		}
		else if (initial_type == 13 && iprint%value_get == 0)
		{
			printf("cx_sep = %e\ncz_sep = %e\ncx_2_sep = %e\ncz_2_sep = %e\n ",cx_sep[(int)(step/value_get)-1],cz_sep[(int)(step/value_get)-1],cx_2_sep[(int)(step/value_get)-1],cz_2_sep[(int)(step/value_get)-1]);
		}
	}
}


/*************middle print out xz cross section****************/

void xz_calculation_condi_7(){

    if(step%value_get == 0){
		int current_output_index = (int)(step/value_get) - 1 ;
	    transfer2xz <<< nz/cpu,nx >>>(xz_d,c);
        CHECK_CUDA(cudaDeviceSynchronize());
	    cudaMemcpy(xz_d_h,xz_d,sizeof(double)*(nx+4)*(nz/cpu+4),cudaMemcpyDeviceToHost);
        CHECK_CUDA(cudaDeviceSynchronize());
	    MPI_Gather((void *)&xz_d_h[0], nx*nz/cpu, MPI_DOUBLE,(void *)&xz_f_h[0], nx*nz/cpu, MPI_DOUBLE,iroot,MPI_COMM_WORLD);

	    if(myid==0){
	    	//center_single_droplet( xz_f_h, cx_coa, cz_coa, current_output_index );	//cpu calculation
	    	//D_single_droplet( xz_f_h, cx_coa, cz_coa, D_value, current_output_index );
			D_2d_least_square(xz_f_h,boundary_x,boundary_z,D_value,current_output_index);
	    }
    }

}

void xz_calculation_condi_13(){


    if(step%value_get == 0){
		int current_output_index = (int)(step /value_get) - 1 ;
        transfer2xz_sym <<< nz/cpu,nx >>>(xz_d,c);
        CHECK_CUDA(cudaDeviceSynchronize());
	    CHECK_CUDA(cudaMemcpy(xz_d_h,xz_d,sizeof(double)*(nx+4)*(nz/cpu+4),cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaDeviceSynchronize());
	    CHECK_MPI(MPI_Gather((void *)&xz_d_h[0], nx*nz/cpu, MPI_DOUBLE,(void *)&xz_f_h[0], nx*nz/cpu, MPI_DOUBLE,iroot,MPI_COMM_WORLD));
        if(myid == 0)
        {
            center_two_droplet_sep( xz_f_h, cx_sep, cz_sep, cx_2_sep, cz_2_sep,current_output_index);
        }

    }

}

void value_print_out_condi_7(){
    if(myid==0){

        Deformation();

    }
}

void value_print_out_condi_13(){

    if(myid==0){
        dxdt();
        dzdt();
        dxdz();
        theta();
	}
}


void xz_calculation(){

    switch (initial_type)
    {
    case 7:
        xz_calculation_condi_7();
        break;
    case 13:
        xz_calculation_condi_13();
        break;		
    default:
        break;
    }

}

void value_print_out(){
    switch (initial_type)
    {
    case 7:
        value_print_out_condi_7();
        break;
    case 13:
        value_print_out_condi_13();
        break;
    default:
        break;
    } 
}


/*************Final print out****************/

void Final_print_out(int n_f){
	p_real<<<bpgBuff,tpbBuff>>>(c,p,a,beta,kappa);
	adjust_array_mid_to_side();
	Mempy_D_To_H_macro_print();
	Gather_Macro_D_To_F(n_f);
	if(myid == 0){
		if(enable_final_print_2d){
			final_print_2d(y_2d_print_position);
		}
		if(enable_final_print__3d){
			final_print_3d();
		}

		if (initial_type == 0 )
		{
			information = fopen("information.txt","a");
			fprintf( information,"dP_ana=%e\n",   2.0*surface_tension/radd);
			fprintf( information,"dP_tot=%e\n",   p_difference(c_f_h,a_f_h));
			fprintf( information,"error =%lf%\n",(p_difference(c_f_h,a_f_h)-2.0*surface_tension/radd)/(2.0*surface_tension/radd)*100.0);
			fclose(information);
		}
	}

}

void cuda_time_final(){
	if( myid == iroot){ 	
		cudaEventRecord(gpu_stop,0);
		cudaEventSynchronize(gpu_stop);
		float cudatime;
		printf("===============================================================\n");
		printf("Iteration terminated!\n");
		cudaEventElapsedTime(&cudatime,gpu_start,gpu_stop);
		printf("GPU total time = %f ms\n",cudatime); //unit = ms
		printf("mlups=%lf \n",(double)(nx*ny*nz)*stepall*pow(10.0,-6.0)/(cudatime/1000.0));
		printf("===============================================================\n");
	}


}


#endif
#ifndef PARAMETER_H
#define PARAMETER_H

//Domain//
#define  nx        192
#define  ny        192		
#define  nz        192
//device parameter//
#define  cpu        4
#define  stepall    196800
#define  iprint     19680             //要是stepall 倍數+偶數


#define  enable_print_out_2d           1
#define  enable_final_print_2d         1
#define  y_2d_print_position           0    //  ny/2 for no symmetry     0 for symmetry 

#define  enable_print_out_3d           0
#define  enable_final_print__3d        0

#define value_get  100    // 偶數
//droplet parameter//

#define  radd       64
#define  thick      4.0      
#define  rho_ratio  1		//liquid to gas density ratio 624.5, 666(100)
#define  mu_ratio   1		//liquid to gas dynamic viscosity ratio 129.8, 119(17.857)
#define  tau_h      0.5 
#define  tau_l      0.25
#define  tau_g      tau_l/mu_ratio*rho_ratio
#define  rho_l      1.0         // 0.765
#define  rho_g      rho_l/rho_ratio
#define  surface_tension     0.005
#define  gra_ac     0.0
#define  uuu        0.006666667
#define  timesss    (double)2             //上面那排的幾倍速度
#define  u_0        uuu*timesss// if use boundary_set 1 and 2
#define  mobility   1
#define  bulk_free_energy 1  	//0 for double obstacle 1 for double well
#define  SF_Type 0

#define  PH_model 0 // 0 Cahn Hilliard ; 1 Allen Cahn
#define  energy_conserve 0 // 1 open; 0 close
#define  boundary_set 0 // 0:periodic ; 1:top and bottom wall ; 2:top bottom wall, left right sym, front back periodic
#define  initial_type 0 

#define  Force_method 1  // 0: He et al. ; 1:buick et al ; 2 guo et al 




//0 stationary droplet 
//1 collision 
//3 two drops merging
//4 one bubble theory (phase separation) 
//5 drop oscillation
//6 wall boundary test
//7 cylinder
//8 collision (after XXXXX time step give u_coll)
//9 one droplet move
//10 one bubble
//11 shear flow droplet collision (need use boundary_set 1)
//12 shear flow single_droplet (need use boundary_set 1)
//13 shear flow droplet coalescence (need use boundary_set 2) 

//initial_type 1
#define  u_coll     0.01875
#define  b_coll     0.55
#define  p_out      0.0
#define  equally    0   // 是否直接平分速度
//initial_type 4
#define  mean_ini_4       -2.7
#define  std_ini_4       1.5
//initial_type 5
#define  u0         0.05
#define  ar         1.2
//initial_type 8
#define  u_coll_7   0.012354167
#define  b_coll_7   0.06
//initial_type 9
#define  u_coll_9   0.01666666667
//initial_type 11 and 13
#define  b_coll_2   0.86
#define  x_coll_2   1.26

#define  start_step 1001 // need odd，很奇怪不要用



#endif

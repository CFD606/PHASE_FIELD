#ifndef linear_CPP
#define linear_CPP

#include<stdio.h>
#include"parameter.h"
#include"math.h"
#include <Eigen/Dense>

/************************* least square ***********************************/


typedef struct {
    double A, B, C, D, E, F;  // Coefficients for the ellipse equation
} EllipseParams;


void Find_out_boundary_of_sharp(double *C, double*boundary_x, double*boundary_z,int* num_points ){
    for (int i = 0; i < nx - 1; i++) {
        for (int k = 0; k < nz - 1; k++) {
            int idx = i + nx * k;
            int idx_right = (i + 1) + nx * k;
            int idx_up = i + nx * (k + 1);
            double x = -1, z = -1;

            // 剛好等於閾值的點直接加入
            if (fabs(C[idx] - 0.5) < 1e-14) {
                x = i;
                z = k;
            } 
            // 水平方向越界的點
            else if ((C[idx] - 0.5) * (C[idx_right] - 0.5) < 0) {
                double t = (0.5 - C[idx]) / (C[idx_right] - C[idx]);
                x = i + t; // 插值計算 X
                z = k;     // 水平邊界
            } 
            // 垂直方向越界的點
            else if ((C[idx] - 0.5) * (C[idx_up] - 0.5) < 0) {
                double t = (0.5 - C[idx]) / (C[idx_up] - C[idx]);
                x = i;     // 垂直邊界
                z = k + t; // 插值計算 Y
            }

            // 如果找到有效的 (x, y) 點，檢查是否重複
            if (x >= 0 && z >= 0) {
                int is_duplicate = 0;
                for (int n = 0; n < (*num_points); n++) {
                    if (fabs(boundary_x[n] - x) < 1e-14 && fabs(boundary_z[n] - z) < 1e-14) {
                        is_duplicate = 1;
                        break;
                    }
                }

                // 如果不是重複點，加入到邊界點集合中
                if (!is_duplicate) {
                    boundary_x[*num_points] = x;
                    boundary_z[*num_points] = z;
                    (*num_points)++;
                }
            }
        }
	}
}

void fit_rotated_ellipse(const double*boundary_x, const double* boundary_z, int num_points,EllipseParams *params)
{
    using namespace Eigen;

    // 1. 構建 (num_points x 6) 矩陣 X
    MatrixXd X(num_points, 6);
    for(int i = 0; i < num_points; i++){
        double xi = boundary_x[i];
        double yi = boundary_z[i];
        X(i, 0) = xi*xi;  // A
        X(i, 1) = xi*yi;  // B
        X(i, 2) = yi*yi;  // C
        X(i, 3) = xi;     // D
        X(i, 4) = yi;     // E
        X(i, 5) = 1.0;    // F
    }

    // 2. 對 X 做 SVD，找最小奇異值對應的奇異向量
    JacobiSVD<MatrixXd> svd(X, ComputeThinU | ComputeThinV);
    // 右奇異向量 matrixV 的維度是 (6 x 6)
    // 取最後一欄 (col(5)) 就是對應最小奇異值的奇異向量
    VectorXd v = svd.matrixV().col(5);

    params->A = v(0);
    params->B = v(1);
    params->C = v(2);
    params->D = v(3);
    params->E = v(4);
    params->F = v(5);

    //return params;
}

void compute_axes(EllipseParams *params,double*DD,int indexx) {
    double  params_A = params->A;
    double  params_B = params->B;
    double  params_C = params->C;
    double  params_D = params->D;
    double  params_E = params->E;
    double  params_F = params->F;
	double center_x,center_y,major_axis,minor_axis;
    // 計算中心座標
    double denominator = (params_B * params_B - 4 * params_A * params_C);
    if (fabs(denominator) < 1e-14) {
        printf("Error: Invalid ellipse parameters for center calculation.\n");
        center_x = 0;
        center_y = 0;
        major_axis = 0;
    	minor_axis = 0;
        return;
    }

    center_x = (2 *  params_C *  params_D -  params_B *  params_E) / denominator;
    center_y = (2 *  params_A *  params_E -  params_B *  params_D) / denominator;

    // 計算主軸與次軸
    double term1 =  params_A * (center_x) * (center_x) +  params_B * (center_x) * (center_y) +  params_C * (center_y) * (center_y);
    double term2 =  params_D * (center_x) +  params_E * (center_y) + params_F;

    double scaling_factor = -1.0 / (term1 + term2);

    double normalized_A = scaling_factor * params_A;
    double normalized_B = scaling_factor * params_B;
    double normalized_C = scaling_factor * params_C;

    double temp = sqrt((normalized_A - normalized_C) * (normalized_A - normalized_C) + normalized_B * normalized_B);
    double major_axis_squared = 2.0 / (normalized_A + normalized_C - temp);
    double minor_axis_squared = 2.0 / (normalized_A + normalized_C + temp);

    if (major_axis_squared < 0 || minor_axis_squared < 0) {
        printf("Error: Invalid ellipse parameters for axis calculation.\n");
        major_axis = 0;
        minor_axis = 0;
        return;
    }

    major_axis = sqrt(major_axis_squared);
    minor_axis = sqrt(minor_axis_squared);
    DD[indexx] = (major_axis-minor_axis)/(major_axis+minor_axis);

}

void D_2d_least_square(double *c, double*x ,double*z,double*D_value ,int indexx){
    
	int num_points = 0 ;
	EllipseParams params;
	Find_out_boundary_of_sharp(c, x, z,&num_points);
	fit_rotated_ellipse(x,z,num_points,&params);
	compute_axes(&params,D_value,indexx);

}
#endif
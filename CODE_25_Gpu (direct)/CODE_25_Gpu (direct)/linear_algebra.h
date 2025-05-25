#ifndef LEAST_SQUARE_H
#define LEAST_SQUARE_H

// 結構體定義
typedef struct {
    double A, B, C, D, E, F;  // Coefficients for the ellipse equation
} EllipseParams;

// 函數宣告
void Find_out_boundary_of_sharp(double *C, double *boundary_x, double *boundary_z, int *num_points);
void fit_rotated_ellipse(const double *boundary_x, const double *boundary_z, int num_points, EllipseParams *params);
void compute_axes(EllipseParams *params, double *DD, int indexx);
void D_2d_least_square(double *c, double *boundary_x, double *boundary_z, double *D_value, int indexx);

#endif // LEAST_SQUARE_H

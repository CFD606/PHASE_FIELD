#ifndef ERROR_CHECK_H 
#define ERROR_CHECK_H 


#define CHECK_CUDA(call)                                                       \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_MPI( call ) 																	\
{   																						\
    int mpi_status = call; 																	\
    if ( 0 != mpi_status ) 																	\
    { 																						\
        char mpi_error_string[MPI_MAX_ERROR_STRING];	 									\
        int mpi_error_string_length = 0; 													\
        MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); 			\
        if ( NULL != mpi_error_string ) 													\
            fprintf(stderr, "ERROR: MPI call \"%s\" in line %d of file %s failed with %s (%d).\n", #call, __LINE__, __FILE__, mpi_error_string, mpi_status);			\
        else 																				\
            fprintf(stderr, "ERROR: MPI call \"%s\" in line %d of file %s failed with %d.\n", #call, __LINE__, __FILE__, mpi_status); 										\
    } 																						\
}


#define CHECK_FILE(fp, filename)                \
    do {                                        \
        if ((fp) == NULL) {                     \
            perror("Error opening file " filename); \
            return;                             \
        }                                       \
    } while (0)



#endif
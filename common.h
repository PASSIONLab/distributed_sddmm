#ifndef COMMON_HEADER
#define COMMON_HEADER

typedef Matrix<double, Dynamic, Dynamic, RowMajor> DenseMatrix;
typedef SpParMat < int64_t, int, SpDCCols<int32_t,int> > PSpMat_s32p64_Int;

#endif
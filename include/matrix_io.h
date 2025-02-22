// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <types.h>
#include <iomanip>
#include <map>
#include <vector.h>
#include <fstream>
#include <matrix.h>
#include <amg_solver.h>
#include <amg_config.h>
#include <distributed/amgx_mpi.h>

namespace amgx
{

namespace io_config
{
enum ReaderProps { NONE = 0, MTX = 1, RHS = 2, SOLN = 4, SIZE = 8, PRINT = 16, GEN_RHS = 32};
static inline bool hasProps( unsigned int query, unsigned int props) { return (query | props) == props; }
inline void addProps(const unsigned int new_props, unsigned int &props) { props |= new_props; }
}

template<class T_Config>
class MatrixIO
{
    public:
        typedef Vector<T_Config> VVector;
        typedef typename T_Config::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef Vector<TConfig_h> Vector_h;
        typedef typename Matrix<T_Config>::MVector MVector;
        typedef typename Matrix<TConfig_h>::MVector MVector_h;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
        typedef Vector<ivec_value_type_h> IVector_h;

        typedef bool (*readerFunc) (std::ifstream &fin, const char *fname
                                    , Matrix<T_Config> &A
                                    , VVector &b
                                    , VVector &x
                                    , const AMG_Config &cfg //@TODO: change behaviour of config::get_parameter to be const, and make const here.
                                    , unsigned int props
                                    , const IVector_h &rank_rows // = IVector_h(0) row indices for given rank
                                   );
        typedef std::map<std::string, readerFunc> readerMap;
        static void registerReader(std::string key, readerFunc func);
        static void unregisterReaders();

        // This is what is called from C-API
        static AMGX_ERROR readSystem(const char *fname
                                     , Matrix<T_Config> &A
                                     , VVector &b
                                     , VVector &x
                                     , const AMG_Config &cfg = AMG_Config()
                                     , unsigned int props = io_config::MTX | io_config::RHS | io_config::SOLN
                                     , const IVector_h &rank_rows = IVector_h(0) // row indices for given rank
                                    );


        static AMGX_ERROR readSystem(const char *fname
                                     , Matrix<T_Config> &A
                                     , const AMG_Config &cfg = AMG_Config()
                                     , unsigned int props = io_config::MTX
                                     , const IVector_h &rank_rows = IVector_h(0) // row indices for given rank
                                    );
        static AMGX_ERROR readSystem(const char *fname
                                     , Matrix<T_Config> &A
                                     , VVector &b
                                     , const AMG_Config &cfg = AMG_Config()
                                     , unsigned int props = io_config::MTX | io_config::RHS
                                     , const IVector_h &rank_rows = IVector_h(0) // row indices for given rank
                                    );
        static std::string readSystemFormat(const char *fname);
        //static AMGX_ERROR readColoring(AuxData* obj, const char* fname);
        //static AMGX_ERROR readGeometry(AuxData* obj, const char* fname);
        //static AMGX_ERROR readGeometry(AuxData* obj, int n, int dimension);


        typedef bool (*writerFunc) (const char *filename, const Matrix<T_Config> *A, const VVector *b, const VVector *x);
        typedef std::map<std::string, writerFunc> writerMap;
        static void registerWriter(std::string key, writerFunc func);
        static void unregisterWriters();

        static AMGX_ERROR writeSystem (const char *filename, const Matrix<T_Config> *A, const VVector *b, const VVector *x);
        static AMGX_ERROR writeSystemWithFormat (const char *filename, const char *format, const Matrix<T_Config> *A, const VVector *b, const VVector *x);

        static bool writeSystemMatrixMarket(const char *fname, const Matrix<T_Config> *tA, const VVector *tb, const VVector *tx);
        static bool writeSystemBinary(const char *fname, const Matrix<T_Config> *tA, const VVector *tb, const VVector *tx);

    private:
        static readerMap &getReaderMap();
        static writerMap &getWriterMap();
};

/*template<typename IVector, typename MVector, typename VVector>
AMGX_ERROR writeSystemBinaryRaw_v2(const char *fname, 
                            int64_t nrows,
                            int64_t nnz,
                            bool ext_diag,
                            const int bdimx,
                            const int bdimy,
                            const IVector& row_offsets, 
                            const IVector& col_indices, 
                            const MVector& mat_values, 
                            const VVector& rhs, 
                            const VVector& sol);
*/

template <typename V>
auto raw(V& v){
    return thrust::raw_pointer_cast(v.data());
}

// datatypes. Ideally, write TConfig values, but just needed something quick.
template <typename T>
uint64_t bin_type()
{
    if constexpr(std::is_same<T, int>::value)           { return 0; }
    else if constexpr(std::is_same<T, int64_t>::value)  { return 1; }
    else if constexpr(std::is_same<T, float>::value)    { return 2; }
    else if constexpr(std::is_same<T, double>::value)   { return 3; }
    else {
        FatalError( "Unsupported data type", AMGX_ERR_BAD_PARAMETERS);
    }
}


// templated binary writer with raw data 
/*
    original writer:
        header_id,  "%%NVAMGBinary\n"
        system_header, 36 bytes of meta info (see writeSystemBinary) for details
        data, always converted to double
    this writer:
        header_id, "%%NVAMGBin_v2\n"
        system_header, 128 bytes, first byte - version, then metadata depending on version
        data, written in type passed to the writer
*/
// arrays are on host
template<typename IVector, typename MVector, typename VVector>
AMGX_ERROR writeSystemBinaryRaw_v2(const char *fname, 
                            int64_t nrows,
                            int64_t nnz,
                            bool ext_diag,
                            const int bdimx,
                            const int bdimy,
                            const IVector& row_offsets, 
                            const IVector& col_indices, 
                            const MVector& mat_values, 
                            const VVector& rhs, 
                            const VVector& sol)
{
    using index_type = typename IVector::value_type;
    using matrix_type = typename MVector::value_type;
    using vector_type = typename VVector::value_type;

    std::string header_id = "%%NVAMGBin_v2\n";
    // format : amgx header id, system_flags header, mtx csr data, (optional) mtx diag data, (optional) rhs data, (optional) sol data, 
    if (!fname)
    {
        FatalError( "Bad filename", AMGX_ERR_BAD_PARAMETERS);
    }

    if (row_offsets.size() < 1 || col_indices.size() < 1 || mat_values.size() < 1)
    {
        FatalError( "Matrix data should not be NULL", AMGX_ERR_BAD_PARAMETERS);
    }

    FILE *fout;
    std::string err = "Writing system to file " + std::string(fname) + "\n";
    amgx_output(err.c_str(), err.length());
    fout = fopen(fname, "wb");

    if (!fout)
    {
        FatalError( "Cannot open output file!", AMGX_ERR_BAD_PARAMETERS);
    }

    bool is_mtx = true;
    bool is_rhs = rhs.size() != 0;
    bool is_soln = sol.size() != 0;
    uint32_t matrix_format = MatrixProps::CSR;

    /*if (A.hasProps(CSR))
    {
        matrix_format = MatrixProps::CSR;
    }
    else if (A.hasProps(COO))
    {
        matrix_format = 1;
    }
    else
    {
        FatalError("Unsupported matrix format", AMGX_ERR_BAD_PARAMETERS);
    }*/

    /*if (types::util<ValueTypeA>::is_complex)
    {
        matrix_format += COMPLEX;
    }*/
    assert (A.get_block_dimx() > 0);
    assert (A.get_block_dimy() > 0);

    // just in case
    const int system_header_size = 128;
    
    std::array<uint64_t, system_header_size> system_flags {
        //header version. if changes - bump the version
        0,
        // version 0:
        // 0 or 1 - if matrix present. Currently always should be 1
        static_cast<uint64_t>(is_mtx),
        // 0 or 1 - if rhs is in the file
        static_cast<uint64_t>(is_rhs), 
        // 0 or 1 - if rhs is in the file
        static_cast<uint64_t>(is_soln), 
        // matrix_format, 0 = CSR (default and only supported rn). data type encoded separately.
        static_cast<uint64_t>(matrix_format), 
        // external diagonal, 0 or 1
        static_cast<uint64_t>(ext_diag),
        // blockdimx and blockdimy
        static_cast<uint64_t>(bdimx), 
        static_cast<uint64_t>(bdimy), 
        // numrows, numnnz
        static_cast<uint64_t>(nrows), 
        static_cast<uint64_t>(nnz),
        // datatypes.
        bin_type<matrix_type>(),
        bin_type<vector_type>(),
        bin_type<index_type>()
    };
        fwrite(header_id.c_str(), sizeof(char), header_id.length(), fout);
        fwrite(system_flags.data(), sizeof(uint64_t), system_header_size, fout);
        
        uint64_t raw_values_number = static_cast<uint64_t>(bdimx) * bdimy * (nnz + (ext_diag ? nrows : 0) );

        if (is_mtx)
        {
            if (row_offsets.size() != nrows + 1 || 
                col_indices.size() != nnz ||
                mat_values.size() != raw_values_number)
            {
                FatalError("matrix dimension do not match", AMGX_ERR_BAD_PARAMETERS);
            }

            if (matrix_format == MatrixProps::CSR)
            {
                fwrite(raw(row_offsets), sizeof(index_type), nrows+1, fout); 
                fwrite(raw(col_indices), sizeof(index_type), nnz, fout); 
                fwrite(raw(mat_values), 
                    sizeof(matrix_type), 
                    static_cast<uint64_t>(bdimx) * bdimy * (nnz + (ext_diag ? nrows : 0) ), 
                    fout); // including diag in the end if exists.
            }
            else
            {
                FatalError("Unsupported matrix format for now", AMGX_ERR_IO);
            }
        } // End of writing matrix


        //write rhs
        if (is_rhs)
        {
            if (rhs.size() != nrows * bdimy)
            {
                FatalError("rhs vector and matrix dimension does not match", AMGX_ERR_BAD_PARAMETERS);
            }

            fwrite(raw(rhs), sizeof(vector_type), rhs.size(), fout);
        }

        // write initial guess if we have it
        if (is_soln)
        {
            if (sol.size() != nrows * bdimx)
            {
                FatalError("solution vector and matrix dimension does not match", AMGX_ERR_BAD_PARAMETERS);
            }

            fwrite(raw(sol), sizeof(vector_type), sol.size(), fout);
        }

        fclose(fout);
        err = "Done writing system to file!\n";
        amgx_output(err.c_str(), err.length());
        return AMGX_OK;
}

} // end namespace amgx

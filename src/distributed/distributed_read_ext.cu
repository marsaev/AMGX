#include "amgx_config.h"
#include <cassert>
#include <string>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdint.h>
#include <iostream>
#include <sstream>
#include "error.h"

namespace amgx{
// temporary functionality. see matrix_io.h for writer.

// datatypes. Ideally, write TConfig values, but just needed something quick.
template <typename T>
uint64_t bin_type(T v = T())
{
    if constexpr(std::is_same<T, int>::value)           { return 0; }
    else if constexpr(std::is_same<T, int64_t>::value)  { return 1; }
    else if constexpr(std::is_same<T, float>::value)    { return 2; }
    else if constexpr(std::is_same<T, double>::value)   { return 3; }
    else {
        return 666;
    }
}


// only real types supported
int distributed_read_large
(int *n,
 int *nnz,
 int *block_dimx,
 int *block_dimy,
 int **row_ptrs,
 void **col_indices_global,
 void **data,
 void **diag_data,
 void **rhs,
 void **sol,
 AMGX_Mode mode,
 const char *filename,
 int allocated_halo_depth,
 int num_partitions,
 int partition_id,
 const int *partition_sizes,
 int partition_vector_size,
 const int *partition_vector)
 {
    assert(allocated_halo_depth == 1);

    FILE *fin = fopen(filename, "rb");
    if (!fin)
        return 1;
    
    AMGX_VecPrecision vec_prec = AMGX_GET_MODE_VAL(AMGX_VecPrecision, mode );
    AMGX_MatPrecision mat_prec = AMGX_GET_MODE_VAL(AMGX_MatPrecision, mode );
    // NOTE: actual MODE here is ignored w.r.t indices, since types are forced by API: row_offsets is int, col_indices is int64
    AMGX_IndPrecision ind_prec = AMGX_GET_MODE_VAL(AMGX_IndPrecision, mode );

    std::string header_id = "%%NVAMGBin_v2\n";
    
    // header id
    auto rres = fread(header_id.data(), 1, header_id.size(), fin );
    assert(rres == header_id.size());

    // header
    const int system_header_size = 128;
    std::array<uint64_t, system_header_size> system_flags;

    rres = fread(system_flags.data(), sizeof(uint64_t), system_header_size, fin );
    assert(rres == system_header_size);

    uint64_t data_offset = ftell(fin);

    //header version. if changes - bump the version
    uint64_t bin_version { system_flags[0] };
    // 0 or 1 - if matrix present. Currently always should be 1
    uint64_t is_mtx{ system_flags[1] };
    // 0 or 1 - if rhs is in the file
    uint64_t is_rhs{ system_flags[2] }; 
    // 0 or 1 - if rhs is in the file
    uint64_t is_soln{ system_flags[3] }; 
    // matrix_format, 0 = CSR (default and only supported rn). data type encoded separately.
    uint64_t matrix_format{ system_flags[4] }; 
    // external diagonal, 0 or 1
    uint64_t ext_diag{ system_flags[5] };
    // blockdimx and blockdimy
    uint64_t bdimx{ system_flags[6] }; 
    uint64_t bdimy{ system_flags[7] }; 
    // numrows, numnnz
    uint64_t gnrows{ system_flags[8] }; 
    uint64_t gnnz{ system_flags[9] };
    // datatypes.
    uint64_t file_matrix_type { system_flags[10] };
    uint64_t file_vector_type{ system_flags[11] };
    uint64_t file_index_type{ system_flags[12] };

    *block_dimx = static_cast<int>(bdimx);
    *block_dimy = static_cast<int>(bdimy);

    assert(ext_diag == 0);

    bool sanity_print { false };
    std::ostringstream null;
    std::ostream& log = sanity_print ? std::cout : null;

    // sanity
    log << " Reading from file: global nrows: " << gnrows << ", global nnz: " << gnnz << 
        ", bdimx x bdimy: " << bdimx << " x " << bdimy <<
        ", matrix_type: " << file_matrix_type <<
        ", vector_type: " << file_vector_type <<
        ", index_type: " << file_index_type <<
        std::endl << std::flush;

    log << "Reading to: " <<
        " target matrix_type: " << mat_prec <<
        " target vector_type: " << vec_prec <<
        " target index_type: " << ind_prec <<
        std::endl << std::flush;

    // Limitation! 
    // API assumes int for local number of rows
    // API assumes int for local nnz, and, thus, local row offsets
    // API assumes int for partition vector size -> global gnrows - int, however int64 if using partitions_offsets
    assert(gnrows < std::numerical_limits<int>::max());

    // Generate global ids of local rows, and save number of local rows
    std::vector<int64_t> local_rows;
    if (partition_vector == nullptr){
        uint64_t rows_per_partition = gnrows / num_partitions;
        // API limitation
        assert( rows_per_partition < std::numerical_limits<int>::max());
        
        uint64_t first_row = rows_per_partition * partition_id;
        uint64_t last_row = std::min(gnrows, rows_per_partition * (partition_id+1));
        // save num local rows to return  values
        *n = static_cast<int>(last_row - first_row);
        local_rows.resize(*n);
        std::iota(local_rows.begin(), local_rows.end(), first_row);
    }
    else{
        assert(gnrows == partition_vector_size);
        // enumerate
        for (int i = 0; i < partition_vector_size; i++)
            if (partition_vector[i] == partition_id)
                local_rows.push_back(i);
        // API limitation
        assert(local_rows.size() < std::numerical_limits<int>::max());
        // save num local rows to return  values
        *n = static_cast<int>(local_rows.size());
    }
    auto num_local_rows = local_rows.size();

    // read helpers
    auto read_dump_data = [&data_offset, partition_id, fin, &log](int num, auto read_type) {
        using val_type = decltype(read_type);
        fseek(fin, data_offset, SEEK_SET);
        std::vector<val_type> current_offsets(num);
        auto res = fread(current_offsets.data(), sizeof(read_type), num, fin);
        assert(res == num);
        // for (auto v: current_offsets){
        //     log << partition_id << ": data: " << v << std::endl << std::flush;
        // }
    };

    // data in file is in read_type, we need to convert to type of target_ptr if needed
    // target_ptr is of type T*
    auto val_reader = [&data_offset, fin, &log](int64_t first_value_offset, int64_t num, auto file_read_type, int bdim, auto target_ptr){
        using file_val_type = decltype(file_read_type);

        std::vector<uint8_t> conversion_buffer;

        fseek(fin, data_offset + sizeof(file_val_type)*(first_value_offset*bdim), SEEK_SET);
        size_t res{};
        // if we don't need converting
        if constexpr (std::is_same<file_val_type, typename std::remove_pointer<decltype(target_ptr)>::type>::value)
        {
            res = fread(target_ptr, sizeof(file_val_type), num*bdim, fin);
        }
        else
        {
            conversion_buffer.reserve(num*bdim*sizeof(file_val_type));
            res = fread(conversion_buffer.data(), sizeof(file_val_type), num*bdim, fin);
            file_val_type* read_data = reinterpret_cast<file_val_type*>(conversion_buffer.data());
            // default-converts on the fly
            std::copy(read_data, read_data + num*bdim, target_ptr);
        }
        assert(res == num*bdim);
        return;
    };

    // specialized version of value reader that reads two values and returns a tuple of {val1, val2-val1}
    // returned values have same type as data in the file
    auto offsets_reader = [&data_offset, partition_id, fin](int row_id, auto file_ro_type, auto target_ro_type) -> 
            std::tuple<decltype(target_ro_type), decltype(target_ro_type)> {
        using file_offset_type = decltype(file_ro_type);
        using target_offset_type = decltype(target_ro_type);
        fseek(fin, data_offset + sizeof(file_offset_type)*row_id, SEEK_SET);
        file_offset_type current_offsets[2];
        auto res = fread(current_offsets, sizeof(file_offset_type), 2, fin);
        assert(res == 2);
        return {
            static_cast<target_offset_type>(current_offsets[0]), 
            static_cast<target_offset_type>(current_offsets[1] - current_offsets[0])
        };
    };

    // read mtx
    
    // starting global row offset and length of a row, used in every reader
    std::vector<std::tuple<int64_t, int64_t>> starting_nnz(*n);
    if (is_mtx)
    {
        // 1. reading rows offsets
        // API assumption - local row-ptrs - int
        *row_ptrs = (int*)malloc(sizeof(int) * (num_local_rows + 1));
        (*row_ptrs)[0] = 0;
        // assumption: we expect only int and int64 for indices type
        assert(bin_type<int>() == file_index_type || bin_type<int64_t>() == file_index_type);
        bool file_has_int32_indices = bin_type<int>() == file_index_type;
        
        // we can add a check here for actual read local_nnz < int32 limit
        for (int r = 0; r < num_local_rows; r++)
        {
            starting_nnz[r] = (file_has_int32_indices ?
                offsets_reader(local_rows[r], int{},     int64_t{}) : 
                offsets_reader(local_rows[r], int64_t{}, int64_t{})
            );
            (*row_ptrs)[r+1] = (*row_ptrs)[r] + 
                std::get<1>(starting_nnz[r]);
        }
        // actual local nnz
        auto num_local_nnz = (*row_ptrs)[num_local_rows];
        // save to return value
        *nnz = static_cast<int>(num_local_nnz);

        fseek(fin, data_offset + sizeof(file_has_int32_indices ? sizeof(int) : sizeof(int64_t))*(gnrows+1), SEEK_SET);
        data_offset += (file_has_int32_indices ? sizeof(int) : sizeof(int64_t))*(gnrows+1);

        // 2. reading col indices
        // allocate returned buffer, for this API - always int64
        *col_indices_global = (int64_t*)malloc(sizeof(int64_t) * (num_local_nnz));
        // raw pointer for writing 
        int64_t* ci_write_ptr = static_cast<int64_t*>(*col_indices_global);

        for (int r = 0; r < num_local_rows; r++)
        {
            if (file_has_int32_indices)
            {
                val_reader(std::get<0>(starting_nnz[r]),
                    std::get<1>(starting_nnz[r]), 
                    int{},
                    1,
                    ci_write_ptr
                );
            }   
            else
            {
                val_reader(std::get<0>(starting_nnz[r]),
                    std::get<1>(starting_nnz[r]), 
                    int64_t{},
                    1,
                    ci_write_ptr
                );
            }
            ci_write_ptr += std::get<1>(starting_nnz[r]);
        }
        data_offset += (file_has_int32_indices ? sizeof(int) : sizeof(int64_t))*(gnnz);

        // 3. reading matrix values
        auto mat_read_dispatcher = [&](auto target_mat_type, int bdim)
        {
            using target_val_type = decltype(target_mat_type);
            *data = malloc(num_local_nnz*bdim*sizeof(target_val_type));
            target_val_type* data_ptr = static_cast<target_val_type*>(*data);

            for (int r = 0; r < num_local_rows; r++)
            {
                if (file_matrix_type == bin_type<float>())
                {
                    val_reader(std::get<0>(starting_nnz[r]),
                        std::get<1>(starting_nnz[r]), 
                        float{},
                        bdim,
                        data_ptr
                    );
                }
                else
                {
                    val_reader(std::get<0>(starting_nnz[r]),
                        std::get<1>(starting_nnz[r]), 
                        double{},
                        bdim,
                        data_ptr
                    );
                }
                    
                data_ptr += std::get<1>(starting_nnz[r]) * bdimx * bdimy;
            }
            // sanity
            // data_ptr = static_cast<target_val_type*>(*data);
            // for (int i = 0; i < num_local_nnz; i++){
            //     log << partition_id << ": mtx_data: " << data_ptr[i] << std::endl << std::flush;
            // }
        };

        if (mat_prec == AMGX_matFloat){
            mat_read_dispatcher(float{}, bdimx*bdimy);
        }
        else if (mat_prec == AMGX_matDouble) {
            mat_read_dispatcher(double{}, bdimx*bdimy);
        }
        else 
        {
            FatalError("Unsupported value type for matrix in file", AMGX_ERR_IO);
        }
        data_offset += ((file_matrix_type == bin_type<float>()) ? sizeof(float) : sizeof(double))*(gnnz);
    } // end of reading matrix

    // read rhs if present
    if (is_rhs)
    {
        // 4. reading rhs vector
        auto rhs_read_dispatcher = [&](auto target_vec_type, int bdim)
        {
            using target_val_type = decltype(target_vec_type);
            *rhs = malloc(num_local_rows*bdim*sizeof(target_val_type));
            target_val_type* data_ptr = static_cast<target_val_type*>(*rhs);
 
            for (int r = 0; r < num_local_rows; r++)
            {
                if (file_vector_type == bin_type<float>())
                {
                    val_reader(local_rows[r],
                        1, 
                        float{},
                        bdim,
                        data_ptr
                    );
                }
                else
                {
                    val_reader(local_rows[r],
                        1, 
                        double{},
                        bdim,
                        data_ptr
                    );
                }
                data_ptr += bdim;
            }
            // sanity
            data_ptr = static_cast<target_val_type*>(*rhs);
            for (int i = 0; i < num_local_rows; i++){
                log << partition_id << ": rhs_data: " << data_ptr[i] << std::endl << std::flush;
            }
        };

        // only supported types right now
        assert(file_vector_type == bin_type<float>() || file_vector_type == bin_type<double>());
 
        if (vec_prec == AMGX_vecFloat){
            rhs_read_dispatcher(float{}, bdimy);
        }
        else if (vec_prec == AMGX_vecDouble) {
            rhs_read_dispatcher(double{}, bdimy);
        }
        else {
            FatalError("Unsupported value type for vector in file", AMGX_ERR_IO);
        }
        data_offset += ((file_vector_type == bin_type<float>()) ? sizeof(float) : sizeof(double))*(gnrows);
        /*if (partition_id == 0)
            read_dump_data(gnnz, double{});*/
    }

    // read rhs if present
    if (is_soln)
    {
        // 5. reading sol/initial guess vector
        auto sol_read_dispatcher = [&](auto target_vec_type, int bdim)
        {
            using target_val_type = decltype(target_vec_type);
            *sol = malloc(num_local_rows*bdim*sizeof(target_val_type));
            target_val_type* data_ptr = static_cast<target_val_type*>(*sol);

            for (int r = 0; r < num_local_rows; r++)
            {
                if (file_vector_type == bin_type<float>())
                    val_reader(local_rows[r],
                        1, 
                        float{},
                        bdim,
                        data_ptr
                    );
                else
                    val_reader(local_rows[r],
                        1, 
                        double{},
                        bdim,
                        data_ptr
                );
                data_ptr += bdim;
            }
            // sanity
            data_ptr = static_cast<target_val_type*>(*sol);
            for (int i = 0; i < num_local_rows; i++){
                log << partition_id << ": sol_data: " << data_ptr[i] << std::endl << std::flush;
            }
        };

        if (vec_prec == AMGX_vecFloat)
            sol_read_dispatcher(float{}, bdimx);
        else
            sol_read_dispatcher(double{}, bdimx);
        data_offset += ((file_vector_type == bin_type<float>()) ? sizeof(float) : sizeof(double))*(gnrows);
    }

    return 0;
 }
} // namespace  amgx
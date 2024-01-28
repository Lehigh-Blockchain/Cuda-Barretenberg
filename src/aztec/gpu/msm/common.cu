

#include "kernel.cu"
#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace pippenger_common {

/**
 * Execute bucket method
 */ 
template <class point_t, class scalar_t>
point_t* pippenger_t<point_t, scalar_t>::execute_bucket_method(
pippenger_t &config, scalar_t *scalars, point_t *points, unsigned bitsize, unsigned c, size_t npoints, cudaStream_t stream) {
    // Initialize dynamic cub_routines object
    config.params = new cub_routines();

    cout << "Entered bucket method execution." << endl;

    // Bucket initialization kernel
    thrust::host_vector<g1_gpu::element> bucketsHost;
    g1_gpu::element* bucketsRaw;
    //thrust::device_ptr<point_t> dptr = thrust::raw_pointer_cast(buckets.data());
    unsigned NUM_THREADS = 1 << 10; 

    unsigned NUM_BLOCKS = (config.num_buckets + NUM_THREADS - 1) / NUM_THREADS;
    //Fill buckets on host
    CUDA_WRAPPER(cudaMallocAsync(&bucketsRaw, config.num_buckets * 3 * 4 * sizeof(uint64_t), stream));
    cudaMemcpyAsync(bucketsHost.data(), bucketsRaw, sizeof(bucketsRaw)*sizeof(point_t), cudaMemcpyHostToHost, cudaStreamDefault);

    cout << "Size of raw buckets pointer: " << sizeof(bucketsRaw) << endl;

    cout << "Copied points from stream to host vector." << endl;

    // Use Thrust copy constructor to create a device vector to send over to the initialize buckets kernel
    thrust::device_vector<g1_gpu::element> deviceBuckets(sizeof(bucketsRaw) * sizeof(g1_gpu::element));
    // thrust::raw_pointer_cast(deviceBuckets.data());
    cudaMemcpyAsync(thrust::raw_pointer_cast(deviceBuckets.data()), bucketsRaw, sizeof(bucketsRaw)*sizeof(point_t),cudaMemcpyDeviceToDevice, cudaStreamDefault);
    cudaStreamSynchronize(cudaStreamDefault);
    cudaFree(bucketsRaw);

    cout << "Size of device buckets after raw pointer copying: " << deviceBuckets.size() << endl;
    //cout << "Size of initialized device buckets should be: " << (sizeof(bucketsRaw)*sizeof(point_t)) << endl;//for debugging

    cout << "Copied points from host vector to device vector." << endl;
    
    ///NB: Calling deviceBuckets.data() here is the same as saying thrust::device_ptr ptr = &deviceBuckets[0]; as in
    ///we retain information entered into deviceBuckets through passing the pointer
    initialize_buckets_kernel<<<NUM_BLOCKS * 4, NUM_THREADS, 0, stream>>>(thrust::raw_pointer_cast(deviceBuckets.data())); 

    cout << "Initialized buckets with the initialize_buckets_kernel" << endl;
    //g1_gpu::element element_b(deviceBuckets[0]);//for debugging
    //std::cout << "First Bucket Contents after init buckets kernel: " << ((g1_gpu::element)deviceBuckets[0]).x.data[0] << std::endl; //for debugging

    //cout << "First bucket contents: " << deviceBuckets[0][0].x.data[0] << endl;//still trying to debug


    // Scalars decomposition kernel
    CUDA_WRAPPER(cudaMallocAsync(&(params->bucket_indices), sizeof(unsigned) * npoints * (windows + 1), stream));
    CUDA_WRAPPER(cudaMallocAsync(&(params->point_indices), sizeof(unsigned) * npoints * (windows + 1), stream));
    split_scalars_kernel<<<NUM_POINTS / NUM_THREADS, NUM_THREADS, 0, stream>>>
        (params->bucket_indices + npoints, params->point_indices + npoints, scalars, npoints, windows, c);

    cout << "Split Scalars kernel launched" << endl;

    cout << "Size of device buckets after launching split scalars kernel: " << deviceBuckets.size() << endl;

    //for debugging
    g1_gpu::element element_a(deviceBuckets[0]);//convert point in first bucket on device to an element
   
    std::cout << "First Bucket Contents after slit scalars kernel: " << element_a.x.data[0] << std::endl; //print element that was point in first bucket on device to see if we are not initializing buckets => illegal memory access was encountered

    // Execute CUB routines for determining bucket sizes, offsets, etc. 
    execute_cub_routines(config, config.params, stream);

    cout << "Cub routines executed after Split Scalars Kernel" << endl;

    // Bucket accumulation kernel
    unsigned NUM_THREADS_2 = 1 << 8;
    unsigned NUM_BLOCKS_2 = ((config.num_buckets + NUM_THREADS_2 - 1) / NUM_THREADS_2) * 4;
    //thrust vector declaration
    
    //accumulate buckets call
    accumulate_buckets_kernel<<<NUM_BLOCKS_2, NUM_THREADS_2, 0, stream>>>
        (thrust::raw_pointer_cast(deviceBuckets.data()), params->bucket_offsets, params->bucket_sizes, params->single_bucket_indices, 
        params->point_indices, points, config.num_buckets);

    cout << "Accumulate Buckets Kernel launched" << endl;
    auto res2 = cudaGetLastError();
    cout << "Cuda Error After Accumulate Buckets: " << res2 << endl;

    cout << "Size of device buckets: " << deviceBuckets.size() << endl;

    // Running sum kernel
    point_t *final_sum;
    CUDA_WRAPPER(cudaMallocAsync(&final_sum, windows * 3 * 4 * sizeof(uint64_t), stream));
    bucket_running_sum_kernel<<<26, 4, 0, stream>>>(thrust::raw_pointer_cast(deviceBuckets.data()), final_sum, c);

    // for (int i = 0; i < 32; i++) {
    //     cout << &final_sum[i] << endl;
    // }

    cout << "Bucket Running Sum kernel lauched" << endl;
    auto res3 = cudaGetLastError();
    cout << "Cuda Error After Bucket Running Sum: " << res3 << endl;

    // Final accumulation kernel
    point_t *res;
    CUDA_WRAPPER(cudaMallocManaged(&res, 3 * 4 * sizeof(uint64_t)));
    final_accumulation_kernel<<<1, 1, 0, stream>>>(final_sum, res, windows, c);

    cout << "Final Accumulation kernel launched" << endl;
    
    // Synchronize stream
    cudaStreamSynchronize(stream);

    cout << "Synchronizing Cuda Stream" << endl;

    // Check for errors codes
    auto res1 = cudaGetLastError();
    cout << "Cuda Error Code: " << res1 << endl;

    cout << "Checking for errors" << endl;

    // Free host and device memory 
    CUDA_WRAPPER(cudaFreeHost(points));
    CUDA_WRAPPER(cudaFreeHost(scalars));
    //CUDA_WRAPPER(cudaFreeAsync(params->buckets, stream));
    CUDA_WRAPPER(cudaFreeAsync(params->bucket_indices, stream));
    CUDA_WRAPPER(cudaFreeAsync(params->point_indices, stream));
    CUDA_WRAPPER(cudaFreeAsync(params->sort_indices_temp_storage, stream));
    CUDA_WRAPPER(cudaFreeAsync(params->single_bucket_indices, stream));
    CUDA_WRAPPER(cudaFreeAsync(params->bucket_sizes, stream));
    CUDA_WRAPPER(cudaFreeAsync(params->nof_buckets_to_compute, stream));
    CUDA_WRAPPER(cudaFreeAsync(params->encode_temp_storage, stream));
    CUDA_WRAPPER(cudaFreeAsync(params->bucket_offsets, stream));
    CUDA_WRAPPER(cudaFreeAsync(params->offsets_temp_storage, stream));
    CUDA_WRAPPER(cudaFree(final_sum));
    CUDA_WRAPPER(cudaFree(res));

    cout << "Freeing memory\nReturning MSM result" << endl;

    return res;
}

/**
 * CUB routines referenced from: https://github.com/ingonyama-zk/icicle (inspired by zkSync's era-bellman-cuda library)
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::execute_cub_routines(pippenger_t &config, cub_routines *params, cudaStream_t stream) {
    // Radix sort algorithm
    size_t sort_indices_temp_storage_bytes; 
    cub::DeviceRadixSort::SortPairs(params->sort_indices_temp_storage, sort_indices_temp_storage_bytes, params->bucket_indices 
                                    + npoints, params->bucket_indices, params->point_indices + npoints, params->point_indices, 
                                    npoints, 0, sizeof(unsigned) * 8, stream);
    CUDA_WRAPPER(cudaMallocAsync(&(params->sort_indices_temp_storage), sort_indices_temp_storage_bytes, stream));
    for (unsigned i = 0; i < config.windows; i++) {
        unsigned offset_out = i * npoints;
        unsigned offset_in = offset_out + npoints;
        cub::DeviceRadixSort::SortPairs(params->sort_indices_temp_storage, sort_indices_temp_storage_bytes, params->bucket_indices 
                                        + offset_in, params->bucket_indices + offset_out, params->point_indices + offset_in, 
                                        params->point_indices + offset_out, npoints, 0, sizeof(unsigned) * 8, stream);
    }

    // Perform length encoding
    CUDA_WRAPPER(cudaMallocAsync(&(params->single_bucket_indices), sizeof(unsigned) * config.num_buckets, stream));

    // TODO: THIS ALLOCATION NEEDS TO BE CHANGED AND WILL VARY RUNTIME OF PIPPENGER FOR SOME REASON
    CUDA_WRAPPER(cudaMallocAsync(&(params->bucket_sizes), sizeof(unsigned) * config.num_buckets * config.num_buckets, stream));
    CUDA_WRAPPER(cudaMallocAsync(&(params->nof_buckets_to_compute), sizeof(unsigned), stream));
    size_t encode_temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(params->encode_temp_storage, encode_temp_storage_bytes, params->bucket_indices, 
                                       params->single_bucket_indices, params->bucket_sizes, params->nof_buckets_to_compute, 
                                       config.windows * npoints, stream);
    CUDA_WRAPPER(cudaMallocAsync(&(params->encode_temp_storage), encode_temp_storage_bytes, stream));
    cub::DeviceRunLengthEncode::Encode(params->encode_temp_storage, encode_temp_storage_bytes, params->bucket_indices, 
                                       params->single_bucket_indices, params->bucket_sizes, params->nof_buckets_to_compute, 
                                       config.windows * npoints, stream);

    // Calculate bucket offsets
    CUDA_WRAPPER(cudaMallocAsync(&(params->bucket_offsets), sizeof(unsigned) * config.num_buckets, stream));
    size_t offsets_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(params->offsets_temp_storage, offsets_temp_storage_bytes, params->bucket_sizes, 
                                  params->bucket_offsets, config.num_buckets, stream);
    CUDA_WRAPPER(cudaMallocAsync(&(params->offsets_temp_storage), offsets_temp_storage_bytes, stream));
    cub::DeviceScan::ExclusiveSum(params->offsets_temp_storage, offsets_temp_storage_bytes, params->bucket_sizes, 
                                  params->bucket_offsets, config.num_buckets, stream);
}

/**
 * Calculate number of windows and buckets
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::calculate_windows(pippenger_t &config, size_t npoints) {
    config.windows = BITSIZE / C; 
    if (BITSIZE % C) {  
        windows++;
    }
    config.num_buckets = windows << C; 
    config.npoints = npoints;
}

/**
 * Calculate the amount of device storage required to store bases 
 */
template <class point_t, class scalar_t>
size_t pippenger_t<point_t, scalar_t>::get_size_bases(pippenger_t &config) {
    return NUM_POINTS * sizeof(point_t);
}

/**
 * Calculate the amount of device storage required to store scalars 
 */
template <class point_t, class scalar_t>
size_t pippenger_t<point_t, scalar_t>::get_size_scalars(pippenger_t &config) {
    // return config.n * sizeof(scalar_t);
    return NUM_POINTS * sizeof(scalar_t);
}

/**
 * Allocate device storage for bases
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::allocate_bases(pippenger_t &config) {
    device_base_ptrs.allocate(get_size_bases(config));
}

/**
 * Allocate device storage for scalars
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::allocate_scalars(pippenger_t &config) {
    device_scalar_ptrs.allocate(get_size_scalars(config));
}

/**
 * Transfer base points to GPU device
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::transfer_bases_to_device(
pippenger_t &config, point_t *device_bases_ptrs, const point_t *points, cudaStream_t stream) {    
    CUDA_WRAPPER(cudaMemcpyAsync(device_bases_ptrs, points, NUM_POINTS * LIMBS * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
}

/**
 * Transfer scalars to GPU device
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::transfer_scalars_to_device(
pippenger_t &config, scalar_t *device_scalar_ptrs, fr *scalars, cudaStream_t stream) {
    CUDA_WRAPPER(cudaMemcpyAsync(device_scalar_ptrs, scalars, NUM_POINTS * LIMBS * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
}

/**
 * Allocate pinned memory using cudaMallocHost
 */
template <class T>
void device_ptr<T>::allocate(size_t bytes) {
    T* d_ptr;
    CUDA_WRAPPER(cudaMallocHost(&d_ptr, bytes));
    d_ptrs.push_back(d_ptr);
}

/**
 * Get size of d_ptrs vector
 */
template <class T>
size_t device_ptr<T>::size() {
    return d_ptrs.size();
}

/**
 * Operator overloading for device_ptr indexing
 */
template <class T>
T* device_ptr<T>::operator[](size_t i) {
    if (i > d_ptrs.size() - 1) {
        cout << "Indexing error!" << endl;
        throw;
    }
    return d_ptrs[i];
}

/**
 * Verify results
 */ 
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::verify_result(point_t *result_1, point_t **result_2) {
    var *result;
    CUDA_WRAPPER(cudaMallocManaged(&result, LIMBS * sizeof(uint64_t)));
    comparator_kernel<<<1, 4>>>(result_1, result_2[0], result);
    cudaDeviceSynchronize();

    assert (result[0] == 1);
    assert (result[1] == 1);
    assert (result[2] == 1);
    assert (result[3] == 1);

    cout << "MSM Result Verified!" << endl;
}

/**
 * Print results
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::print_result(g1_gpu::element *result_1, g1_gpu::element **result_2) {
    for (int i = 0; i < LIMBS; i++) {
        printf("result_naive_msm is: %zu\n", result_1[0].x.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result_naive_msm is: %zu\n", result_1[0].y.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result_naive_msm is: %zu\n", result_1[0].z.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result_bucket_method_msm is: %zu\n", result_2[0][0].x.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result_bucket_method_msm is: %zu\n", result_2[0][0].y.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result_bucket_method_msm is: %zu\n", result_2[0][0].z.data[i]);
    }
}

}
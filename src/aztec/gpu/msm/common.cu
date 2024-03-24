

#include "kernel.cu"
#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace pippenger_common {

point_t host_buckets[26624 * sizeof(point_t)];
unsigned bucket_offsets_host[26624];

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
    point_t *buckets;
    
    

    unsigned NUM_THREADS = 1 << 10; 

    unsigned NUM_BLOCKS = (config.num_buckets + NUM_THREADS - 1) / NUM_THREADS;
    //Fill buckets on host
    CUDA_WRAPPER(cudaMallocAsync(&buckets, config.num_buckets * 3 * 4 * sizeof(uint64_t), stream));
    



    cout << "Copied points from host vector to device vector." << endl;
    
    ///NB: Calling deviceBuckets.data() here is the same as saying thrust::device_ptr ptr = &deviceBuckets[0]; as in
    ///we retain information entered into deviceBuckets through passing the pointer
    initialize_buckets_kernel<<<NUM_BLOCKS * 4, NUM_THREADS, 0, stream>>>(buckets); ///*thrust::raw_pointer_cast(deviceBuckets.data())*/ was used previously

    cout << "Buckets initialized, printing..." << endl;

    cout << "Size of buckets: " << num_buckets << endl; 
    cout << "Size of point_t: " << sizeof(point_t) << endl;

    cout << "Size of host buckets: " << sizeof(host_buckets) << endl;

    transfer_field_elements_to_host(config, host_buckets, buckets, stream);
   
    cout << "Slob on my knob." << endl;

    auto error = cudaGetLastError();
    cout << "Cuda Error After Bucket Running Sum: " << error << endl;

    

    // for (int i = 0; i < sizeof(config.bucket_offsets); i++) {
    //     cout << "Bucket " << i << " with data: ";
    //     for (int j = 0; j < bucket_offsets[i]; j++) {
    //         cout << "Point " << j << " in bucket " << i << " with x: ";
    //         for (int j = 0; j < 4; j++) {
    //             if (host_buckets[i].x.data[j] != 0) {
    //                 cout << host_buckets[i].x.data[j] << " " << endl;
    //             }
    //         }
    //         cout << ", y: ";
    //         for (int j = 0; j < 4; j++) {
    //             if (host_buckets[i].y.data[j] != 0) {
    //                 cout << host_buckets[i].y.data[j] << " " << endl;
    //             }
    //         }
    //         cout << ", z: ";
    //         for (int j = 0; j < 4; j++) {
    //             if (host_buckets[i].z.data[j] != 0) {
    //                 cout << host_buckets[i].z.data[j] << " " << endl;
    //             }
    //         }
    //         cout << endl;
    //     }
    // }

    // for (int i = 0; i < sizeof(host_buckets); i++) {
    //     cout << "Bucket " << i << " with data: ";
    //         cout << "x: ";
    //         for (int j = 0; j < 4; j++) {
    //             if (host_buckets[i].x.data[j] != 0) {
    //                 cout << host_buckets[i].x.data[j] << " " << endl;
    //             }
    //         }
    //         cout << ", y: ";
    //         for (int j = 0; j < 4; j++) {
    //             if (host_buckets[i].y.data[j] != 0) {
    //                 cout << host_buckets[i].y.data[j] << " " << endl;
    //             }
    //         }
    //         cout << ", z: ";
    //         for (int j = 0; j < 4; j++) {
    //             if (host_buckets[i].z.data[j] != 0) {
    //                 cout << host_buckets[i].z.data[j] << " " << endl;
    //             }
    //         }
    //         cout << endl;
    // }
    
    cout << "Initialized buckets with the initialize_buckets_kernel" << endl;

    // Scalars decomposition kernel
    CUDA_WRAPPER(cudaMallocAsync(&(params->bucket_indices), sizeof(unsigned) * npoints * (windows + 1), stream));
    CUDA_WRAPPER(cudaMallocAsync(&(params->point_indices), sizeof(unsigned) * npoints * (windows + 1), stream));
    split_scalars_kernel<<<NUM_POINTS / NUM_THREADS, NUM_THREADS, 0, stream>>>
        (params->bucket_indices + npoints, params->point_indices + npoints, scalars, npoints, windows, c);

    cout << "Split Scalars kernel launched" << endl;

    cout << "Lauching cub routines" << endl;

    //auto res2 = cudaGetLastError();
    //cout << "Cuda Error After Cub Routines: " << res2 << endl;
    
    // Execute CUB routines for determining bucket sizes, offsets, etc. 
    execute_cub_routines(config, config.params, stream);

    cout << "slob on my cob" << endl;

    transfer_offsets_to_host(config, bucket_offsets_host, params->bucket_offsets, stream);

    // for (int i = 0; i < sizeof(bucket_offsets_host); i++) {
    //     cout << "Offset for bucket " << i << ": " << bucket_offsets_host[i] << endl;;
    // }

    cout << "Size of bucket offsets: " << sizeof(bucket_offsets_host) << endl;

    cout << "Cub routines executed after Split Scalars Kernel" << endl;

    // Bucket accumulation kernel
    unsigned NUM_THREADS_2 = 1 << 6;
    unsigned NUM_BLOCKS_2 = ((config.num_buckets + NUM_THREADS_2 - 1) / NUM_THREADS_2);
    //thrust vector declaration
    

    cout << "Entering Thrust implementation" << endl;


    //THRUST STUFF NOW
    thrust::host_vector<g1_gpu::element> bucketsHost;
    //thrust::device_ptr<point_t> dptr = thrust::raw_pointer_cast(&buckets);
    cudaMemcpyAsync(bucketsHost.data(), buckets, sizeof(buckets)*sizeof(point_t), cudaMemcpyHostToHost, cudaStreamDefault);

    //cout << "Size of raw buckets pointer: " << sizeof(buckets) << endl;

    cout << "Copied points from stream to host vector." << endl;

    // Use Thrust copy constructor to create a device vector to send over to the initialize buckets kernel
    thrust::device_vector<point_t> deviceBuckets(sizeof(buckets) * sizeof(g1_gpu::element));//this may have to be changed back from point_t to element
    // thrust::raw_pointer_cast(deviceBuckets.data());
    cudaMemcpyAsync(thrust::raw_pointer_cast(deviceBuckets.data()), buckets, sizeof(buckets)*sizeof(point_t),cudaMemcpyDeviceToDevice, cudaStreamDefault);
    cudaStreamSynchronize(cudaStreamDefault);
    //cudaFree(bucketsRaw);



    //accumulate buckets call
    accumulate_buckets_kernel<<<NUM_BLOCKS_2, NUM_THREADS_2, 0, stream>>>
        (thrust::raw_pointer_cast(deviceBuckets.data()), params->bucket_offsets, params->bucket_sizes, params->single_bucket_indices, 
        params->point_indices, points, config.num_buckets);

    cout << "Accumulate Buckets Kernel launched" << endl;
    


    //CONVERT THRUST BACK TO RAW POINTERS
    buckets = thrust::raw_pointer_cast(&deviceBuckets[0]);//conversion back to raw pointer for buckets
    ///NB: no need to free memory occupied by deviceBuckets as it will be done automatically when common is no longer in scope
    cout << "Buckets size after raw pointer cast: " << sizeof(*buckets) * sizeof(point_t) << endl;//for debugging
    ///NB: this could be a problem


    // Running sum kernel
    point_t *final_sum;
    cout << "Windows: " << windows << endl;
    CUDA_WRAPPER(cudaMallocAsync(&final_sum, windows * 3 * 4 * sizeof(uint64_t), stream));
    bucket_running_sum_kernel<<<26, 4, 0, stream>>>(buckets, final_sum, c, config.num_buckets);

    cudaStreamSynchronize(stream);

    auto res23 = cudaGetLastError();
    cout << "Cuda Error After Accumulate Buckets: " << res23 << endl;

    cudaStreamSynchronize(stream);

    cout << "Bucket Running Sum kernel lauched" << endl;
    auto res3 = cudaGetLastError();
    cout << "Cuda Error After Bucket Running Sum: " << res3 << endl;

    cudaStreamSynchronize(stream);

    // Final accumulation kernel
    point_t *res;
    CUDA_WRAPPER(cudaMallocManaged(&res, 3 * 4 * sizeof(uint64_t)));
    final_accumulation_kernel<<<1, 1, 0, stream>>>(final_sum, res, windows, c);

    cout << "Final Accumulation kernel launched" << endl;

    cudaStreamSynchronize(stream);

    auto res4 = cudaGetLastError();
    cout << "Cuda Error After Final Accumulation: " << res4 << endl;
    
    // Synchronize stream
    cudaStreamSynchronize(stream);

    cout << "Synchronizing Cuda Stream" << endl;

    // Check for errors codes
    auto res1 = cudaGetLastError();
    cout << "Cuda Error Code After Sychronization of Stream: " << res1 << endl;

    cout << "Checking for errors" << endl;


    //Free host and device memory 
    CUDA_WRAPPER(cudaFreeHost(points));
    CUDA_WRAPPER(cudaFreeHost(scalars));
    CUDA_WRAPPER(cudaFreeAsync(buckets, stream));
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
 * Transfer base points to CPU device
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::transfer_bases_to_host(
pippenger_t &config, point_t *device_bases_ptrs, const point_t *point_buckets, cudaStream_t stream) {    
    CUDA_WRAPPER(cudaMemcpyAsync(device_bases_ptrs, point_buckets, NUM_POINTS * LIMBS * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
}

/**
 * Transfer bucket offsets to CPU device
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::transfer_offsets_to_host(
pippenger_t &config, unsigned *host_offsets, unsigned *device_offsets, cudaStream_t stream) {    
    CUDA_WRAPPER(cudaMemcpyAsync(host_offsets, device_offsets, num_buckets * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
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
 * Transfer field elements to host device for debugging purposes
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::transfer_field_elements_to_host(
pippenger_t &config, point_t* host_buckets, point_t* buckets, cudaStream_t stream) {
    CUDA_WRAPPER(cudaMemcpyAsync(host_buckets, buckets, num_buckets * sizeof(point_t), cudaMemcpyDeviceToHost, stream)); // multiply by 4 to account for the 4 elements being printed in field.cuh
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

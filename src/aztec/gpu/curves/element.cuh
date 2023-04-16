#include <cstdint>
#include <stdio.h>
#include <cstdio>
#include <chrono>
#include <iostream>

using namespace std;

namespace gpu_barretenberg {
namespace group_elements { 
    
/* -------------------------- Jacobian Representation ---------------------------------------------- */

/**
 * Implements elliptic curve group arithmetic using Jacobian coordinates
 */
template < typename fq_gpu, typename fr_gpu> 
class element {
    public:    
        fq_gpu x, y, z;

        __device__ element() noexcept {}
        
        __device__ element(const fq_gpu &a, const fq_gpu &b, const fq_gpu &c) noexcept;
        
        __device__ element(const element& other) noexcept;
};

/* -------------------------- Affine Representation ---------------------------------------------- */

/**
 * Implements elliptic curve group arithmetic using Affine coordinates
 */
template < typename fq_gpu, typename fr_gpu> 
class affine_element {
    public:    
        fq_gpu x, y;

        __device__ affine_element() noexcept {}

        __device__ affine_element(const fq_gpu &a, const fq_gpu &b) noexcept;

        __device__ affine_element(const affine_element &other) noexcept;        
};

/* -------------------------- Projective Coordinate Representation ---------------------------------------------- */

/**
 * Implements elliptic curve group arithmetic using Projective coordinates
 */
template < typename fq_gpu, typename fr_gpu> 
class projective_element {
    public:    
        fq_gpu x, y, z;

        __device__ projective_element() noexcept {}

        __device__ projective_element(const fq_gpu &a, const fq_gpu &b, const fq_gpu &c) noexcept;

        __device__ projective_element(const projective_element &other) noexcept;  
};

}
}
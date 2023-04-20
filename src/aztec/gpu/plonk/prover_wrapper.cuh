#include "kzg_wrapper.cu"
#include "queue_wrapper.cu"

using namespace std;
using namespace waffle;
using namespace kzg_gpu_wrapper;

namespace prover_gpu_wrapper {

/**
 * Polymorphic 'ProverWrapper' class that represents the top-level prover functions and 
 * derives the 'ProverBase' base class.
 */
class ProverWrapper : public ProverBase<standard_settings> {
    public:    
        virtual plonk_proof& construct_proof() override;
        virtual void execute_first_round() override;
};

// Might need to inherent ProverWrapper from KzgWrapper.

}
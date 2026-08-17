// Spin-1 transverse projector for the (+---) metric used by FV and tensor.
#ifndef CTPWA_FRAMEWORK_TENSORS_SPIN_PROJECTOR_CUH
#define CTPWA_FRAMEWORK_TENSORS_SPIN_PROJECTOR_CUH

#include "framework/tensors/Tensor.cuh"

namespace ctpwa {

__device__ inline tensor transverse_projector(const FV& momentum)
{
    return tensor::Gnormal()
           - tensor(momentum, momentum) / (momentum * momentum);
}

} // namespace ctpwa

#endif // CTPWA_FRAMEWORK_TENSORS_SPIN_PROJECTOR_CUH

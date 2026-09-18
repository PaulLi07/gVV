#pragma once

#include "core/math/TensorOps.cuh"

// Reusable covariant orbital tensors built from a parent momentum and a
// relative daughter momentum. These functions return bare STF geometry;
// Blatt-Weisskopf and LS-normalization factors belong to the process Wave.

namespace ctpwa {

__device__ inline FV orbital_pwave(
    const FV& parent,
    const FV& relative_momentum)
{
    return transverse_projector(parent) * relative_momentum;
}

__device__ inline tensor orbital_dwave(
    const FV& parent,
    const FV& relative_momentum)
{
    const tensor projector = transverse_projector(parent);
    const FV projected = projector * relative_momentum;
    return tensor(projected, projected)
           - projector * ((projector * relative_momentum) * relative_momentum)
                 / 3.0;
}

// Contract the bare rank-four G-wave orbital tensor with a rank-two source:
//
//   result^{mu nu} = t_bare^(4)^{mu nu lambda tau}
//                    source_{lambda tau}.
//
// A standalone rank-four object is intentionally avoided. Projecting source
// to spin two does not change this contraction because t_bare^(4) is itself
// symmetric, parent-transverse, and traceless in every index pair. The outer
// spin-two projector sometimes written around this result is therefore
// redundant. No B4 or orbital/CG normalization is included here.
__device__ inline tensor contract_orbital_gwave(
    const FV& parent,
    const FV& relative_momentum,
    const tensor& source)
{
    const tensor projector = transverse_projector(parent);
    const FV projected = projector * relative_momentum;
    const double projected2 = projected * projected;
    const tensor spin2_source = spin2_project(parent, source);
    const FV source_times_projected = spin2_source * projected;
    const double scalar = projected * source_times_projected;

    const tensor mixed = tensor(projected, source_times_projected)
                         + tensor(source_times_projected, projected);

    return tensor(projected, projected) * scalar
           - (projector * scalar + mixed * 2.0) * (projected2 / 7.0)
           + spin2_source
                 * (2.0 * projected2 * projected2 / 35.0);
}

} // namespace ctpwa

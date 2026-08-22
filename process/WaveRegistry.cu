// Host catalogue for complete GVV Waves. Model and propagator compilation are
// deliberately kept out of this registration unit.
#include "process/WaveRegistry.cuh"

#include <stdexcept>

const std::vector<GVVWaveMetadata>& gvv_wave_registry()
{
    // This is the only host registration point for complete GVV waves. The
    // matching device dispatch is the single gvv_wave_tensor function in
    // WaveRegistry.cuh; individual formulae stay in process/waves/.
    static const std::vector<GVVWaveMetadata> registry = {
        {"gvv.scalar_00", "0++", "0^{++}(00)", "scalar", GVV_SCALAR_00},
        {"gvv.scalar_22", "0++", "0^{++}(22)", "scalar", GVV_SCALAR_22},
        {"gvv.pseudoscalar_11", "0-+", "0^{-+}(11)",
         "pseudoscalar", GVV_PSEUDOSCALAR_11}
    };
    return registry;
}

const GVVWaveMetadata& gvv_registered_wave(const std::string& id)
{
    for (const GVVWaveMetadata& wave : gvv_wave_registry()) {
        if (wave.id == id) {
            return wave;
        }
    }
    throw std::runtime_error(
        "model references unregistered GVV wave '" + id + "'");
}

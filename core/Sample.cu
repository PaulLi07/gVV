#include "core/AmplitudeKernels.cuh"
#include "core/Sample.h"
#include "TFile.h"
#include "TTree.h"
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

// Reads the configured ROOT tree into host [px,py,pz,E] arrays, uploads them,
// and caches each sample's parameter-independent Wave Gram matrix.

namespace {

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

GVVBranchConfig::GVVBranchConfig()
    : tree_name("Pwa"),
      branches{{
          "p4_pip1", "p4_pim1", "p4_pi01",
          "p4_pip2", "p4_pim2", "p4_pi02", "p4_gam"}},
      input_order(GVV_PX_PY_PZ_E)
{
}

GVVSample::GVVSample(const std::string& label)
    : label_(label),
      entries_(0),
      F_matrix_(nullptr),
      wave_coefficients_(nullptr),
      amp2_(nullptr),
      omega_factors_(nullptr),
      omega_factor_sigma_(-1.0),
      number_active_waves_(0),
      number_terms_(0)
{
    device_p4_.fill(nullptr);
}

GVVSample::~GVVSample()
{
    if (omega_factors_ != nullptr) cudaFree(omega_factors_);
    for (double* pointer : device_p4_) {
        if (pointer != nullptr) {
            cudaFree(pointer);
        }
    }
    if (F_matrix_ != nullptr) {
        cudaFree(F_matrix_);
    }
    if (wave_coefficients_ != nullptr) {
        cudaFree(wave_coefficients_);
    }
    if (amp2_ != nullptr) {
        cudaFree(amp2_);
    }
}

void GVVSample::Load(
    const std::string& file_name,
    const GVVBranchConfig& branches)
{
    if (entries_ != 0 || F_matrix_ != nullptr || amp2_ != nullptr) {
        throw std::runtime_error("sample has already been loaded: " + label_);
    }
    if (branches.tree_name.empty()
        || (branches.input_order != GVV_PX_PY_PZ_E
            && branches.input_order != GVV_E_PX_PY_PZ)) {
        throw std::invalid_argument("invalid GVV ROOT branch configuration");
    }

    std::unordered_set<std::string> branch_names;
    for (const std::string& branch : branches.branches) {
        if (branch.empty() || !branch_names.insert(branch).second) {
            throw std::invalid_argument(
                "GVV ROOT branch names must be non-empty and unique");
        }
    }

    TFile input(file_name.c_str(), "READ");
    if (input.IsZombie()) {
        throw std::runtime_error("cannot open ROOT file: " + file_name);
    }
    TTree* tree = dynamic_cast<TTree*>(input.Get(branches.tree_name.c_str()));
    if (tree == nullptr) {
        throw std::runtime_error(
            "tree '" + branches.tree_name + "' is missing in " + file_name);
    }

    double values[GVV_NFINAL_PARTICLES][4] = {{0.0}};
    for (int particle = 0; particle < GVV_NFINAL_PARTICLES; ++particle) {
        const std::string& branch = branches.branches[particle];
        if (tree->GetBranch(branch.c_str()) == nullptr) {
            throw std::runtime_error(
                "branch '" + branch + "' is missing in " + file_name);
        }
        if (tree->SetBranchAddress(branch.c_str(), values[particle]) < 0) {
            throw std::runtime_error(
                "cannot bind branch '" + branch + "' in " + file_name);
        }
    }

    const Long64_t number_entries = tree->GetEntries();
    if (number_entries <= 0
        || number_entries
               > static_cast<Long64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("invalid entry count in " + file_name);
    }

    entries_ = static_cast<int>(number_entries);
    for (int particle = 0; particle < GVV_NFINAL_PARTICLES; ++particle) {
        host_p4_[particle].resize(
            static_cast<std::size_t>(number_entries) * 4);
    }

    for (Long64_t event = 0; event < number_entries; ++event) {
        if (tree->GetEntry(event) <= 0) {
            throw std::runtime_error(
                "cannot read event " + std::to_string(event)
                + " from " + file_name);
        }
        const std::size_t offset = static_cast<std::size_t>(event) * 4;
        for (int particle = 0; particle < GVV_NFINAL_PARTICLES; ++particle) {
            double* destination = host_p4_[particle].data() + offset;
            if (branches.input_order == GVV_PX_PY_PZ_E) {
                for (int component = 0; component < 4; ++component) {
                    destination[component] = values[particle][component];
                }
            } else {
                destination[0] = values[particle][1];
                destination[1] = values[particle][2];
                destination[2] = values[particle][3];
                destination[3] = values[particle][0];
            }
        }
    }

    std::cout << "Loaded " << label_ << ": " << entries_
              << " events from " << file_name << '\n';
}

void GVVSample::UploadAndBuildF(
    const std::vector<int>& active_wave_types,
    int number_terms)
{
    if (entries_ <= 0) {
        throw std::runtime_error("cannot upload empty sample " + label_);
    }
    if (F_matrix_ != nullptr || wave_coefficients_ != nullptr
        || amp2_ != nullptr) {
        throw std::runtime_error("sample has already been prepared: " + label_);
    }
    if (active_wave_types.empty() || number_terms <= 0) {
        throw std::invalid_argument(
            "model has no active wave or Term for sample " + label_);
    }
    number_active_waves_ = static_cast<int>(active_wave_types.size());
    number_terms_ = number_terms;

    for (int particle = 0; particle < GVV_NFINAL_PARTICLES; ++particle) {
        const std::size_t bytes = host_p4_[particle].size() * sizeof(double);
        check_cuda(
            cudaMallocManaged(&device_p4_[particle], bytes),
            "cudaMallocManaged sample momentum");
        check_cuda(
            cudaMemcpy(
                device_p4_[particle],
                host_p4_[particle].data(),
                bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy sample momentum");
    }

    check_cuda(
        cudaMallocManaged(
            &F_matrix_,
            static_cast<std::size_t>(entries_)
                * number_active_waves_ * number_active_waves_
                * sizeof(double)),
        "cudaMallocManaged GVV F matrix");
    check_cuda(
        cudaMallocManaged(
            &wave_coefficients_,
            static_cast<std::size_t>(entries_) * number_active_waves_
                * sizeof(DeviceComplex)),
        "cudaMallocManaged GVV Wave coefficient workspace");
    check_cuda(
        cudaMallocManaged(
            &amp2_, static_cast<std::size_t>(entries_) * sizeof(double)),
        "cudaMallocManaged GVV intensity");
    int* device_wave_types = nullptr;
    try {
        check_cuda(
            cudaMallocManaged(
                &device_wave_types,
                active_wave_types.size() * sizeof(int)),
            "cudaMallocManaged active GVV waves");
        check_cuda(
            cudaMemcpy(
                device_wave_types,
                active_wave_types.data(),
                active_wave_types.size() * sizeof(int),
                cudaMemcpyHostToDevice),
            "cudaMemcpy active GVV waves");
        CalGVVFmatrix(
            Momenta(),
            device_wave_types,
            number_active_waves_,
            F_matrix_,
            entries_);
        check_cuda(cudaFree(device_wave_types), "cudaFree active GVV waves");
    } catch (...) {
        if (device_wave_types != nullptr) {
            cudaFree(device_wave_types);
        }
        throw;
    }
}

const std::string& GVVSample::Label() const
{
    return label_;
}

int GVVSample::Entries() const
{
    return entries_;
}

const double* GVVSample::HostMomentum(int particle, int event) const
{
    // Projection traversal owns these loop bounds; avoid rechecking them for
    // every particle of every event after the sample has been loaded.
    return host_p4_[particle].data()
           + static_cast<std::size_t>(event) * 4;
}

GVVDeviceMomenta GVVSample::Momenta() const
{
    return GVVDeviceMomenta(
        device_p4_[0], device_p4_[1], device_p4_[2],
        device_p4_[3], device_p4_[4], device_p4_[5], device_p4_[6]);
}

const double* GVVSample::FMatrix() const
{
    return F_matrix_;
}

DeviceComplex* GVVSample::WaveCoefficientBuffer()
{
    return wave_coefficients_;
}

double* GVVSample::IntensityBuffer()
{
    return amp2_;
}

int GVVSample::NumberActiveWaves() const
{
    return number_active_waves_;
}

int GVVSample::NumberTerms() const
{
    return number_terms_;
}

void GVVSample::UpdateOmegaFactors(ctpwa::TabulatedFunctionView width_table, double sigma)
{
    if (sigma == omega_factor_sigma_) return;
    if (sigma != 0.0) {
        if (omega_factors_ == nullptr) {
            check_cuda(cudaMallocManaged(&omega_factors_,
                static_cast<std::size_t>(entries_) * sizeof(DeviceComplex)),
                "cudaMallocManaged common omega factors");
        }
        CalGVVOmegaFactors(Momenta(), width_table, sigma, omega_factors_, entries_);
    }
    omega_factor_sigma_ = sigma;
}

const DeviceComplex* GVVSample::OmegaFactorBuffer() const
{
    return omega_factor_sigma_ > 0.0 ? omega_factors_ : nullptr;
}

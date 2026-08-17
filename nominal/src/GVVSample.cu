#include "../include/GVVSample.h"

#include "TFile.h"
#include "TTree.h"

#include <cuda_runtime.h>

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

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
      term_coefficients_(nullptr),
      amp2_(nullptr),
      number_active_waves_(0),
      number_terms_(0)
{
    device_p4_.fill(nullptr);
}

GVVSample::~GVVSample()
{
    for (double* pointer : device_p4_) {
        if (pointer != nullptr) {
            cudaFree(pointer);
        }
    }
    if (F_matrix_ != nullptr) {
        cudaFree(F_matrix_);
    }
    if (term_coefficients_ != nullptr) {
        cudaFree(term_coefficients_);
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

    TFile input(file_name.c_str(), "READ");
    if (input.IsZombie()) {
        throw std::runtime_error("cannot open ROOT file: " + file_name);
    }
    TTree* tree = dynamic_cast<TTree*>(input.Get(branches.tree_name.c_str()));
    if (tree == nullptr) {
        throw std::runtime_error(
            "tree '" + branches.tree_name + "' is missing in " + file_name);
    }

    double values[7][4] = {{0.0}};
    for (int particle = 0; particle < 7; ++particle) {
        const std::string& branch = branches.branches[particle];
        if (tree->GetBranch(branch.c_str()) == nullptr) {
            throw std::runtime_error(
                "branch '" + branch + "' is missing in " + file_name);
        }
        tree->SetBranchAddress(branch.c_str(), values[particle]);
    }

    const Long64_t number_entries = tree->GetEntries();
    if (number_entries <= 0
        || number_entries
               > static_cast<Long64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("invalid entry count in " + file_name);
    }

    entries_ = static_cast<int>(number_entries);
    for (int particle = 0; particle < 7; ++particle) {
        host_p4_[particle].resize(
            static_cast<std::size_t>(number_entries) * 4);
    }

    for (Long64_t event = 0; event < number_entries; ++event) {
        tree->GetEntry(event);
        const std::size_t offset = static_cast<std::size_t>(event) * 4;
        for (int particle = 0; particle < 7; ++particle) {
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
    if (F_matrix_ != nullptr || term_coefficients_ != nullptr
        || amp2_ != nullptr) {
        throw std::runtime_error("sample has already been prepared: " + label_);
    }
    if (active_wave_types.empty() || number_terms <= 0) {
        throw std::invalid_argument(
            "model has no active wave or Term for sample " + label_);
    }
    number_active_waves_ = static_cast<int>(active_wave_types.size());
    number_terms_ = number_terms;

    for (int particle = 0; particle < 7; ++particle) {
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
            &term_coefficients_,
            static_cast<std::size_t>(entries_) * number_terms_
                * sizeof(DeviceComplex)),
        "cudaMallocManaged GVV Term coefficient workspace");
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

void GVVSample::UploadAndBuildF()
{
    UploadAndBuildF(
        {GVV_SCALAR_00, GVV_SCALAR_22, GVV_PSEUDOSCALAR_11},
        GVV_NTERMS);
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
    if (particle < 0 || particle >= 7) {
        throw std::out_of_range("invalid GVV particle index");
    }
    if (event < 0 || event >= entries_) {
        throw std::out_of_range("invalid GVV event index");
    }
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

DeviceComplex* GVVSample::TermCoefficientBuffer()
{
    return term_coefficients_;
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

#include "../include/NLL_estimator.h"

#include "TFile.h"
#include "TLorentzVector.h"
#include "TTree.h"
#include "TVector3.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kProjectionMaxComponents = 15;
constexpr double kProjectionAxisTolerance = 1.0e-12;

enum ProjectionParticleIndex {
    kPip1 = 0,
    kPim1 = 1,
    kPi01 = 2,
    kPip2 = 3,
    kPim2 = 4,
    kPi02 = 5,
    kGamma = 6
};

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

TLorentzVector make_four_vector(const double* values)
{
    TLorentzVector result;
    result.SetPxPyPzE(values[0], values[1], values[2], values[3]);
    return result;
}

void store_four_vector(const TLorentzVector& source, double destination[4])
{
    destination[0] = source.Px();
    destination[1] = source.Py();
    destination[2] = source.Pz();
    destination[3] = source.E();
}

TVector3 safe_unit(const TVector3& vector, const TVector3& fallback)
{
    if (vector.Mag2() > kProjectionAxisTolerance) {
        return vector.Unit();
    }
    return fallback.Unit();
}

TVector3 transverse_axis(const TVector3& reference, const TVector3& z_axis)
{
    TVector3 result = reference.Cross(z_axis);
    if (result.Mag2() <= kProjectionAxisTolerance) {
        const TVector3 fallback =
            std::fabs(z_axis.Z()) < 0.9
                ? TVector3(0.0, 0.0, 1.0)
                : TVector3(1.0, 0.0, 0.0);
        result = fallback.Cross(z_axis);
    }
    return safe_unit(result, TVector3(0.0, 1.0, 0.0));
}

double wrap_angle(double angle)
{
    constexpr double pi = 3.14159265358979323846;
    constexpr double two_pi = 2.0 * pi;
    while (angle <= -pi) {
        angle += two_pi;
    }
    while (angle > pi) {
        angle -= two_pi;
    }
    return angle;
}

double omega_decay_plane_phi(
    const TLorentzVector& pip_in_x,
    const TLorentzVector& pim_in_x,
    const TLorentzVector& omega_in_x,
    const TVector3& x_helicity_axis)
{
    const TVector3 z_axis = safe_unit(
        omega_in_x.Vect(), TVector3(0.0, 0.0, 1.0));
    const TVector3 y_axis = transverse_axis(x_helicity_axis, z_axis);
    const TVector3 local_x_axis = safe_unit(
        y_axis.Cross(z_axis), TVector3(1.0, 0.0, 0.0));

    TLorentzVector pip = pip_in_x;
    TLorentzVector pim = pim_in_x;
    const TVector3 boost_to_omega = -omega_in_x.BoostVector();
    pip.Boost(boost_to_omega);
    pim.Boost(boost_to_omega);
    const TVector3 normal = pip.Vect().Cross(pim.Vect());
    if (normal.Mag2() <= kProjectionAxisTolerance) {
        return 0.0;
    }
    const TVector3 unit_normal = normal.Unit();
    return std::atan2(
        unit_normal.Dot(y_axis), unit_normal.Dot(local_x_axis));
}

struct ProjectionEvent {
    double p4_pip1[4];
    double p4_pim1[4];
    double p4_pi01[4];
    double p4_pip2[4];
    double p4_pim2[4];
    double p4_pi02[4];
    double p4_gam[4];
    double p4_omega1[4];
    double p4_omega2[4];
    double p4_X[4];

    double m_omega1;
    double m_omega2;
    double m_omegaomega;
    double m_gammaomega1;
    double m_gammaomega2;
    double m_pip1_pim1;
    double m_pip1_pi01;
    double m_pim1_pi01;
    double m_pip2_pim2;
    double m_pip2_pi02;
    double m_pim2_pi02;

    double cos_theta_gamma;
    double cos_theta_omega;
    double omega1_decay_plane_angle;
    double omega2_decay_plane_angle;
    double delta_phi_decay_planes;

    void Book(TTree& tree)
    {
        tree.Branch("p4_pip1", p4_pip1, "p4_pip1[4]/D");
        tree.Branch("p4_pim1", p4_pim1, "p4_pim1[4]/D");
        tree.Branch("p4_pi01", p4_pi01, "p4_pi01[4]/D");
        tree.Branch("p4_pip2", p4_pip2, "p4_pip2[4]/D");
        tree.Branch("p4_pim2", p4_pim2, "p4_pim2[4]/D");
        tree.Branch("p4_pi02", p4_pi02, "p4_pi02[4]/D");
        tree.Branch("p4_gam", p4_gam, "p4_gam[4]/D");
        tree.Branch("p4_omega1", p4_omega1, "p4_omega1[4]/D");
        tree.Branch("p4_omega2", p4_omega2, "p4_omega2[4]/D");
        tree.Branch("p4_X", p4_X, "p4_X[4]/D");

        tree.Branch("m_omega1", &m_omega1, "m_omega1/D");
        tree.Branch("m_omega2", &m_omega2, "m_omega2/D");
        tree.Branch("m_omegaomega", &m_omegaomega, "m_omegaomega/D");
        tree.Branch(
            "m_gammaomega1", &m_gammaomega1, "m_gammaomega1/D");
        tree.Branch(
            "m_gammaomega2", &m_gammaomega2, "m_gammaomega2/D");
        tree.Branch("m_pip1_pim1", &m_pip1_pim1, "m_pip1_pim1/D");
        tree.Branch("m_pip1_pi01", &m_pip1_pi01, "m_pip1_pi01/D");
        tree.Branch("m_pim1_pi01", &m_pim1_pi01, "m_pim1_pi01/D");
        tree.Branch("m_pip2_pim2", &m_pip2_pim2, "m_pip2_pim2/D");
        tree.Branch("m_pip2_pi02", &m_pip2_pi02, "m_pip2_pi02/D");
        tree.Branch("m_pim2_pi02", &m_pim2_pi02, "m_pim2_pi02/D");

        tree.Branch(
            "cos_theta_gamma", &cos_theta_gamma, "cos_theta_gamma/D");
        tree.Branch(
            "cos_theta_omega", &cos_theta_omega, "cos_theta_omega/D");
        tree.Branch(
            "omega1_decay_plane_angle",
            &omega1_decay_plane_angle,
            "omega1_decay_plane_angle/D");
        tree.Branch(
            "omega2_decay_plane_angle",
            &omega2_decay_plane_angle,
            "omega2_decay_plane_angle/D");
        tree.Branch(
            "delta_phi_decay_planes",
            &delta_phi_decay_planes,
            "delta_phi_decay_planes/D");
    }

    void Load(const GVVSample& sample, int event)
    {
        const TLorentzVector pip1 = make_four_vector(
            sample.HostMomentum(kPip1, event));
        const TLorentzVector pim1 = make_four_vector(
            sample.HostMomentum(kPim1, event));
        const TLorentzVector pi01 = make_four_vector(
            sample.HostMomentum(kPi01, event));
        const TLorentzVector pip2 = make_four_vector(
            sample.HostMomentum(kPip2, event));
        const TLorentzVector pim2 = make_four_vector(
            sample.HostMomentum(kPim2, event));
        const TLorentzVector pi02 = make_four_vector(
            sample.HostMomentum(kPi02, event));
        const TLorentzVector gamma = make_four_vector(
            sample.HostMomentum(kGamma, event));

        const TLorentzVector omega1 = pip1 + pim1 + pi01;
        const TLorentzVector omega2 = pip2 + pim2 + pi02;
        const TLorentzVector x_state = omega1 + omega2;
        const TLorentzVector psi = x_state + gamma;

        store_four_vector(pip1, p4_pip1);
        store_four_vector(pim1, p4_pim1);
        store_four_vector(pi01, p4_pi01);
        store_four_vector(pip2, p4_pip2);
        store_four_vector(pim2, p4_pim2);
        store_four_vector(pi02, p4_pi02);
        store_four_vector(gamma, p4_gam);
        store_four_vector(omega1, p4_omega1);
        store_four_vector(omega2, p4_omega2);
        store_four_vector(x_state, p4_X);

        m_omega1 = omega1.M();
        m_omega2 = omega2.M();
        m_omegaomega = x_state.M();
        m_gammaomega1 = (gamma + omega1).M();
        m_gammaomega2 = (gamma + omega2).M();
        m_pip1_pim1 = (pip1 + pim1).M();
        m_pip1_pi01 = (pip1 + pi01).M();
        m_pim1_pi01 = (pim1 + pi01).M();
        m_pip2_pim2 = (pip2 + pim2).M();
        m_pip2_pi02 = (pip2 + pi02).M();
        m_pim2_pi02 = (pim2 + pi02).M();

        std::array<TLorentzVector, 7> in_psi = {{
            pip1, pim1, pi01, pip2, pim2, pi02, gamma}};
        const TVector3 boost_to_psi = -psi.BoostVector();
        for (TLorentzVector& vector : in_psi) {
            vector.Boost(boost_to_psi);
        }
        const TLorentzVector omega1_psi =
            in_psi[kPip1] + in_psi[kPim1] + in_psi[kPi01];
        const TLorentzVector omega2_psi =
            in_psi[kPip2] + in_psi[kPim2] + in_psi[kPi02];
        const TLorentzVector x_psi = omega1_psi + omega2_psi;
        cos_theta_gamma = safe_unit(
            in_psi[kGamma].Vect(), TVector3(0.0, 0.0, 1.0))
                              .Dot(TVector3(0.0, 0.0, 1.0));

        std::array<TLorentzVector, 7> in_x = in_psi;
        const TVector3 boost_to_x = -x_psi.BoostVector();
        for (TLorentzVector& vector : in_x) {
            vector.Boost(boost_to_x);
        }
        const TLorentzVector omega1_x =
            in_x[kPip1] + in_x[kPim1] + in_x[kPi01];
        const TLorentzVector omega2_x =
            in_x[kPip2] + in_x[kPim2] + in_x[kPi02];
        const TVector3 x_helicity_axis = safe_unit(
            -in_x[kGamma].Vect(), TVector3(0.0, 0.0, 1.0));
        cos_theta_omega = safe_unit(
            omega1_x.Vect(), TVector3(0.0, 0.0, 1.0))
                              .Dot(x_helicity_axis);

        omega1_decay_plane_angle = omega_decay_plane_phi(
            in_x[kPip1], in_x[kPim1], omega1_x, x_helicity_axis);
        omega2_decay_plane_angle = omega_decay_plane_phi(
            in_x[kPip2], in_x[kPim2], omega2_x, x_helicity_axis);
        delta_phi_decay_planes = wrap_angle(
            omega1_decay_plane_angle - omega2_decay_plane_angle);
    }
};

} // namespace

NLL_estimator::NLL_estimator(const GVVBranchConfig& branches)
    : branches_(branches),
      device_resonances_(nullptr),
      device_terms_(nullptr),
      device_couplings_(nullptr),
      prepared_(false)
{
    for (int index = 0; index < GVV_NRESONANCES; ++index) {
        resonances_[index] = gvv_default_resonance(index);
    }
    for (int index = 0; index < GVV_NTERMS; ++index) {
        terms_[index] = gvv_default_term(index);
        couplings_[index] = DeviceComplex(0.10, 0.0);
    }
    // Removes the global phase and normalization degeneracy.  The scalar
    // phase reference keeps its initialized positive magnitude and zero
    // phase; that magnitude remains a fit parameter.
    couplings_[GVV_SCALE_AND_PHASE_REFERENCE_TERM] =
        DeviceComplex(1.0, 0.0);
    couplings_[GVV_SCALAR_PHASE_REFERENCE_TERM] =
        DeviceComplex(0.10, 0.0);
}

NLL_estimator::~NLL_estimator()
{
    if (device_resonances_ != nullptr) {
        cudaFree(device_resonances_);
    }
    if (device_terms_ != nullptr) {
        cudaFree(device_terms_);
    }
    if (device_couplings_ != nullptr) {
        cudaFree(device_couplings_);
    }
}

std::unique_ptr<GVVSample> NLL_estimator::LoadSample(
    const std::string& file_name,
    const std::string& label) const
{
    std::unique_ptr<GVVSample> sample(new GVVSample(label));
    sample->Load(file_name, branches_);
    return sample;
}

void NLL_estimator::LoadNormalizationMC(const std::string& file_name)
{
    normalization_mc_ = LoadSample(file_name, "normalization MC");
}

void NLL_estimator::LoadData(const std::string& file_name)
{
    data_ = LoadSample(file_name, "data");
}

void NLL_estimator::AddBackground(
    const std::string& file_name,
    double likelihood_coefficient,
    const std::string& label)
{
    BackgroundSample background;
    background.sample = LoadSample(file_name, label);
    background.likelihood_coefficient = likelihood_coefficient;
    backgrounds_.push_back(std::move(background));
}

void NLL_estimator::UploadModel()
{
    check_cuda(
        cudaMallocManaged(
            &device_resonances_,
            GVV_NRESONANCES * sizeof(GVVResonanceParameters)),
        "cudaMallocManaged GVV resonances");
    check_cuda(
        cudaMallocManaged(&device_terms_, GVV_NTERMS * sizeof(GVVTermSpec)),
        "cudaMallocManaged GVV terms");
    check_cuda(
        cudaMallocManaged(
            &device_couplings_, GVV_NTERMS * sizeof(DeviceComplex)),
        "cudaMallocManaged GVV couplings");
    SynchronizeModel();
}

void NLL_estimator::SynchronizeModel()
{
    check_cuda(
        cudaMemcpy(
            device_resonances_,
            resonances_.data(),
            GVV_NRESONANCES * sizeof(GVVResonanceParameters),
            cudaMemcpyHostToDevice),
        "cudaMemcpy GVV resonances");
    check_cuda(
        cudaMemcpy(
            device_terms_,
            terms_.data(),
            GVV_NTERMS * sizeof(GVVTermSpec),
            cudaMemcpyHostToDevice),
        "cudaMemcpy GVV terms");
    check_cuda(
        cudaMemcpy(
            device_couplings_,
            couplings_.data(),
            GVV_NTERMS * sizeof(DeviceComplex),
            cudaMemcpyHostToDevice),
        "cudaMemcpy GVV couplings");
}

void NLL_estimator::Prepare()
{
    if (normalization_mc_ == nullptr || data_ == nullptr) {
        throw std::runtime_error(
            "normalization MC and data must be loaded before Prepare()");
    }

    omega_width_table_.Build();
    omega_width_table_.Upload();
    UploadModel();

    normalization_mc_->UploadAndBuildF();
    data_->UploadAndBuildF();
    for (BackgroundSample& background : backgrounds_) {
        background.sample->UploadAndBuildF();
    }
    prepared_ = true;
    std::cout << "GVV samples, F matrices, and omega width table prepared\n";
}

double NLL_estimator::EvaluateSample(
    GVVSample& sample,
    double likelihood_coefficient,
    double normalization)
{
    CalGVVPDF(
        sample.Momenta(),
        device_resonances_,
        device_terms_,
        device_couplings_,
        omega_width_table_.DeviceView(),
        sample.FMatrix(),
        sample.IntensityBuffer(),
        sample.Entries());

    double logarithm_sum = 0.0;
    for (int event = 0; event < sample.Entries(); ++event) {
        const double pdf = sample.IntensityBuffer()[event] / normalization;
        if (!(pdf > 0.0) || !std::isfinite(pdf)) {
            return -std::numeric_limits<double>::infinity();
        }
        logarithm_sum += std::log(pdf);
    }
    return likelihood_coefficient * logarithm_sum;
}

double NLL_estimator::Cal_log_likelihood()
{
    if (!prepared_) {
        throw std::runtime_error("call Prepare() before evaluating likelihood");
    }
    SynchronizeModel();

    CalGVVPDF(
        normalization_mc_->Momenta(),
        device_resonances_,
        device_terms_,
        device_couplings_,
        omega_width_table_.DeviceView(),
        normalization_mc_->FMatrix(),
        normalization_mc_->IntensityBuffer(),
        normalization_mc_->Entries());

    double normalization = 0.0;
    for (int event = 0; event < normalization_mc_->Entries(); ++event) {
        const double intensity = normalization_mc_->IntensityBuffer()[event];
        if (!(intensity >= 0.0) || !std::isfinite(intensity)) {
            return -1.0e100;
        }
        normalization += intensity;
    }
    normalization /= normalization_mc_->Entries();
    if (!(normalization > 0.0) || !std::isfinite(normalization)) {
        return -1.0e100;
    }

    double log_likelihood = EvaluateSample(*data_, +1.0, normalization);
    if (!std::isfinite(log_likelihood)) {
        return -1.0e100;
    }
    for (BackgroundSample& background : backgrounds_) {
        const double contribution = EvaluateSample(
            *background.sample,
            background.likelihood_coefficient,
            normalization);
        if (!std::isfinite(contribution)) {
            return -1.0e100;
        }
        log_likelihood += contribution;
    }
    return log_likelihood;
}

void NLL_estimator::SetCoupling(int term_index, double real, double imag)
{
    if (term_index < 0 || term_index >= GVV_NTERMS) {
        throw std::out_of_range("invalid GVV term index");
    }
    const int parameterization = gvv_coupling_parameterization(term_index);
    if (parameterization == GVV_COUPLING_FIXED_SCALE_AND_PHASE
        && (real != 1.0 || imag != 0.0)) {
        throw std::invalid_argument(
            "scale-and-phase reference coupling must remain 1+0i");
    }
    if (parameterization == GVV_COUPLING_POSITIVE_REAL
        && (!(real > 0.0) || imag != 0.0)) {
        throw std::invalid_argument(
            "phase-reference coupling must remain positive real");
    }
    couplings_[term_index] = DeviceComplex(real, imag);
}

void NLL_estimator::SetLogCouplingMagnitude(
    int term_index,
    double log_magnitude)
{
    if (term_index < 0 || term_index >= GVV_NTERMS
        || gvv_coupling_parameterization(term_index)
               != GVV_COUPLING_POSITIVE_REAL) {
        throw std::invalid_argument(
            "log coupling magnitude is only valid for a phase reference");
    }
    if (!std::isfinite(log_magnitude)) {
        throw std::invalid_argument("log coupling magnitude must be finite");
    }
    const double magnitude = std::exp(log_magnitude);
    if (!(magnitude > 0.0) || !std::isfinite(magnitude)) {
        throw std::invalid_argument(
            "log coupling magnitude is outside the numerical range");
    }
    couplings_[term_index] = DeviceComplex(magnitude, 0.0);
}

DeviceComplex NLL_estimator::Coupling(int term_index) const
{
    if (term_index < 0 || term_index >= GVV_NTERMS) {
        throw std::out_of_range("invalid GVV term index");
    }
    return couplings_[term_index];
}

void NLL_estimator::SetLogSDRatio(int resonance_index, double log_ratio)
{
    if (resonance_index < 0 || resonance_index >= GVV_NRESONANCES
        || !resonances_[resonance_index].fit_sd_ratio) {
        throw std::invalid_argument("S/D ratio is not fitted for this resonance");
    }
    const double bounded = std::max(-30.0, std::min(30.0, log_ratio));
    resonances_[resonance_index].sd_ratio = std::exp(bounded);
}

void NLL_estimator::SetLogFlatteRatio(int resonance_index, double log_ratio)
{
    if (resonance_index < 0 || resonance_index >= GVV_NRESONANCES
        || !resonances_[resonance_index].fit_flatte_ratio) {
        throw std::invalid_argument(
            "Flatte omega-omega ratio is not fitted for this resonance");
    }
    const double bounded = std::max(-30.0, std::min(30.0, log_ratio));
    resonances_[resonance_index].flatte_ratio = std::exp(bounded);
}

const GVVResonanceParameters& NLL_estimator::Resonance(
    int resonance_index) const
{
    if (resonance_index < 0 || resonance_index >= GVV_NRESONANCES) {
        throw std::out_of_range("invalid GVV resonance index");
    }
    return resonances_[resonance_index];
}

int NLL_estimator::NumberFitParameters() const
{
    int count = gvv_number_coupling_fit_parameters();
    for (const GVVResonanceParameters& resonance : resonances_) {
        if (resonance.fit_sd_ratio) {
            ++count;
        }
        if (resonance.fit_flatte_ratio) {
            ++count;
        }
    }
    return count;
}

int NLL_estimator::DataEntries() const
{
    return data_ == nullptr ? 0 : data_->Entries();
}

int NLL_estimator::NormalizationMCEntries() const
{
    return normalization_mc_ == nullptr ? 0 : normalization_mc_->Entries();
}

void NLL_estimator::PrintModelSummary() const
{
    std::cout << "GVV model: " << GVV_NRESONANCES << " physical components, "
              << GVV_NTERMS << " coherent terms\n";
    for (int index = 0; index < GVV_NRESONANCES; ++index) {
        const GVVResonanceParameters& resonance = resonances_[index];
        std::cout << "  " << std::setw(10) << gvv_resonance_name(index)
                  << "  m=" << resonance.mass
                  << "  model=" << gvv_propagator_name(
                         resonance.propagator_model);
        if (resonance.propagator_model == GVV_PROP_SUBTRACTED_FLATTE) {
            std::cout << "  Gamma_rest=" << resonance.pole_width;
        } else {
            std::cout << "  Gamma=" << resonance.pole_width;
        }
        if (resonance.fit_sd_ratio) {
            std::cout << "  fit r_D/S=" << resonance.sd_ratio;
        }
        if (resonance.propagator_model == GVV_PROP_SUBTRACTED_FLATTE) {
            std::cout << "  R_omegaomega=" << resonance.flatte_ratio;
            if (resonance.fit_flatte_ratio) {
                std::cout << " (fitted as log R)";
            } else {
                std::cout << " (fixed)";
            }
        }
        std::cout << '\n';
    }
    std::cout << "  scale-and-phase reference amplitude: "
              << gvv_term_name(GVV_SCALE_AND_PHASE_REFERENCE_TERM)
              << " = 1 + 0i\n"
              << "  scalar phase reference amplitude: "
              << gvv_term_name(GVV_SCALAR_PHASE_REFERENCE_TERM)
              << " = rho + 0i, rho > 0 and fitted as log(rho)\n";
}

void NLL_estimator::Project_fit_result(
    const std::string& save_name,
    int best_start,
    long long best_seed,
    double minimum)
{
    if (!prepared_ || normalization_mc_ == nullptr || data_ == nullptr) {
        throw std::runtime_error(
            "call Prepare() before writing the GVV projection");
    }
    if (GVV_NTERMS > kProjectionMaxComponents) {
        throw std::runtime_error(
            "GVV_NTERMS exceeds projection weight_component capacity");
    }

    const std::array<DeviceComplex, GVV_NTERMS> fitted_couplings = couplings_;
    const int number_mc = normalization_mc_->Entries();
    auto evaluate_mc = [&](const std::array<DeviceComplex, GVV_NTERMS>& values) {
        couplings_ = values;
        SynchronizeModel();
        CalGVVPDF(
            normalization_mc_->Momenta(),
            device_resonances_,
            device_terms_,
            device_couplings_,
            omega_width_table_.DeviceView(),
            normalization_mc_->FMatrix(),
            normalization_mc_->IntensityBuffer(),
            number_mc);
        std::vector<double> result(number_mc, 0.0);
        std::copy(
            normalization_mc_->IntensityBuffer(),
            normalization_mc_->IntensityBuffer() + number_mc,
            result.begin());
        return result;
    };

    std::vector<double> total_intensity;
    std::vector<double> scalar_intensity;
    std::vector<double> pseudoscalar_intensity;
    std::array<
        std::array<std::vector<double>, GVV_NTERMS>,
        GVV_NTERMS> pair_intensity;

    try {
        total_intensity = evaluate_mc(fitted_couplings);

        std::array<DeviceComplex, GVV_NTERMS> selected;
        selected.fill(DeviceComplex(0.0, 0.0));
        for (int term = 0; term < GVV_NTERMS; ++term) {
            if (terms_[term].wave_type == GVV_SCALAR_00
                || terms_[term].wave_type == GVV_SCALAR_22) {
                selected[term] = fitted_couplings[term];
            }
        }
        scalar_intensity = evaluate_mc(selected);

        selected.fill(DeviceComplex(0.0, 0.0));
        for (int term = 0; term < GVV_NTERMS; ++term) {
            if (terms_[term].wave_type == GVV_PSEUDOSCALAR_11) {
                selected[term] = fitted_couplings[term];
            }
        }
        pseudoscalar_intensity = evaluate_mc(selected);

        for (int first = 0; first < GVV_NTERMS; ++first) {
            for (int second = first; second < GVV_NTERMS; ++second) {
                selected.fill(DeviceComplex(0.0, 0.0));
                selected[first] = fitted_couplings[first];
                selected[second] = fitted_couplings[second];
                pair_intensity[first][second] = evaluate_mc(selected);
            }
        }
    } catch (...) {
        couplings_ = fitted_couplings;
        SynchronizeModel();
        throw;
    }
    couplings_ = fitted_couplings;
    SynchronizeModel();

    const double sum_pdf = std::accumulate(
        total_intensity.begin(), total_intensity.end(), 0.0);
    if (!(sum_pdf > 0.0) || !std::isfinite(sum_pdf)) {
        throw std::runtime_error(
            "invalid normalization intensity sum for projection");
    }

    double effective_yield = static_cast<double>(data_->Entries());
    for (const BackgroundSample& background : backgrounds_) {
        effective_yield += background.likelihood_coefficient
                           * background.sample->Entries();
    }
    if (!(effective_yield > 0.0) || !std::isfinite(effective_yield)) {
        throw std::runtime_error(
            "non-positive effective signal yield for projection");
    }

    TFile output(save_name.c_str(), "RECREATE");
    if (output.IsZombie()) {
        throw std::runtime_error(
            "cannot create projection ROOT file: " + save_name);
    }

    ProjectionEvent event_values;
    TTree tree_mc("MC", "accepted normalization MC with fitted weights");
    event_values.Book(tree_mc);
    double weight = 0.0;
    double weight_0pp = 0.0;
    double weight_0mp = 0.0;
    double weight_int_0pp_0mp = 0.0;
    double weight_component
        [kProjectionMaxComponents][kProjectionMaxComponents];
    tree_mc.Branch("weight", &weight, "weight/D");
    tree_mc.Branch("weight_0pp", &weight_0pp, "weight_0pp/D");
    tree_mc.Branch("weight_0mp", &weight_0mp, "weight_0mp/D");
    tree_mc.Branch(
        "weight_int_0pp_0mp",
        &weight_int_0pp_0mp,
        "weight_int_0pp_0mp/D");
    tree_mc.Branch(
        "weight_component",
        weight_component,
        "weight_component[15][15]/D");

    double maximum_closure_residual = 0.0;
    double sum_projection_weight = 0.0;
    for (int event = 0; event < number_mc; ++event) {
        weight = total_intensity[event] / sum_pdf * effective_yield;
        weight_0pp = scalar_intensity[event] / sum_pdf * effective_yield;
        weight_0mp = pseudoscalar_intensity[event] / sum_pdf * effective_yield;
        weight_int_0pp_0mp = weight - weight_0pp - weight_0mp;
        sum_projection_weight += weight;

        for (int first = 0; first < kProjectionMaxComponents; ++first) {
            for (int second = 0;
                 second < kProjectionMaxComponents;
                 ++second) {
                weight_component[first][second] = -1.0;
            }
        }

        double reconstructed_intensity = 0.0;
        for (int first = 0; first < GVV_NTERMS; ++first) {
            const double diagonal = pair_intensity[first][first][event];
            reconstructed_intensity += diagonal;
            weight_component[first][first] =
                diagonal / sum_pdf * effective_yield;
            for (int second = first + 1;
                 second < GVV_NTERMS;
                 ++second) {
                const double interference =
                    pair_intensity[first][second][event]
                    - pair_intensity[first][first][event]
                    - pair_intensity[second][second][event];
                reconstructed_intensity += interference;
                const double interference_weight =
                    interference / sum_pdf * effective_yield;
                weight_component[first][second] = interference_weight;
                weight_component[second][first] = interference_weight;
            }
        }
        const double closure_scale = std::max(
            1.0, std::fabs(total_intensity[event]));
        maximum_closure_residual = std::max(
            maximum_closure_residual,
            std::fabs(reconstructed_intensity - total_intensity[event])
                / closure_scale);

        event_values.Load(*normalization_mc_, event);
        tree_mc.Fill();
    }
    if (maximum_closure_residual > 1.0e-7) {
        throw std::runtime_error(
            "projection component closure check failed");
    }

    TTree tree_data("data", "selected data");
    event_values.Book(tree_data);
    for (int event = 0; event < data_->Entries(); ++event) {
        event_values.Load(*data_, event);
        tree_data.Fill();
    }

    TTree tree_background("bg", "combined two-dimensional sidebands");
    event_values.Book(tree_background);
    int sideband_id = 0;
    double weight_bg = 0.0;
    tree_background.Branch("sideband_id", &sideband_id, "sideband_id/I");
    tree_background.Branch("weight_bg", &weight_bg, "weight_bg/D");
    for (std::size_t background_index = 0;
         background_index < backgrounds_.size();
         ++background_index) {
        const BackgroundSample& background = backgrounds_[background_index];
        sideband_id = static_cast<int>(background_index) + 1;
        // The likelihood coefficients are (-0.5,+0.25).  The background
        // estimate added to signal MC is therefore (+0.5,-0.25).
        weight_bg = -background.likelihood_coefficient;
        for (int event = 0; event < background.sample->Entries(); ++event) {
            event_values.Load(*background.sample, event);
            tree_background.Fill();
        }
    }

    TTree component_map("component_map", "GVV component index map");
    int component_index = 0;
    int resonance_index = 0;
    int wave_type = 0;
    char component_name[64] = {0};
    char component_jpc[16] = {0};
    component_map.Branch(
        "component_index", &component_index, "component_index/I");
    component_map.Branch(
        "resonance_index", &resonance_index, "resonance_index/I");
    component_map.Branch("wave_type", &wave_type, "wave_type/I");
    component_map.Branch("name", component_name, "name/C");
    component_map.Branch("jpc", component_jpc, "jpc/C");
    for (int term = 0; term < GVV_NTERMS; ++term) {
        component_index = term;
        resonance_index = terms_[term].resonance_index;
        wave_type = terms_[term].wave_type;
        std::snprintf(
            component_name,
            sizeof(component_name),
            "%s",
            gvv_term_name(term));
        std::snprintf(
            component_jpc,
            sizeof(component_jpc),
            "%s",
            wave_type == GVV_PSEUDOSCALAR_11 ? "0-+" : "0++");
        component_map.Fill();
    }

    TTree metadata("metadata", "GVV projection provenance");
    int n_terms = GVV_NTERMS;
    int n_data = data_->Entries();
    int n_normalization_mc = normalization_mc_->Entries();
    int n_background_samples = static_cast<int>(backgrounds_.size());
    int n_sb1 = backgrounds_.size() > 0
                    ? backgrounds_[0].sample->Entries()
                    : 0;
    int n_sb2 = backgrounds_.size() > 1
                    ? backgrounds_[1].sample->Entries()
                    : 0;
    double sb1_likelihood_coefficient = backgrounds_.size() > 0
                                            ? backgrounds_[0]
                                                  .likelihood_coefficient
                                            : 0.0;
    double sb2_likelihood_coefficient = backgrounds_.size() > 1
                                            ? backgrounds_[1]
                                                  .likelihood_coefficient
                                            : 0.0;
    long long stored_best_seed = best_seed;
    metadata.Branch("n_terms", &n_terms, "n_terms/I");
    metadata.Branch("n_data", &n_data, "n_data/I");
    metadata.Branch(
        "n_normalization_mc",
        &n_normalization_mc,
        "n_normalization_mc/I");
    metadata.Branch(
        "n_background_samples",
        &n_background_samples,
        "n_background_samples/I");
    metadata.Branch("n_SB1", &n_sb1, "n_SB1/I");
    metadata.Branch("n_SB2", &n_sb2, "n_SB2/I");
    metadata.Branch(
        "SB1_likelihood_coefficient",
        &sb1_likelihood_coefficient,
        "SB1_likelihood_coefficient/D");
    metadata.Branch(
        "SB2_likelihood_coefficient",
        &sb2_likelihood_coefficient,
        "SB2_likelihood_coefficient/D");
    metadata.Branch(
        "effective_signal_yield",
        &effective_yield,
        "effective_signal_yield/D");
    metadata.Branch("best_start", &best_start, "best_start/I");
    metadata.Branch("best_seed", &stored_best_seed, "best_seed/L");
    metadata.Branch("minimum_nll", &minimum, "minimum_nll/D");
    metadata.Branch(
        "maximum_component_closure_residual",
        &maximum_closure_residual,
        "maximum_component_closure_residual/D");
    metadata.Fill();

    if (output.Write() <= 0) {
        throw std::runtime_error(
            "failed to write projection ROOT file: " + save_name);
    }
    output.Close();

    std::cout << std::setprecision(12)
              << "Projection written to " << save_name
              << "; sum(weight)=" << sum_projection_weight
              << ", target=" << effective_yield
              << ", max component closure residual="
              << maximum_closure_residual << '\n';
}

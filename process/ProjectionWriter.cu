// GVV projection ROOT schema and event-observable serialization.
#include "process/ProjectionWriter.h"
#include "process/FitLikelihood.h"

#include "TFile.h"
#include "TLorentzVector.h"
#include "TTree.h"
#include "TVector3.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kProjectionAxisTolerance = 1.0e-12;

void fill_tree(TTree& tree)
{
    if (tree.Fill() < 0) {
        throw std::runtime_error(
            "failed to fill projection tree '" + std::string(tree.GetName())
            + "'");
    }
}

template <std::size_t Size>
void copy_checked(char (&destination)[Size], const std::string& source)
{
    if (source.size() >= Size) {
        throw std::runtime_error(
            "projection metadata string is too long: " + source);
    }
    std::snprintf(destination, Size, "%s", source.c_str());
}

enum ProjectionParticleIndex {
    kPip1 = 0,
    kPim1 = 1,
    kPi01 = 2,
    kPip2 = 3,
    kPim2 = 4,
    kPi02 = 5,
    kGamma = 6
};

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

        std::array<TLorentzVector, GVV_NFINAL_PARTICLES> in_psi = {{
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

        std::array<TLorentzVector, GVV_NFINAL_PARTICLES> in_x = in_psi;
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

void write_gvv_projection(
    FitLikelihood& likelihood,
    const std::string& save_name,
    int best_start,
    long long best_seed,
    double minimum)
{
    const GVVCompiledModel& model = likelihood.Model();
    const GVVSample& normalization_mc =
        likelihood.NormalizationMCSample();
    const GVVSample& data = likelihood.DataSample();
    const int number_terms = likelihood.NumberTerms();
    const std::vector<DeviceComplex> fitted_couplings =
        model.initial_couplings;
    const int number_mc = normalization_mc.Entries();
    auto evaluate_mc = [&](const std::vector<DeviceComplex>& values) {
        return likelihood.EvaluateNormalizationMCIntensity(values);
    };

    std::vector<double> total_intensity;
    std::vector<std::string> group_ids;
    for (const GVVTermMetadata& term : model.term_metadata) {
        if (std::find(group_ids.begin(), group_ids.end(), term.jpc)
            == group_ids.end()) {
            group_ids.push_back(term.jpc);
        }
    }
    std::vector<std::vector<double>> group_intensity(group_ids.size());
    std::vector<std::vector<std::vector<double>>> pair_intensity(
        number_terms,
        std::vector<std::vector<double>>(number_terms));

    total_intensity = evaluate_mc(fitted_couplings);

    std::vector<DeviceComplex> selected(
        number_terms, DeviceComplex(0.0, 0.0));
    for (std::size_t group = 0; group < group_ids.size(); ++group) {
        std::fill(
            selected.begin(), selected.end(), DeviceComplex(0.0, 0.0));
        for (int term = 0; term < number_terms; ++term) {
            if (model.term_metadata[term].jpc == group_ids[group]) {
                selected[term] = fitted_couplings[term];
            }
        }
        group_intensity[group] = evaluate_mc(selected);
    }

    for (int first = 0; first < number_terms; ++first) {
        for (int second = first; second < number_terms; ++second) {
            std::fill(
                selected.begin(),
                selected.end(),
                DeviceComplex(0.0, 0.0));
            selected[first] = fitted_couplings[first];
            selected[second] = fitted_couplings[second];
            pair_intensity[first][second] = evaluate_mc(selected);
        }
    }
    const double sum_pdf = std::accumulate(
        total_intensity.begin(), total_intensity.end(), 0.0);
    if (!(sum_pdf > 0.0) || !std::isfinite(sum_pdf)) {
        throw std::runtime_error(
            "invalid normalization intensity sum for projection");
    }

    double effective_yield = static_cast<double>(data.Entries());
    for (std::size_t index = 0;
         index < likelihood.NumberBackgroundSamples();
         ++index) {
        effective_yield +=
            likelihood.BackgroundLikelihoodCoefficient(index)
            * likelihood.BackgroundSampleAt(index).Entries();
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
    std::vector<double> weight_group(group_ids.size(), 0.0);
    std::vector<double> weight_component(
        static_cast<std::size_t>(number_terms) * number_terms, 0.0);
    tree_mc.Branch("weight", &weight, "weight/D");
    tree_mc.Branch("weight_0pp", &weight_0pp, "weight_0pp/D");
    tree_mc.Branch("weight_0mp", &weight_0mp, "weight_0mp/D");
    tree_mc.Branch(
        "weight_int_0pp_0mp",
        &weight_int_0pp_0mp,
        "weight_int_0pp_0mp/D");
    tree_mc.Branch("weight_group", &weight_group);
    tree_mc.Branch(
        "weight_component",
        &weight_component);

    double maximum_closure_residual = 0.0;
    double sum_projection_weight = 0.0;
    for (int event = 0; event < number_mc; ++event) {
        weight = total_intensity[event] / sum_pdf * effective_yield;
        double sum_group_weight = 0.0;
        weight_0pp = 0.0;
        weight_0mp = 0.0;
        for (std::size_t group = 0; group < group_ids.size(); ++group) {
            weight_group[group] = group_intensity[group][event]
                                  / sum_pdf * effective_yield;
            sum_group_weight += weight_group[group];
            if (group_ids[group] == "0++") weight_0pp = weight_group[group];
            if (group_ids[group] == "0-+") weight_0mp = weight_group[group];
        }
        // Historical branch name retained for nominal plotting. With future
        // groups this stores the total interference between all JPC groups.
        weight_int_0pp_0mp = weight - sum_group_weight;
        sum_projection_weight += weight;

        std::fill(
            weight_component.begin(), weight_component.end(), -1.0);

        double reconstructed_intensity = 0.0;
        for (int first = 0; first < number_terms; ++first) {
            const double diagonal = pair_intensity[first][first][event];
            reconstructed_intensity += diagonal;
            weight_component[
                static_cast<std::size_t>(first) * number_terms + first] =
                diagonal / sum_pdf * effective_yield;
            for (int second = first + 1;
                 second < number_terms;
                 ++second) {
                const double interference =
                    pair_intensity[first][second][event]
                    - pair_intensity[first][first][event]
                    - pair_intensity[second][second][event];
                reconstructed_intensity += interference;
                const double interference_weight =
                    interference / sum_pdf * effective_yield;
                weight_component[
                    static_cast<std::size_t>(first) * number_terms + second] =
                    interference_weight;
                weight_component[
                    static_cast<std::size_t>(second) * number_terms + first] =
                    interference_weight;
            }
        }
        const double closure_scale = std::max(
            1.0, std::fabs(total_intensity[event]));
        maximum_closure_residual = std::max(
            maximum_closure_residual,
            std::fabs(reconstructed_intensity - total_intensity[event])
                / closure_scale);

        event_values.Load(normalization_mc, event);
        fill_tree(tree_mc);
    }
    if (maximum_closure_residual > 1.0e-7) {
        throw std::runtime_error(
            "projection component closure check failed");
    }

    TTree tree_data("data", "selected data");
    event_values.Book(tree_data);
    for (int event = 0; event < data.Entries(); ++event) {
        event_values.Load(data, event);
        fill_tree(tree_data);
    }

    TTree tree_background("bg", "combined two-dimensional sidebands");
    event_values.Book(tree_background);
    int sideband_id = 0;
    double weight_bg = 0.0;
    tree_background.Branch("sideband_id", &sideband_id, "sideband_id/I");
    tree_background.Branch("weight_bg", &weight_bg, "weight_bg/D");
    for (std::size_t background_index = 0;
         background_index < likelihood.NumberBackgroundSamples();
         ++background_index) {
        const GVVSample& background =
            likelihood.BackgroundSampleAt(background_index);
        sideband_id = static_cast<int>(background_index) + 1;
        // The likelihood coefficients are (-0.5,+0.25).  The background
        // estimate added to signal MC is therefore (+0.5,-0.25).
        weight_bg =
            -likelihood.BackgroundLikelihoodCoefficient(background_index);
        for (int event = 0; event < background.Entries(); ++event) {
            event_values.Load(background, event);
            fill_tree(tree_background);
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
    for (int term = 0; term < number_terms; ++term) {
        component_index = term;
        resonance_index = model.terms[term].resonance_index;
        wave_type = model.term_metadata[term].registered_wave_type;
        copy_checked(component_name, model.term_metadata[term].id);
        copy_checked(component_jpc, model.term_metadata[term].jpc);
        fill_tree(component_map);
    }

    TTree group_map("group_map", "GVV JPC group index map");
    int group_index = 0;
    char group_jpc[16] = {0};
    group_map.Branch("group_index", &group_index, "group_index/I");
    group_map.Branch("jpc", group_jpc, "jpc/C");
    for (std::size_t group = 0; group < group_ids.size(); ++group) {
        group_index = static_cast<int>(group);
        copy_checked(group_jpc, group_ids[group]);
        fill_tree(group_map);
    }

    TTree metadata("metadata", "GVV projection provenance");
    int n_terms = number_terms;
    int n_groups = static_cast<int>(group_ids.size());
    int n_data = data.Entries();
    int n_normalization_mc = normalization_mc.Entries();
    int n_background_samples =
        static_cast<int>(likelihood.NumberBackgroundSamples());
    int n_sb1 = likelihood.NumberBackgroundSamples() > 0
                    ? likelihood.BackgroundSampleAt(0).Entries()
                    : 0;
    int n_sb2 = likelihood.NumberBackgroundSamples() > 1
                    ? likelihood.BackgroundSampleAt(1).Entries()
                    : 0;
    double sb1_likelihood_coefficient =
        likelihood.NumberBackgroundSamples() > 0
            ? likelihood.BackgroundLikelihoodCoefficient(0)
            : 0.0;
    double sb2_likelihood_coefficient =
        likelihood.NumberBackgroundSamples() > 1
            ? likelihood.BackgroundLikelihoodCoefficient(1)
            : 0.0;
    long long stored_best_seed = best_seed;
    metadata.Branch("n_terms", &n_terms, "n_terms/I");
    metadata.Branch("n_groups", &n_groups, "n_groups/I");
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
    fill_tree(metadata);

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

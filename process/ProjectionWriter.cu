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
constexpr std::size_t kProjectionComponentScratchBytes =
    64ULL * 1024ULL * 1024ULL;

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
    const std::string& output_tag,
    const std::string& model_signature,
    int best_start,
    long long best_seed,
    double minimum)
{
    const GVVCompiledModel& model = likelihood.Model();
    const GVVSample& normalization_mc =
        likelihood.NormalizationMCSample();
    const GVVSample& data = likelihood.DataSample();
    const int number_terms = likelihood.NumberTerms();
    const int number_mc = normalization_mc.Entries();
    const int number_pairs = ctpwa::component_pair_count(number_terms);
    // The packed values exist once in FitLikelihood's managed scratch and
    // once in this host batch while ROOT rows are serialized.
    const std::size_t component_bytes_per_event =
        2ULL * static_cast<std::size_t>(number_pairs) * sizeof(double)
        + static_cast<std::size_t>(number_terms) * sizeof(DeviceComplex);
    const std::size_t capacity_by_bytes = std::max<std::size_t>(
        1, kProjectionComponentScratchBytes / component_bytes_per_event);
    const int component_batch_capacity = static_cast<int>(
        std::min<std::size_t>(number_mc, capacity_by_bytes));

    std::vector<std::string> group_ids;
    for (const GVVTermMetadata& term : model.term_metadata) {
        if (std::find(group_ids.begin(), group_ids.end(), term.jpc)
            == group_ids.end()) {
            group_ids.push_back(term.jpc);
        }
    }
    std::vector<int> term_groups(number_terms, -1);
    for (int term = 0; term < number_terms; ++term) {
        term_groups[term] = static_cast<int>(std::find(
            group_ids.begin(), group_ids.end(),
            model.term_metadata[term].jpc) - group_ids.begin());
    }
    const std::vector<double> total_intensity =
        likelihood.EvaluateNormalizationMCIntensity();
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
    std::vector<double> weight_group(group_ids.size(), 0.0);
    std::vector<double> weight_component(
        static_cast<std::size_t>(number_terms) * number_terms, 0.0);
    tree_mc.Branch("weight", &weight, "weight/D");
    tree_mc.Branch("weight_group", &weight_group);
    tree_mc.Branch(
        "weight_component",
        &weight_component);

    double maximum_closure_residual = 0.0;
    double sum_projection_weight = 0.0;
    for (int batch_begin = 0;
         batch_begin < number_mc;
         batch_begin += component_batch_capacity) {
        const int batch_events = std::min(
            component_batch_capacity, number_mc - batch_begin);
        const std::vector<double> packed_components =
            likelihood.EvaluateNormalizationMCComponentBatch(
                batch_begin, batch_events);
        if (packed_components.size()
            != static_cast<std::size_t>(batch_events) * number_pairs) {
            throw std::runtime_error(
                "projection component batch has an inconsistent size");
        }

        for (int local_event = 0;
             local_event < batch_events;
             ++local_event) {
            const int event = batch_begin + local_event;
            const double projection_scale = effective_yield / sum_pdf;
            weight = total_intensity[event] * projection_scale;
            sum_projection_weight += weight;
            std::fill(weight_group.begin(), weight_group.end(), 0.0);
            std::fill(
                weight_component.begin(), weight_component.end(), -1.0);

            double reconstructed_intensity = 0.0;
            const std::size_t event_offset =
                static_cast<std::size_t>(local_event) * number_pairs;
            // Each packed entry is already K_ii or the complete K_ij+K_ji
            // interference. Group curves retain only pairs internal to that
            // group; cross-group interference remains in the total model.
            for (int first = 0; first < number_terms; ++first) {
                for (int second = first;
                     second < number_terms;
                     ++second) {
                    const double contribution = packed_components[
                        event_offset + ctpwa::component_pair_index(
                            first, second, number_terms)];
                    reconstructed_intensity += contribution;
                    if (term_groups[first] == term_groups[second]) {
                        weight_group[term_groups[first]] +=
                            contribution * projection_scale;
                    }
                    const double component_weight =
                        contribution * projection_scale;
                    weight_component[
                        static_cast<std::size_t>(first) * number_terms
                        + second] = component_weight;
                    weight_component[
                        static_cast<std::size_t>(second) * number_terms
                        + first] = component_weight;
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

    TTree tree_background("bg", "weighted background samples");
    event_values.Book(tree_background);
    int background_index = 0;
    double weight_bg = 0.0;
    tree_background.Branch(
        "background_index", &background_index, "background_index/I");
    tree_background.Branch("weight_bg", &weight_bg, "weight_bg/D");
    for (std::size_t sample_index = 0;
         sample_index < likelihood.NumberBackgroundSamples();
         ++sample_index) {
        const GVVSample& background =
            likelihood.BackgroundSampleAt(sample_index);
        background_index = static_cast<int>(sample_index);
        // The projection adds the negative of the signed likelihood term.
        weight_bg =
            -likelihood.BackgroundLikelihoodCoefficient(sample_index);
        for (int event = 0; event < background.Entries(); ++event) {
            event_values.Load(background, event);
            fill_tree(tree_background);
        }
    }

    TTree background_map("background_map", "background sample index map");
    int background_entries = 0;
    double likelihood_coefficient = 0.0;
    double projection_weight = 0.0;
    char background_label[128] = {0};
    background_map.Branch(
        "background_index", &background_index, "background_index/I");
    background_map.Branch("label", background_label, "label/C");
    background_map.Branch(
        "n_events", &background_entries, "n_events/I");
    background_map.Branch(
        "likelihood_coefficient",
        &likelihood_coefficient,
        "likelihood_coefficient/D");
    background_map.Branch(
        "projection_weight", &projection_weight, "projection_weight/D");
    int n_background_events = 0;
    for (std::size_t sample_index = 0;
         sample_index < likelihood.NumberBackgroundSamples();
         ++sample_index) {
        const GVVSample& background =
            likelihood.BackgroundSampleAt(sample_index);
        background_index = static_cast<int>(sample_index);
        background_entries = background.Entries();
        likelihood_coefficient =
            likelihood.BackgroundLikelihoodCoefficient(sample_index);
        projection_weight = -likelihood_coefficient;
        copy_checked(background_label, background.Label());
        n_background_events += background_entries;
        fill_tree(background_map);
    }

    TTree component_map("component_map", "GVV component index map");
    int component_index = 0;
    int resonance_index = 0;
    int wave_type = 0;
    char component_name[64] = {0};
    char component_label[128] = {0};
    char component_resonance_id[64] = {0};
    char component_wave_id[64] = {0};
    char component_wave_label[128] = {0};
    char component_jpc[16] = {0};
    component_map.Branch(
        "component_index", &component_index, "component_index/I");
    component_map.Branch(
        "resonance_index", &resonance_index, "resonance_index/I");
    component_map.Branch("wave_type", &wave_type, "wave_type/I");
    component_map.Branch("name", component_name, "name/C");
    component_map.Branch("label", component_label, "label/C");
    component_map.Branch(
        "resonance_id", component_resonance_id, "resonance_id/C");
    component_map.Branch("wave_id", component_wave_id, "wave_id/C");
    component_map.Branch(
        "wave_label", component_wave_label, "wave_label/C");
    component_map.Branch("jpc", component_jpc, "jpc/C");
    for (int term = 0; term < number_terms; ++term) {
        component_index = term;
        resonance_index = model.terms[term].resonance_index;
        wave_type = model.term_metadata[term].registered_wave_type;
        copy_checked(component_name, model.term_metadata[term].id);
        copy_checked(component_label, model.term_metadata[term].label);
        copy_checked(
            component_resonance_id,
            model.resonance_metadata[resonance_index].id);
        copy_checked(component_wave_id, model.term_metadata[term].wave_id);
        copy_checked(
            component_wave_label, model.term_metadata[term].wave_latex);
        copy_checked(component_jpc, model.term_metadata[term].jpc);
        fill_tree(component_map);
    }

    TTree group_map("group_map", "GVV JPC group index map");
    int group_index = 0;
    char group_jpc[16] = {0};
    char group_label[32] = {0};
    group_map.Branch("group_index", &group_index, "group_index/I");
    group_map.Branch("jpc", group_jpc, "jpc/C");
    group_map.Branch("label", group_label, "label/C");
    for (std::size_t group = 0; group < group_ids.size(); ++group) {
        group_index = static_cast<int>(group);
        copy_checked(group_jpc, group_ids[group]);
        const std::string label = group_ids[group].size() >= 3
            ? group_ids[group].substr(0, group_ids[group].size() - 2)
                  + "^{" + group_ids[group].substr(group_ids[group].size() - 2)
                  + "}"
            : group_ids[group];
        copy_checked(group_label, label);
        fill_tree(group_map);
    }

    TTree metadata("metadata", "GVV projection provenance");
    int projection_schema_version = 2;
    int n_terms = number_terms;
    int n_groups = static_cast<int>(group_ids.size());
    int n_data = data.Entries();
    int n_normalization_mc = normalization_mc.Entries();
    int n_background_samples =
        static_cast<int>(likelihood.NumberBackgroundSamples());
    long long stored_best_seed = best_seed;
    char stored_output_tag[64] = {0};
    char stored_model_signature[64] = {0};
    char background_method[64] = {0};
    copy_checked(stored_output_tag, output_tag);
    copy_checked(stored_model_signature, model_signature);
    copy_checked(background_method, "signed_weighted_samples");
    metadata.Branch(
        "schema_version",
        &projection_schema_version,
        "schema_version/I");
    metadata.Branch("output_tag", stored_output_tag, "output_tag/C");
    metadata.Branch(
        "model_signature", stored_model_signature, "model_signature/C");
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
    metadata.Branch(
        "n_background_events",
        &n_background_events,
        "n_background_events/I");
    metadata.Branch(
        "background_method", background_method, "background_method/C");
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

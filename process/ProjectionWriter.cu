// GVV fitted-projection ROOT producer.
//
// This process-specific module derives plotting observables from each event,
// converts the fitted normalization-MC intensity and Term-pair components into
// projection weights, and serializes the stable Fit-to-Plotting ROOT contract.
// It does not perform the Minuit fit or write the human/machine fit summaries.
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

// -----------------------------------------------------------------------------
// Common serialization and geometry utilities
// -----------------------------------------------------------------------------

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

double cosine_to_axis(const TVector3& vector, const TVector3& axis)
{
    if (vector.Mag2() <= kProjectionAxisTolerance) {
        return 0.0;
    }
    return vector.Unit().Dot(axis);
}

struct OmegaDecayObservables {
    double cos_theta_decay_plane = 0.0;
    double phi_decay_plane = 0.0;
    double cos_theta_pip = 0.0;
    double cos_theta_pim = 0.0;
    double cos_theta_pi0 = 0.0;
    double decay_plane_normal_magnitude = 0.0;
};

OmegaDecayObservables calculate_omega_decay_observables(
    const TLorentzVector& pip_in_x,
    const TLorentzVector& pim_in_x,
    const TLorentzVector& pi0_in_x,
    const TLorentzVector& omega_in_x,
    const TVector3& x_parent_z_axis)
{
    // omega helicity frame: z follows the omega flight direction in the X
    // rest frame, y is normal to the X -> omega omega decay plane, and x
    // completes the right-handed basis.
    const TVector3 z_axis = safe_unit(
        omega_in_x.Vect(), TVector3(0.0, 0.0, 1.0));
    const TVector3 y_axis = transverse_axis(x_parent_z_axis, z_axis);
    const TVector3 x_axis = safe_unit(
        y_axis.Cross(z_axis), TVector3(1.0, 0.0, 0.0));

    TLorentzVector pip = pip_in_x;
    TLorentzVector pim = pim_in_x;
    TLorentzVector pi0 = pi0_in_x;
    const TVector3 boost_to_omega = -omega_in_x.BoostVector();
    pip.Boost(boost_to_omega);
    pim.Boost(boost_to_omega);
    pi0.Boost(boost_to_omega);

    OmegaDecayObservables result;
    result.cos_theta_pip = cosine_to_axis(pip.Vect(), z_axis);
    result.cos_theta_pim = cosine_to_axis(pim.Vect(), z_axis);
    result.cos_theta_pi0 = cosine_to_axis(pi0.Vect(), z_axis);

    // The oriented three-pion decay-plane normal is p(pi+) cross p(pi-).
    // Its magnitude carries the Dalitz-dependent omega decay-analyser scale.
    const TVector3 decay_plane_normal = pip.Vect().Cross(pim.Vect());
    result.decay_plane_normal_magnitude = decay_plane_normal.Mag();
    if (decay_plane_normal.Mag2() <= kProjectionAxisTolerance) {
        return result;
    }
    const TVector3 unit_normal = decay_plane_normal.Unit();
    result.cos_theta_decay_plane = unit_normal.Dot(z_axis);
    result.phi_decay_plane = std::atan2(
        unit_normal.Dot(y_axis), unit_normal.Dot(x_axis));
    return result;
}

// -----------------------------------------------------------------------------
// Event-level kinematic observables and their ROOT branch schema
// -----------------------------------------------------------------------------

struct ProjectionEvent {
    // Four-momenta use the ROOT convention [px, py, pz, E]. They are stored
    // in the same frame as the input event record.
    double p4_pip1[4];   // Four-momentum of pi+ from the first omega candidate.
    double p4_pim1[4];   // Four-momentum of pi- from the first omega candidate.
    double p4_pi01[4];   // Four-momentum of pi0 from the first omega candidate.
    double p4_pip2[4];   // Four-momentum of pi+ from the second omega candidate.
    double p4_pim2[4];   // Four-momentum of pi- from the second omega candidate.
    double p4_pi02[4];   // Four-momentum of pi0 from the second omega candidate.
    double p4_gam[4];    // Four-momentum of the radiative photon.
    double p4_omega1[4]; // Four-momentum of omega1 = pi+_1 + pi-_1 + pi0_1.
    double p4_omega2[4]; // Four-momentum of omega2 = pi+_2 + pi-_2 + pi0_2.
    double p4_X[4];      // Four-momentum of X = omega1 + omega2.

    double m_omega1;       // Invariant mass M(pi+_1 pi-_1 pi0_1).
    double m_omega2;       // Invariant mass M(pi+_2 pi-_2 pi0_2).
    double m_omegaomega;   // Invariant mass M(omega1 omega2), i.e. M(X).
    double m_gammaomega1;  // Invariant mass M(gamma omega1).
    double m_gammaomega2;  // Invariant mass M(gamma omega2).
    double m_pip1_pim1;    // Invariant mass M(pi+_1 pi-_1).
    double m_pip1_pi01;    // Invariant mass M(pi+_1 pi0_1).
    double m_pim1_pi01;    // Invariant mass M(pi-_1 pi0_1).
    double m_pip2_pim2;    // Invariant mass M(pi+_2 pi-_2).
    double m_pip2_pi02;    // Invariant mass M(pi+_2 pi0_2).
    double m_pim2_pi02;    // Invariant mass M(pi-_2 pi0_2).

    // Angular observables follow the coordinate convention documented in
    // CalculateAngularObservables() below. Azimuths are stored in radians.
    double cos_theta_gamma; // Photon polar cosine in the psi rest frame.

    // Direction of omega1 in the X = omega1 omega2 helicity frame. The omega2
    // direction is back-to-back and therefore is not stored separately.
    double cos_theta_omega1; // Polar cosine of omega1 relative to z_X.
    double phi_omega1;       // Azimuth of omega1 relative to the production plane.

    // Direction of each oriented 3-pion decay-plane normal in its parent
    // omega helicity frame. The normal is p(pi+) cross p(pi-).
    double cos_theta_decay_plane_omega1; // Polar cosine of the omega1 normal.
    double phi_decay_plane_omega1;       // Azimuth of the omega1 normal.
    double cos_theta_decay_plane_omega2; // Polar cosine of the omega2 normal.
    double phi_decay_plane_omega2;       // Azimuth of the omega2 normal.
    double delta_phi_decay_planes;       // Wrapped phi_plane1 - phi_plane2.

    // Pion polar cosines in the corresponding omega helicity frame.
    double cos_theta_pip_omega1; // pi+ from omega1 relative to z_1.
    double cos_theta_pim_omega1; // pi- from omega1 relative to z_1.
    double cos_theta_pi0_omega1; // pi0 from omega1 relative to z_1.
    double cos_theta_pip_omega2; // pi+ from omega2 relative to z_2.
    double cos_theta_pim_omega2; // pi- from omega2 relative to z_2.
    double cos_theta_pi0_omega2; // pi0 from omega2 relative to z_2.

    // Magnitude of p(pi+) cross p(pi-) in each omega rest frame. For project
    // four-momenta in GeV, these quantities have units of GeV^2.
    double decay_plane_normal_magnitude_omega1;
    double decay_plane_normal_magnitude_omega2;

    void Book(TTree& tree)
    {
        // Stored final-state and composite four-momenta.
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

        // Stored invariant masses.
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

        // Photon and omega1 production angles.
        tree.Branch(
            "cos_theta_gamma", &cos_theta_gamma, "cos_theta_gamma/D");
        tree.Branch(
            "cos_theta_omega1", &cos_theta_omega1, "cos_theta_omega1/D");
        tree.Branch("phi_omega1", &phi_omega1, "phi_omega1/D");

        // Complete polar and azimuthal angles of both decay-plane normals.
        tree.Branch(
            "cos_theta_decay_plane_omega1",
            &cos_theta_decay_plane_omega1,
            "cos_theta_decay_plane_omega1/D");
        tree.Branch(
            "phi_decay_plane_omega1",
            &phi_decay_plane_omega1,
            "phi_decay_plane_omega1/D");
        tree.Branch(
            "cos_theta_decay_plane_omega2",
            &cos_theta_decay_plane_omega2,
            "cos_theta_decay_plane_omega2/D");
        tree.Branch(
            "phi_decay_plane_omega2",
            &phi_decay_plane_omega2,
            "phi_decay_plane_omega2/D");
        tree.Branch(
            "delta_phi_decay_planes",
            &delta_phi_decay_planes,
            "delta_phi_decay_planes/D");

        // Polar angles of all six pions in their parent omega helicity frames.
        tree.Branch(
            "cos_theta_pip_omega1",
            &cos_theta_pip_omega1,
            "cos_theta_pip_omega1/D");
        tree.Branch(
            "cos_theta_pim_omega1",
            &cos_theta_pim_omega1,
            "cos_theta_pim_omega1/D");
        tree.Branch(
            "cos_theta_pi0_omega1",
            &cos_theta_pi0_omega1,
            "cos_theta_pi0_omega1/D");
        tree.Branch(
            "cos_theta_pip_omega2",
            &cos_theta_pip_omega2,
            "cos_theta_pip_omega2/D");
        tree.Branch(
            "cos_theta_pim_omega2",
            &cos_theta_pim_omega2,
            "cos_theta_pim_omega2/D");
        tree.Branch(
            "cos_theta_pi0_omega2",
            &cos_theta_pi0_omega2,
            "cos_theta_pi0_omega2/D");

        // Unnormalized omega decay-plane analyser magnitudes.
        tree.Branch(
            "decay_plane_normal_magnitude_omega1",
            &decay_plane_normal_magnitude_omega1,
            "decay_plane_normal_magnitude_omega1/D");
        tree.Branch(
            "decay_plane_normal_magnitude_omega2",
            &decay_plane_normal_magnitude_omega2,
            "decay_plane_normal_magnitude_omega2/D");
    }

    void Load(const GVVSample& sample, int event)
    {
        // Read the seven final-state particles from the process sample.
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

        // Reconstruct the two omega candidates, their parent X, and psi.
        const TLorentzVector omega1 = pip1 + pim1 + pi01;
        const TLorentzVector omega2 = pip2 + pim2 + pi02;
        const TLorentzVector x_state = omega1 + omega2;
        const TLorentzVector psi = x_state + gamma;

        // Serialize input and composite four-momenta in [px, py, pz, E].
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

        // Calculate the omega, omega-omega, gamma-omega, and pion-pair masses.
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

        CalculateAngularObservables(
            pip1, pim1, pi01, pip2, pim2, pi02, gamma, psi);
    }

private:
    void CalculateAngularObservables(
        const TLorentzVector& pip1,
        const TLorentzVector& pim1,
        const TLorentzVector& pi01,
        const TLorentzVector& pip2,
        const TLorentzVector& pim2,
        const TLorentzVector& pi02,
        const TLorentzVector& gamma,
        const TLorentzVector& psi)
    {
        // Coordinate-system convention
        // ----------------------------
        // 1. psi rest frame: the global +z_beam axis is the e+e- beam axis.
        //    theta_gamma is the radiative-photon polar angle relative to it.
        // 2. X rest frame: +z_X points opposite to the radiative photon,
        //    equivalently along the X flight direction in the psi rest frame.
        //    The production plane is spanned by z_beam and z_X. We choose
        //      y_X = unit(z_beam cross z_X),
        //      x_X = unit(y_X cross z_X).
        //    theta_omega1 and phi_omega1 locate omega1 in this basis.
        // 3. omega_i rest frame: +z_i follows the omega_i flight direction in
        //    the X rest frame, y_i = unit(z_X cross z_i), and
        //    x_i = unit(y_i cross z_i). The oriented decay-plane normal is
        //      n_i = unit(p(pi+_i) cross p(pi-_i)).
        //    Its polar and azimuthal angles are measured in (x_i,y_i,z_i).

        const TVector3 beam_z_axis(0.0, 0.0, 1.0);

        // Boost every final-state particle into the psi rest frame.
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

        // cos(theta_gamma): photon polar angle relative to the beam +z axis.
        cos_theta_gamma = safe_unit(
            in_psi[kGamma].Vect(), beam_z_axis).Dot(beam_z_axis);

        // Boost from the psi rest frame into the omega-omega (X) rest frame.
        std::array<TLorentzVector, GVV_NFINAL_PARTICLES> in_x = in_psi;
        const TVector3 boost_to_x = -x_psi.BoostVector();
        for (TLorentzVector& vector : in_x) {
            vector.Boost(boost_to_x);
        }
        const TLorentzVector omega1_x =
            in_x[kPip1] + in_x[kPim1] + in_x[kPi01];
        const TLorentzVector omega2_x =
            in_x[kPip2] + in_x[kPim2] + in_x[kPi02];
        const TVector3 x_z_axis = safe_unit(
            -in_x[kGamma].Vect(), beam_z_axis);
        const TVector3 x_y_axis = transverse_axis(beam_z_axis, x_z_axis);
        const TVector3 x_x_axis = safe_unit(
            x_y_axis.Cross(x_z_axis), TVector3(1.0, 0.0, 0.0));

        // Complete omega1 direction in the X helicity frame.
        const TVector3 omega1_direction = safe_unit(
            omega1_x.Vect(), TVector3(0.0, 0.0, 1.0));
        cos_theta_omega1 = omega1_direction.Dot(x_z_axis);
        phi_omega1 = std::atan2(
            omega1_direction.Dot(x_y_axis),
            omega1_direction.Dot(x_x_axis));

        // Complete decay-plane-normal and pion polar-angle observables for
        // both omega candidates in their respective helicity frames.
        const OmegaDecayObservables omega1_decay =
            calculate_omega_decay_observables(
                in_x[kPip1], in_x[kPim1], in_x[kPi01],
                omega1_x, x_z_axis);
        const OmegaDecayObservables omega2_decay =
            calculate_omega_decay_observables(
                in_x[kPip2], in_x[kPim2], in_x[kPi02],
                omega2_x, x_z_axis);

        cos_theta_decay_plane_omega1 =
            omega1_decay.cos_theta_decay_plane;
        phi_decay_plane_omega1 = omega1_decay.phi_decay_plane;
        cos_theta_pip_omega1 = omega1_decay.cos_theta_pip;
        cos_theta_pim_omega1 = omega1_decay.cos_theta_pim;
        cos_theta_pi0_omega1 = omega1_decay.cos_theta_pi0;
        decay_plane_normal_magnitude_omega1 =
            omega1_decay.decay_plane_normal_magnitude;

        cos_theta_decay_plane_omega2 =
            omega2_decay.cos_theta_decay_plane;
        phi_decay_plane_omega2 = omega2_decay.phi_decay_plane;
        cos_theta_pip_omega2 = omega2_decay.cos_theta_pip;
        cos_theta_pim_omega2 = omega2_decay.cos_theta_pim;
        cos_theta_pi0_omega2 = omega2_decay.cos_theta_pi0;
        decay_plane_normal_magnitude_omega2 =
            omega2_decay.decay_plane_normal_magnitude;

        // Signed difference of the two local decay-plane azimuths, wrapped to
        // (-pi, pi]. This is not an unsigned geometric plane-opening angle.
        delta_phi_decay_planes = wrap_angle(
            phi_decay_plane_omega1 - phi_decay_plane_omega2);
    }
};

// -----------------------------------------------------------------------------
// Model grouping, batch sizing, and display-label bookkeeping
// -----------------------------------------------------------------------------

struct ProjectionGrouping {
    std::vector<std::string> group_ids;
    std::vector<int> term_groups;
};

ProjectionGrouping build_projection_grouping(
    const GVVCompiledModel& model,
    int number_terms)
{
    ProjectionGrouping grouping;

    // Preserve the first-appearance order of JPC labels in model.json.
    for (const GVVTermMetadata& term : model.term_metadata) {
        if (std::find(
                grouping.group_ids.begin(),
                grouping.group_ids.end(),
                term.jpc)
            == grouping.group_ids.end()) {
            grouping.group_ids.push_back(term.jpc);
        }
    }

    // Record the JPC group index of each active Term.
    grouping.term_groups.assign(number_terms, -1);
    for (int term = 0; term < number_terms; ++term) {
        grouping.term_groups[term] = static_cast<int>(std::find(
            grouping.group_ids.begin(),
            grouping.group_ids.end(),
            model.term_metadata[term].jpc) - grouping.group_ids.begin());
    }

    return grouping;
}

int projection_component_batch_capacity(
    int number_mc,
    int number_terms,
    int number_pairs)
{
    // The packed values exist once in FitLikelihood's managed scratch and
    // once in this host batch while ROOT rows are serialized.
    const std::size_t component_bytes_per_event =
        2ULL * static_cast<std::size_t>(number_pairs) * sizeof(double)
        + static_cast<std::size_t>(number_terms) * sizeof(DeviceComplex);
    const std::size_t capacity_by_bytes = std::max<std::size_t>(
        1, kProjectionComponentScratchBytes / component_bytes_per_event);
    return static_cast<int>(
        std::min<std::size_t>(number_mc, capacity_by_bytes));
}

std::string make_group_label(const std::string& jpc)
{
    return jpc.size() >= 3
        ? jpc.substr(0, jpc.size() - 2)
              + "^{" + jpc.substr(jpc.size() - 2) + "}"
        : jpc;
}

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
    // -------------------------------------------------------------------------
    // 1. Collect the fitted model and input samples
    // -------------------------------------------------------------------------

    const GVVCompiledModel& model = likelihood.Model();
    const GVVSample& normalization_mc =
        likelihood.NormalizationMCSample();
    const GVVSample& data = likelihood.DataSample();
    const int number_terms = likelihood.NumberTerms();
    const int number_mc = normalization_mc.Entries();
    const int number_pairs = ctpwa::component_pair_count(number_terms);

    const ProjectionGrouping grouping =
        build_projection_grouping(model, number_terms);
    const std::vector<std::string>& group_ids = grouping.group_ids;
    const std::vector<int>& term_groups = grouping.term_groups;
    const int component_batch_capacity =
        projection_component_batch_capacity(
            number_mc, number_terms, number_pairs);

    // -------------------------------------------------------------------------
    // 2. Establish the common MC normalization for all fitted weights
    // -------------------------------------------------------------------------

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

    // Every fitted MC, JPC-group, and Term-pair weight uses this one scale.
    const double projection_scale = effective_yield / sum_pdf;

    // -------------------------------------------------------------------------
    // 3. Create the output file and the accepted-MC projection tree
    // -------------------------------------------------------------------------

    TFile output(save_name.c_str(), "RECREATE");
    if (output.IsZombie()) {
        throw std::runtime_error(
            "cannot create projection ROOT file: " + save_name);
    }

    ProjectionEvent event_values;
    TTree tree_mc("MC", "accepted normalization MC with fitted weights");
    event_values.Book(tree_mc);

    // weight is the full coherent fitted model. weight_group contains each
    // JPC group's internal coherent sum, excluding cross-group interference.
    // weight_component stores the symmetric Term-pair matrix: diagonals are
    // individual intensities and off-diagonals are complete interference
    // contributions K_ij + K_ji.
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

    // Evaluate Term-pair components in bounded batches, then immediately
    // serialize one accepted-MC row per event.
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

    // -------------------------------------------------------------------------
    // 4. Serialize selected data and dynamic signed-background samples
    // -------------------------------------------------------------------------

    TTree tree_data("Data", "selected data");
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
        // Plotting adds the negative of the signed likelihood contribution
        // to the fitted signal-MC histogram.
        weight_bg =
            -likelihood.BackgroundLikelihoodCoefficient(sample_index);
        for (int event = 0; event < background.Entries(); ++event) {
            event_values.Load(background, event);
            fill_tree(tree_background);
        }
    }

    // -------------------------------------------------------------------------
    // 5. Serialize background, Term-component, and JPC-group index maps
    // -------------------------------------------------------------------------

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
        const std::string label = make_group_label(group_ids[group]);
        copy_checked(group_label, label);
        fill_tree(group_map);
    }

    // -------------------------------------------------------------------------
    // 6. Serialize fit and schema provenance
    // -------------------------------------------------------------------------

    TTree metadata("metadata", "GVV projection provenance");
    int projection_schema_version = 4;
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

    // -------------------------------------------------------------------------
    // 7. Commit all trees to disk and report the normalization closure
    // -------------------------------------------------------------------------

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

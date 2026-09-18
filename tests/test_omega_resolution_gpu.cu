#define GVV_POST_NO_MAIN
#define GVV_FIT_NO_MAIN
// End-to-end GPU regression: cache invalidation, signed likelihood,
// projection components and Post integration share the same omega factors.
#include "fit/Fit.cu"
#include "core/Model.h"
#include "core/physics/OmegaResolution.cuh"
#include "core/IO.h"
#include "post/Post.cu"
#include <TFile.h>
#include <TTree.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unistd.h>

namespace {
constexpr int kEvents = 6;
using Masses = std::array<std::array<double,2>,kEvents>;
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
bool close(double a, double b, double tolerance=2.e-8)
{
    return std::fabs(a-b) <= tolerance * std::max({1.0,std::fabs(a),std::fabs(b)});
}
std::complex<double> complex(DeviceComplex v) { return {v.real,v.imag}; }
Masses fixture(const std::string& path)
{
    // Numerical four-vector fixtures, not simulated detector events. Pion
    // momenta vary across both omega peaks and rotate relative to the beam.
    const double base[7][4] = {
        {0.12,-0.03,-0.20,0.30}, {-0.15,0.01,0.04,0.27}, {0.03,0.02,0.16,0.22},
        {-0.10,-0.02,0.18,0.29}, {0.12,-0.02,-0.03,0.28}, {-0.02,0.04,-0.15,0.23},
        {0.61,-0.43,1.92,2.060097}};
    TFile file(path.c_str(),"RECREATE");
    TTree tree("Pwa","omega resolution numerical fixture");
    GVVBranchConfig branches;
    double p4[7][4] = {};
    for(int p=0;p<7;++p) {
        const auto& name=branches.branches[p];
        tree.Branch(name.c_str(),p4[p],(name+"[4]/D").c_str());
    }
    Masses masses{};
    for(int event=0;event<kEvents;++event) {
        const double angle=0.3*event, scale=0.96+0.016*event;
        for(int p=0;p<7;++p) {
            const double spatial_scale=p==6?1.0:scale;
            const double mass2=std::max(0.0,base[p][3]*base[p][3]-base[p][0]*base[p][0]
                -base[p][1]*base[p][1]-base[p][2]*base[p][2]);
            p4[p][0]=spatial_scale*(std::cos(angle)*base[p][0]-std::sin(angle)*base[p][1]);
            p4[p][1]=spatial_scale*(std::sin(angle)*base[p][0]+std::cos(angle)*base[p][1]);
            p4[p][2]=spatial_scale*base[p][2];
            p4[p][3]=std::sqrt(mass2+p4[p][0]*p4[p][0]+p4[p][1]*p4[p][1]+p4[p][2]*p4[p][2]);
        }
        for(int omega=0;omega<2;++omega) {
            double sum[4]={};
            for(int p=3*omega;p<3*omega+3;++p)
                for(int c=0;c<4;++c) sum[c]+=p4[p][c];
            masses[event][omega]=sum[3]*sum[3]-sum[0]*sum[0]-sum[1]*sum[1]-sum[2]*sum[2];
        }
        tree.Fill();
    }
    tree.Write();
    return masses;
}
}
int main()
{
    const std::string stem="/tmp/gvv-omega-gpu-"+std::to_string(getpid());
    const std::string input=stem+".root", projection=stem+"-projection.root";
    try {
        int devices=0;
        require(cudaGetDeviceCount(&devices)==cudaSuccess && devices>0,"CUDA device required");
        const auto masses=fixture(input);
        auto model=gvv_load_compiled_model("config/model.json");
        const auto layout=gvv_fit_parameter_layout(model);
        std::vector<double> values;
        for(const auto& binding:layout) values.push_back(binding.fit.initial_value);
        require(layout.back().target==GVVFitParameterTarget::OmegaResolutionSigma,"sigma layout changed");
        FitLikelihood fit(model);
        fit.LoadNormalizationMC(input);
        fit.LoadData(input);
        fit.AddBackground(input,-0.5,"SB1");
        fit.AddBackground(input,0.25,"SB2");
        fit.Prepare();
        GVVSample norm("test normalization");
        norm.Load(input, GVVBranchConfig{});
        GVVAmplitude amplitude(model);
        amplitude.Prepare(norm);
        auto parameters = model.initial_parameters;
        auto intensity = [&]() {
            amplitude.SetParameters(parameters);
            const double* values = amplitude.EvaluateIntensity(norm);
            return std::vector<double>(values, values + kEvents);
        };
        parameters.omega_resolution_sigma=0.0;
        const auto baseline=intensity();
        const auto baseline_pairs=amplitude.EvaluateComponentBatch(norm,0,kEvents);
        const int pairs=baseline_pairs.size()/kEvents;
        OmegaWidthTable table;
        table.Build();
        GVVComponentEvaluator post(input,input,GVVBranchConfig(),model,layout);
        for(double sigma:{0.0,0.0001,0.005,0.009,0.005,0.0}) {
            fit.Parameters().omega_resolution_sigma=sigma;
            parameters.omega_resolution_sigma=sigma;
            const double ll=fit.LogLikelihood();
            const auto actual=intensity();
            const auto components=amplitude.EvaluateComponentBatch(norm,0,kEvents);
            const auto subset=amplitude.EvaluateComponentBatch(norm,2,3);
            std::vector<double> expected(kEvents),integrals(pairs,0.0);
            for(int e=0;e<kEvents;++e) {
                double ratio=1.0;
                for(int omega=0;omega<2;++omega) {
                    const double s=masses[e][omega];
                    ratio*=std::norm(complex(gvv_smeared_omega_propagator(s,table.HostView(),sigma))
                        /complex(gvv_omega_propagator(s,table.HostView())));
                }
                expected[e]=baseline[e]*ratio;
                require(close(actual[e],expected[e]),"CPU/GPU omega reweighting disagrees");
                double component_sum=0.0;
                for(int p=0;p<pairs;++p) {
                    const double predicted=baseline_pairs[e*pairs+p]*ratio;
                    require(close(components[e*pairs+p],predicted),"component did not receive common omega factor");
                    integrals[p]+=predicted;
                    component_sum+=components[e*pairs+p];
                    if(e>=2 && e<5) require(close(subset[(e-2)*pairs+p],components[e*pairs+p]),
                        "omega cache indexed relative to component batch instead of whole sample");
                }
                require(close(component_sum,actual[e]),"projection component closure failed");
            }
            const double mean=std::accumulate(expected.begin(),expected.end(),0.0)/kEvents;
            double expected_ll=0.0;
            for(double intensity:expected) expected_ll+=0.75*std::log(intensity/mean);
            require(close(ll,expected_ll),"data/background/normalization sigma update is inconsistent");
            require((norm.OmegaFactorBuffer()==nullptr)==(sigma==0.0),
                "cache did not switch between legacy and smeared paths");
            if(sigma>0.0) {
                values.back()=std::log(sigma);
                const auto result=post.Evaluate(values);
                post.ValidateTotal(values,result);
                for(int p=0;p<pairs;++p) require(close(result.truth[p],integrals[p])
                    && close(result.selected[p],integrals[p]),"Post differs from fit/projection amplitude");
            } else {
                for(int e=0;e<kEvents;++e) require(actual[e]==baseline[e],"sigma=0 restoration is not exact");
            }
        }
        fit.Parameters().omega_resolution_sigma=0.007;
        ctpwa::FitAttempt best; best.start_index=0; best.seed=1; best.minimum=0.0;
        fit.WriteProjection(projection,"omega-test",gvv_model_signature(model.definition),best);
        TFile file(projection.c_str(),"READ");
        auto* metadata=dynamic_cast<TTree*>(file.Get("metadata"));
        require(metadata!=nullptr,"projection metadata missing");
        double sigma=-1.0,mean=-1.0;
        require(metadata->SetBranchAddress("omega_resolution_sigma",&sigma)>=0
                && metadata->SetBranchAddress("omega_resolution_mean",&mean)>=0,"resolution metadata missing");
        metadata->GetEntry(0);
        require(sigma==0.007 && mean==0.0,"projection did not record current sigma");
        file.Close();
        std::remove(input.c_str()); std::remove(projection.c_str());
        std::cout<<"Omega resolution GPU likelihood/projection/Post tests passed\n";
    } catch(const std::exception& error) {
        std::remove(input.c_str()); std::remove(projection.c_str());
        std::cerr<<error.what()<<'\n'; return 1;
    }
}

#include "causal_rig_fit.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc,char** argv){try{
    if(argc!=7)return 2;std::string mode=argv[6];if(mode!="fresh"&&mode!="causal"&&mode!="guarded")return 2;
    std::ifstream left(argv[1]),right(argv[2]),calibration(argv[3]),input(argv[4]);
    if(!left||!right||!calibration||!input||std::ifstream(argv[5]).good())return 3;
    std::array<fhn::HandRig,2> rigs{fhn::HandRig::load(left),fhn::HandRig::load(right)};
    const auto cameras=fhn::load_rig_cameras(calibration);fhn::CausalRigFitter tracker;
    std::ofstream output(argv[5]);if(!output)return 4;output<<std::setprecision(17);
    std::int64_t stamp;int source_id;unsigned count=0;
    while(input>>stamp){fhn::RigPose seed;fhn::RigFitOptions options;std::array<fhn::PixelObservation,21> observations{};
        if(!(input>>source_id>>seed.side)||seed.side<0||seed.side>1)return 5;
        if(!(input>>seed.wrist.q.w>>seed.wrist.q.x>>seed.wrist.q.y>>seed.wrist.q.z>>seed.wrist.t.x>>seed.wrist.t.y>>seed.wrist.t.z))return 5;
        for(auto& angle:seed.angles)if(!(input>>angle))return 5;
        if(!(input>>options.iterations>>options.huber>>options.angle_prior))return 5;
        for(auto& o:observations)for(int v=0;v<2;++v){int present;
            if(!(input>>present>>o.point[v].x>>o.point[v].y>>o.sigma[v])||present<0||present>1)return 5;o.present[v]=present;}
        const auto begin=std::chrono::steady_clock::now();fhn::CausalRigResult diagnostics;
        if(mode!="fresh")diagnostics=tracker.fit(rigs[seed.side],seed,cameras,observations,stamp,source_id,options,mode=="guarded");
        else{diagnostics.fit=fhn::fit_rig_pixels(rigs[seed.side],seed,cameras,observations,options);diagnostics.fresh_seed_cost=diagnostics.history_seed_cost=diagnostics.fit.initial_cost;}
        double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();const auto& r=diagnostics.fit;const auto& pose=r.pose;
        output<<r.sufficient<<' '<<r.observations<<' '<<r.iterations<<' '<<r.updates<<' '<<r.initial_cost<<' '<<r.cost<<' '<<r.observed_rmse_px<<' '<<us;
        output<<' '<<pose.side<<' '<<pose.wrist.q.w<<' '<<pose.wrist.q.x<<' '<<pose.wrist.q.y<<' '<<pose.wrist.q.z<<' '<<pose.wrist.t.x<<' '<<pose.wrist.t.y<<' '<<pose.wrist.t.z;
        for(auto angle:pose.angles)output<<' '<<angle;
        for(auto point:rigs[seed.side].landmarks(pose.angles,pose.wrist,pose.side))output<<' '<<point.x<<' '<<point.y<<' '<<point.z;
        output<<' '<<diagnostics.history_eligible<<' '<<diagnostics.used_history<<' '<<diagnostics.fresh_seed_cost<<' '<<diagnostics.history_seed_cost<<' '<<diagnostics.fit_count<<'\n';++count;
    }
    if(!input.eof()||!output.good()||!count)return 6;std::cout<<"poses="<<count<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 7;}}

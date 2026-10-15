// Saved-observation fitter; output includes diagnostics and fitted landmarks.
#include "fit_rig_pixels.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc,char** argv){try{
    if(argc!=5&&argc!=6)return 2;std::string mode=argc==6?argv[5]:"finite";int projection_mode=mode=="finite"?0:mode=="analytic"?1:mode=="hybrid"?2:-1;if(projection_mode<0)return 2;std::ifstream profile(argv[1]),calibration(argv[2]),input(argv[3]);if(!profile||!calibration||!input||std::ifstream(argv[4]).good())return 3;
    const auto rig=fhn::HandRig::load(profile);auto cameras=fhn::load_rig_cameras(calibration);std::ofstream output(argv[4]);if(!output)return 4;output<<std::setprecision(17);
    unsigned count=0;fhn::RigPose seed;
    while(input>>seed.side){fhn::RigFitOptions options;std::array<fhn::PixelObservation,21> obs{};
        if(!(input>>seed.wrist.q.w>>seed.wrist.q.x>>seed.wrist.q.y>>seed.wrist.q.z>>seed.wrist.t.x>>seed.wrist.t.y>>seed.wrist.t.z))return 5;
        for(auto& a:seed.angles)if(!(input>>a))return 5;
        if(!(input>>options.iterations>>options.huber>>options.angle_prior))return 5;
        for(auto& o:obs)for(int v=0;v<2;++v){int present;if(!(input>>present>>o.point[v].x>>o.point[v].y>>o.sigma[v])||present<0||present>1)return 5;o.present[v]=present;}
        auto begin=std::chrono::steady_clock::now();auto result=fhn::fit_rig_pixels(rig,seed,cameras,obs,options,projection_mode);
        double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();auto points=rig.landmarks(result.pose.angles,result.pose.wrist,result.pose.side);
        output<<result.sufficient<<' '<<result.observations<<' '<<result.iterations<<' '<<result.updates<<' '<<result.initial_cost<<' '<<result.cost<<' '<<result.observed_rmse_px<<' '<<us;
        auto& p=result.pose;output<<' '<<p.side<<' '<<p.wrist.q.w<<' '<<p.wrist.q.x<<' '<<p.wrist.q.y<<' '<<p.wrist.q.z<<' '<<p.wrist.t.x<<' '<<p.wrist.t.y<<' '<<p.wrist.t.z;for(auto a:p.angles)output<<' '<<a;for(auto v:points)output<<' '<<v.x<<' '<<v.y<<' '<<v.z;output<<'\n';++count;
    }if(!input.eof()||!output.good()||!count)return 6;std::cout<<"poses="<<count<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 7;}}

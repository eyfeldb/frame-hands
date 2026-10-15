// Read one retained ring sample per forward camera; no image export or input.
#include "io.h"
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <unistd.h>

int main(int argc,char** argv) {
    const bool exhaustive_only=argc>1 && std::string(argv[1])=="--exhaustive";
    Ring ring;Nets nets;std::string error;
    if(!ring.open((run_dir()+"/cam-ring").c_str(),error) ||
       !nets.load("/home/steamos/devkit-game/FrameHands/upstream/models/ncnn",false,error)) {
        std::fprintf(stderr,"%s\n",error.c_str());return 1;
    }
    std::printf("diagnostic_only=1 ring_live=%d historical_samples_allowed=1\n",ring.alive());
    std::map<std::string,Camera> calibration;
    if(!load_calibration(calibration,error)){std::fprintf(stderr,"%s\n",error.c_str());return 1;}
    Pool pool(3,{5,6,7});
    if(argc>1 && std::string(argv[1])=="--sequence"){
        std::map<std::string,std::vector<uint8_t>> data;
        std::map<std::string,Image> images;
        for(int ci=0;ci<ring.cameras();++ci){
            const auto& c=ring.camera(ci);fh_ring_slot_t meta{};
            if(c.node!=0 && c.node!=3)continue;
            std::string name=c.node==0?"slam_right":"slam_left";
            if(ring.read(ci,ring.latest(ci),data[name],&meta))
                images[name]={data[name].data(),int(c.width),int(c.height),int(c.width)};
        }
        Contrast palm;Contrast::parse("clahe:2",palm);nets.set_contrast(palm,Contrast{Contrast::None});
        Tracker tracker(calibration,nets,pool);
        std::puts("static_pair_sequence=1 synthetic_timestamps=1 not_live_motion_or_latency_evidence=1");
        int both=0,duplicates=0;
        for(int frame=0;frame<180;++frame){
            auto hands=tracker.step(images,1000000000ll+int64_t(frame)*33333333);
            int left=0,right=0,stereo=0;
            for(auto hand:hands){(hand->right()?right:left)++;stereo+=hand->nviews>=2;}
            both+=left==1 && right==1;duplicates+=left>1 || right>1;
            if(frame%15==0)std::printf("  simulated_frame=%d left=%d right=%d stereo=%d\n",frame,left,right,stereo);
        }
        std::printf("static_pair_both_hands_frames=%d duplicate_side_frames=%d\n",both,duplicates);
        return 0;
    }
    for(int ci=0;ci<ring.cameras();++ci){
        const auto& cam=ring.camera(ci);
        if(cam.width!=1056 || cam.height!=1024)continue;
        std::vector<uint8_t> pixels;fh_ring_slot_t meta{};
        if(!ring.read(ci,ring.latest(ci),pixels,&meta))continue;
        Image img{pixels.data(),int(cam.width),int(cam.height),int(cam.width)};
        std::printf("video%d frame=%llu age_ms=%.1f mean=%.1f\n",cam.node,
            (unsigned long long)meta.frame,(mono_ns()-meta.publish_ns)/1e6,
            std::accumulate(pixels.begin(),pixels.end(),0.0)/pixels.size());
        for(const char* mode:{"default","none","stretch","clahe:2"}){
            Contrast contrast;Contrast::parse(std::string(mode)=="default"?"clahe:2":mode,contrast);
            nets.set_contrast(contrast,std::string(mode)=="default"?Contrast{Contrast::None}:contrast);
            if(exhaustive_only){
                Tracker tracker(calibration,nets,pool);
                auto views=tracker.exhaustive({{cam.node==0?"slam_right":"slam_left",img}});
                std::printf("  exhaustive contrast=%s accepted_views=%zu\n",mode,views.size());
                for(const auto& v:views)std::printf("    presence=%.3f right=%.3f roi=%.1f,%.1f,%.1f rotation=%.1f\n",
                    v.lm.presence,v.lm.right,v.roi.center[0],v.roi.center[1],v.roi.size,v.roi.rotation*180/M_PI);
                continue;
            }
            for(int quarter=0;quarter<24;++quarter){
                int palms=0,landmarks=0;double highest=0,best_right=-1;
                for(int grid:{1,2,3}){
                    double size=cam.width*(grid==1?1.0:grid==2?.65:.45);
                    for(int gx=0;gx<grid;++gx)for(int gy=0;gy<grid;++gy){
                        V2 center{grid==1?cam.width/2.:size/2+(cam.width-size)*gx/(grid-1),
                                  grid==1?cam.height/2.:size/2+(cam.height-size)*gy/(grid-1)};
                        auto found=nets.palms(img,center,size,quarter*M_PI/12);
                        palms+=int(found.size());
                        for(const auto& palm:found){
                            auto lm=nets.landmarks(img,palm.roi());
                            if(lm.presence>highest){highest=lm.presence;best_right=lm.right;}
                            landmarks+=lm.presence>=.5;
                        }
                    }
                }
                std::printf("  contrast=%s rotation=%d palms=%d accepted_crops=%d max_presence=%.3f best_right=%.3f\n",
                    mode,quarter*15,palms,landmarks,highest,best_right);
            }
        }
    }
}

// Offline, exact-recorded-frame diagnostic. Never publishes or captures images.
#include "io.h"
#include "record.h"
#include <json/json.h>
#include <fstream>
#include <algorithm>

int main(int argc,char**argv) {
    if(argc!=4 && argc!=7){std::fprintf(stderr,"usage: probe RECORD_DIR SET_INDEX OUTPUT_JSON [CAMERA CENTER_X CENTER_Y]\n");return 1;}
    SetReader input;Nets nets;std::string error;
    std::map<std::string,Camera> cameras;
    if(!input.open(argv[1],error)||!load_calibration(cameras,error)||
       !nets.load("/home/steamos/devkit-game/FrameHands/upstream/models/ncnn",false,error)){
        std::fprintf(stderr,"%s\n",error.c_str());return 1;
    }
    std::vector<fh_set_cam_t> metadata;std::vector<std::vector<uint8_t>> pixels;
    const int index=std::stoi(argv[2]);
    for(int i=0;i<=index;++i)if(!input.next(metadata,pixels))return 2;
    std::map<std::string,Image> images;
    for(size_t i=0;i<metadata.size();++i){auto& c=metadata[i];images[c.name]={pixels[i].data(),int(c.width),int(c.height),int(c.width)};}
    Pool pool(3,{5,6,7});Tracker tracker(cameras,nets,pool);
    Json::Value result;result["record_index"]=index;result["kind"]="offline diagnostic, not live output";
    auto append=[&](const char*method,const std::string&camera,const Landmarks&lm){
        Json::Value item;item["method"]=method;item["camera"]=camera;item["presence"]=lm.presence;item["right"]=lm.right;
        for(const V2&p:lm.pts){Json::Value point;point.append(p[0]);point.append(p[1]);item["points"].append(point);}
        result["views"].append(item);
    };
    const auto exhaustive=tracker.exhaustive(images);
    for(const Seen&view:exhaustive)append("exhaustive",view.cam,view.lm);
    for(const Seen&view:exhaustive){
        if(view.cam!="slam_right" || view.lm.right<.7 || view.lm.presence<.9)continue;
        V3 points[21];
        if(!tracker.single_view(cameras.at(view.cam),view.lm,1.0,points))continue;
        const auto& target=cameras.at("slam_left");
        V2 projected[21];
        for(int k=0;k<21;++k)projected[k]=target.project(points[k],nullptr);
        const Roi roi=roi_from_points(projected);
        auto lm=nets.landmarks(images.at("slam_left"),roi);
        const V2 center=(projected[0]+projected[5]+projected[9]+projected[13]+projected[17])*.2;
        std::printf("handoff palm center=%.1f,%.1f off_axis=%.2f\n",center[0],center[1],target.off_axis(center));
        std::vector<double> residuals;
        V3 triangulated[21];
        for(int k=0;k<21;++k){
            const V3 origins[2]={cameras.at(view.cam).origin,target.origin};
            const V3 dirs[2]={cameras.at(view.cam).ray(view.lm.pts[k]),target.ray(lm.pts[k])};
            const double weights[2]={view.lm.presence,lm.presence};double residual;
            triangulated[k]=triangulate(origins,dirs,weights,2,&residual);residuals.push_back(residual);
        }
        std::sort(residuals.begin(),residuals.end());
        V3 target_mono[21];tracker.single_view(target,lm,1.0,target_mono);
        const double source_ratio=norm(triangulated[9]-cameras.at(view.cam).origin)/norm(points[9]-cameras.at(view.cam).origin);
        const double target_ratio=norm(triangulated[9]-target.origin)/norm(target_mono[9]-target.origin);
        const double cost=residuals[10]/.01+std::abs(std::log(source_ratio/1.16))+std::abs(std::log(target_ratio/1.16))+std::abs(lm.right-view.lm.right);
        std::printf("pair median_ray_residual=%.6f ratios=%.3f,%.3f cost=%.3f\n",residuals[10],source_ratio,target_ratio,cost);
        result["projection_diagnostic"]["off_axis_degrees"]=target.off_axis(center);
        result["projection_diagnostic"]["median_ray_residual_m"]=residuals[10];
        result["projection_diagnostic"]["source_depth_ratio"]=source_ratio;
        result["projection_diagnostic"]["target_depth_ratio"]=target_ratio;
        result["projection_diagnostic"]["pair_cost"]=cost;
        append("automatic_mono_projection_diagnostic","slam_left",lm);
        std::printf("projected center=%.1f,%.1f size=%.1f angle=%.1f presence=%.3f\n",roi.center[0],roi.center[1],roi.size,roi.rotation*180/M_PI,lm.presence);
        result["projected_roi_x"]=roi.center[0];result["projected_roi_y"]=roi.center[1];
        result["projected_roi_size"]=roi.size;result["projected_roi_angle"]=roi.rotation*180/M_PI;
    }
    if(argc==7){
        const std::string camera=argv[4];
        if(!images.count(camera))return 4;
        const V2 center{std::stod(argv[5]),std::stod(argv[6])};
        Landmarks best{}; double best_size=0,best_angle=0;
        for(double size:{200.,300.,400.,500.})for(int angle=0;angle<24;++angle){
            Landmarks found=nets.landmarks(images.at(camera),{center,size,angle*M_PI/12});
            if(found.presence>best.presence){best=found;best_size=size;best_angle=angle*15.;}
        }
        append("manual_seed_best_of_96_not_automatic",camera,best);
        result["manual_roi_size"]=best_size;result["manual_roi_angle"]=best_angle;
        result["manual_roi_x"]=center[0];result["manual_roi_y"]=center[1];
        std::printf("manual size=%.0f angle=%.0f\n",best_size,best_angle);
    }
    // Human-marked missed-hand ROI in set 438: establishes whether the landmark
    // network can fit this hand when detection/search is removed from the test.
    if(index==438 && images.count("slam_right")){
        Landmarks best{},automatic{};double best_size=0,best_angle=0,auto_size=0,auto_angle=0;
        for(double size:{300.,400.,500.})for(int angle=0;angle<24;++angle){
            Landmarks found=nets.landmarks(images.at("slam_right"),{{540,450},size,angle*M_PI/12});
            if(found.presence>best.presence){best=found;best_size=size;best_angle=angle*15.;}
            for(const Palm& palm:nets.palms(images.at("slam_right"),{540,450},size,angle*M_PI/12)){
                Landmarks detected=nets.landmarks(images.at("slam_right"),palm.roi());
                if(norm(detected.pts[9]-V2{540,450})<180 && detected.presence>automatic.presence){
                    automatic=detected;auto_size=size;auto_angle=angle*15.;
                }
            }
        }
        append("manual_seed_best_of_72","slam_right",best);
        append("manual_search_center_automatic_palm","slam_right",automatic);
        result["manual_roi_size"]=best_size;result["manual_roi_angle"]=best_angle;
        result["auto_search_size"]=auto_size;result["auto_search_angle"]=auto_angle;
        std::printf("manual size=%.0f angle=%.0f; palm search size=%.0f angle=%.0f\n",best_size,best_angle,auto_size,auto_angle);
    }
    std::ofstream output(argv[3]);if(!output)return 3;
    output<<result<<"\n";
    for(const auto&view:result["views"])std::printf("%s %s presence=%.3f right=%.3f\n",view["method"].asCString(),view["camera"].asCString(),view["presence"].asDouble(),view["right"].asDouble());
}

// Offline factorial startup probe: original/stereo wrist x default/learned curls.
// Current observations only. Native costs, limits, thumb and remaining articulation
// are unchanged; variance here has only the existing native relative-view meaning.
extern "C" int fh_seed_observation_state(void *handle,const float *packet,float size,int stereo,int curls,float *diagnostic) {
    using namespace xrt::tracking::hand::mercury;
    using namespace xrt::tracking::hand::mercury::lm;
    if(!handle||!packet||!diagnostic||(stereo!=0&&stereo!=1)||(curls!=0&&curls!=1)||!std::isfinite(size)||size<.06f||size>.12f)return -1;
    auto &state=*static_cast<KinematicHandLM *>(handle);if(!state.first_frame)return -1;
    for(int i=0;i<242;i++)if(!std::isfinite(packet[i]))return -1;
    one_frame_input input{};int index=0,active=0;
    for(auto &view:input.views){view.active=packet[index++]!=0;active+=view.active;view.look_dir={packet[index],packet[index+1],packet[index+2],packet[index+3]};index+=4;view.stereographic_radius=packet[index++];
        float norm=view.look_dir.x*view.look_dir.x+view.look_dir.y*view.look_dir.y+view.look_dir.z*view.look_dir.z+view.look_dir.w*view.look_dir.w;
        if(view.active&&(view.stereographic_radius<=0||std::fabs(norm-1)>.001f))return -1;
        for(auto &p:view.keypoints_in_scaled_stereographic){p.pos_2d={packet[index],packet[index+1]};p.depth_relative_to_midpxm=packet[index+2];p.confidence_xy=packet[index+3];p.confidence_depth=packet[index+4];index+=5;if(p.confidence_xy<0||p.confidence_depth<0)return -1;}
        for(auto &c:view.curls){c.value=packet[index++];c.variance=packet[index++];if(c.variance<=0)return -1;}
    }
    if(!active)return -1;
    for(int i=0;i<30;i++)diagnostic[i]=0;
    float stereo_diagnostic[10]{};int stereo_status=stereo?fh_seed_stereo_state(handle,packet,size,stereo_diagnostic):1;
    if(stereo_status<0)return -1;
    if(stereo_status!=0){
        // Match the pinned native reset setup; no solver step is taken here.
        xrt_pose guess=XRT_POSE_IDENTITY;hand_init_guess(input,size,state.is_right,state.left_in_right,guess);
        OptimizerHandInit(state.last_frame,state.this_frame_pre_rotation);
        if(guess.position.z>.05f)state.this_frame_pre_position={0.f,0.f,-.3f};
        else state.this_frame_pre_position={guess.position.x,guess.position.y,guess.position.z};
        state.this_frame_pre_rotation={guess.orientation.x,guess.orientation.y,guess.orientation.z,guess.orientation.w};
    }
    for(int finger=0;finger<4;finger++){
        auto &f=state.last_frame.finger[finger];const double initial[3]={f.proximal_swing.x,f.rots[0],f.rots[1]};
        const auto &limits=the_limit.fingers[finger];const double lower[3]={limits.pxm_swing_x.min,limits.curls[0].min,limits.curls[1].min};const double upper[3]={limits.pxm_swing_x.max,limits.curls[0].max,limits.curls[1].max};
        double observed=get_avg_curl_value(input,finger+1);if(!std::isfinite(observed))return -1;
        double target=std::clamp(observed,lower[0]+lower[1]+lower[2],upper[0]+upper[1]+upper[2]);
        diagnostic[10+5*finger]=float(observed);diagnostic[11+5*finger]=float(target);
        if(curls){
            // Euclidean projection of native default angles onto the bounded
            // angle-sum plane. No fitted truth or invented joint ratios.
            double lo=lower[0]-initial[0],hi=upper[0]-initial[0];
            for(int j=1;j<3;j++){lo=std::min(lo,lower[j]-initial[j]);hi=std::max(hi,upper[j]-initial[j]);}
            for(int iteration=0;iteration<60;iteration++){double midpoint=(lo+hi)*.5,total=0;for(int j=0;j<3;j++)total+=std::clamp(initial[j]+midpoint,lower[j],upper[j]);if(total<target)lo=midpoint;else hi=midpoint;}
            float values[3];for(int j=0;j<3;j++)values[j]=float(std::clamp(initial[j]+(lo+hi)*.5,lower[j],upper[j]));
            f.proximal_swing.x=values[0];f.rots[0]=values[1];f.rots[1]=values[2];
        }
        diagnostic[12+5*finger]=f.proximal_swing.x;diagnostic[13+5*finger]=f.rots[0];diagnostic[14+5*finger]=f.rots[1];
    }
    OptimizerHandPackIntoVector(state.last_frame,false,state.TinyOptimizerInput.data());
    const auto &q=state.this_frame_pre_rotation;const auto &p=state.this_frame_pre_position;
    diagnostic[0]=q.x;diagnostic[1]=q.y;diagnostic[2]=q.z;diagnostic[3]=q.w;diagnostic[4]=p.x;diagnostic[5]=p.y;diagnostic[6]=p.z;
    diagnostic[7]=float(stereo_status);diagnostic[8]=stereo_diagnostic[7];diagnostic[9]=stereo_diagnostic[9];
    // first_frame remains true. Python suppresses reset only, not regularization.
    return 0;
}

// Included only after the pinned lm_main.cpp in an optional offline translation
// unit. Reuses native forward kinematics without running or modifying a solver.
extern "C" __declspec(dllexport) int fh_generate_hand(const float *wrist,const float *angles,float size,int side,float *output) {
    using namespace xrt::tracking::hand::mercury::lm;
    if(!wrist||!angles||!output||(side!=0&&side!=1)||!std::isfinite(size)||size<.06f||size>.12f)return -1;
    for(int i=0;i<7;i++)if(!std::isfinite(wrist[i]))return -1;
    float norm=0;for(int i=0;i<4;i++)norm+=wrist[i]*wrist[i];if(std::fabs(norm-1)>0.001f)return -1;
    for(int i=0;i<4;i++)for(int j=0;j<3;j++){
        float value=angles[i*3+j];const auto &bound=j==0?the_limit.fingers[i].pxm_swing_x:the_limit.fingers[i].curls[j-1];
        if(!std::isfinite(value)||value<bound.min||value>bound.max)return -1;
    }
    try {
        KinematicHandLM state{};state.is_right=side!=0;
        OptimizerHand<float> hand{};Quat<float> rotation{};
        rotation.x=wrist[0];rotation.y=wrist[1];rotation.z=wrist[2];rotation.w=wrist[3];
        OptimizerHandInit(hand,rotation);hand.hand_size=size;hand.wrist_final_orientation=rotation;
        hand.wrist_final_location={wrist[4],wrist[5],wrist[6]};
        for(int i=0;i<4;i++){hand.finger[i].proximal_swing.x=angles[i*3];hand.finger[i].rots[0]=angles[i*3+1];hand.finger[i].rots[1]=angles[i*3+2];}
        Translations55<float> translations{};Orientations54<float> orientations{};xrt_hand_joint_set joints{};
        eval_to_viz_hand(state,hand,translations,orientations,joints);
        int n=0;for(const auto &joint:joints.values.hand_joint_set_default){const auto &p=joint.relation.pose;
            output[n++]=p.position.x;output[n++]=p.position.y;output[n++]=p.position.z;
            output[n++]=p.orientation.x;output[n++]=p.orientation.y;output[n++]=p.orientation.z;output[n++]=p.orientation.w;}
        if(n!=182)return -2;for(int i=0;i<n;i++)if(!std::isfinite(output[i]))return -3;
        return 0;
    }catch(...){return -4;}
}

// Synthetic diagnostics only: called through the optional bridge on a fresh
// solver. Known truth is never used by recorded inference or live entry points.
extern "C" int fh_seed_synthetic_state(void *handle,const float *wrist,const float *angles,float size) {
    using namespace xrt::tracking::hand::mercury::lm;
    if(!handle)return -1;
    auto &state=*static_cast<KinematicHandLM *>(handle);
    if(!state.first_frame)return -1;
    float validation[182];
    if(fh_generate_hand(wrist,angles,size,state.is_right?1:0,validation))return -1;
    Quat<float> rotation{};rotation.x=wrist[0];rotation.y=wrist[1];rotation.z=wrist[2];rotation.w=wrist[3];
    OptimizerHandInit(state.last_frame,rotation);
    state.last_frame.wrist_final_orientation=rotation;
    state.last_frame.wrist_final_location={wrist[4],wrist[5],wrist[6]};
    state.last_frame.hand_size=size;
    for(int i=0;i<4;i++){state.last_frame.finger[i].proximal_swing.x=angles[i*3];state.last_frame.finger[i].rots[0]=angles[i*3+1];state.last_frame.finger[i].rots[1]=angles[i*3+2];}
    state.this_frame_pre_position=state.last_frame.wrist_final_location;
    state.this_frame_pre_rotation=rotation;state.optimize_hand_size=false;state.target_hand_size=size;
    OptimizerHandPackIntoVector(state.last_frame,false,state.TinyOptimizerInput.data());
    // Keep first_frame true: the first fit in BOTH cold/seeded arms has no
    // temporal regularization. Only the subsequent fits use the requested value.
    return 0;
}

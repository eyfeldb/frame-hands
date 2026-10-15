// Offline startup candidate. Consumes only current calibrated observation rays;
// no known pose, external tracked pose, history, truth angles or identity changes.
#include <Eigen/SVD>
extern "C" int fh_seed_stereo_state(void *handle,const float *packet,float size,float *diagnostic) {
    using namespace xrt::tracking::hand::mercury::lm;
    if(!handle||!packet||!diagnostic||!std::isfinite(size)||size<.06f||size>.12f)return -1;
    auto &state=*static_cast<KinematicHandLM *>(handle);if(!state.first_frame)return -1;
    for(int i=0;i<242;i++)if(!std::isfinite(packet[i]))return -1;
    for(int i=0;i<10;i++)diagnostic[i]=0;
    if(packet[0]==0||packet[121]==0)return 1;
    const int ml[5]={0,5,9,13,17};const int xr[5]={1,7,12,17,22};
    Eigen::Quaterniond calibration(state.left_in_right.orientation.w,state.left_in_right.orientation.x,state.left_in_right.orientation.y,state.left_in_right.orientation.z);
    Eigen::Vector3d translation(state.left_in_right.position.x,state.left_in_right.position.y,state.left_in_right.position.z);
    Eigen::Vector3d origin=-(calibration.conjugate()*translation);
    Eigen::Matrix<double,3,5> observed;
    double max_gap=0,min_sine=1;
    for(int j=0;j<5;j++) {
        Eigen::Vector3d rays[2];
        for(int v=0;v<2;v++) {
            const float *p=packet+121*v;Eigen::Quaterniond crop(p[4],p[1],p[2],p[3]);
            if(std::fabs(crop.squaredNorm()-1)>.001||p[5]<=0)return -1;
            const float *point=p+6+5*ml[j];if(point[3]<.5f)return 2;
            const double x=point[0]*p[5],y=point[1]*p[5],denom=1+x*x+y*y;
            rays[v]=crop*Eigen::Vector3d(2*x/denom,2*y/denom,(-1+x*x+y*y)/denom);
            if(v==1)rays[v]=calibration.conjugate()*rays[v];
            rays[v].normalize();
        }
        const double c=rays[0].dot(rays[1]),denom=1-c*c;
        if(denom<.0004)return 3; // sine(angle)<0.02; weak stereo depth
        const double b0=rays[0].dot(origin),b1=rays[1].dot(origin);
        const double left=(b0-c*b1)/denom,right=(c*b0-b1)/denom;
        if(left<.1||right<.1||left>1.5||right>1.5)return 4;
        const Eigen::Vector3d a=left*rays[0],b=origin+right*rays[1];
        const double gap=(a-b).norm();if(gap>.01)return 5;
        max_gap=std::max(max_gap,gap);min_sine=std::min(min_sine,std::sqrt(denom));observed.col(j)=(a+b)*.5;
    }
    // Native right geometry reflects camera X AFTER its internal rotation.
    // Undo that reflection before estimating the internal rigid palm transform.
    if(state.is_right)observed.row(0)*=-1;
    KinematicHandLM template_state{};OptimizerHand<float> hand{};Quat<float> identity{};identity.w=1;
    OptimizerHandInit(hand,identity);hand.hand_size=size;hand.wrist_final_orientation=identity;hand.wrist_final_location={0,0,0};
    Translations55<float> translations{};Orientations54<float> orientations{};xrt_hand_joint_set joints{};
    eval_to_viz_hand(template_state,hand,translations,orientations,joints);
    Eigen::Matrix<double,3,5> model;
    for(int j=0;j<5;j++){const auto &p=joints.values.hand_joint_set_default[xr[j]].relation.pose.position;model.col(j)=Eigen::Vector3d(p.x,p.y,p.z);}
    const Eigen::Vector3d source_center=model.rowwise().mean(),target_center=observed.rowwise().mean();
    const auto source=(model.colwise()-source_center).eval(),target=(observed.colwise()-target_center).eval();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(source*target.transpose(),Eigen::ComputeFullU|Eigen::ComputeFullV);
    if(svd.singularValues()[0]<=0||svd.singularValues()[1]/svd.singularValues()[0]<.02)return 6;
    Eigen::Matrix3d sign=Eigen::Matrix3d::Identity();sign(2,2)=(svd.matrixV()*svd.matrixU().transpose()).determinant()<0?-1:1;
    const Eigen::Matrix3d rotation=svd.matrixV()*sign*svd.matrixU().transpose();
    Eigen::Vector3d wrist=target_center-rotation*source_center;
    const double rms=std::sqrt(((rotation*model).colwise()+wrist-observed).squaredNorm()/5);
    if(!std::isfinite(rms)||rms>.015)return 7;
    Eigen::Quaterniond q(rotation);q.normalize();if(state.is_right)wrist.x()*=-1;
    Quat<float> native_rotation{};native_rotation.x=float(q.x());native_rotation.y=float(q.y());native_rotation.z=float(q.z());native_rotation.w=float(q.w());
    // Keep native default articulation, and first-frame (unregularized) behavior.
    OptimizerHandInit(state.last_frame,native_rotation);state.last_frame.wrist_final_orientation=native_rotation;
    state.last_frame.wrist_final_location={float(wrist.x()),float(wrist.y()),float(wrist.z())};state.last_frame.hand_size=size;
    state.this_frame_pre_position=state.last_frame.wrist_final_location;state.this_frame_pre_rotation=native_rotation;state.optimize_hand_size=false;state.target_hand_size=size;
    OptimizerHandPackIntoVector(state.last_frame,false,state.TinyOptimizerInput.data());
    diagnostic[0]=float(q.x());diagnostic[1]=float(q.y());diagnostic[2]=float(q.z());diagnostic[3]=float(q.w());
    for(int i=0;i<3;i++)diagnostic[4+i]=float(wrist[i]);
    diagnostic[7]=float(max_gap);diagnostic[8]=float(min_sine);diagnostic[9]=float(rms);
    return 0;
}

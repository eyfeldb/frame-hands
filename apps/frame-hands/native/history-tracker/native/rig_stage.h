#pragma once
#include "causal_rig_fit.h"
namespace fhn {
// Portable wrist initialization and both fitting stages. Millimeters, rig-indexed
// observations, current capture head space. No synthesized detection or visibility.
inline bool palm_seed(const HandRig& reference,int side,
                      const std::array<RigObservation,21>& observations,RigPose& seed){
    constexpr std::array<int,5> indices{5,8,11,14,17};
    if(side<0||side>1)throw std::invalid_argument("Invalid side");
    std::array<V,5> x{},y{};V xc{},yc{};
    for(int i=0;i<5;++i){int j=indices[i];if(!observations[j].present)return false;
        observations[j].validate();x[i]=reference.rest[j];if(side)x[i].x=-x[i].x;
        y[i]=observations[j].point;xc=xc+x[i]*.2;yc=yc+y[i]*.2;}
    double spread=0;for(int i=1;i<5;++i)for(int j=i+1;j<5;++j)
        spread=std::max(spread,(y[i]-y[0]).cross(y[j]-y[0]).len());
    if(spread<1e-3)return false;
    double s[3][3]{};
    for(int i=0;i<5;++i){V a=x[i]-xc,b=y[i]-yc;double av[]{a.x,a.y,a.z},bv[]{b.x,b.y,b.z};
        for(int r=0;r<3;++r)for(int c=0;c<3;++c)s[r][c]+=av[r]*bv[c];}
    // Horn's symmetric quaternion objective; Jacobi gives its largest eigenvector.
    double a[4][4]={{s[0][0]+s[1][1]+s[2][2],s[1][2]-s[2][1],s[2][0]-s[0][2],s[0][1]-s[1][0]},
        {s[1][2]-s[2][1],s[0][0]-s[1][1]-s[2][2],s[0][1]+s[1][0],s[0][2]+s[2][0]},
        {s[2][0]-s[0][2],s[0][1]+s[1][0],-s[0][0]+s[1][1]-s[2][2],s[1][2]+s[2][1]},
        {s[0][1]-s[1][0],s[0][2]+s[2][0],s[1][2]+s[2][1],-s[0][0]-s[1][1]+s[2][2]}};
    double v[4][4]{};for(int i=0;i<4;++i)v[i][i]=1;
    for(int sweep=0;sweep<80;++sweep){int p=0,q=1;double largest=0,scale=0;
        for(int i=0;i<4;++i){scale=std::max(scale,std::abs(a[i][i]));for(int j=i+1;j<4;++j)if(std::abs(a[i][j])>largest){largest=std::abs(a[i][j]);p=i;q=j;}}
        if(largest<1e-14*std::max(1.,scale))break;
        double theta=.5*std::atan2(2*a[p][q],a[q][q]-a[p][p]),c=std::cos(theta),sine=std::sin(theta);
        double pp=a[p][p],qq=a[q][q],pq=a[p][q];
        a[p][p]=c*c*pp-2*c*sine*pq+sine*sine*qq;a[q][q]=sine*sine*pp+2*c*sine*pq+c*c*qq;a[p][q]=a[q][p]=0;
        for(int k=0;k<4;++k){if(k!=p&&k!=q){double kp=a[k][p],kq=a[k][q];a[k][p]=a[p][k]=c*kp-sine*kq;a[k][q]=a[q][k]=sine*kp+c*kq;}
            double vp=v[k][p],vq=v[k][q];v[k][p]=c*vp-sine*vq;v[k][q]=sine*vp+c*vq;}
    }
    int best=0;for(int i=1;i<4;++i)if(a[i][i]>a[best][best])best=i;
    seed={};seed.side=side;seed.wrist.q=Q{v[0][best],v[1][best],v[2][best],v[3][best]}.unit();
    seed.wrist.t=yc-seed.wrist.q.rotate(xc);return true;
}
struct RigStageResult {bool initialized=false;RigPose seed{};RigFitResult initial{};CausalRigResult refined{};RigFitOptions options{};};
class RigStage {
    CausalRigFitter temporal_;
    std::array<std::int64_t,2> last_stamp_{};
    struct LastFit {bool valid=false;int source_id=-1;std::int64_t stamp=0;V wrist{};};
    std::array<LastFit,2> last_fit_{};   // previous fitted wrist (capture head frame), single-view depth prior
public:
    RigStageResult fit(const HandRig& reference,const HandRig& anatomy,int side,
        const std::array<RigObservation,21>& points,const std::array<RigCamera,2>& cameras,
        const std::array<PixelObservation,21>& pixels,std::int64_t stamp,int source_id){
        if(side<0||side>1||stamp<=0||source_id<0)throw std::invalid_argument("Invalid stage metadata");
        if(stamp<=last_stamp_[side])throw std::invalid_argument("Nonmonotonic stage timestamp");
        last_stamp_[side]=stamp;
        for(const auto& o:points)o.validate();for(const auto& o:pixels)o.validate();
        RigStageResult result;
        // Insufficient stereo must not produce a fresh fitted hand. Caller retains
        // explicit source availability; this stage supplies no monocular fallback.
        const int stereo=observed_landmarks(pixels);   // stereo pairs, or any-view landmarks in mono mode
        if(stereo<6||!palm_seed(reference,side,points,result.seed)){temporal_.clear(side);last_fit_[side]={};return result;}
        for(int i=0;i<20;++i)result.seed.angles[i]=std::clamp(0.,anatomy.joints[i].lower,anatomy.joints[i].upper);
        result.initial=fit_rig(anatomy,result.seed,points);
        if(!result.initial.sufficient)return result;
        // Single camera: depth along the ray is only weakly observable from apparent size, so keep the
        // wrist near the 3D seed (the tracker's depth-continuous single-view estimate) with a soft prior.
        // The reference is this track's previous fitted wrist when it is recent (depth stays continuous
        // across stereo<->single-view transitions, where the single-view point cloud carries the model's own
        // relative depth); otherwise the fresh 3D seed. Lateral position is pixel-constrained regardless.
        // Stereo fits may opt into the same previous-fit wrist prior (FRAME_HANDS_RIG_STEREO_SIGMA_MM, 0 = off)
        // to damp triangulation jitter when both cameras see the hand.
        RigFitOptions options{};
        const bool mono=mono_fit_enabled()&&stereo_pairs(pixels)<6;
        const double sigma=mono?mono_translation_sigma_mm():stereo_translation_sigma_mm();
        if(sigma>0){options.translation_prior=1/(sigma*sigma);
            const auto& h=last_fit_[side];const V fresh=result.initial.pose.wrist.t;
            if(h.valid&&h.source_id==source_id&&stamp-h.stamp<=100000000){
                // Anchored: lean toward the previous fit for smoothness, but leak toward the fresh 3D seed every
                // frame and never stray far from it. A pure previous-fit reference let a single-camera hand random-
                // walk in depth (the wearer saw still hands float by 10-50 cm over seconds on 2026-10-10).
                V ref=h.wrist+(fresh-h.wrist)*mono_anchor_leak();
                const V off=ref-fresh;const double d=off.len(),cap=mono_anchor_cap_mm();
                if(d>cap)ref=fresh+off*(cap/d);
                options.translation_reference=ref;
            }else options.translation_reference=fresh;}
        result.options=options;
        result.refined=temporal_.fit(anatomy,result.initial.pose,cameras,pixels,stamp,source_id,options,true);
        result.initialized=result.refined.fit.sufficient;
        last_fit_[side]={result.initialized,source_id,stamp,result.refined.fit.pose.wrist.t};
        return result;
    }
};
}

#pragma once
#include "fit_rig_pixels.h"
namespace fhn {
struct StereoRay {V origin{},direction{};};
inline bool rig_pixel_ray(const RigCamera& c,V pixel,StereoRay& out){
    c.validate();if(!pixel.finite())throw std::invalid_argument("Nonfinite ray pixel");
    double x=(pixel.x-c.cx)/c.fx,y=(pixel.y-c.cy)/c.fy,r=std::hypot(x,y),theta=r;
    for(int i=0;i<16;++i){double t2=theta*theta;double value=theta*(1+t2*(c.k[0]+t2*(c.k[1]+t2*(c.k[2]+t2*c.k[3]))));double derivative=1+t2*(3*c.k[0]+t2*(5*c.k[1]+t2*(7*c.k[2]+t2*9*c.k[3])));if(!std::isfinite(derivative)||derivative<=1e-8)return false;double d=(value-r)/derivative;theta-=d;if(std::abs(d)<1e-12)break;}
    if(!std::isfinite(theta)||theta<0||theta>=1.5707963267948966)return false;
    double scale=r>1e-12?std::sin(theta)/r:1;V eye{x*scale,y*scale,std::cos(theta)};
    out.origin=(c.rotation[0]*c.translation.x+c.rotation[1]*c.translation.y+c.rotation[2]*c.translation.z)*-1;
    out.direction=c.rotation[0]*eye.x+c.rotation[1]*eye.y+c.rotation[2]*eye.z;
    if(!out.direction.finite()||out.direction.len()<.999)return false;
    out.direction=out.direction*(1/out.direction.len());V roundtrip;
    return c.project(out.origin+out.direction*1000,roundtrip)&&(roundtrip-pixel).len()<1e-3;
}
struct StereoDisagreement {bool valid=false;std::array<double,2> pixel_distance{};double rms_px=0;};
inline StereoDisagreement stereo_disagreement(const std::array<RigCamera,2>& cams,const PixelObservation& observation){
    observation.validate();StereoDisagreement out;if(!observation.present[0]||!observation.present[1])return out;
    std::array<StereoRay,2> rays;for(int v=0;v<2;++v)if(!rig_pixel_ray(cams[v],observation.point[v],rays[v]))return out;
    V baseline=rays[1].origin-rays[0].origin;if(baseline.len()<1e-3)return out;
    for(int v=0;v<2;++v){V normal=baseline.cross(rays[1-v].direction);double norm=normal.len();if(norm<1e-6)return out;normal=normal*(1/norm);V planar=rays[v].direction-normal*normal.dot(rays[v].direction);V projected;if(!cams[v].project(rays[v].origin+planar*1000,projected))return out;out.pixel_distance[v]=(projected-observation.point[v]).len();}
    out.rms_px=std::hypot(out.pixel_distance[0],out.pixel_distance[1])/std::sqrt(2.);out.valid=std::isfinite(out.rms_px);return out;
}
// Isolated weighting hypothesis, not calibrated confidence or a presence decision.
// Reuse the existing3-sigma robust threshold, cap scale at4; never remove a point.
inline double stereo_uncertainty_factor(const StereoDisagreement& disagreement,const PixelObservation& observation){
    observation.validate();if(!disagreement.valid)return 1;
    double nominal=std::hypot(observation.sigma[0],observation.sigma[1]);
    return std::clamp(disagreement.rms_px/(3*nominal),1.,4.);
}
}

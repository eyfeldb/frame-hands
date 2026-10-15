#pragma once
#include "fit_hand_rig.h"
#include "rig_bound_step.h"
namespace fhn {
// Calibrated four-radial-coefficient KB fisheye projection. No inferred camera defaults.
struct RigCamera {
    std::array<V,3> rotation{{{1,0,0},{0,1,0},{0,0,1}}};V translation{};double fx=0,fy=0,cx=0,cy=0;std::array<double,4> k{};
    void validate()const{
        for(int i=0;i<3;++i){if(!rotation[i].finite())throw std::invalid_argument("Invalid camera rotation");for(int j=0;j<3;++j)if(std::abs(rotation[i].dot(rotation[j])-(i==j?1.:0.))>1e-5)throw std::invalid_argument("Nonrigid camera rotation");}
        if(rotation[0].dot(rotation[1].cross(rotation[2]))<.99999||!translation.finite()||!std::isfinite(fx)||!std::isfinite(fy)||fx<=0||fy<=0||!std::isfinite(cx)||!std::isfinite(cy))throw std::invalid_argument("Invalid camera transform/intrinsics");
        for(auto x:k)if(!std::isfinite(x))throw std::invalid_argument("Invalid camera distortion");
    }
    bool project(V head,V& pixel)const{
        V e=V{rotation[0].dot(head),rotation[1].dot(head),rotation[2].dot(head)}+translation;if(!e.finite()||e.z<=1e-6)return false;
        double radius=std::hypot(e.x,e.y),theta=std::atan2(radius,e.z),t2=theta*theta;
        double radial=1+t2*(k[0]+t2*(k[1]+t2*(k[2]+t2*k[3])));
        double scale=radius>1e-12?theta*radial/radius:1/e.z;
        pixel={fx*e.x*scale+cx,fy*e.y*scale+cy,0};return pixel.finite();
    }
    bool projection_jacobian(V head,std::array<V,2>& rows)const{
        V e=V{rotation[0].dot(head),rotation[1].dot(head),rotation[2].dot(head)}+translation;
        if(!e.finite()||e.z<=1e-6)return false;
        double r=std::hypot(e.x,e.y),scale,radial_derivative,depth_derivative;
        if(r<1e-4*e.z){
            // Smooth optical-axis limit, avoiding subtraction of nearly equal terms.
            double t=r/e.z,t2=t*t,a=k[0]-1./3.,b=k[1]-k[0]+1./5.;
            scale=(1+a*t2+b*t2*t2)/e.z;
            radial_derivative=(2*a+4*b*t2)/(e.z*e.z*e.z);
            depth_derivative=-(1+3*a*t2+5*b*t2*t2)/(e.z*e.z);
        }else{
            double theta=std::atan2(r,e.z),t2=theta*theta;
            double value=theta*(1+t2*(k[0]+t2*(k[1]+t2*(k[2]+t2*k[3]))));
            double derivative=1+t2*(3*k[0]+t2*(5*k[1]+t2*(7*k[2]+t2*9*k[3])));
            scale=value/r;radial_derivative=(derivative*e.z/(r*r+e.z*e.z)-scale)/(r*r);
            depth_derivative=-derivative/(r*r+e.z*e.z);
        }
        V u{fx*(scale+e.x*e.x*radial_derivative),fx*e.x*e.y*radial_derivative,fx*e.x*depth_derivative};
        V v{fy*e.x*e.y*radial_derivative,fy*(scale+e.y*e.y*radial_derivative),fy*e.y*depth_derivative};
        rows[0]=rotation[0]*u.x+rotation[1]*u.y+rotation[2]*u.z;
        rows[1]=rotation[0]*v.x+rotation[1]*v.y+rotation[2]*v.z;
        return rows[0].finite()&&rows[1].finite();
    }

};
inline std::array<RigCamera,2> load_rig_cameras(std::istream& input){
    std::string magic,extra;int version=0,count=0;std::array<RigCamera,2> result;
    if(!(input>>magic>>version>>count)||magic!="FHCAMKB"||version!=1||count!=2)throw std::invalid_argument("Invalid camera header");
    for(auto& c:result){for(auto& r:c.rotation)if(!(input>>r.x>>r.y>>r.z))throw std::invalid_argument("Truncated rotation");if(!(input>>c.translation.x>>c.translation.y>>c.translation.z>>c.fx>>c.fy>>c.cx>>c.cy))throw std::invalid_argument("Truncated camera");for(auto& k:c.k)if(!(input>>k))throw std::invalid_argument("Truncated distortion");c.validate();}
    if(input>>extra)throw std::invalid_argument("Trailing camera data");return result;
}
struct PixelObservation {std::array<V,2> point{};std::array<double,2> sigma{3,3};std::array<bool,2> present{};
    void validate()const{for(int v=0;v<2;++v)if(present[v]&&(!point[v].finite()||!std::isfinite(sigma[v])||sigma[v]<.001))throw std::invalid_argument("Invalid pixel observation");}
};
struct PixelResidual {
    std::array<V,2> v{};
    PixelResidual operator-(const PixelResidual& b)const{return{{v[0]-b.v[0],v[1]-b.v[1]}};}
    PixelResidual operator*(double s)const{return{{v[0]*s,v[1]*s}};}
    double dot(const PixelResidual& b)const{return v[0].dot(b.v[0])+v[1].dot(b.v[1]);}
    double len()const{return std::sqrt(dot(*this));}
};
struct PixelFitResult {RigPose pose{};int observations=0,iterations=0,updates=0;bool sufficient=false;double initial_cost=0,cost=0,observed_rmse_px=0;};
inline PixelFitResult fit_rig_pixels(const HandRig& rig,const RigPose& seed,const std::array<RigCamera,2>& cams,const std::array<PixelObservation,21>& obs,RigFitOptions options={},int projection_mode=0,const std::array<double,20>* angle_reference=nullptr,bool resolve_bounds=false){
    if(projection_mode<0||projection_mode>2)throw std::invalid_argument("Invalid projection derivative mode");
    bool analytic_projection=projection_mode!=0;
    const auto& prior=angle_reference?*angle_reference:seed.angles;
    for(int i=0;i<20;++i)if(!std::isfinite(prior[i])||prior[i]<rig.joints[i].lower-1e-8||prior[i]>rig.joints[i].upper+1e-8)throw std::invalid_argument("Invalid angle prior reference");
    if(options.iterations<1||options.iterations>100||!std::isfinite(options.huber)||options.huber<=0||!std::isfinite(options.angle_prior)||options.angle_prior<0)throw std::invalid_argument("Invalid fitting options");
    auto points=rig.landmarks(seed.angles,seed.wrist,seed.side);for(auto& c:cams)c.validate();PixelFitResult out;out.pose=seed;
    for(auto& o:obs){o.validate();if(o.present[0]&&o.present[1])++out.observations;}
    if(out.observations<6)return out;out.sufficient=true; // Count gate only, not an observability/visibility verdict.
    auto residuals=[&](const std::array<V,21>& points,std::array<PixelResidual,21>& r){
        for(int i=0;i<21;++i)for(int v=0;v<2;++v)if(obs[i].present[v]){V uv;if(!cams[v].project(points[i],uv))return false;r[i].v[v]=(uv-V{obs[i].point[v].x,obs[i].point[v].y,0})*(1/obs[i].sigma[v]);}return true;};
    auto cost=[&](const RigPose& p,const std::array<PixelResidual,21>& r){double value=0;for(auto& x:r){double n=x.len();value+=n<=options.huber?.5*n*n:options.huber*(n-.5*options.huber);}for(int i=0;i<20;++i){double d=p.angles[i]-prior[i];value+=.5*options.angle_prior*d*d;}return value;};
    std::array<PixelResidual,21> residual{};if(!residuals(points,residual))throw std::invalid_argument("Seed behind camera");
    out.initial_cost=out.cost=cost(seed,residual);if(!std::isfinite(out.cost))throw std::invalid_argument("Unbounded pixel cost");double damping=.001;
    for(int iteration=0;iteration<options.iterations;++iteration){out.iterations=iteration+1;double H[26][26]{},g[26]{};std::array<std::array<PixelResidual,21>,26> jac{};bool derivative_valid=true;
        std::array<std::array<std::array<V,2>,2>,21> camera_jac{};
        if(analytic_projection){for(int i=0;i<21;++i)for(int v=0;v<2;++v)if(obs[i].present[v]&&!cams[v].projection_jacobian(points[i],camera_jac[i][v]))derivative_valid=false;if(!derivative_valid)break;}
        for(int j=0;j<26;++j){std::array<double,26>d{};double h=j<3?.01:1e-5;
            if(j>=6&&out.pose.angles[j-6]+h>rig.joints[j-6].upper)h=-h;if(j>=6&&out.pose.angles[j-6]+h<rig.joints[j-6].lower)continue;
            d[j]=h;auto trial=rig_step(rig,out.pose,d);auto y=rig.landmarks(trial.angles,trial.wrist,trial.side);std::array<PixelResidual,21> r{};
            if(analytic_projection&&!(projection_mode==2&&j<3)){for(int i=0;i<21;++i){V displacement=(y[i]-points[i])*(1/h);for(int v=0;v<2;++v)if(obs[i].present[v])jac[j][i].v[v]={camera_jac[i][v][0].dot(displacement)/obs[i].sigma[v],camera_jac[i][v][1].dot(displacement)/obs[i].sigma[v],0};}}
            else{if(!residuals(y,r)){derivative_valid=false;break;}for(int i=0;i<21;++i)jac[j][i]=(r[i]-residual[i])*(1/h);}
        }
        if(!derivative_valid)break; // Retain last valid pose; caller must inspect residual quality.
        for(int i=0;i<21;++i){auto r=residual[i];double n=r.len(),w=n>options.huber?options.huber/n:1.;for(int j=0;j<26;++j){g[j]+=w*jac[j][i].dot(r);for(int k=0;k<=j;++k)H[j][k]+=w*jac[j][i].dot(jac[k][i]);}}
        for(int j=0;j<20;++j){H[j+6][j+6]+=options.angle_prior;g[j+6]+=options.angle_prior*(out.pose.angles[j]-prior[j]);}
        for(int j=0;j<26;++j){for(int k=0;k<j;++k)H[k][j]=H[j][k];H[j][j]+=damping*std::max(1e-6,H[j][j]);g[j]=-g[j];}
        std::array<double,26>d{};bool solved=resolve_bounds?rig_solve_feasible(H,g,rig,out.pose,d):rig_solve(H,g,d);if(!solved){damping*=10;continue;}double fraction=1.;for(int j=0;j<26;++j){double limit=j<3?20.:(j<6?.25:.35);if(std::abs(d[j])>limit)fraction=std::min(fraction,limit/std::abs(d[j]));}for(auto& x:d)x*=fraction;
        auto trial=rig_step(rig,out.pose,d);auto y=rig.landmarks(trial.angles,trial.wrist,trial.side);std::array<PixelResidual,21> r{};
        double value=residuals(y,r)?cost(trial,r):std::numeric_limits<double>::infinity();
        if(std::isfinite(value)&&value<out.cost){double improvement=out.cost-value;out.pose=trial;points=y;residual=r;out.cost=value;++out.updates;damping=std::max(1e-9,damping*.3);if(improvement<1e-9)break;}
        else{damping*=10;if(damping>1e10)break;}
    }
    double square=0;int count=0;for(int i=0;i<21;++i)for(int v=0;v<2;++v)if(obs[i].present[v]){auto d=residual[i].v[v]*obs[i].sigma[v];square+=d.dot(d);++count;}out.observed_rmse_px=std::sqrt(square/count);return out;
}
}

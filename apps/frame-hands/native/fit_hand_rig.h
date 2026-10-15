#pragma once
#include "hand_rig.h"
#include <limits>
namespace fhn {
// Experimental bounded pose fitting in millimeters. No identity, calibration or visibility policy.
struct RigPose { std::array<double,20> angles{}; Rigid wrist{}; int side=0; };
struct RigObservation {
    V point{};
    // Lower Cholesky factor of covariance (mm^2); absent observations are ignored.
    std::array<double,6> lower{3,0,3,0,0,3}; bool present=false;
    V whiten(V d)const {double x=d.x/lower[0],y=(d.y-lower[1]*x)/lower[2];return{x,y,(d.z-lower[3]*x-lower[4]*y)/lower[5]};}
    void validate()const {
        if(!present)return;
        if(!point.finite())throw std::invalid_argument("Nonfinite observed point");
        for(double x:lower)if(!std::isfinite(x))throw std::invalid_argument("Nonfinite uncertainty");
        if(lower[0]<.001||lower[2]<.001||lower[5]<.001)throw std::invalid_argument("Nonpositive or tiny covariance factor");
    }
};
struct RigFitOptions {int iterations=30;double huber=3.,angle_prior=.01;};
struct RigFitResult {
    RigPose pose{};int observations=0,iterations=0,updates=0;bool sufficient=false;
    double initial_cost=0,cost=0,observed_rmse_mm=0;
};
inline Q rig_exp(V v){double n=v.len();if(n<1e-12)return Q{1,.5*v.x,.5*v.y,.5*v.z}.unit();double s=std::sin(n*.5)/n;return{std::cos(n*.5),v.x*s,v.y*s,v.z*s};}
inline RigPose rig_step(const HandRig& rig,RigPose p,const std::array<double,26>& d){
    p.wrist.t=p.wrist.t+V{d[0],d[1],d[2]};p.wrist.q=(rig_exp({d[3],d[4],d[5]})*p.wrist.q).unit();
    for(int i=0;i<20;++i)p.angles[i]=std::clamp(p.angles[i]+d[i+6],rig.joints[i].lower,rig.joints[i].upper);return p;
}
inline bool rig_solve(double a[26][26],double b[26],std::array<double,26>& x){
    // Pivoted elimination. LM diagonal regularization handles unobserved joint directions.
    for(int k=0;k<26;++k){int pivot=k;for(int i=k+1;i<26;++i)if(std::abs(a[i][k])>std::abs(a[pivot][k]))pivot=i;
        if(!std::isfinite(a[pivot][k])||std::abs(a[pivot][k])<1e-14)return false;
        if(pivot!=k){for(int j=k;j<26;++j)std::swap(a[k][j],a[pivot][j]);std::swap(b[k],b[pivot]);}
        for(int i=k+1;i<26;++i){double f=a[i][k]/a[k][k];for(int j=k+1;j<26;++j)a[i][j]-=f*a[k][j];b[i]-=f*b[k];}
    }
    for(int i=25;i>=0;--i){double v=b[i];for(int j=i+1;j<26;++j)v-=a[i][j]*x[j];x[i]=v/a[i][i];if(!std::isfinite(x[i]))return false;}return true;
}
inline RigFitResult fit_rig(const HandRig& rig,const RigPose& seed,const std::array<RigObservation,21>& obs,RigFitOptions options={}){
    if(options.iterations<1||options.iterations>100||!std::isfinite(options.huber)||options.huber<=0||!std::isfinite(options.angle_prior)||options.angle_prior<0)throw std::invalid_argument("Invalid fitting options");
    auto points=rig.landmarks(seed.angles,seed.wrist,seed.side);RigFitResult out;out.pose=seed;
    for(const auto& o:obs){o.validate();if(o.present)++out.observations;}
    if(out.observations<6)return out;out.sufficient=true;
    auto cost=[&](const RigPose& p,const std::array<V,21>& x){double value=0;
        for(int i=0;i<21;++i)if(obs[i].present){double n=obs[i].whiten(x[i]-obs[i].point).len();value+=n<=options.huber?.5*n*n:options.huber*(n-.5*options.huber);}
        for(int i=0;i<20;++i){double d=p.angles[i]-seed.angles[i];value+=.5*options.angle_prior*d*d;}return value;};
    out.initial_cost=out.cost=cost(seed,points);if(!std::isfinite(out.cost))throw std::invalid_argument("Unbounded measurement cost");
    double damping=.001;
    for(int iteration=0;iteration<options.iterations;++iteration){
        out.iterations=iteration+1;double H[26][26]{},g[26]{};std::array<std::array<V,21>,26> jac{};
        for(int j=0;j<26;++j){std::array<double,26>d{};double h=j<3?.01:1e-5;
            if(j>=6&&out.pose.angles[j-6]+h>rig.joints[j-6].upper)h=-h;
            if(j>=6&&out.pose.angles[j-6]+h<rig.joints[j-6].lower)continue;
            d[j]=h;auto trial=rig_step(rig,out.pose,d);auto y=rig.landmarks(trial.angles,trial.wrist,trial.side);
            for(int i=0;i<21;++i)if(obs[i].present)jac[j][i]=obs[i].whiten((y[i]-points[i])*(1/h));
        }
        for(int i=0;i<21;++i)if(obs[i].present){V r=obs[i].whiten(points[i]-obs[i].point);double norm=r.len(),w=norm>options.huber?options.huber/norm:1.;
            for(int j=0;j<26;++j){g[j]+=w*jac[j][i].dot(r);for(int k=0;k<=j;++k)H[j][k]+=w*jac[j][i].dot(jac[k][i]);}}
        for(int j=0;j<20;++j){H[j+6][j+6]+=options.angle_prior;g[j+6]+=options.angle_prior*(out.pose.angles[j]-seed.angles[j]);}
        for(int j=0;j<26;++j){for(int k=0;k<j;++k)H[k][j]=H[j][k];H[j][j]+=damping*std::max(1e-6,H[j][j]);g[j]=-g[j];}
        std::array<double,26>d{};if(!rig_solve(H,g,d)){damping*=10;continue;}
        double fraction=1.;for(int j=0;j<26;++j){double limit=j<3?20.:(j<6?.25:.35);if(std::abs(d[j])>limit)fraction=std::min(fraction,limit/std::abs(d[j]));}for(auto& x:d)x*=fraction;
        auto trial=rig_step(rig,out.pose,d);auto y=rig.landmarks(trial.angles,trial.wrist,trial.side);double value=cost(trial,y);
        if(std::isfinite(value)&&value<out.cost){double improvement=out.cost-value;out.pose=trial;points=y;out.cost=value;++out.updates;damping=std::max(1e-9,damping*.3);if(improvement<1e-9)break;}
        else{damping*=10;if(damping>1e10)break;}
    }
    double square=0;for(int i=0;i<21;++i)if(obs[i].present){auto d=points[i]-obs[i].point;square+=d.dot(d);}out.observed_rmse_mm=std::sqrt(square/out.observations);return out;
}
}

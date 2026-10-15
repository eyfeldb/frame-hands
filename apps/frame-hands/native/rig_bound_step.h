#pragma once
#include "fit_hand_rig.h"
namespace fhn {
// Resolve the damped linear step when an already-bound joint would move outward.
// This is a local active-set experiment, not a full trust-region reflective solver.
// Reconsider all joints on every nonlinear iteration; no persistent locking.
inline bool rig_solve_feasible(const double input[26][26],const double rhs[26],
    const HandRig& rig,const RigPose& pose,std::array<double,26>& result,int* blocked_count=nullptr){
    std::array<bool,26> blocked{};
    for(int pass=0;pass<=20;++pass){
        double matrix[26][26],vector[26];
        for(int j=0;j<26;++j){vector[j]=blocked[j]?0.:rhs[j];
            for(int k=0;k<26;++k)matrix[j][k]=(blocked[j]||blocked[k])?(j==k?1.:0.):input[j][k];}
        result={};if(!rig_solve(matrix,vector,result))return false;
        bool changed=false;
        for(int j=0;j<20;++j){int k=j+6;
            bool outward=(pose.angles[j]<=rig.joints[j].lower+1e-9&&result[k]<0)||
                         (pose.angles[j]>=rig.joints[j].upper-1e-9&&result[k]>0);
            if(outward&&!blocked[k]){blocked[k]=true;changed=true;}
        }
        if(!changed){if(blocked_count){*blocked_count=0;for(bool b:blocked)*blocked_count+=b;}return true;}
    }
    return false;
}
}

#include "rig_bound_step.h"
#include <iostream>
#include <stdexcept>
using namespace fhn;
void require(bool value){if(!value)throw std::runtime_error("bound-step control failed");}
int main(){
 HandRig rig;RigPose pose;for(auto& j:rig.joints){j.lower=0;j.upper=1;}
 double H[26][26]{},rhs[26]{};for(int i=0;i<26;++i)H[i][i]=1;
 H[6][0]=H[0][6]=.9;rhs[6]=1;rhs[0]=2; // unconstrained joint wants negative; wrist overshoots if merely clamped
 std::array<double,26>d{};int blocked=0;require(rig_solve_feasible(H,rhs,rig,pose,d,&blocked));
 require(blocked==1&&d[6]==0&&std::abs(d[0]-2)<1e-12);
 // Upper-bound mirror has the same coupled correction.
 pose.angles[0]=1;rhs[6]=-1;rhs[0]=-2;require(rig_solve_feasible(H,rhs,rig,pose,d,&blocked));
 require(blocked==1&&d[6]==0&&std::abs(d[0]+2)<1e-12);
 // Release a formerly blocked joint when the step points inward.
 rhs[6]=-2;rhs[0]=-1;require(rig_solve_feasible(H,rhs,rig,pose,d,&blocked));require(blocked==0&&d[6]<0);
 // Interior poses exactly reproduce the original linear solve; inputs are immutable.
 pose.angles.fill(.5);double copy[26][26],b[26];for(int i=0;i<26;++i){b[i]=rhs[i];for(int j=0;j<26;++j)copy[i][j]=H[i][j];}
 std::array<double,26>control{};require(rig_solve(copy,b,control));require(rig_solve_feasible(H,rhs,rig,pose,d,&blocked));
 require(blocked==0);for(int i=0;i<26;++i)require(control[i]==d[i]);require(H[0][6]==.9&&rhs[6]==-2);
 std::cout<<"4 bound-step controls passed\n";
}

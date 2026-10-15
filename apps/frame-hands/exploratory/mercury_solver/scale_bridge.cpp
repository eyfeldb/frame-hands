// Offline C ABI for pinned, unmodified Mercury hand-size refinement heuristics.
#include "hg_hand_size_opt.hpp"
#include <cmath>
using namespace xrt::tracking::hand::mercury;
#define API extern "C" __declspec(dllexport)
static void state(HandSizeRefinement &s,float *out){float *x,*y;s.get_refinement_schedule_ptrs(x,y);out[0]=s.getTargetHandSize();out[1]=s.getHandSizeErrorMul();out[2]=s.isOptimizing()?1.f:0.f;out[3]=*x;out[4]=*y;}
API void *fh_scale_create(){try{return new HandSizeRefinement();}catch(...){return nullptr;}}
API void fh_scale_destroy(void *p){delete static_cast<HandSizeRefinement *>(p);}
API int fh_scale_pre(void *p,const int *found,const int *seen,float *out){
    if(!p||!found||!seen||!out)return -1;
    std::array<HandSizeRefinement::ViewData,2> views{};std::array<HandSizeRefinement::HandData,2> hands{};
    for(int v=0;v<2;v++)for(int h=0;h<2;h++)views[v].viewHand[h].foundROIthisFrame=found[v*2+h]!=0;
    for(int h=0;h<2;h++){hands[h].this_frame_hand_detected=found[h]||found[2+h];hands[h].hand_seen_before=seen[h]!=0;}
    auto &s=*static_cast<HandSizeRefinement *>(p);bool optimize=s.framePreOptimizer(views,hands);state(s,out);out[2]=optimize?1.f:0.f;return 0;
}
API int fh_scale_post(void *p,float size,float error,const float *confidence){
    if(!p||!confidence||!std::isfinite(size)||size<.06f||size>.120001f||!std::isfinite(error)||error<0)return -1;
    one_frame_input input{};int i=0;
    for(auto &view:input.views)for(auto &joint:view.keypoints_in_scaled_stereographic){float c=confidence[i++];if(!std::isfinite(c)||c<0)return -1;joint.confidence_xy=c;}
    static_cast<HandSizeRefinement *>(p)->frameHandPostOptimizer(size,error,input);return 0;
}
API int fh_scale_finish(void *p,float *out){if(!p||!out)return -1;auto &s=*static_cast<HandSizeRefinement *>(p);s.framePost();state(s,out);return 0;}

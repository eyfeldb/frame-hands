#pragma once
#include "stereo_disagreement.h"
namespace fhn {
// Isolated current-image hypothesis. No calibrated confidence, identity or gesture.
struct StereoModelChoice {
 std::array<PixelObservation,21> pixels{};int choice=0;
 std::array<double,4> score{},maximum_chain{};std::array<bool,4> eligible{};
};
inline StereoModelChoice choose_stereo_model_views(const std::array<RigCamera,2>& cameras,
 const std::array<PixelObservation,21>& full,const std::array<PixelObservation,21>& lite,
 const std::array<double,2>& full_presence,const std::array<double,2>& lite_presence){
 for(const auto* presence:{&full_presence,&lite_presence})for(double p:*presence)if(!std::isfinite(p)||p<0||p>1)throw std::invalid_argument("Invalid model presence");
 for(const auto& p:full)p.validate();for(const auto& p:lite)p.validate();
 StereoModelChoice out;out.pixels=full;out.score.fill(std::numeric_limits<double>::infinity());out.maximum_chain=out.score;
 constexpr int chains[5][4]={{5,6,7,0},{8,9,10,1},{11,12,13,2},{14,15,16,3},{17,18,19,4}};
 std::array<std::array<PixelObservation,21>,4> alternatives{};
 for(int option=0;option<4;++option){auto& pixels=alternatives[option];pixels=full;bool presence=true;
  // Deterministic order: Full/Full, Full/Lite, Lite/Full, Lite/Lite.
  for(int v=0;v<2;++v){bool use_lite=(option&(v==0?2:1))!=0;presence&=(use_lite?lite_presence[v]:full_presence[v])>=.8;
   if(use_lite)for(int i=0;i<21;++i){pixels[i].point[v]=lite[i].point[v];pixels[i].present[v]=lite[i].present[v];pixels[i].sigma[v]=lite[i].sigma[v];}}
  double total=0,maximum=0;bool valid=true;
  for(const auto& chain:chains){double square=0;
   for(int i:chain){auto d=stereo_disagreement(cameras,pixels[i]);if(!d.valid){valid=false;break;}double z=d.rms_px/std::hypot(pixels[i].sigma[0],pixels[i].sigma[1]);square+=z*z;}
   if(!valid)break;total+=square;maximum=std::max(maximum,std::sqrt(square/4));
  }
  if(valid){out.score[option]=std::sqrt(total/20);out.maximum_chain[option]=maximum;out.eligible[option]=presence&&maximum<=3;}
 }
 // Never perturb a geometrically consistent Full pair, even if another scores better.
 if(out.maximum_chain[0]<=3)return out;
 int best=-1;for(int i=1;i<4;++i)if(out.eligible[i]&&(best<0||out.score[i]<out.score[best]))best=i;
 if(best>=0){out.choice=best;out.pixels=alternatives[best];}return out;
}
}

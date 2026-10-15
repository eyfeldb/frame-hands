#pragma once
#include <cstdint>
#include <cstring>
namespace fhs {
struct DisplayLease {char magic[8]={'F','H','D','I','S','P','1',0};uint64_t stamp_ms=0;uint32_t hands=0,reserved=0;};
static_assert(sizeof(DisplayLease)==24);
inline bool display_lease(const DisplayLease& p,uint64_t now,int side){return side>=0&&side<2&&!std::memcmp(p.magic,"FHDISP1",8)&&!p.reserved&&!(p.hands&~3u)&&p.stamp_ms&&p.stamp_ms<=now&&now-p.stamp_ms<=100&&(p.hands&(1u<<side));}
}

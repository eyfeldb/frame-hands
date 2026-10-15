#pragma once
#include <cstdint>
namespace fhs {
// Diagnostic only: stop an otherwise empty SteamVR-only trial after 8 s of
// continuously visible source hands without any application-facing skeleton.
struct ActionWatchdog {
    uint64_t since=0;bool watching=false;
    bool update(uint64_t now,bool source_visible,bool skeleton_visible){
        if(!source_visible||skeleton_visible){watching=false;return false;}
        if(!watching){watching=true;since=now;}
        return now>=since&&now-since>=8000;
    }
};
}

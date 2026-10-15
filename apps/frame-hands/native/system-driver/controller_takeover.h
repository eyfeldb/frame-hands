#pragma once
#include <cstdint>
namespace fhnav {
// No movement/touch takeover. After selecting Hands, release existing buttons
// before a new press can take ownership (including the launch-menu trigger).
struct ButtonGate {
    bool armed=false;uint64_t last=0;
    void reset(){armed=false;last=0;}
    bool sample(bool available,uint64_t buttons){
        if(!available){reset();return false;}
        if(!buttons)armed=true;
        bool press=armed&&(buttons&~last);last=buttons;return press;
    }
};
}

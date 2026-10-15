#pragma once
#include <cstdint>
#include <cstring>
namespace fhs {
enum class InputMode:uint32_t { Auto=0, Hands=1, Controllers=2 };
struct ModeCommand {
    char magic[8]={'F','H','M','O','D','E','1',0};
    uint64_t sequence=0;
    InputMode mode=InputMode::Auto;
    uint32_t reserved=0;
};
static_assert(sizeof(ModeCommand)==24);
struct ModeSelection {
    InputMode mode=InputMode::Auto;uint64_t sequence=0;
    bool accept(const ModeCommand& c){
        if(std::memcmp(c.magic,"FHMODE1",8)||!c.sequence||c.sequence<=sequence||c.reserved||uint32_t(c.mode)>2)return false;
        sequence=c.sequence;mode=c.mode;return true;
    }
    void controller_press(){if(mode==InputMode::Hands)mode=InputMode::Controllers;}
};
}

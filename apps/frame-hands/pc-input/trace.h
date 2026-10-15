#pragma once
#include "protocol.h"
#include <cstddef>

namespace fhs {
struct ViewHand{uint32_t id=0,flags=0;float raw[21][3]{},bones[31][3]{},axes[3][3]{},candidate[21][3]{};uint32_t candidate_id=0;};
struct View{char magic[8]={'F','H','V','I','E','W','2',0};uint64_t time=0;ViewHand hands[2];};
static_assert(sizeof(ViewHand)==924&&sizeof(View)==1864);
// Fixed little-endian trace record, including rejected incoming measurements.
// flags: bit 0 input read succeeded, bit 1 mailbox fresh, bit 2 stability enabled.
// Pose flags: bit 0 connected, bit 1 valid. Matrices are row-major 3x4, metres.
struct Trace {
    char magic[8]={'F','H','T','R','A','C','E','1'};
    uint32_t version=1,size=3136;
    uint64_t time=0;
    uint32_t flags=0,head_flags=0,capture_flags=0,published=0;
    float head[12]{},captured_head[12]{};
    Packet incoming{},mailbox{};
    View view{};
};
static_assert(sizeof(Trace)==3136&&offsetof(Trace,incoming)==136&&offsetof(Trace,mailbox)==704&&offsetof(Trace,view)==1272);
}

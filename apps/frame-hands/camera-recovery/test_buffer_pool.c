/* Exercise the actual reader selection and learner, without camera access. */
#define main reader_program_main
#include "camd/camd.c"
#undef main
#include <assert.h>

static xr_camera_t camera(int node, const char *sensor)
{
    xr_camera_t c = {0};
    c.node=node; c.width=1056; c.height=1024;
    c.pixfmt=V4L2_PIX_FMT_NV12; c.nplanes=2;
    c.planesize[0]=1152*1024; c.planesize[1]=1152*512;
    snprintf(c.sensor,sizeof(c.sensor),"%s",sensor);
    return c;
}

static void learner_fixture(cam_t *c, bool split, int correct_block)
{
    memset(c,0,sizeof(*c));
    c->nslots=32; c->maxindex=15;
    for(int s=0;s<32;s++) {
        c->slot_group[s]=split ? s/16 : 0;
        c->slot_pos[s]=split ? s%16 : s;
    }
    for(int i=0;i<16;i++) {
        c->nobs[i]=10;
        c->votes[i][correct_block+i]=10;
    }
}

int main(void)
{
    xr_camera_t a=camera(0,"og01a1b 4-0036");
    xr_camera_t b=camera(3,"og01a1b 4-0060");
    xr_camera_t eye=camera(9,"ov7251 7-0060");
    xr_group_t ga={0},gb={0};
    ga.cam=&a; gb.cam=&b;
    ga.planesize[0]=gb.planesize[0]=a.planesize[0];
    ga.planesize[1]=gb.planesize[1]=a.planesize[1];
    // Reproduce the defect: an assigned group excluded the opposite group
    // before any evidence was collected, even if ownership was guessed wrong.
    bool legacy_includes_b=group_fits(&gb,&a,a.planesize[0]) && gb.cam==&a;
    assert(!legacy_includes_b);
    assert(candidate_group(&ga,&a,a.planesize[0],true));
    assert(candidate_group(&gb,&a,a.planesize[0],true));
    assert(candidate_group(&ga,&b,b.planesize[0],true));
    assert(candidate_group(&gb,&b,b.planesize[0],true));
    gb.cam=&eye;
    assert(!candidate_group(&gb,&a,a.planesize[0],true));
    gb.cam=&b; gb.planesize[1]++;
    assert(!candidate_group(&gb,&a,a.planesize[0],true));
    gb.planesize[1]--; gb.planesize[0]=a.planesize[0]-1;
    assert(!candidate_group(&gb,&a,a.planesize[0],true));
    gb.planesize[0]=a.planesize[0];
    a.pixfmt=b.pixfmt=V4L2_PIX_FMT_GREY;
    assert(!candidate_group(&gb,&a,a.planesize[0],true)); // other paths unchanged

    static cam_t c;
    for(int split=0;split<2;split++) for(int block=0;block<=16;block+=16) {
        learner_fixture(&c,split,block);
        resolve_map(&c);
        assert(c.mapped);
        for(int i=0;i<16;i++) assert(c.slot_of[i]==block+i);
        learner_fixture(&c,split,block);
        for(int i=0;i<16;i++) c.votes[i][16-block+i]=10;
        resolve_map(&c);
        assert(!c.mapped); // simultaneous ambiguous groups must not be guessed
        learner_fixture(&c,split,block);
        memset(c.votes,0,sizeof(c.votes));
        resolve_map(&c);
        assert(!c.mapped); // no evidence is not a mapping
    }
    puts("PASS: false ownership exclusion reproduced; pooled/split layouts resolve; ambiguous, missing, wrong-sensor and wrong-size evidence rejected");
}

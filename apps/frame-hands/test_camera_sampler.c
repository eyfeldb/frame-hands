// Exercise the exact reader sampler on synthetic padded camera memory.
#define main ft_camd_program_main
#include "camd/camd.c"
#undef main
#include <assert.h>

int main(void) {
    cam_t c={0};
    c.lay.width=1056;c.lay.height=1024;c.lay.pitch=1152;
    c.need=(size_t)c.lay.pitch*c.lay.height;
    uint8_t *pixels=calloc(c.need,1);assert(pixels);
    uint64_t old_before[NSAMP],old_after[NSAMP],before[NSAMP],after[NSAMP];
    sample_words(pixels,c.need,old_before);sample_camera(&c,pixels,before);
    for(unsigned y=0;y<c.lay.height;++y)
        memset(pixels+(size_t)y*c.lay.pitch+128,100,c.lay.width-256);
    sample_words(pixels,c.need,old_after);sample_camera(&c,pixels,after);
    assert(!memcmp(old_before,old_after,sizeof old_before)); // reproduce alias
    int changed=0;
    assert(fresh(after,before,&changed));
    assert(changed>NSAMP/2);
    memcpy(before,after,sizeof before);
    for(unsigned y=0;y<c.lay.height;++y)
        memset(pixels+(size_t)y*c.lay.pitch+c.lay.width,255,c.lay.pitch-c.lay.width);
    sample_camera(&c,pixels,after);
    assert(!memcmp(before,after,sizeof before)); // padding cannot fake a frame
    free(pixels);
    puts("PASS: old sampler misses active changes; corrected sampler sees fresh luma and ignores padding");
    return 0;
}

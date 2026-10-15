/* Actual callback regression with synthetic memory and read-only ioctl stub. */
#define ioctl test_ioctl
#define main reader_program_main
#include "camd/camd.c"
#undef main
#undef ioctl
#include <assert.h>

static int sync_balance;
int test_ioctl(int fd, unsigned long request, ...)
{
    va_list args;va_start(args,request);void *arg=va_arg(args,void*);va_end(args);
    if(request==VIDIOC_QUERYBUF) {
        struct v4l2_buffer *b=arg;
        b->memory=V4L2_MEMORY_DMABUF;b->length=2;
        b->m.planes[0].m.fd=125;b->m.planes[0].length=1769472;
        b->m.planes[1].m.fd=126;b->m.planes[1].length=151552;
        return 0;
    }
    assert(request==DMA_BUF_IOCTL_SYNC);
    struct dma_buf_sync *s=arg;
    sync_balance+=(s->flags&DMA_BUF_SYNC_END)?-1:1;
    assert(sync_balance>=0 && sync_balance<=1);
    return 0;
}

int main(void)
{
    static cam_t c;static xr_camera_t cam;static fh_ring_cam_t rc;
    cam.planesize[0]=1769472;cam.planesize[1]=151552;
    c.cam=&cam;c.direct_queue=true;c.mapped=true;c.query_fd=20;
    c.nslots=2;c.maxindex=1;c.source_fd[0]=125;c.source_fd[1]=165;
    c.fd[0]=21;c.fd[1]=22;c.slot_of[0]=0;c.slot_of[1]=1;
    c.lay.width=1056;c.lay.height=1024;c.lay.pitch=1152;
    c.out_w=1056;c.out_h=1024;c.need=1152*1024;
    uint8_t *correct=calloc(c.need,1),*wrong=malloc(c.need);
    assert(correct && wrong);memset(wrong,180,c.need);
    c.map[0]=correct;c.map[1]=wrong;
    c.rc=&rc;rc.nslots=3;rc.stride=1056;
    rc.slot_bytes=sizeof(fh_ring_slot_t)+1056*1024;
    c.ring_slots=calloc(rc.nslots,rc.slot_bytes);assert(c.ring_slots);
    // Another camera's fresh image must never replace this camera's stale one.
    on_frame(&c,0,1,1,1);
    assert(c.stale==1 && rc.published==0 && c.repairs==0 && sync_balance==0);
    memset(correct,120,c.need);
    on_frame(&c,0,2,2,2);
    assert(rc.published==1 && c.repairs==0 && sync_balance==0);
    fh_ring_slot_t *slot=(fh_ring_slot_t*)(c.ring_slots+rc.slot_bytes);
    assert(slot->v4l2_seq==2 && slot->capture_ns==2 && ((uint8_t*)(slot+1))[0]==120);
    // Repeated unchanged data is dropped, not stamped as another observation.
    on_frame(&c,0,3,3,3);
    assert(rc.published==1 && c.stale==2 && sync_balance==0);
    free(correct);free(wrong);free(c.ring_slots);
    puts("PASS actual callback: exact camera only, balanced DMA sync, fresh provenance, no stale republishing or cross-camera rescue");
}

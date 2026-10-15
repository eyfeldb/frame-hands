"""Create a separate exact-queue candidate from the reviewed camera source."""
import hashlib
import json
from pathlib import Path
import shutil
from prepare import prepare


def replace(code,before,after):
    if code.count(before)!=1:raise RuntimeError('Source anchor changed: '+before[:80])
    return code.replace(before,after)


def prepare_direct(source,output):
    prepare(source,output)
    path=output/'camd/camd.c';code=path.read_text()
    code=replace(code,'    int             nslots;',
        '    bool            direct_queue;\n    int             query_fd;\n    int             source_fd[MAX_SLOTS];\n    int             nslots;')
    code=replace(code,'    c->cam  = cam;', '    c->query_fd = -1;\n    c->cam  = cam;')
    code=replace(code,'            c->fd[s]         = fd;',
        '            c->source_fd[s]  = grp->buf[b].xfd;\n            c->fd[s]         = fd;')
    anchor='/*\n * Candidate buffers for a camera:'
    code=replace(code,anchor,r'''
#include "direct_queue.h"

static int exact_slot(cam_t *c, unsigned index)
{
    int source_fd=-1;
    int err=queue_buffer_fd(c->query_fd,index,c->cam->planesize[0],
                            c->cam->planesize[1],&source_fd);
    if(err) { errno=err; return -1; }
    int slot=unique_queue_slot(c->source_fd,c->nslots,source_fd);
    if(slot<0) { errno=ESTALE; return -1; }
    return slot;
}

static void initialize_exact_queue(cam_t *c)
{
    if(!forward_pool(c->cam))return;
    c->query_fd=open(c->cam->path,O_RDONLY|O_NONBLOCK|O_CLOEXEC);
    if(c->query_fd<0)die("%s: cannot query queue: %s",c->slug,strerror(errno));
    int depth=0;
    for(int i=0;i<MAX_INDEX;i++) {
        int slot=exact_slot(c,(unsigned)i);
        if(slot<0) {
            if(errno==EINVAL && i>0)break;
            die("%s: exact mapping index %d: %s",c->slug,i,strerror(errno));
        }
        for(int j=0;j<i;j++)
            if(c->slot_of[j]==slot)die("%s: duplicate buffer in queue",c->slug);
        c->slot_of[i]=slot;
        depth++;
    }
    if(depth<2 || depth==MAX_INDEX)die("%s: unsupported queue depth %d",c->slug,depth);
    c->maxindex=depth-1;
    // A forward buffer cannot simultaneously belong to both camera queues.
    for(int p=0;p<ncams;p++) {
        cam_t *other=&cams[p];
        if(other==c || !other->direct_queue)continue;
        for(int i=0;i<depth;i++)for(int j=0;j<=other->maxindex;j++)
            if(c->source_fd[c->slot_of[i]]==other->source_fd[other->slot_of[j]])
                die("%s: shared buffer identity across camera queues",c->slug);
    }
    c->direct_queue=true;c->mapped=true;c->shared=false;
    printf("%s: kernel queue mapping, depth %d, source fds:",c->slug,depth);
    for(int i=0;i<depth;i++)printf(" %d",c->source_fd[c->slot_of[i]]);
    printf("\n");fflush(stdout);
}

''' + anchor)
    # Finish setup after all candidates have been mapped read-only, before privileges drop.
    anchor='           c->lay.width, c->lay.height, c->lay.pitch, c->nslots, forward_pool(cam) ? " (forward candidate pool)" : (own ? "" : " (shared run)"));'
    code=replace(code,anchor,anchor+'\n    initialize_exact_queue(c);')
    code=replace(code,'    if (!c->mapped) {\n        learn(c, (int)index);',r'''
    if(c->direct_queue) {
        int slot=exact_slot(c,(unsigned)index);
        if(slot<0 || index>c->maxindex) {
            fprintf(stderr,"%s: queue identity changed/unavailable at index %lld; stopping\n",c->slug,(long long)index);
            exit(3);
        }
        c->slot_of[index]=slot;
    }
    if (!c->mapped) {
        learn(c, (int)index);''')
    code=replace(code,'        found = find_fresh(c, cur);',
        '        // Never substitute another camera or index when the kernel gives exact identity.\n'
        '        found = c->direct_queue ? -1 : find_fresh(c, cur);')
    code=replace(code,'        buf_sync(c->fd[slot], DMA_BUF_SYNC_END);\n        // Never substitute',
        '        if(!c->direct_queue)buf_sync(c->fd[slot], DMA_BUF_SYNC_END);\n        // Never substitute')
    code=replace(code,'        c->stale++;\n        c->rc->dropped++;',
        '        if(c->direct_queue) {\n'
        '            c->stale++;c->rc->dropped++;\n'
        '            if(++c->stale_run>=STALE_RELEARN*MAX_RELEARNS) {\n'
        '                fprintf(stderr,"%s: exact buffers stopped changing; stopping\\n",c->slug);exit(3);\n'
        '            }\n            return;\n        }\n'
        '        c->stale++;\n        c->rc->dropped++;')
    path.write_text(code,encoding='utf-8',newline='\n')
    shutil.copyfile(Path(__file__).with_name('direct_queue.h'),output/'camd/direct_queue.h')
    manifest=json.loads((output/'source-manifest.json').read_text())
    manifest['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((output/'camd').iterdir()) if p.is_file()}
    (output/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

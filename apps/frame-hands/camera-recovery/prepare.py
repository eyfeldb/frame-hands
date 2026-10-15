"""Patch a copy of the exact accepted reader; never modify installed sources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

BASE_HASH = '7ee5430af630be1567676658f8f853f011d1747e701d657661bdffd6acd57d24'
HELPER = r'''
/* Sensor markers and group->cam come from descriptor-order heuristics, not
 * from the kernel's actual queued DMA-BUF ownership. For the two calibrated
 * forward NV12 cameras, keep all same-model, size-compatible candidates even
 * when discovery guessed an "own" group. Event learning and the external
 * stereo geometry gate still decide whether these images are usable.
 * Other sensor paths retain their previous selection policy. */
static bool forward_pool(const xr_camera_t *cam)
{
    char model[16];
    model_of(cam->sensor, model, sizeof(model));
    return !strcmp(model, "og01a1b") && cam->pixfmt == V4L2_PIX_FMT_NV12 &&
           cam->width == 1056 && cam->height == 1024;
}

static bool candidate_group(const xr_group_t *g, const xr_camera_t *cam,
                            size_t need, bool own)
{
    if (!group_fits(g, cam, need)) return false;
    if (own && !forward_pool(cam)) return g->cam == cam;
    char model[16], other[16];
    model_of(cam->sensor, model, sizeof(model));
    model_of(g->cam ? g->cam->sensor : g->sensor, other, sizeof(other));
    return model[0] && !strcmp(model, other);
}

'''


def prepare(source, output):
    code = (source/'camd/camd.c').read_bytes()
    if hashlib.sha256(code).hexdigest() != BASE_HASH:
        raise RuntimeError('Reader source differs from the reviewed baseline')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source/'camd', output/'camd')
    code = code.decode()
    marker = '/*\n * Candidate buffers for a camera:'
    assert code.count(marker) == 1
    code = code.replace(marker, HELPER + marker)
    start = code.index('        if (!group_fits(grp, cam, c->need))')
    end = code.index('        for (int b = 0;', start)
    code = code[:start] + '''        if (!candidate_group(grp, cam, c->need, own))
            continue;

        printf("  candidate %s group=%d buffers=%d guessed_owner=%s\\n",
               c->slug, g, grp->nbufs, grp->cam ? grp->cam->path : "unbound");

''' + code[end:]
    # Report the complete descriptor grouping without pixels or sensitive data.
    code = code.replace('printf("XRService pid %d\\ncameras:\\n", xr.pid);',
                        'xr_print(&xr, stdout);\n    printf("camera candidates:\\n");')
    code = code.replace('own ? "" : " (shared run)"',
                        'forward_pool(cam) ? " (forward candidate pool)" : (own ? "" : " (shared run)")')
    (output/'camd/camd.c').write_text(code, encoding='utf-8', newline='\n')
    manifest = dict(baseline_camd_sha256=BASE_HASH,
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((output/'camd').iterdir()) if p.is_file()})
    (output/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)

"""Pin and adapt the public UmeTrack research source; never changes a driver."""
import argparse
import hashlib
from pathlib import Path
import subprocess

COMMIT = 'e263ec0844abf73457468a050a75853444d33d69'


def prepare(repo):
    actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise ValueError(f'Expected UmeTrack {COMMIT}, got {actual}')
    target = repo/'lib/common/hand_skinning.py'
    source = target.read_text(encoding='utf-8')
    old = 'from pytorch3d.transforms import so3_exp_map'
    new = 'from rotations import so3_exp_map'
    if source.count(old) == 1:
        target.write_text(source.replace(old, new), encoding='utf-8')
    elif source.count(new) != 1:
        raise ValueError('Unexpected skinning implementation')
    # A state_dict is enough; explicitly prohibit arbitrary checkpoint objects.
    target = repo/'lib/models/model_loader.py'
    source = target.read_text(encoding='utf-8')
    old = 'torch.load(fp)'
    new = "torch.load(fp, map_location='cpu', weights_only=True)"
    if source.count(old) == 1:
        target.write_text(source.replace(old, new), encoding='utf-8')
    elif source.count(new) != 1:
        raise ValueError('Unexpected model loader')
    weights = repo/'pretrained_models/pretrained_weights.torch'
    print('UmeTrack', actual, 'weights SHA256', hashlib.sha256(weights.read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo', type=Path)
    prepare(parser.parse_args().repo)

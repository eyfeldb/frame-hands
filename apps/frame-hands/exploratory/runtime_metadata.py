"""Read optional installed-source metadata without developer-tool dependencies."""
from pathlib import Path
import re

def installed_revision(repository):
    try:
        git=Path(repository)/'.git'
        if git.is_file():
            directive=git.read_text().strip()
            if not directive.startswith('gitdir: '):return 'unavailable'
            git=(git.parent/directive[8:]).resolve()
        head=(git/'HEAD').read_text().strip()
        if head.startswith('ref: '):
            name=head[5:]
            ref=(git/name).resolve()
            if not ref.is_relative_to(git.resolve()):return 'unavailable'
            if ref.is_file():head=ref.read_text().strip()
            else:
                for line in (git/'packed-refs').read_text().splitlines():
                    fields=line.split()
                    if len(fields)==2 and fields[1]==name:
                        head=fields[0];break
        return head if re.fullmatch('[0-9a-f]{40}',head) else 'unavailable'
    except (OSError,ValueError):
        # Artifact SHA256s remain authoritative even in a deployment without .git.
        return 'unavailable'

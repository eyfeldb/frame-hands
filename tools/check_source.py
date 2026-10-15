"""Check public source syntax, file boundaries, and high-confidence secret patterns."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORE = {'.git', '.local', 'build', 'dist', '__pycache__', '.venv', '.zig-cache'}
BLOCKED = {'.mp4', '.mkv', '.avi', '.jsonl', '.npz', '.npy', '.pt', '.pth', '.pem', '.key', '.exe', '.dll', '.so', '.pdb', '.bin'}


def main():
    # Once in Git, inspect tracked files even if someone added an ignored build/data file.
    from_git = False
    if (ROOT / '.git').exists():
        tracked = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z'])
        names = [p.decode() for p in tracked.split(b'\0') if p]
        paths = [ROOT / n for n in names] if names else list(ROOT.rglob('*'))
        from_git = bool(names)
    else:
        paths = list(ROOT.rglob('*'))
    errors = []
    count = 0
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if not from_git and any(p in IGNORE for p in relative.parts):
            continue
        count += 1
        if path.suffix in BLOCKED:
            errors.append(f'{relative}: generated/private-data file type')
            continue
        if path.suffix == '.png':
            continue
        try:
            text = path.read_text(encoding='utf-8-sig')
            if path.suffix == '.py':
                ast.parse(text, filename=str(relative))
            if path.suffix in {'.json', '.vrdrivermanifest'}:
                json.loads(text)
            patterns = [r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',
                        r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})',
                        r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
                        r'https?://[^\s/:]+:[^\s/@]+@',
                        r'(?i)C:[/\\]+Users[/\\]+(?!Public\b|Default\b)[^/\\\s\x22\x27]+']
            if any(re.search(pattern, text) for pattern in patterns):
                errors.append(f'{relative}: possible secret or personal path; inspect locally')
        except (UnicodeError, SyntaxError, ValueError) as exc:
            errors.append(f'{relative}: {type(exc).__name__}')
    if errors:
        print('\n'.join(errors))
        return 1
    print(f'Checked {count} source files: syntax, JSON, private file types and known secret patterns passed.')
    for directory, test in [('apps/frame-hands/dashboard', 'test_service'),
                            ('apps/frame-hands/native/system-driver', 'test_pc_transport')]:
        if test == 'test_service' and os.name == 'nt':
            print('Dashboard lifecycle tests require Linux (fcntl/getuid); run by Linux CI.')
            continue
        subprocess.run([sys.executable, '-m', 'unittest', test], cwd=ROOT / directory, check=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

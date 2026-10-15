"""Exclusive native experiment guard; services alone do not establish idleness."""
from contextlib import contextmanager
from pathlib import Path
import os,subprocess

PREFIXES=('fe-eyegrab','fe-collector','frame-camera','ft-hands','ft-camd','ft-ringplay')
def conflicts(processes,own_pid):
 return [(pid,name) for pid,name in processes if pid!=own_pid and name.startswith(PREFIXES)]

def process_names(proc=Path('/proc')):
 rows=[]
 for entry in proc.iterdir():
  if not entry.name.isdecimal():continue
  try:rows.append((int(entry.name),(entry/'comm').read_text().strip()))
  except (FileNotFoundError,ProcessLookupError):continue
  except PermissionError:raise RuntimeError('Cannot inspect native process ownership')
 return rows

def require_no_optical_processes():
 active=conflicts(process_names(),os.getpid())
 if active:raise RuntimeError('Optical process active: '+repr(active))

@contextmanager
def native_workflow(root=Path('/home/steamos/devkit-game')):
 import fcntl
 locks=[]
 try:
  for relative in ('FrameEyes/workflow.lock','FrameHands/workflow.lock','FrameHands/trial.lock'):
   lock=(root/relative).open('a');locks.append(lock)
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:raise RuntimeError('Another workflow owns '+relative+'; no benchmark started') from None
  if subprocess.run(['systemctl','is-active','--quiet','frame-eyes-reader-brightness.service']).returncode==0:raise RuntimeError('Eye service active')
  require_no_optical_processes()
  yield
 finally:
  for lock in reversed(locks):lock.close()

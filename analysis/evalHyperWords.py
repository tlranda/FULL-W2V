#!/usr/bin/env python3

from os import path
import sys, subprocess, argparse

parser = argparse.ArgumentParser(description='Automates the process of evaluating a word embedding.')
parser.add_argument('file', type=str, help='Word embedding file.')
parser.add_argument('-skip-regen', '--skip-regen', action='store_false', help="Skip regenerating the numpy and .py cache files (default always regenerates)")
parser.add_argument('-skip-similarity', '--skip-similarity', action='store_true', help="Skip similarity analyses (default does not skip)")
parser.add_argument('-skip-analogy', '--skip-analogy', action='store_true', help="Skip analogy analyses (default does not skip)")
parser.add_argument('-preserve-base-embedding', '--preserve-base-embedding', action='store_true', help="Don't delete the base embedding after creating .npy and .vocab caches")
parser.add_argument('-vocab', '--vocab', type=str, default=None, help='Vocabulary file to recount from')
parser.add_argument('-verbose', '--verbose', action='store_true', help='Output bonus info for analysis')
parser.add_argument('-simlex-bin', '--simlex-bin', action='store_true', help='Set output to only simlex bin')
parser.add_argument('-cutoff', '--cutoff', type=int, default=200, help='Cutoff for evaluation')
args = parser.parse_args()
if args.simlex_bin:
  args.verbose=True

script_home = path.dirname(__file__)
hyperdir = path.join(script_home, "hyperwords")

# Hyperwords requires embedding to be *.words
base_embedding = path.relpath(args.file)
words_ready = base_embedding.endswith(".words")
if words_ready:
  embedding = base_embedding
  base_embedding = base_embedding[:-6]
  # Attempting to skip regen but it doesn't exist--soft default back to not skipping
  if not args.skip_regen and not (path.exists(embedding+".npy") and path.exists(embedding+".vocab")):
    args.skip_regen = True
  elif not args.skip_regen and not args.simlex_bin:
    print("Using existing HyperWords adjusted embedding based on {0} and {1}".format(embedding+".npy", embedding+".vocab"))
else:
  # Create suitable name
  embedding = path.relpath(base_embedding+".words")
  if not args.simlex_bin:
    print("{0} is not a *.words file (HyperWords requirement), copying to {1}...".format(base_embedding, embedding))
  if path.exists(embedding): # Potentially unsafe to copy
    print("{0} already exists. Overwrite? y/n: ".format(embedding), end='')
    choice = input()
    while choice.lower() not in ['y', 'n']:
      print("Unrecognized input ({0})! {1} already exists. Overwrite? y/n: ".format(choice, embedding), end='')
      choice = input()
    if choice.lower() == 'y':
      args.skip_regen = True # This will be a new file, it must have embeddings regenerated
      import shutil
      shutil.copyfile(base_embedding, embedding)
      del shutil
    else:
      args.skip_regen = False
  else: # No collision, safe to copy
    import shutil
    shutil.copyfile(base_embedding, embedding)
    del shutil

# Create embedding for hyperwords
if args.skip_regen:
  # Special case: Base embedding does not exist
  if not path.exists(embedding):
    if not args.simlex_bin:
      print("No base embedding found to regenerate!!")
    if path.exists(embedding+'.npy') and path.exists(embedding+'.vocab'):
      if not args.simlex_bin:
        print("Continuing with cached materials {0}.npy and {0}.vocab".format(embedding))
    else:
      print("No cached {0}.npy or {0}.vocab to use, please ensure the correct file path was specified.".format(embedding))
      exit()
  else:
    # Apply numpy fixup for Hyperwords
    if not args.simlex_bin:
      print("Adjusting embedding for HyperWords Use...")
    completed_proc = subprocess.run(['python2', path.relpath(hyperdir+'/hyperwords/text2numpy.py'), embedding])
    if completed_proc.returncode != 0:
      print("FAILURE! Aborting.")
      exit()
    # Preserve disk space after cache by removing the original ascii file
    if not args.preserve_base_embedding:
      import os
      os.remove(embedding)

# Perform hyperwords evaluations
if not args.skip_similarity:
  extension = ['--vocab', args.vocab] if args.vocab is not None else []
  if args.verbose:
    extension.extend(['--verbose', '1'])
  if not args.simlex_bin:
    print("Similarity Results (WS353, SimLex999)\n-------------------------------------")
    cmd = ['python2', path.relpath(hyperdir+'/hyperwords/ws_eval.py'), 'embedding', base_embedding, path.relpath(hyperdir+'/testsets/ws/ws353.txt')]
    cmd.extend(extension)
    completed_proc = subprocess.run(cmd)
    if completed_proc.returncode != 0:
      if completed_proc.stdout is not None:
        print(f'stdout ws353: {completed_proc.stdout}')
      if completed_proc.stderr is not None:
        print(f'stderr ws353: {completed_proc.stderr}')
      print("FAILURE! Aborting.")
      exit()
    print()
    cmd = ['python2', path.relpath(hyperdir+'/hyperwords/ws_eval.py'), 'embedding', base_embedding, path.relpath(hyperdir+'/testsets/ws/SimLex999.txt')]
    cmd.extend(extension)
    if args.cutoff > 0:
      cmd.extend(['--cutoff', str(args.cutoff)])
    completed_proc = subprocess.run(cmd)
    if completed_proc.returncode != 0:
      if completed_proc.stdout is not None:
        print(f'stdout simlex999: {completed_proc.stdout}')
      if completed_proc.stderr is not None:
        print(f'stderr simlex999: {completed_proc.stderr}')
      print("FAILURE! Aborting.")
      exit()
    if not args.simlex_bin:
      print()

if not args.skip_analogy and not args.simlex_bin:
  print("Google Analogy Results\n----------------------")
  completed_proc = subprocess.run(['python2', path.relpath(hyperdir+'/hyperwords/analogy_eval.py'), 'embedding', base_embedding, path.relpath(hyperdir+'/testsets/analogy/google.txt')])
  if completed_proc.returncode != 0:
    if completed_proc.stdout is not None:
      print(completed_proc.stdout)
    if completed_proc.stderr is not None:
      print(completed_proc.stderr)
    print("FAILURE! Aborting.")
    exit()
  print()


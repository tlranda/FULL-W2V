import argparse as AP, subprocess
from random import shuffle

parser = AP.ArgumentParser("Reverses sentence content within target cache")
parser.add_argument('input', type=str, help='Input file (corpus or serialized cache)')
parser.add_argument('output', type=str, help='Output file, will be overwritten if already exists')
parser.add_argument('--cache', action='store_true', help='Input file is a cache (first line shouldn\'t be affected by shuffling/randomization)')
#parser.add_argument('--shuffle-order', action='store_true', help='Output sentences in shuffled order. Default maintains orginal sentence ordering.')
parser.add_argument('--reverse-sentences', action='store_true', help='Output sentence content in reverse order. Default maintains orginal content order.')
parser.add_argument('--scramble-sentences', action='store_true', help='Output sentence content in random order. Default maintains original content order.')
parser.add_argument('--debug', type=int, default=100, help='Frequency of debug outputs')
args = parser.parse_args()

def validate_args():
  from os import path
  if not path.exists(args.input):
    print("No such file: {0}".format(args.input))
    raise e

def load_sentences():
  try:
    infile = open(args.input, 'r')
  except OSError as e:
    print("Could not open input file {0}".format(args.input))
    raise e
  sentences = []
  if args.cache:
    first_line = infile.readline()
    max_lines = int(first_line.split()[0])
  else:
    first_line = None
    proc = subprocess.run(['wc', '-l', args.input], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
      print("Failed to read length of corpus!")
      exit(1)
    max_lines = int(proc.stdout.decode().split()[0])
  print("Loading...")
  line = 0
  for _ in infile.readlines():
    sentences.append(_.rstrip().split(" "))
    line += 1
    if args.debug > 0 and line % args.debug == 0:
      print("Line {0}/{1} ({2:.2f}%)".format(line, max_lines, 100*line/max_lines), end='\r')
  print("\nRead")
  infile.close()
  return first_line, sentences, max_lines

def update_sentences(first_line, sentences, max_lines):
  #if args.shuffle_order:
  #  shuffle(sentences)
  with open(args.output, 'w') as of:
    print("Exporting with changes...")
    if first_line is not None:
      of.write(first_line)
    line = 0
    for _ in sentences:
      line += 1
      if args.debug > 0 and line % args.debug == 0:
        print("Line {0}/{1} ({2:.2f}%)".format(line, max_lines, 100*line/max_lines), end='\r')
      if args.scramble_sentences:
        shuffle(_)
      if args.reverse_sentences:
        _.reverse()
      _ = " ".join(_)+"\n"
      of.write(_)
  print("\nExported")

if __name__ == '__main__':
  validate_args()
  first_line, corpus, max_lines = load_sentences()
  update_sentences(first_line, corpus, max_lines)


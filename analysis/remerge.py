import numpy as np
from argparse import ArgumentParser as AP

def build():
    p = AP("Recover Word2Vec embedding files from hyperwords .vocab and .npy files")
    p.add_argument('npy', type=str, help='Path to npy file')
    p.add_argument('vocab', type=str, help='Path to vocab file')
    p.add_argument('-out', type=str, default=None, help='Name to export as')
    return p

def parse(p, a=None):
    if a is None:
        a = p.parse_args()
    else:
        from glob import glob
        globbed = []
        for arg in a:
            if '*' in arg:
                globbed.extend(glob(arg))
            else:
                globbed.append(arg)
        a = p.parse_args(globbed)
    return a

def reconstruct(npy_name, vocab_name, out=None):
    vals = np.load(npy_name)
    with open(vocab_name, 'r') as f:
        strings = [_.rstrip() for _ in f.readlines()]
    # Auto-naming procedure
    if out is None:
        out = npy_name[:npy_name.index('.npy')]
    with open(out, 'w') as f:
        f.write(f"{len(strings)} {vals.shape[1]}"+"\n")
        for word, array in zip(strings, vals):
            f.write(f"{word} {' '.join([str(_) for _ in array])}"+"\n")

if __name__ == '__main__':
    args = parse(build())
    reconstruct(args.npy, args.vocab, args.out)

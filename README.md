## The write up
https://issabawwab.pages.dev/posts/micrograd-from-scratch/

## MNIST data
The `data/` dir is gitignored, get the files before running:
```
mkdir -p data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    curl -sSL "https://raw.githubusercontent.com/fgnt/mnist/master/$f.gz" -o "data/$f.gz"
done
gunzip data/*.gz
```

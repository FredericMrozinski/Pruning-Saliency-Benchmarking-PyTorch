# Pruning-Saliency-Benchmarking-PyTorch

This repository houses a small Python script that compares magnitude based weight pruning in neural networks with Fisher-information based weight pruning. It accompanies my following blog-post:

[https://fredericmrozinski.github.io/built-blog/posts/neuron-pruning-1/](https://fredericmrozinski.github.io/built-blog/posts/neuron-pruning-1/)

# How to run the script

1. Install all packages as listed in `requirements.txt`.
2. Download the ImageNet **validation** dataset ("Validation images (all tasks)") from [https://image-net.org/challenges/LSVRC/2012/2012-downloads.php](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and place the downloaded file into `res` (no need to unzip).
3. Downlaod the labels ("Development kit (Task 1 & 2)") from the same website and also place in `res`.
4. Execute the script! The results will be logged to TensorBoard.

# Development state

While the script, to the best of my knowledge, is currently stable, I want to extend it further, soon, to test it on more models.

# Questions or issues?

Please contact me at [fm.public@tuta.com](mailto:fm.public@tuta.com). I'll be more than happy to help!

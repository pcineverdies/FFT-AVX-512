# FFT-AVX-512

This repository is the result of my bachelor's thesis, **Speed-up of Fast Fourier Transform using the Intel AVX-512 SIMD instructions**.
This work lead to an artcile which can be found [here](https://link.springer.com/chapter/10.1007/978-3-031-30333-3_34).

After an initial analysis of the AVX's ISA, I applied techinques such as *Loop unrolling* and *SLP* to improve the performance of a trivial implementation of the recursive algorithm. 

I used and compared different ways to memorize complex numbers, which I called *complex interleaved* (real and immaginary parts are staggered inside a vector) and *block interleaved* (real and immaginary parts are separated in two different vectors). The idea was taken from [here](https://ieeexplore.ieee.org/document/8091024).

The main perk of this project is its simplicity: you can use my version to get into AVX-512's instruction set, and even try to boost it up choosing better intrinsics (or applying more advanced techniques). 

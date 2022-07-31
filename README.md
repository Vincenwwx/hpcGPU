## Implementation of Decoupled Look-back LSD Radix Sort[^1] Using OpenCL on GPU
## Introduction

This project is based on the _Onesweep[^2]_ paper by Andy Adinets and Duane Merril from NVIDIA, but implemented with OpenCL.

### Basics

- Data: array of _n_ elements, with each element of _k_ bits
- Digit: _d_ bits, size of every digit can be determined arbitarily
- Digit place: k / d 
- Digit binning: 2^d

In our example, we take n = 40, with each tile containing 10 elments, therefore 4 tiles or blocks are needed. 

Besides, each key is of 32 bits and each digit takes 8 bits, and thus 4 digits/key and the number digit binnings is 2^8 = 256


## Project Definition
1. Implement the Radix sort on GPU and compare the performance with the CPU code.
2. The length of the input array (of integers, float or binary) to be sorted should be 1 million. 

## Implementation

### Kernel 1 -  Upfront histogram 

The first kernel is responsible for generating a global histogram for the the whole data of __all digit places__ with a single pass.

![Upfront Histogram](https://user-images.githubusercontent.com/49132368/182047122-22470312-f0ab-4c51-8092-65c75f9ae564.png)

_Figure 1 - Upfront Histogram_

> Memory operation comlexity: _n_ reads

### Kernel 2 - Exclusive Sum

Calculate the exclusive prefix sum along the upfront histogram from the lowest to the highest digit binning.

![Exclusive sum](https://user-images.githubusercontent.com/49132368/182047131-028b1d32-e2db-4794-a5b1-7e69b58a149b.png)

_Figure 2 - Exclusive Sum_

> Memory operation complexity: #_digit binnings_ * #_digit places_. Since _O_ << n, can be neglected

### Kernel 3 -  Chained Scan Digit Binnings

This kernel will be executed #_digit places_ times/iterations. 

For each iteration: it will first calculate the starting index of each digit binning in every data tile using the __decoupled look-back__[^3] technique. And there will be #_digit binnings_ look-back lanes for each iteration.
![Chained](https://user-images.githubusercontent.com/49132368/182047634-a329bf05-64ed-44ac-b033-a4de664cd79a.png)
_Figure 3 - Chained Scans_

![Decoupled Look-back Technique](https://user-images.githubusercontent.com/49132368/182047710-fb8a8d3c-396b-47c9-84d2-73874d15b1ab.png)

_Figure 4 - Decoupled Look-back Technique_ 

Then, each tile/block will sequentially scan itself to further get the global index for every internal data elements, and the scans happening among the blocks are conducted simultaneously.

Now that we have had new indexes for all data elements, we re-scatter the elements into the data array.

> Memory operation complexity: 2 * #_digit places_* n 

## Reference
[^1] [Redix sort in Wiki](https://en.wikipedia.org/wiki/Radix_sort)

[^2] [Onesweep: A Faster Least Significant Digit Radix Sort for GPUs](https://arxiv.org/pdf/2206.01784.pdf)

[^3] [Single-pass Parallel Prefix Scan with Decoupled Look-back](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back)
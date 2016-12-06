# Adaptels [Achanta+, CoRR2016] in C++

this is a C++ implementation of Adaptels proposed in the paper [Achanta+, CoRR2016] to create superpixels of a 2D RGB image.

some equations are APPROXIMATED and does not yield the exact results.

input image are converted to:
* RGB : to CIE L\*a\*b\*
* Gray : as is

pixel model:
* Laplacian (double exponential) distribution, with some approximation.


### Reference
 * "Uniform Information Segmentation"., Radhakrishna Achanta, Pablo Márquez-Neila, Pascal Fua, Sabine Süsstrunk, CoRR 2016

**this is not an official implementation by the authors of the paper.**

# Build

## Build on Linux/Mac OS X

TODO

## Build on Windows
```
> cd Adaptels
> mkdir build
> cd build
> cmake -DCMAKE_PREFIX_PATH=C:\Programs\library\opencv-3.1.0\build -G "Visual Studio 14 2015 Win64" ..
> start Adaptels.sln
Build x64 Release.
```

# Usage

```
[usage] Adaptels.exe <image file> <information threshold> <color=1 gray=0>
e.g.)
> Adaptels.exe lena.png 30.0 0
```
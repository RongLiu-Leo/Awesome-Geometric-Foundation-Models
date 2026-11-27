# Awesome-Geometric-Foundation-Models

A curated guide to geometric foundation models for 3D and 4D vision, with papers, architecture diagrams, and walkthroughs of the original code.

[Browse the interactive guide](https://rongliu-leo.github.io/Awesome-Geometric-Foundation-Models/) · [Research roadmap](https://rongliu-leo.github.io/Awesome-Geometric-Foundation-Models/#roadmap) · [Compare models](https://rongliu-leo.github.io/Awesome-Geometric-Foundation-Models/#compare/all) · [Contribute a paper](CONTRIBUTING.md)

## Papers

Ordered by the first public version of each paper. Select a model name to see how it turns inputs into geometry. Profiles with a code walkthrough trace the forward pass to specific lines in the authors’ implementation.

<!-- catalog:start -->

### 2023

- [**DUSt3R**](https://rongliu-leo.github.io/Awesome-Geometric-Foundation-Models/#model/dust3r) · [Paper](<https://arxiv.org/abs/2312.14132>) · [Code](<https://github.com/naver/dust3r/blob/main/dust3r/model.py>) · [Project](<https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/>) — Two image grids become two dense pointmaps expressed in the first camera’s coordinate frame.

### 2024

- [**MASt3R**](https://rongliu-leo.github.io/Awesome-Geometric-Foundation-Models/#model/mast3r) · [Paper](<https://arxiv.org/abs/2406.09756>) · [Code](<https://github.com/naver/mast3r/blob/main/mast3r/catmlp_dpt_head.py>) · [Project](<https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/>) — A descriptor head adds explicit pixel matching to DUSt3R’s two-view pointmap architecture.

### 2025

- [**VGGT**](https://rongliu-leo.github.io/Awesome-Geometric-Foundation-Models/#model/vggt) · [Paper](<https://arxiv.org/abs/2503.11651>) · [Code](<https://github.com/facebookresearch/vggt/blob/main/vggt/models/vggt.py>) · [Project](<https://vgg-t.github.io/>) — Alternating frame and global attention supplies one scene representation to several prediction heads.
- [**π³**](https://rongliu-leo.github.io/Awesome-Geometric-Foundation-Models/#model/pi3) · [Paper](<https://arxiv.org/abs/2507.13347>) · [Code](<https://github.com/yyfz/Pi3/blob/main/pi3/models/pi3.py>) · [Project](<https://yyfz.github.io/pi3/>) — Every view takes the same path through the network, removing the need to pick an anchor.
- [**MapAnything**](https://rongliu-leo.github.io/Awesome-Geometric-Foundation-Models/#model/mapanything) · [Paper](<https://arxiv.org/abs/2509.13414>) · [Code](<https://github.com/facebookresearch/map-anything/blob/main/mapanything/models/mapanything/model.py>) · [Project](<https://map-anything.github.io/>) — Optional depth, poses and calibration help one network reconstruct scenes at an estimated metric scale.
- [**Depth Anything 3**](https://rongliu-leo.github.io/Awesome-Geometric-Foundation-Models/#model/da3) · [Paper](<https://arxiv.org/abs/2511.10647>) · [Code](<https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/model/da3.py>) · [Project](<https://depth-anything-3.github.io/>) — One vision transformer estimates depth and cameras across views, trained with supervision on both depth and rays.

### 2026

- [**VGGT-Ω**](https://rongliu-leo.github.io/Awesome-Geometric-Foundation-Models/#model/omega) · [Paper](<https://arxiv.org/abs/2605.15195>) · [Code](<https://github.com/facebookresearch/vggt-omega/blob/main/vggt_omega/models/vggt_omega.py>) · [Project](<https://vggt-omega.github.io/>) — Selected layers pass compact register and camera tokens between views to reduce training memory.
- [**Déjà View**](https://rongliu-leo.github.io/Awesome-Geometric-Foundation-Models/#model/dvlt) · [Paper](<https://arxiv.org/abs/2605.30215>) · [Code](<https://github.com/nv-tlabs/dvlt/blob/main/src/dvlt/model/dvlt/model.py>) · [Project](<https://research.nvidia.com/labs/dvl/projects/dvlt/>) — One transformer block revisits the same views, with an adjustable budget for refining geometry.
- [**π⁴**](https://rongliu-leo.github.io/Awesome-Geometric-Foundation-Models/#model/pi4) · [Paper](<https://arxiv.org/abs/2611.23144>) · [Code](<https://github.com/yyfz/Pi4/blob/main/pi4/models/pi4.py>) · [Project](<https://yyfz.github.io/pi4/>) — Two looped decoders reconstruct visible surfaces and track moving points through the recovered 3D scene.

<!-- catalog:end -->

## Read offline

Download or clone this repository and open `index.html` in a browser. Diagrams, comparisons, and included code excerpts work offline.

## Credits and licensing

This guide was prepared for noncommercial research and education. Original material in this project is licensed under the [Apache License 2.0](LICENSE). Third-party code excerpts retain their upstream licenses, including applicable noncommercial terms. Each walkthrough includes the relevant license texts and attributions under **Source credits and licenses**.

/* Model catalog. See CONTRIBUTING.md for the entry format. */
window.AtlasData = [
  {
    id: "dvlt", name: "Déjà View", title: "Looping Transformers for Multi-View 3D Reconstruction", preprint: "2026-05-28", venue: "NeurIPS 2026", family: "Looped refinement", color: "blue",
    group: "Joint multi-view", compareWith: "vggt",
    summary: "One transformer block revisits the same views, with an adjustable budget for refining geometry.",
    stages: [
      ["Encode views", "A DINOv2 encoder turns each RGB view into patch features."],
      ["Repeat one block", "Each step mixes tokens within frames and across views. A time signal conditions the shared block."],
      ["Decode depth and rays", "Dense heads predict depth, rays and confidence from the refined features."],
      ["Recover the scene", "Fit cameras from rays and combine each ray with its depth to locate a 3D point."]
    ],
    strength: "The released 117M-parameter model lets you trade inference time for refinement by choosing 8–16 steps with one checkpoint.",
    limit: "Each additional step costs compute. The authors observe instability beyond the trained step range, and moving objects have no explicit motion representation.",
    inputs: "Uncalibrated RGB views of a scene.", exchange: "One weight-tied block alternates frame and global attention over repeated steps.",
    output: "Depth, rays and confidence. Rays yield cameras and points.", scale: "Relative scale in a frame anchored to the first view.", motion: "Scene reconstruction. The model has no trajectory head.",
    project: "https://research.nvidia.com/labs/dvl/projects/dvlt/", paper: "https://arxiv.org/abs/2605.30215", code: "https://github.com/nv-tlabs/dvlt/blob/main/src/dvlt/model/dvlt/model.py", symbol: "DVLTModel",
    moreCode: "https://github.com/nv-tlabs/dvlt/blob/main/src/dvlt/model/dvlt/blocks.py", moreSymbol: "LoopedAABlock", figure: "https://research.nvidia.com/labs/dvl/projects/dvlt/_astro/method_overview.Bejem5VV_Z1QUyvh.png",
    codeNote: "In _solve_inference_linspace_k, the requested step count sets the time grid. Each interval updates the same hidden state."
  }, {
    id: "dust3r", name: "DUSt3R", title: "Geometric 3D Vision Made Easy", preprint: "2023-12-21", venue: "CVPR 2024", family: "Pairwise geometry", color: "sage",
    group: "Pairwise", compareWith: "mast3r",
    summary: "Two image grids become two dense pointmaps expressed in the first camera’s coordinate frame.",
    stages: [
      ["Encode a pair", "A shared CroCo-initialized vision transformer encodes the two images independently."],
      ["Exchange features", "Two decoder branches use cross-attention to read the other image’s features."],
      ["Regress pointmaps", "Each branch predicts a 3D position and confidence per pixel in the first camera’s coordinates."],
      ["Align more views", "For a collection, build a graph of overlapping pairs and optimize their alignment into one scene."]
    ],
    strength: "Two uncalibrated images suffice for dense correspondences in 3D. The shared coordinate frame also supports recovery of relative cameras.",
    limit: "Each pair establishes its own frame and scale. A larger scene needs enough overlapping pairs and an additional alignment step.",
    inputs: "Two uncalibrated RGB images per network call.", exchange: "Twin cross-attention decoders exchange features between two views.",
    output: "Two XYZ pointmaps and pixel confidence. Cameras can be recovered from the pointmaps.", scale: "Relative. Both pointmaps use the first camera frame.", motion: "Matches can be derived from 3D geometry. There is no temporal motion decoder.",
    project: "https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/", paper: "https://arxiv.org/abs/2312.14132", code: "https://github.com/naver/dust3r/blob/main/dust3r/model.py", symbol: "AsymmetricCroCo3DStereo.forward",
    moreCode: "https://github.com/naver/dust3r/tree/main/dust3r/cloud_opt", moreSymbol: "Global alignment", figure: "",
    codeNote: "The second head’s output key names its coordinate system. Pixels belong to image two, while their XYZ positions refer to camera one."
  }, {
    id: "mast3r", name: "MASt3R", title: "Grounding Image Matching in 3D with MASt3R", preprint: "2024-06-14", venue: "ECCV 2024", family: "Dense matching", color: "sand",
    group: "Pairwise", compareWith: "dust3r",
    summary: "A descriptor head adds explicit pixel matching to DUSt3R’s two-view pointmap architecture.",
    stages: [
      ["Encode a pair", "A shared visual encoder processes an image pair, as in DUSt3R."],
      ["Decode together", "Two communicating decoder branches build geometry-aware features."],
      ["Predict two outputs", "A dense head estimates metric pointmaps. A second head maps encoder and decoder features to pixel descriptors."],
      ["Match descriptors", "Fast reciprocal nearest-neighbor search accepts descriptors that select one another across the pair."]
    ],
    strength: "Explicit descriptors make wide-baseline matching and map-free relocalization practical. Reciprocal search avoids exhaustively comparing every pair of pixels.",
    limit: "The network still operates on pairs. Reconstructing a larger collection adds pair selection and a global alignment or optimization stage.",
    inputs: "Two RGB views per network call.", exchange: "DUSt3R-style two-view decoding plus a learned local descriptor branch.", output: "Metric pointmaps, confidence and dense descriptors for pixel matches.",
    scale: "The metric checkpoint learns absolute scale from training data.", motion: "Reciprocal matches are two-view pixel correspondences. Long tracks require additional association.",
    project: "https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/", paper: "https://arxiv.org/abs/2406.09756", code: "https://github.com/naver/mast3r/blob/main/mast3r/catmlp_dpt_head.py", symbol: "Cat_MLP_LocalFeatures_DPT_Pts3d.forward",
    moreCode: "https://github.com/naver/mast3r/blob/main/mast3r/fast_nn.py", moreSymbol: "fast_reciprocal_NNs", figure: "",
    codeNote: "The dense head combines predicted points with local features. The separate reciprocal matcher consumes the resulting descriptors."
  }, {
    id: "vggt", name: "VGGT", title: "Visual Geometry Grounded Transformer", preprint: "2025-03-14", venue: "CVPR 2025", family: "Joint multi-view", color: "blue",
    group: "Joint multi-view", compareWith: "omega",
    summary: "Alternating frame and global attention supplies one scene representation to several prediction heads.",
    stages: [
      ["Tokenize views", "DINOv2 patch features join camera and register tokens for each image."],
      ["Mix across views", "A stack of 24 frame/global attention pairs connects image detail with the rest of the scene."],
      ["Read scene geometry", "Camera, depth and pointmap heads read the aggregated features. Dense predictions include confidence."],
      ["Track query pixels", "With query points, a tracking head predicts image coordinates and visibility across the other views."]
    ],
    strength: "A single forward pass supplies cameras, depth, pointmaps and optional pixel tracks for a view set. The user can also unproject depth using the predicted cameras.",
    limit: "Global attention grows quadratically with the combined token count. The authors identify strong non-rigid deformation and extreme rotations as difficult cases.",
    inputs: "One or more RGB images, with optional 2D query pixels for tracking.", exchange: "Distinct transformer layers alternate per-frame and all-view self-attention.",
    output: "Cameras, depth, points and confidence, plus optional query tracks.", scale: "Relative scale with the first camera as the reference.", motion: "The tracking head returns 2D pixel coordinates, visibility and confidence.",
    project: "https://vgg-t.github.io/", paper: "https://arxiv.org/abs/2503.11651", code: "https://github.com/facebookresearch/vggt/blob/main/vggt/models/vggt.py", symbol: "VGGT.forward",
    moreCode: "https://github.com/facebookresearch/vggt/blob/main/vggt/models/aggregator.py", moreSymbol: "Aggregator", figure: "https://vgg-t.github.io/resources/architecture_v4.png",
    codeNote: "Follow aggregated_tokens_list from the aggregator into each task head. Passing query_points activates the optional pixel tracker."
  }, {
    id: "pi3", name: "π³", title: "Permutation-Equivariant Visual Geometry Learning", preprint: "2025-07-17", venue: "ICLR 2026", family: "Reference-free", color: "lilac",
    group: "Joint multi-view", compareWith: "pi4", aliases: ["pi3", "pi^3"],
    summary: "Every view takes the same path through the network, removing the need to pick an anchor.",
    stages: [
      ["Encode each view", "DINOv2 produces patch features. The model attaches the same learned register embeddings to each view."],
      ["Exchange symmetrically", "Alternating frame and global attention treats the input views alike."],
      ["Decode local geometry", "Separate heads return camera-local points, camera-to-world poses and confidence."],
      ["Transform points", "Apply each predicted camera pose to its local points to assemble a shared point cloud."]
    ],
    strength: "Reordering a photo collection reorders its per-view predictions consistently. Reconstruction avoids sensitivity to the choice of a first reference image.",
    limit: "The original π³ estimates relative scale. The paper also documents grid artifacts in uncertain regions and failures on transparent surfaces.",
    inputs: "An unordered collection of RGB views.", exchange: "Symmetric per-view processing with alternating frame and global attention.",
    output: "Camera poses, local pointmaps and confidence. Poses transform points to a common frame.", scale: "Relative scale for original π³. Pi3X is a separate variant.", motion: "Reconstructs per-view geometry and cameras. Point trajectories require a separate method.",
    project: "https://yyfz.github.io/pi3/", paper: "https://arxiv.org/abs/2507.13347", code: "https://github.com/yyfz/Pi3/blob/main/pi3/models/pi3.py", symbol: "Pi3.forward",
    moreCode: "", moreSymbol: "", figure: "https://yyfz.github.io/pi3/assets/images/main.png",
    codeNote: "The point head multiplies image-plane coordinates by depth to construct local XYZ. A later camera transform places those points in the scene."
  }, {
    id: "omega", name: "VGGT-Ω", title: "VGGT-Ω", preprint: "2026-05-14", venue: "CVPR 2026", family: "Register attention", color: "blue",
    group: "Joint multi-view", compareWith: "vggt", aliases: ["vggt omega"],
    summary: "Selected layers pass compact register and camera tokens between views to reduce training memory.",
    stages: [
      ["Encode each frame", "DINOv3-initialized features join one camera token and 16 register tokens per view."],
      ["Exchange selectively", "Some inter-frame layers exchange camera and register tokens only. The remaining global layers also exchange patches."],
      ["Decode cameras and depth", "A camera head and a streamlined dense head estimate poses, intrinsics, depth and confidence."],
      ["Unproject the depths", "Depth and cameras yield scene points. The released model also exposes camera and register features."]
    ],
    strength: "The authors report roughly 30% of VGGT’s training GPU memory. The revised training recipe covers both static scenes and moving subjects.",
    limit: "The released heads provide depth and cameras without explicit point tracks. Strong motion blur and abrupt field-of-view changes can degrade reconstruction.",
    inputs: "RGB images or video frames.", exchange: "Frame attention plus a mix of global and compact camera/register attention across views.",
    output: "Camera parameters, depth, confidence and camera/register features.", scale: "Depth and camera translation in a normalized scene frame.", motion: "Trained on dynamic scenes. Its released forward pass outputs per-frame geometry without trajectories.",
    project: "https://vggt-omega.github.io/", paper: "https://arxiv.org/abs/2605.15195", code: "https://github.com/facebookresearch/vggt-omega/blob/main/vggt_omega/models/vggt_omega.py", symbol: "VGGTOmega.forward",
    moreCode: "https://github.com/facebookresearch/vggt-omega/blob/main/vggt_omega/models/aggregator.py", moreSymbol: "Aggregator._run_inter_frame_attention_block", figure: "",
    codeNote: "The aggregator can update only camera and register tokens across frames. VGGTOmega.forward reads depth and confidence from a single dense head."
  },
  {
    id: "da3",
    name: "Depth Anything 3",
    title: "Depth Anything 3: Recovering the Visual Space from Any Views",
    preprint: "2025-11-13",
    venue: "ICLR 2026",
    family: "Depth & rays",
    color: "lilac",
    group: "Joint multi-view", compareWith: "mapanything",
    summary: "One vision transformer estimates depth and cameras across views, trained with supervision on both depth and rays.",
    stages: [
      ["Images and cameras", "Encode image patches. Known poses and intrinsics can supply camera tokens."],
      ["View attention", "Early DINOv2 layers process each image. Later layers alternate within-view and cross-view attention."],
      ["Depth and rays", "Two DPT branches decode pixel depth and an auxiliary ray field on a coarser grid."],
      ["Scene geometry", "The default camera head reads camera tokens. The exporter unprojects depth with the recovered intrinsics and poses to form scene points."]
    ],
    strength: "The same backbone handles a single photograph and a collection of views. Known camera parameters can constrain the reconstruction, and compatible checkpoints can also predict 3D Gaussians for rendering.",
    limit: "The main any-view models recover relative scale from RGB. Producing metric reconstructions with the Nested checkpoint adds a second network to estimate scale.",
    inputs: "One RGB image, an image set or a video, optionally with camera calibration and poses.",
    exchange: "A single DINOv2 backbone changes which image patches can attend to one another in its later layers.",
    output: "DA3-Large returns depth, confidence, intrinsics and camera poses. An optional inference path fits cameras from the internal ray field.",
    scale: "Main checkpoints give relative depth. Nested aligns any-view results with a monocular metric-depth model.",
    motion: "Video inputs yield depth and camera motion. The architecture has no moving-object trajectory head.",
    project: "https://depth-anything-3.github.io/",
    paper: "https://arxiv.org/abs/2511.10647",
    code: "https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/model/da3.py",
    symbol: "DepthAnything3Net.forward",
    moreCode: "https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/model/dualdpt.py",
    moreSymbol: "DualDPT",
    figure: "https://arxiv.org/html/2511.10647v1/figs/pdfs/pipeline.png",
    codeNote: "Follow the backbone features into the depth head. The use_ray_pose switch selects either ray-based camera recovery or the default camera decoder."
  },
  {
    id: "mapanything",
    name: "MapAnything",
    title: "MapAnything: Universal Feed-Forward Metric 3D Reconstruction",
    preprint: "2025-09-16",
    venue: "3DV 2026",
    family: "Geometry priors",
    color: "sand",
    group: "Joint multi-view", compareWith: "da3",
    summary: "Optional depth, poses and calibration help one network reconstruct scenes at an estimated metric scale.",
    stages: [
      ["Encode each view", "DINOv2 encodes RGB. Separate encoders process any available camera rays, poses and depth."],
      ["Combine features", "Sum the features for each view. Mark the first view as reference and add a scene-scale token."],
      ["Exchange across views", "Alternating attention connects patches within each image and across the full set."],
      ["Assemble metric 3D", "A dense head predicts unit rays and ray depths. Pose and scale heads place the points in a common metric frame."]
    ],
    strength: "The same weights can fill sparse metric depth or reconstruct from images alone. In the paper’s ETH3D example, supplying calibration and metric poses improves the estimated scene scale.",
    limit: "Scale from RGB alone can be imprecise on unfamiliar scenes. The model accepts geometric measurements without explicitly modeling their uncertainty.",
    inputs: "One or more RGB views. Each sparse or dense camera-axis depth map needs intrinsics or rays for its view. When supplying poses, include the first view’s pose.",
    exchange: "Image and geometric features are summed per view before alternating attention exchanges information across the set.",
    output: "Unit rays, distances along them, camera poses, confidence and one scene-scale factor define a metric point cloud.",
    scale: "A dedicated token predicts scene scale. Metric depth or camera translations can help anchor the estimate.",
    motion: "Per-view geometry describes visible surfaces. The published representation has no explicit scene-flow output.",
    project: "https://map-anything.github.io/",
    paper: "https://arxiv.org/abs/2509.13414",
    code: "https://github.com/facebookresearch/map-anything/blob/main/mapanything/models/mapanything/model.py",
    symbol: "MapAnything.forward",
    moreCode: "https://github.com/facebookresearch/map-anything/blob/main/mapanything/utils/geometry.py",
    moreSymbol: "depthmap_to_world_frame",
    figure: "https://arxiv.org/html/2509.13414v1/MapAnything_Method.png",
    codeNote: "Follow the optional-input fusion in forward, then the scale token and the dense and pose heads. The geometry utilities show how camera coordinates become world points."
  },
  {
    id: "pi4",
    name: "π⁴",
    title: "π⁴: A Foundation Model for 4D Scene Geometry and Motion",
    preprint: "2026-11-17",
    venue: "Technical report",
    family: "Scene motion",
    color: "sage",
    group: "4D motion", compareWith: "pi3", aliases: ["pi4", "pi^4"],
    summary: "Two looped decoders reconstruct visible surfaces and track moving points through the recovered 3D scene.",
    stages: [
      ["Shared image features", "A DINOv3 encoder extracts patch features from each uncalibrated RGB view."],
      ["Looped geometry", "Sixteen shared-weight iterations fuse views. A sparse 3D U-Net refines local pointmaps alongside pose and scale estimates."],
      ["Conditioned motion", "Geometry features guide eight iterations of a separate motion decoder to estimate 2D tracks, visibility and moving-object masks."],
      ["Lift to 3D", "Sample each target view’s reconstructed surface at the tracked pixel to recover its metric 3D position."]
    ],
    strength: "One pass links reconstructed surfaces to dense 2D and 3D tracks. The geometry branch treats views symmetrically, so it can also reconstruct an unordered photo collection.",
    limit: "A large moving foreground against a weakly textured background can make object motion look like camera motion. The authors also report less consistent camera trajectories on very long sequences.",
    inputs: "An uncalibrated image, unordered collection or video. Tracking uses a query frame and at least two views.",
    exchange: "Each decoder reuses within-view and cross-view attention blocks. The geometry decoder’s features condition the motion decoder.",
    output: "Local pointmaps, poses, scale and confidence, plus query-conditioned tracks, visibility and moving-object masks for multiple images.",
    scale: "Multiply transformed local points by the predicted scene scale. The released 3D tracks already include that scale.",
    motion: "Dense image-space correspondences become 3D trajectories by sampling the reconstructed surface in each target view.",
    project: "https://yyfz.github.io/pi4/",
    paper: "https://arxiv.org/abs/2611.23144",
    code: "https://github.com/yyfz/Pi4/blob/main/pi4/models/pi4.py",
    symbol: "Pi4.forward",
    moreCode: "https://github.com/yyfz/Pi4/blob/main/pi4/models/decoder.py",
    moreSymbol: "MotionDecoder",
    figure: "https://arxiv.org/html/2611.23144v1/pipeline.png",
    codeNote: "In forward, query_frame enables the motion decoder. Follow lift_tracks to see how it samples target-view geometry and applies the predicted camera pose and metric scale."
  }
];

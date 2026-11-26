window.AtlasWalkData = {
  "models": {
    "dvlt": {
      "id": "dvlt",
      "repo": "nv-tlabs/dvlt",
      "commit": "d9f1c0c481eea8b6a399b507b30c9f484e99fd69",
      "licenseFile": "LICENSE",
      "inputs": [
        "batch[DataField.IMAGES] · [B,S,3,H,W] — RGB in [0,1] for B scenes and S views per scene. Use image sizes compatible with the 14-pixel patch grid.",
        "inference_steps · scalar setting — K controls how many refinement steps run for the same images and weights."
      ],
      "outputs": [
        "depth / depth_conf · [B,S,H,W,1] / [B,S,H,W] — positive depth and its confidence in the inner network’s output.",
        "rays · [B,S,H,W,6] — each pixel carries a direction in the first three channels and an origin in the last three, both in scene coordinates.",
        "world_points · [B,S,H,W,3] — the inner network returns origin + direction × depth in the relative-scale scene frame. Its world_points_conf reuses depth confidence.",
        "cameras / world_points_direct / world_points · per-scene cameras / [B,S,H,W,3] / [B,S,H,W,3] — DVLT.predict fits cameras from rays and exposes both direct and camera-based reconstructions."
      ],
      "steps": [
        {
          "title": "Encode RGB and prepend camera and register tokens",
          "note": "DINOv2 processes each view independently. The network flattens batch and view dimensions before encoding, prepends special tokens, then passes the assembled sequence through the solver and decoder.",
          "blocks": [
            {
              "file": "src/dvlt/model/dvlt/model.py",
              "start": 374,
              "end": 380,
              "repo": "nv-tlabs/dvlt",
              "commit": "d9f1c0c481eea8b6a399b507b30c9f484e99fd69",
              "text": "    def _normalize_images(self, images: Tensor) -> Tensor:\n        return (images - self._resnet_mean) / self._resnet_std\n\n    def _encode_images(self, images: Tensor) -> Tensor:\n        B, S, C, H, W = images.shape\n        images = self._normalize_images(images).view(B * S, C, H, W)\n        return self.patch_embed_encoder(images, is_training=True)[\"x_norm_patchtokens\"]",
              "symbol": "DVLTModel"
            },
            {
              "file": "src/dvlt/model/dvlt/model.py",
              "start": 621,
              "end": 629,
              "repo": "nv-tlabs/dvlt",
              "commit": "d9f1c0c481eea8b6a399b507b30c9f484e99fd69",
              "text": "    def forward_inference(self, images):\n        B, S, _, H, W = images.shape\n        z_0 = self._encode_images(images)\n        register_token = self.register_token.expand(B, S, -1, -1).reshape(B * S, self.num_register_tokens, -1)\n        camera_token = _slice_expand_flatten(self.camera_token, B, S)\n        x = torch.cat([camera_token, register_token, z_0], dim=1)\n        rope_pos = self._get_rope_positions(B * S, H, W, images.device)\n        features = self._solve_inference(x, rope_pos, B, S)\n        return self._decode(features, H, W, B, S, rope_pos)",
              "symbol": "DVLTModel.forward_inference"
            }
          ]
        },
        {
          "title": "Run K refinement steps with one set of weights",
          "note": "The inference loop gives each step a position on the time grid. _interval_step always selects recurrent_blocks[0], conditioning that same block on the current time or time interval.",
          "blocks": [
            {
              "file": "src/dvlt/model/dvlt/model.py",
              "start": 542,
              "end": 550,
              "repo": "nv-tlabs/dvlt",
              "commit": "d9f1c0c481eea8b6a399b507b30c9f484e99fd69",
              "text": "    def _solve_inference_linspace_k(self, x, rope_pos, B, S):\n        \"\"\"Inference for k_sampling='linspace': uniform grid with configurable K.\"\"\"\n        K = self.inference_steps\n        ts = torch.linspace(0.0, 1.0, K).tolist()\n        for i in range(K):\n            t_now = ts[i]\n            t_next = ts[i + 1] if i + 1 \u003c K else 1.0\n            x = self._interval_step(x, t_now, t_next, rope_pos, B, S)\n        return x",
              "symbol": "DVLTModel._solve_inference_linspace_k"
            },
            {
              "file": "src/dvlt/model/dvlt/model.py",
              "start": 413,
              "end": 421,
              "repo": "nv-tlabs/dvlt",
              "commit": "d9f1c0c481eea8b6a399b507b30c9f484e99fd69",
              "text": "        block = self.recurrent_blocks[0]\n        if self.time_conditioning == \"continuous\":\n            t_batch = torch.full((B,), t_now, device=x.device, dtype=torch.float32)\n            t_frame = t_batch.unsqueeze(1).expand(-1, S).reshape(B * S)\n            x = block(x, t_frame, t_batch, rope_pos, B, S)\n        else:\n            t_pair = torch.tensor([[t_now, t_next]], device=x.device, dtype=torch.float32)\n            x = block(x, t_pair.expand(B * S, -1), t_pair.expand(B, -1), rope_pos, B, S)\n        return x",
              "symbol": "DVLTModel._interval_step"
            }
          ]
        },
        {
          "title": "Change what each token can attend to",
          "note": "P counts all tokens per view, including the special tokens. Frame attention sees P tokens at once. Reshaping to S × P lets the global block exchange information between all views before the next refinement.",
          "blocks": [
            {
              "file": "src/dvlt/model/dvlt/blocks.py",
              "start": 128,
              "end": 138,
              "repo": "nv-tlabs/dvlt",
              "commit": "d9f1c0c481eea8b6a399b507b30c9f484e99fd69",
              "text": "        self.frame_attn = DepthScaledAttentionBlock(**kw, rope=rope)\n        self.global_attn = DepthScaledAttentionBlock(**kw, rope=None)\n\n    def forward(self, x, k_frame, k_batch, rope_pos, B, S):\n        \"\"\"k_frame, k_batch: int/float tensors or None depending on mode.\"\"\"\n        _, P, C = x.shape\n        x = x.reshape(B * S, P, C)\n        x = self.frame_attn(x, k_frame, pos=rope_pos.reshape(B * S, P, 2))\n        x = x.reshape(B, S * P, C)\n        x = self.global_attn(x, k_batch, pos=None)\n        return x.reshape(B * S, P, C)",
              "symbol": "LoopedAABlock"
            }
          ]
        },
        {
          "title": "Decode a six-channel ray field and a two-channel depth field",
          "note": "Both heads apply transformer layers and discard the special tokens for dense prediction. The released checkpoint uses a convolutional upsampler for depth and a linear projection plus pixel shuffle for rays. A depth output channel also supplies confidence.",
          "blocks": [
            {
              "file": "src/dvlt/model/dvlt/model.py",
              "start": 328,
              "end": 338,
              "repo": "nv-tlabs/dvlt",
              "commit": "d9f1c0c481eea8b6a399b507b30c9f484e99fd69",
              "text": "        self.ray_decoder = DecoderHead(\n            out_dim=6, head_type=decoder_head_type, init_values=decoder_init_values, **_decoder_kw\n        )\n        _depth_decoder_kw = {\n            **_decoder_kw,\n            \"embed_dim\": depth_decoder_embed_dim if depth_decoder_embed_dim is not None else decoder_embed_dim,\n            \"depth\": depth_decoder_depth if depth_decoder_depth is not None else decoder_depth,\n            \"num_heads\": depth_decoder_num_heads if depth_decoder_num_heads is not None else decoder_num_heads,\n            \"init_values\": decoder_init_values,\n        }\n        self.depth_decoder = DecoderHead(out_dim=2, head_type=depth_head_type or decoder_head_type, **_depth_decoder_kw)",
              "symbol": "DVLTModel.__init__"
            },
            {
              "file": "src/dvlt/model/dvlt/heads.py",
              "start": 226,
              "end": 235,
              "repo": "nv-tlabs/dvlt",
              "commit": "d9f1c0c481eea8b6a399b507b30c9f484e99fd69",
              "text": "    def forward(self, x, H, W, patch_start_idx=0, pos=None):\n        B = x.shape[0]\n        ph, pw = H // self.patch_size, W // self.patch_size\n\n        x = self.proj_in(x)\n        for blk in self.blocks:\n            x = blk(x, pos=pos)\n        features = x\n        x = self.norm(x)[:, patch_start_idx:]\n",
              "symbol": "DecoderHead.forward"
            },
            {
              "file": "src/dvlt/model/dvlt/heads.py",
              "start": 236,
              "end": 256,
              "repo": "nv-tlabs/dvlt",
              "commit": "d9f1c0c481eea8b6a399b507b30c9f484e99fd69",
              "text": "        if self.head_type == \"conv\":\n            aspect_ratio = W / H\n            x = x.permute(0, 2, 1).reshape(B, -1, ph, pw)\n            for block in self.upsample_blocks:\n                x = _concat_uv(x, aspect_ratio)\n                for layer in block:\n                    if self._checkpoint_conv and self.training:\n                        x = grad_checkpoint(layer, x, use_reentrant=self._use_reentrant)\n                    else:\n                        x = layer(x)\n            x = F.interpolate(x, (H, W), mode=\"bilinear\", align_corners=False)\n            x = _concat_uv(x, aspect_ratio)\n            with torch.amp.autocast(\"cuda\", enabled=False):\n                x = self.output_block(x.float())\n        else:\n            with torch.amp.autocast(\"cuda\", enabled=False):\n                x = self.head(x.float())\n                x = x.transpose(-1, -2).view(B, -1, ph, pw)\n                x = F.pixel_shuffle(x, self.patch_size)\n\n        return x, features",
              "symbol": "DecoderHead.forward"
            }
          ]
        },
        {
          "title": "Compose the direct pointmap at every pixel",
          "note": "The two decoders read the same refined features. The network reshapes six ray channels and one positive depth channel to image resolution, then adds the ray origin to the scaled direction. Depth confidence becomes the point confidence.",
          "blocks": [
            {
              "file": "src/dvlt/model/dvlt/model.py",
              "start": 571,
              "end": 581,
              "repo": "nv-tlabs/dvlt",
              "commit": "d9f1c0c481eea8b6a399b507b30c9f484e99fd69",
              "text": "        for start in range(0, BS, chunk):\n            end = min(start + chunk, BS)\n            f = features[start:end]\n            p = rope_pos[start:end]\n            _dec_kw = dict(H=H, W=W, patch_start_idx=self.patch_start_idx, pos=p)\n            ray_out_c, ray_features_c = self.ray_decoder(f, **_dec_kw)\n            depth_out_c, _ = self.depth_decoder(f, **_dec_kw)\n            ray_chunks.append(ray_out_c)\n            depth_chunks.append(depth_out_c)\n            if need_cam_tokens:\n                cam_token_chunks.append(ray_features_c[:, 0])",
              "symbol": "DVLTModel._decode"
            },
            {
              "file": "src/dvlt/model/dvlt/model.py",
              "start": 586,
              "end": 602,
              "repo": "nv-tlabs/dvlt",
              "commit": "d9f1c0c481eea8b6a399b507b30c9f484e99fd69",
              "text": "        pred_rays, _ = activate_head(ray_out, activation=\"identity\", conf_activation=None)\n        pred_rays = pred_rays.view(B, S, H, W, 6)\n\n        pred_depth, depth_conf = activate_head(depth_out, activation=\"exp_clamped\", conf_activation=\"exp_plus_one\")\n        pred_depth = pred_depth.view(B, S, H, W, 1)\n        depth_conf = depth_conf.view(B, S, H, W)\n\n        # World points derived from depth + ray (origin + direction * depth).\n        pred_points = pred_rays[..., 3:6] + pred_rays[..., :3] * pred_depth\n\n        predictions = {\n            \"world_points\": pred_points,\n            \"world_points_conf\": depth_conf,\n            \"depth\": pred_depth,\n            \"depth_conf\": depth_conf,\n            \"rays\": pred_rays,\n        }",
              "symbol": "DVLTModel._decode"
            }
          ]
        },
        {
          "title": "Trace the second pointmap returned by the public wrapper",
          "note": "DVLT.predict fits a camera to each ray map. By default its world_points unprojects depth through those fitted cameras. It preserves the inner model’s ray-composed XYZ as world_points_direct, and a configuration switch can select it for world_points as well.",
          "blocks": [
            {
              "file": "src/dvlt/model/dvlt/model.py",
              "start": 940,
              "end": 951,
              "repo": "nv-tlabs/dvlt",
              "commit": "d9f1c0c481eea8b6a399b507b30c9f484e99fd69",
              "text": "    def _postprocess_predictions(self, batch, predictions):\n        H, W = batch[DataField.IMAGES].shape[-2:]\n        rays = predictions[\"rays\"]\n        depth_conf = predictions[\"depth_conf\"]\n        pose_conf = depth_conf if self.use_depth_conf_for_pose else torch.ones_like(depth_conf)\n        extrinsics_c2w, intrinsics = rays_to_pose(rays, pose_conf, H, W, self.patch_size)\n        cameras = [extri_intri_to_cameras(e, i, (H, W)) for e, i in zip(extrinsics_c2w, intrinsics, strict=False)]\n        preds = {\n            PredictionField.CAMERAS: cameras,\n            PredictionField.DEPTHS: predictions[\"depth\"].squeeze(-1),\n            PredictionField.DEPTHS_CONF: depth_conf,\n        }",
              "symbol": "DVLT._postprocess_predictions"
            },
            {
              "file": "src/dvlt/model/dvlt/model.py",
              "start": 958,
              "end": 971,
              "repo": "nv-tlabs/dvlt",
              "commit": "d9f1c0c481eea8b6a399b507b30c9f484e99fd69",
              "text": "        rays_world_points = predictions[\"world_points\"]\n        if self.world_points_from_rays:\n            preds[PredictionField.WORLD_POINTS] = rays_world_points\n        else:\n            unproj_world_points, _, _ = depth_to_world_coords_points(\n                preds[PredictionField.DEPTHS], extrinsics_c2w, intrinsics\n            )\n            preds[PredictionField.WORLD_POINTS] = unproj_world_points\n        preds[PredictionField.WORLD_POINTS_DIRECT] = rays_world_points\n        preds[PredictionField.WORLD_POINTS_DIRECT_CONF] = predictions[\"world_points_conf\"]\n        return preds\n\n    def predict(self, batch, accelerator):\n        return self._postprocess_predictions(batch, self.model(batch))",
              "symbol": "DVLT"
            }
          ]
        }
      ]
    },
    "omega": {
      "id": "omega",
      "repo": "facebookresearch/vggt-omega",
      "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
      "licenseFile": "LICENSE",
      "inputs": [
        "images · [S,3,H,W] or [B,S,3,H,W] — S RGB views per scene, in [0,1]. The supplied loader resizes to multiples of 16 and pads views to a common shape."
      ],
      "outputs": [
        "depth / depth_conf · [B,S,H,W,1] / [B,S,H,W] — camera-axis depth in the learned normalized scene scale and positive confidence.",
        "pose_enc · [B,S,9] — camera translation, quaternion and two fields of view. The camera utility decodes world-to-camera extrinsics and intrinsics.",
        "camera_and_register_tokens · [B,S,17,2048] — final camera and register features for the default 1024-wide model, after combining frame and inter-frame features.",
        "world_points_from_depth · [S,H,W,3] — the demo’s derived pointmap for one scene, computed after the network returns depth and pose."
      ],
      "steps": [
        {
          "title": "Normalize the images and form per-view token sequences",
          "note": "VGGTOmega adds a batch axis to a single image set. Its aggregator normalizes each RGB channel, runs the DINOv3 patch encoder, and prepends one camera token and 16 registers to each view’s patch features.",
          "blocks": [
            {
              "file": "vggt_omega/models/vggt_omega.py",
              "start": 32,
              "end": 41,
              "repo": "facebookresearch/vggt-omega",
              "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "text": "        self.autocast = autocast\n        self.aggregator = Aggregator(patch_size=patch_size, embed_dim=embed_dim, use_checkpoint=use_checkpoint)\n        _warn_if_rope_not_max(self.aggregator)\n        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None\n        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None\n        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None\n\n    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:\n        if len(images.shape) == 4:\n            images = images.unsqueeze(0)",
              "symbol": "VGGTOmega"
            },
            {
              "file": "vggt_omega/models/aggregator.py",
              "start": 118,
              "end": 133,
              "repo": "facebookresearch/vggt-omega",
              "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "text": "        batch_size, num_frames, num_channels, height, width = images.shape\n        if num_channels != 3:\n            raise ValueError(f\"Expected 3 input channels, got {num_channels}\")\n\n        images = (images - self._resnet_mean) / self._resnet_std\n        images = images.view(batch_size * num_frames, num_channels, height, width)\n\n        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)\n        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)\n\n        patch_tokens = self.patch_embed(images)\n        if isinstance(patch_tokens, dict):\n            patch_tokens = patch_tokens[\"x_norm_patchtokens\"]\n\n        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)\n        _, num_tokens, embed_dim = tokens.shape",
              "symbol": "Aggregator.forward"
            }
          ]
        },
        {
          "title": "Select the cross-view exchange at each layer",
          "note": "The default stack has 24 frame/inter-frame pairs. Five zero-based layer indices enable register attention, while other inter-frame layers use all tokens. The aggregator saves four intermediate layers for the output heads.",
          "blocks": [
            {
              "file": "vggt_omega/models/aggregator.py",
              "start": 23,
              "end": 33,
              "repo": "facebookresearch/vggt-omega",
              "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "text": "    def __init__(\n        self,\n        patch_size: int = 16,\n        embed_dim: int = 1024,\n        depth: int = 24,\n        num_heads: int = 16,\n        mlp_ratio: float = 4.0,\n        num_register_tokens: int = 16,\n        register_attention_block_indices: list[int] = [2, 6, 9, 14, 20],\n        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),\n        use_checkpoint: bool = False,",
              "symbol": "Aggregator.__init__"
            },
            {
              "file": "vggt_omega/models/aggregator.py",
              "start": 87,
              "end": 95,
              "repo": "facebookresearch/vggt-omega",
              "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "text": "        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))\n        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))\n        self.patch_token_start = 1 + num_register_tokens\n\n        self.inter_frame_attention_types = [\"global\"] * depth\n        for idx in register_attention_block_indices:\n            if idx \u003c 0 or idx >= depth:\n                raise ValueError(f\"register_attention_block_indices contains invalid block index {idx}\")\n            self.inter_frame_attention_types[idx] = \"register\"",
              "symbol": "Aggregator.__init__"
            },
            {
              "file": "vggt_omega/models/aggregator.py",
              "start": 154,
              "end": 168,
              "repo": "facebookresearch/vggt-omega",
              "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "text": "            tokens = self._run_inter_frame_attention_block(\n                tokens,\n                batch_size,\n                num_frames,\n                num_tokens,\n                embed_dim,\n                block_idx,\n                self.inter_frame_attention_types[block_idx],\n            )\n            if block_idx in self.cached_layer_indices:\n                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))\n            else:\n                outputs.append(None)\n\n        return outputs, self.patch_token_start",
              "symbol": "Aggregator.forward"
            }
          ]
        },
        {
          "title": "Follow the compact register exchange",
          "note": "The global path flattens the entire view set for self-attention. In the register path the layer extracts only the camera/register prefix, applies the inter-frame block to those tokens, and rejoins the patch tokens.",
          "blocks": [
            {
              "file": "vggt_omega/models/aggregator.py",
              "start": 194,
              "end": 199,
              "repo": "facebookresearch/vggt-omega",
              "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "text": "        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)\n\n        if attention_type == \"global\":\n            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)\n            tokens = self._run_block(self.inter_frame_blocks[block_idx], tokens, None)\n            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)",
              "symbol": "Aggregator._run_inter_frame_attention_block"
            },
            {
              "file": "vggt_omega/models/aggregator.py",
              "start": 204,
              "end": 219,
              "repo": "facebookresearch/vggt-omega",
              "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "text": "        patch_token_start = self.patch_token_start\n        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(\n            batch_size,\n            num_frames * patch_token_start,\n            embed_dim,\n        )\n        patch_tokens = tokens[:, :, patch_token_start:].reshape(\n            batch_size,\n            num_frames * (num_tokens - patch_token_start),\n            embed_dim,\n        )\n\n        camera_and_register_tokens = self._run_block(\n            self.inter_frame_blocks[block_idx], camera_and_register_tokens, None\n        )\n        tokens = torch.cat([camera_and_register_tokens, patch_tokens], dim=1)",
              "symbol": "Aggregator._run_inter_frame_attention_block"
            }
          ]
        },
        {
          "title": "Restore a dense depth image",
          "note": "The dense head reshapes patch tokens into spatial grids, projects and resizes features from the four cached stages, and fuses them. Separate pixel-shuffle branches restore the depth and confidence images. Their exponential activations make the two returned fields positive.",
          "blocks": [
            {
              "file": "vggt_omega/models/heads/dense_head.py",
              "start": 130,
              "end": 139,
              "repo": "facebookresearch/vggt-omega",
              "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "text": "            x = x.reshape(batch_size * num_frames, -1, x.shape[-1])\n            x = self.norm(x)\n            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))\n            x = self.projects[feature_idx](x)\n            x = self._apply_pos_embed(x, width, height)\n            x = self.resize_layers[feature_idx](x)\n            multi_scale_features.append(x)\n\n        fused = self.scratch_forward(multi_scale_features)\n        fused = self._apply_pos_embed(fused, width, height)",
              "symbol": "DenseHead._forward_impl"
            },
            {
              "file": "vggt_omega/models/heads/dense_head.py",
              "start": 141,
              "end": 153,
              "repo": "facebookresearch/vggt-omega",
              "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "text": "        depth_logits = self.proj(fused)\n        depth_logits = F.pixel_shuffle(depth_logits, self.final_shuffle_factor)\n        depth_logits = depth_logits.permute(0, 2, 3, 1)\n\n        confidence_logits = self.proj_conf(fused)\n        confidence_logits = F.pixel_shuffle(confidence_logits, self.final_shuffle_factor)\n        confidence_logits = confidence_logits.permute(0, 2, 3, 1).squeeze(-1)\n\n        depth = torch.exp(depth_logits)\n        depth_conf = 1.0 + torch.exp(confidence_logits)\n\n        depth = depth.view(batch_size, num_frames, *depth.shape[1:])\n        depth_conf = depth_conf.view(batch_size, num_frames, *depth_conf.shape[1:])",
              "symbol": "DenseHead._forward_impl"
            }
          ]
        },
        {
          "title": "Read camera tokens and decode a camera matrix",
          "note": "The camera head mixes special tokens across views and reads token zero of each view. Its nine values encode translation, quaternion and two fields of view. The camera utility constructs a world-to-camera transform and focal lengths in pixels.",
          "blocks": [
            {
              "file": "vggt_omega/models/heads/camera_head.py",
              "start": 63,
              "end": 73,
              "repo": "facebookresearch/vggt-omega",
              "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "text": "        camera_and_register_tokens = tokens[:, :, :patch_token_start]\n        camera_and_register_tokens = self.token_norm(camera_and_register_tokens)\n\n        camera_and_register_tokens = camera_and_register_tokens.reshape(batch_size, num_frames * patch_token_start, -1)\n        rope_sincos = None\n        for block in self.trunk:\n            camera_and_register_tokens = block(camera_and_register_tokens, rope_sincos)\n\n        camera_and_register_tokens = camera_and_register_tokens.reshape(batch_size, num_frames, patch_token_start, -1)\n        camera_tokens = self.trunk_norm(camera_and_register_tokens[:, :, 0])\n        return _apply_camera_activation(self.camera_branch(camera_tokens))",
              "symbol": "CameraHead.forward"
            },
            {
              "file": "vggt_omega/models/heads/camera_head.py",
              "start": 76,
              "end": 80,
              "repo": "facebookresearch/vggt-omega",
              "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "text": "def _apply_camera_activation(raw_camera: torch.Tensor) -> torch.Tensor:\n    translation = raw_camera[..., :3]\n    quaternion = raw_camera[..., 3:7]\n    fov = F.relu(raw_camera[..., 7:]) + 0.01\n    return torch.cat([translation, quaternion, fov], dim=-1)",
              "symbol": "_apply_camera_activation"
            },
            {
              "file": "vggt_omega/utils/pose_enc.py",
              "start": 29,
              "end": 43,
              "repo": "facebookresearch/vggt-omega",
              "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "text": "def encoding_to_camera(pose_encoding, image_size_hw, build_intrinsics=True):\n    \"\"\"Decode VGGT-Omega pose encoding into extrinsics and intrinsics.\"\"\"\n    T = pose_encoding[..., :3]\n    quat = pose_encoding[..., 3:7]\n    fov_h = pose_encoding[..., 7]\n    fov_w = pose_encoding[..., 8]\n\n    R = quat_to_mat(quat)\n    extrinsics = torch.cat([R, T[..., None]], dim=-1)\n\n    intrinsics = None\n    if build_intrinsics:\n        H, W = image_size_hw\n        fy = (H / 2.0) / torch.tan(fov_h / 2.0)\n        fx = (W / 2.0) / torch.tan(fov_w / 2.0)",
              "symbol": "encoding_to_camera"
            }
          ]
        },
        {
          "title": "Unproject depth to get a pointmap in the demo",
          "note": "The network emits depth, confidence and pose encoding. After converting the pose, the demo computes camera XYZ from pixel coordinates and depth, then applies the inverse world-to-camera transform to obtain scene XYZ.",
          "blocks": [
            {
              "file": "vggt_omega/models/vggt_omega.py",
              "start": 59,
              "end": 72,
              "repo": "facebookresearch/vggt-omega",
              "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "text": "            if self.camera_head is not None:\n                predictions[\"pose_enc\"] = self.camera_head(\n                    aggregated_tokens_list,\n                    patch_token_start=patch_token_start,\n                )\n\n            if self.dense_head is not None:\n                depth, depth_conf = self.dense_head(\n                    aggregated_tokens_list,\n                    images=images,\n                    patch_token_start=patch_token_start,\n                )\n                predictions[\"depth\"] = depth\n                predictions[\"depth_conf\"] = depth_conf",
              "symbol": "VGGTOmega.forward"
            },
            {
              "file": "demo_gradio.py",
              "start": 83,
              "end": 95,
              "repo": "facebookresearch/vggt-omega",
              "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "text": "    fx = intrinsic[:, 0, 0][:, None, None]\n    fy = intrinsic[:, 1, 1][:, None, None]\n    cx = intrinsic[:, 0, 2][:, None, None]\n    cy = intrinsic[:, 1, 2][:, None, None]\n\n    camera_points = np.stack(\n        [\n            (x - cx) / fx * depth,\n            (y - cy) / fy * depth,\n            depth,\n        ],\n        axis=-1,\n    )",
              "symbol": "unproject_depth_map_to_point_map"
            },
            {
              "file": "demo_gradio.py",
              "start": 97,
              "end": 103,
              "repo": "facebookresearch/vggt-omega",
              "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "text": "    rotation = extrinsic[:, :3, :3]\n    translation = extrinsic[:, :3, 3]\n    return np.einsum(\n        \"sij,shwj->shwi\",\n        np.transpose(rotation, (0, 2, 1)),\n        camera_points - translation[:, None, None, :],\n    )",
              "symbol": "unproject_depth_map_to_point_map"
            }
          ]
        }
      ]
    },
    "pi3": {
      "id": "pi3",
      "repo": "yyfz/Pi3",
      "creditedRepos": [
        "naver/dust3r"
      ],
      "commit": "a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99",
      "licenseFile": "LICENSE",
      "inputs": [
        "imgs · [B,N,3,H,W] — RGB in [0,1] for B scenes with N views. The 14-pixel patch grid requires suitable image dimensions. The network applies ImageNet normalization itself."
      ],
      "outputs": [
        "local_points · [B,N,H,W,3] — an XYZ pointmap in each view’s own camera frame, at relative scene scale.",
        "camera_poses · [B,N,4,4] — camera-to-world rigid transforms for each view in the common scene frame.",
        "points · [B,N,H,W,3] — camera-local points transformed into the shared reference-free scene frame.",
        "conf · [B,N,H,W,1] — raw per-pixel confidence logits. Apply sigmoid to obtain scores in [0,1]."
      ],
      "steps": [
        {
          "title": "Turn the RGB views into patch embeddings",
          "note": "Original π³ builds a DINOv2 ViT-L/14 encoder. Its forward pass normalizes each channel, folds the view axis into the batch, and retains the normalized patch tokens for each image.",
          "blocks": [
            {
              "file": "pi3/models/pi3.py",
              "start": 24,
              "end": 29,
              "repo": "yyfz/Pi3",
              "commit": "a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99",
              "text": "        # ----------------------\n        #        Encoder\n        # ----------------------\n        self.encoder = dinov2_vitl14_reg(pretrained=False)\n        self.patch_size = 14\n        del self.encoder.mask_token",
              "symbol": "Pi3.__init__"
            },
            {
              "file": "pi3/models/pi3.py",
              "start": 173,
              "end": 186,
              "repo": "yyfz/Pi3",
              "commit": "a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99",
              "text": "    def forward(self, imgs):\n        imgs = (imgs - self.image_mean) / self.image_std\n\n        B, N, _, H, W = imgs.shape\n        patch_h, patch_w = H // 14, W // 14\n        \n        # encode by dinov2\n        imgs = imgs.reshape(B*N, _, H, W)\n        hidden = self.encoder(imgs, is_training=True)\n\n        if isinstance(hidden, dict):\n            hidden = hidden[\"x_norm_patchtokens\"]\n\n        hidden, pos = self.decode(hidden, N, H, W)",
              "symbol": "Pi3.forward"
            }
          ]
        },
        {
          "title": "Alternate per-view and all-view attention",
          "note": "The same five learned registers are repeated for every view. In the default large decoder, 36 separate blocks alternate between each view’s token sequence and a sequence spanning all N views. The last two layer outputs are concatenated along the feature axis.",
          "blocks": [
            {
              "file": "pi3/models/pi3.py",
              "start": 59,
              "end": 65,
              "repo": "yyfz/Pi3",
              "commit": "a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99",
              "text": "        elif decoder_size == 'large':\n            dec_embed_dim = 1024\n            dec_num_heads = 16\n            mlp_ratio = 4\n            dec_depth = 36\n        else:\n            raise NotImplementedError",
              "symbol": "Pi3.__init__"
            },
            {
              "file": "pi3/models/pi3.py",
              "start": 138,
              "end": 144,
              "repo": "yyfz/Pi3",
              "commit": "a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99",
              "text": "        hidden = hidden.reshape(B*N, hw, -1)\n\n        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])\n\n        # Concatenate special tokens with patch tokens\n        hidden = torch.cat([register_token, hidden], dim=1)\n        hw = hidden.shape[1]",
              "symbol": "Pi3.decode"
            },
            {
              "file": "pi3/models/pi3.py",
              "start": 156,
              "end": 171,
              "repo": "yyfz/Pi3",
              "commit": "a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99",
              "text": "        for i in range(len(self.decoder)):\n            blk = self.decoder[i]\n\n            if i % 2 == 0:\n                pos = pos.reshape(B*N, hw, -1)\n                hidden = hidden.reshape(B*N, hw, -1)\n            else:\n                pos = pos.reshape(B, N*hw, -1)\n                hidden = hidden.reshape(B, N*hw, -1)\n\n            hidden = blk(hidden, xpos=pos)\n\n            if i+1 in [len(self.decoder)-1, len(self.decoder)]:\n                final_output.append(hidden.reshape(B*N, hw, -1))\n\n        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)",
              "symbol": "Pi3.decode"
            }
          ]
        },
        {
          "title": "Branch into point, confidence and camera decoders",
          "note": "The point and confidence branches start with matching transformer architectures and have separate weights. Linear heads read three channels for geometry and one for confidence. A third transformer reduces its feature width to 512 for pose prediction.",
          "blocks": [
            {
              "file": "pi3/models/pi3.py",
              "start": 96,
              "end": 109,
              "repo": "yyfz/Pi3",
              "commit": "a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99",
              "text": "        self.point_decoder = TransformerDecoder(\n            in_dim=2*self.dec_embed_dim, \n            dec_embed_dim=1024,\n            dec_num_heads=16,\n            out_dim=1024,\n            rope=self.rope,\n        )\n        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)\n\n        # ----------------------\n        #     Conf Decoder\n        # ----------------------\n        self.conf_decoder = deepcopy(self.point_decoder)\n        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)",
              "symbol": "Pi3.__init__"
            },
            {
              "file": "pi3/models/pi3.py",
              "start": 114,
              "end": 122,
              "repo": "yyfz/Pi3",
              "commit": "a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99",
              "text": "        self.camera_decoder = TransformerDecoder(\n            in_dim=2*self.dec_embed_dim, \n            dec_embed_dim=1024,\n            dec_num_heads=16,                # 8\n            out_dim=512,\n            rope=self.rope,\n            use_checkpoint=False\n        )\n        self.camera_head = CameraHead(dim=512)",
              "symbol": "Pi3.__init__"
            },
            {
              "file": "pi3/models/pi3.py",
              "start": 186,
              "end": 190,
              "repo": "yyfz/Pi3",
              "commit": "a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99",
              "text": "        hidden, pos = self.decode(hidden, N, H, W)\n\n        point_hidden = self.point_decoder(hidden, xpos=pos)\n        conf_hidden = self.conf_decoder(hidden, xpos=pos)\n        camera_hidden = self.camera_decoder(hidden, xpos=pos)",
              "symbol": "Pi3.forward"
            }
          ]
        },
        {
          "title": "Map the patch grid to camera-local XYZ",
          "note": "LinearPts3d emits patch_size² values per output channel, then pixel shuffle moves them onto the H × W image grid. The point branch interprets its channels as two image-plane coordinates and log depth. Multiplication by positive depth produces local XYZ.",
          "credit": "The LinearPts3d excerpt adapts DUSt3R’s linear head, © 2024–present NAVER Corporation. π³ changes the projection configuration and output processing. The adapted excerpt is shared under CC BY-NC-SA 4.0.",
          "references": [
            {
              "label": "Original DUSt3R linear head",
              "url": "https://github.com/naver/dust3r/blob/4c24a6ebf04809f2cfe59915e51779c8984aaa40/dust3r/heads/linear_head.py"
            },
            {
              "label": "CC BY-NC-SA 4.0",
              "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"
            }
          ],
          "blocks": [
            {
              "file": "pi3/models/layers/transformer_head.py",
              "start": 64,
              "end": 81,
              "repo": "yyfz/Pi3",
              "commit": "a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99",
              "text": "    def __init__(self, patch_size, dec_embed_dim, output_dim=3,):\n        super().__init__()\n        self.patch_size = patch_size\n\n        self.proj = nn.Linear(dec_embed_dim, (output_dim)*self.patch_size**2)\n\n    def forward(self, decout, img_shape):\n        H, W = img_shape\n        tokens = decout[-1]\n        B, S, D = tokens.shape\n\n        # extract 3D points\n        feat = self.proj(tokens)  # B,S,D\n        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)\n        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W\n\n        # permute + norm depth\n        return feat.permute(0, 2, 3, 1)",
              "symbol": "LinearPts3d"
            },
            {
              "file": "pi3/models/pi3.py",
              "start": 192,
              "end": 202,
              "repo": "yyfz/Pi3",
              "commit": "a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99",
              "text": "        with torch.amp.autocast(device_type='cuda', enabled=False):\n            # local points\n            point_hidden = point_hidden.float()\n            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)\n            xy, z = ret.split([2, 1], dim=-1)\n            z = torch.exp(z)\n            local_points = torch.cat([xy * z, z], dim=-1)\n\n            # confidence\n            conf_hidden = conf_hidden.float()\n            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)",
              "symbol": "Pi3.forward"
            }
          ]
        },
        {
          "title": "Predict camera-to-world transforms and return both pointmaps",
          "note": "The camera head pools the patch grid, regresses translation and rotation, and orthogonalizes the rotation before assembling a 4 × 4 transform. Multiplying homogeneous local XYZ by this matrix gives points in one shared scene frame.",
          "references": [
            {
              "label": "Camera head implementation in π³",
              "url": "https://github.com/yyfz/Pi3/blob/a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99/pi3/models/layers/camera_head.py#L55-L72"
            }
          ],
          "blocks": [
            {
              "file": "pi3/models/pi3.py",
              "start": 204,
              "end": 216,
              "repo": "yyfz/Pi3",
              "commit": "a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99",
              "text": "            # camera\n            camera_hidden = camera_hidden.float()\n            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)\n\n            # unproject local points using camera poses\n            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]\n\n        return dict(\n            points=points,\n            local_points=local_points,\n            conf=conf,\n            camera_poses=camera_poses,\n        )",
              "symbol": "Pi3.forward"
            }
          ]
        }
      ]
    },
    "dust3r": {
      "id": "dust3r",
      "repo": "naver/dust3r",
      "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
      "licenseFile": "LICENSE",
      "inputs": [
        "view1['img'], view2['img'] · each [B, 3, Hᵢ, Wᵢ] — one processed RGB tensor per camera. The official loader normalizes channels to [−1, 1] and crops to the model’s patch grid.",
        "true_shape · [B, 2] per view — processed height and width. View dictionaries from the loader also include instance identifiers used to detect symmetric pairs."
      ],
      "outputs": [
        "res1['pts3d'] · [B, H₁, W₁, 3] — camera-one XYZ for pixels in image one, at relative scene scale.",
        "res2['pts3d_in_other_view'] · [B, H₂, W₂, 3] — camera-one XYZ for pixels in image two.",
        "res1['conf'], res2['conf'] · [B, Hᵢ, Wᵢ] — learned scores for each point prediction. Camera parameters and a larger aligned scene are recovered downstream."
      ],
      "steps": [
        {
          "title": "Prepare a pair of RGB views",
          "note": "The example here follows the released 512-pixel DPT checkpoint (ViT-L encoder, ViT-B decoders). Each image has its own pixel grid. The loader records the processed size, then the model reads both view dictionaries.",
          "blocks": [
            {
              "file": "dust3r/utils/image.py",
              "start": 117,
              "end": 123,
              "repo": "naver/dust3r",
              "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
              "text": "            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))\n\n        W2, H2 = img.size\n        if verbose:\n            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')\n        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(\n            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))",
              "symbol": "load_images"
            },
            {
              "file": "dust3r/model.py",
              "start": 153,
              "end": 159,
              "repo": "naver/dust3r",
              "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
              "text": "    def _encode_symmetrized(self, view1, view2):\n        img1 = view1['img']\n        img2 = view2['img']\n        B = img1.shape[0]\n        # Recover true_shape when available, otherwise assume that the img shape is the true one\n        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))\n        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))",
              "symbol": "AsymmetricCroCo3DStereo._encode_symmetrized"
            }
          ]
        },
        {
          "title": "Encode image patches with shared weights",
          "note": "A 16 × 16 patch becomes one 1,024-channel token. Matching image sizes let the model stack the pair along the batch axis and use one encoder call. The two images remain independent through these 24 encoder blocks.",
          "blocks": [
            {
              "file": "dust3r/model.py",
              "start": 128,
              "end": 140,
              "repo": "naver/dust3r",
              "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
              "text": "    def _encode_image(self, image, true_shape):\n        # embed the image into patches  (x has size B x Npatches x C)\n        x, pos = self.patch_embed(image, true_shape=true_shape)\n\n        # add positional embedding without cls token\n        assert self.enc_pos_embed is None\n\n        # now apply the transformer encoder and normalization\n        for blk in self.enc_blocks:\n            x = blk(x, pos)\n\n        x = self.enc_norm(x)\n        return x, pos, None",
              "symbol": "AsymmetricCroCo3DStereo._encode_image"
            },
            {
              "file": "dust3r/model.py",
              "start": 142,
              "end": 151,
              "repo": "naver/dust3r",
              "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
              "text": "    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):\n        if img1.shape[-2:] == img2.shape[-2:]:\n            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),\n                                             torch.cat((true_shape1, true_shape2), dim=0))\n            out, out2 = out.chunk(2, dim=0)\n            pos, pos2 = pos.chunk(2, dim=0)\n        else:\n            out, pos, _ = self._encode_image(img1, true_shape1)\n            out2, pos2, _ = self._encode_image(img2, true_shape2)\n        return out, out2, pos, pos2",
              "symbol": "AsymmetricCroCo3DStereo._encode_image_pairs"
            }
          ]
        },
        {
          "title": "Exchange features between decoder branches",
          "note": "Both streams are projected to 768 channels. Each of 12 paired decoder stages sees the previous features of both views. Reversing the feature tuple lets the second branch query from image two while reading image one.",
          "blocks": [
            {
              "file": "dust3r/model.py",
              "start": 68,
              "end": 74,
              "repo": "naver/dust3r",
              "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
              "text": "        self.croco_args = fill_default_args(croco_kwargs, super().__init__)\n        super().__init__(**croco_kwargs)\n\n        # dust3r specific initialization\n        self.dec_blocks2 = deepcopy(self.dec_blocks)\n        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)\n        self.set_freeze(freeze)",
              "symbol": "AsymmetricCroCo3DStereo.__init__"
            },
            {
              "file": "dust3r/model.py",
              "start": 175,
              "end": 186,
              "repo": "naver/dust3r",
              "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
              "text": "        # project to decoder dim\n        f1 = self.decoder_embed(f1)\n        f2 = self.decoder_embed(f2)\n\n        final_output.append((f1, f2))\n        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):\n            # img1 side\n            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)\n            # img2 side\n            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)\n            # store the result\n            final_output.append((f1, f2))",
              "symbol": "AsymmetricCroCo3DStereo._decoder"
            }
          ]
        },
        {
          "title": "Turn four token layers into a dense image grid",
          "credit": "The DPT output adapter builds on CroCo (© 2022–present NAVER Corporation, CC BY-NC-SA 4.0) and MultiMAE (© EPFL VILAB, CC BY-NC 4.0). The NAVER implementation selects different token layers and aligns the refinement grids.",
          "references": [
            {
              "label": "CroCo source license",
              "url": "https://github.com/naver/croco/blob/master/LICENSE"
            },
            {
              "label": "Original MultiMAE adapter",
              "url": "https://github.com/EPFL-VILAB/MultiMAE/blob/main/multimae/output_adapters.py#L661-L690"
            },
            {
              "label": "MultiMAE license",
              "url": "https://github.com/EPFL-VILAB/MultiMAE/blob/main/LICENSE"
            }
          ],
          "note": "The DPT head reads the encoder and three decoder depths. It reshapes their patch sequences into 2D feature grids, projects them and fuses them progressively. Its final head produces three geometry channels and a confidence channel per pixel.",
          "blocks": [
            {
              "file": "dust3r/heads/dpt_head.py",
              "start": 104,
              "end": 114,
              "repo": "naver/dust3r",
              "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
              "text": "    out_nchan = 3\n    ed = net.enc_embed_dim\n    dd = net.dec_embed_dim\n    return PixelwiseTaskWithDPT(num_channels=out_nchan + has_conf,\n                                feature_dim=feature_dim,\n                                last_dim=last_dim,\n                                hooks_idx=[0, l2*2//4, l2*3//4, l2],\n                                dim_tokens=[ed, dd, dd, dd],\n                                postprocess=postprocess,\n                                depth_mode=net.depth_mode,\n                                conf_mode=net.conf_mode,",
              "symbol": "create_dpt_head"
            },
            {
              "file": "dust3r/heads/dpt_head.py",
              "start": 43,
              "end": 54,
              "repo": "naver/dust3r",
              "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
              "text": "        # Hook decoder onto 4 layers from specified ViT layers\n        layers = [encoder_tokens[hook] for hook in self.hooks]\n\n        # Extract only task-relevant tokens and ignore global tokens.\n        layers = [self.adapt_tokens(l) for l in layers]\n\n        # Reshape tokens to spatial representation\n        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]\n\n        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]\n        # Project layers to chosen feature dim\n        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]",
              "symbol": "DPTOutputAdapter_fix.forward"
            },
            {
              "file": "dust3r/heads/dpt_head.py",
              "start": 56,
              "end": 65,
              "repo": "naver/dust3r",
              "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
              "text": "        # Fuse layers using refinement stages\n        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]\n        path_3 = self.scratch.refinenet3(path_4, layers[2])\n        path_2 = self.scratch.refinenet2(path_3, layers[1])\n        path_1 = self.scratch.refinenet1(path_2, layers[0])\n\n        # Output head\n        out = self.head(path_1)\n\n        return out",
              "symbol": "DPTOutputAdapter_fix.forward"
            }
          ]
        },
        {
          "title": "Decode XYZ and name the coordinate frame",
          "note": "Postprocessing splits geometry from confidence. With this checkpoint’s exponential depth mode, the three-vector keeps its direction and its length becomes expm1(length). Both heads predict in camera one’s frame. Renaming the second output records that fact.",
          "blocks": [
            {
              "file": "dust3r/heads/postprocess.py",
              "start": 14,
              "end": 19,
              "repo": "naver/dust3r",
              "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
              "text": "    fmap = out.permute(0, 2, 3, 1)  # B,H,W,3\n    res = dict(pts3d=reg_dense_depth(fmap[:, :, :, 0:3], mode=depth_mode))\n\n    if conf_mode is not None:\n        res['conf'] = reg_dense_conf(fmap[:, :, :, 3], mode=conf_mode)\n    return res",
              "symbol": "postprocess"
            },
            {
              "file": "dust3r/heads/postprocess.py",
              "start": 36,
              "end": 44,
              "repo": "naver/dust3r",
              "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
              "text": "    # distance to origin\n    d = xyz.norm(dim=-1, keepdim=True)\n    xyz = xyz / d.clip(min=1e-8)\n\n    if mode == 'square':\n        return xyz * d.square()\n\n    if mode == 'exp':\n        return xyz * torch.expm1(d)",
              "symbol": "reg_dense_depth"
            },
            {
              "file": "dust3r/model.py",
              "start": 199,
              "end": 211,
              "repo": "naver/dust3r",
              "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
              "text": "    def forward(self, view1, view2):\n        # encode the two images --> B,S,D\n        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)\n\n        # combine all ref images into object-centric representation\n        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)\n\n        with torch.cuda.amp.autocast(enabled=False):\n            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)\n            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)\n\n        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame\n        return res1, res2",
              "symbol": "AsymmetricCroCo3DStereo.forward"
            }
          ]
        },
        {
          "title": "Align a collection after pair inference",
          "note": "The demo forms image pairs and runs the two-view network first. For more than two views it optimizes the pairwise estimates together, producing scene-level poses and points. This solver runs outside the network’s forward method.",
          "blocks": [
            {
              "file": "dust3r/demo.py",
              "start": 155,
              "end": 163,
              "repo": "naver/dust3r",
              "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
              "text": "    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)\n    output = inference(pairs, model, device, batch_size=1, verbose=not silent)\n\n    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer\n    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)\n    lr = 0.01\n\n    if mode == GlobalAlignerMode.PointCloudOptimizer:\n        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)",
              "symbol": "get_reconstructed_scene"
            },
            {
              "file": "README.md",
              "start": 214,
              "end": 219,
              "repo": "naver/dust3r",
              "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
              "text": "    # retrieve useful values from scene:\n    imgs = scene.imgs\n    focals = scene.get_focals()\n    poses = scene.get_im_poses()\n    pts3d = scene.get_pts3d()\n    confidence_masks = scene.get_masks()",
              "symbol": "README.md"
            }
          ]
        }
      ]
    },
    "mast3r": {
      "id": "mast3r",
      "repo": "naver/mast3r",
      "commit": "f5209afc300cec36239a7ac992263f36847bbba0",
      "licenseFile": "LICENSE",
      "inputs": [
        "view1['img'], view2['img'] · each [B, 3, Hᵢ, Wᵢ] — a normalized RGB pair prepared by the inherited DUSt3R loader. The metric CatMLP+DPT checkpoint uses 16 × 16 patches.",
        "true_shape · [B, 2] per view — processed image height and width for restoring the pixel grids. Loader-provided instance identifiers allow reuse of symmetric pairs."
      ],
      "outputs": [
        "pts3d / pts3d_in_other_view · [B, Hᵢ, Wᵢ, 3] — points for both image grids in the first camera’s frame, with learned metric scale for this checkpoint.",
        "desc · [B, Hᵢ, Wᵢ, 24] — a unit-length descriptor per pixel. conf and desc_conf each have shape [B, Hᵢ, Wᵢ].",
        "matches_im0, matches_im1 · each [M, 2] — optional downstream (x, y) pixel coordinates for M reciprocal matches."
      ],
      "steps": [
        {
          "title": "Run the inherited stereo backbone",
          "note": "AsymmetricMASt3R inherits DUSt3R’s forward pass. The pinned base implementation reads a prepared RGB pair, encodes both images and calls a two-stream decoder. The metric checkpoint uses a 24-layer shared ViT-L encoder and two 12-layer decoders.",
          "blocks": [
            {
              "file": "mast3r/model.py",
              "start": 39,
              "end": 44,
              "repo": "naver/mast3r",
              "commit": "f5209afc300cec36239a7ac992263f36847bbba0",
              "text": "class AsymmetricMASt3R(AsymmetricCroCo3DStereo):\n    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, **kwargs):\n        self.desc_mode = desc_mode\n        self.two_confs = two_confs\n        self.desc_conf_mode = desc_conf_mode\n        super().__init__(**kwargs)",
              "symbol": "AsymmetricMASt3R"
            },
            {
              "file": "dust3r/model.py",
              "start": 199,
              "end": 211,
              "repo": "naver/dust3r",
              "commit": "3cc8c88c413bb9e34c41db0e0eef99c2ee010b12",
              "text": "    def forward(self, view1, view2):\n        # encode the two images --> B,S,D\n        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)\n\n        # combine all ref images into object-centric representation\n        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)\n\n        with torch.cuda.amp.autocast(enabled=False):\n            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)\n            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)\n\n        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame\n        return res1, res2",
              "symbol": "AsymmetricCroCo3DStereo.forward"
            },
            {
              "file": "dust3r/model.py",
              "start": 128,
              "end": 140,
              "repo": "naver/dust3r",
              "commit": "3cc8c88c413bb9e34c41db0e0eef99c2ee010b12",
              "text": "    def _encode_image(self, image, true_shape):\n        # embed the image into patches  (x has size B x Npatches x C)\n        x, pos = self.patch_embed(image, true_shape=true_shape)\n\n        # add positional embedding without cls token\n        assert self.enc_pos_embed is None\n\n        # now apply the transformer encoder and normalization\n        for blk in self.enc_blocks:\n            x = blk(x, pos)\n\n        x = self.enc_norm(x)\n        return x, pos, None",
              "symbol": "AsymmetricCroCo3DStereo._encode_image"
            }
          ]
        },
        {
          "title": "Exchange views and attach a compound head to each",
          "note": "At each paired decoder stage, image one reads image two’s prior state and image two reads image one’s. MASt3R replaces the per-view readouts with a DPT geometry branch and an MLP descriptor branch. The DPT branch taps four depths.",
          "blocks": [
            {
              "file": "dust3r/model.py",
              "start": 175,
              "end": 186,
              "repo": "naver/dust3r",
              "commit": "3cc8c88c413bb9e34c41db0e0eef99c2ee010b12",
              "text": "        # project to decoder dim\n        f1 = self.decoder_embed(f1)\n        f2 = self.decoder_embed(f2)\n\n        final_output.append((f1, f2))\n        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):\n            # img1 side\n            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)\n            # img2 side\n            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)\n            # store the result\n            final_output.append((f1, f2))",
              "symbol": "AsymmetricCroCo3DStereo._decoder"
            },
            {
              "file": "mast3r/model.py",
              "start": 60,
              "end": 67,
              "repo": "naver/mast3r",
              "commit": "f5209afc300cec36239a7ac992263f36847bbba0",
              "text": "        if self.desc_conf_mode is None:\n            self.desc_conf_mode = conf_mode\n        # allocate heads\n        self.downstream_head1 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))\n        self.downstream_head2 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))\n        # magic wrapper\n        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)\n        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)",
              "symbol": "AsymmetricMASt3R.set_downstream_head"
            },
            {
              "file": "mast3r/catmlp_dpt_head.py",
              "start": 217,
              "end": 229,
              "repo": "naver/mast3r",
              "commit": "f5209afc300cec36239a7ac992263f36847bbba0",
              "text": "        out_nchan = 3\n        ed = net.enc_embed_dim\n        dd = net.dec_embed_dim\n        return Cat_MLP_LocalFeatures_DPT_Pts3d(net, local_feat_dim=local_feat_dim, has_conf=has_conf,\n                                               num_channels=out_nchan + has_conf,\n                                               feature_dim=feature_dim,\n                                               last_dim=last_dim,\n                                               hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],\n                                               dim_tokens=[ed, dd, dd, dd],\n                                               postprocess=postprocess,\n                                               depth_mode=net.depth_mode,\n                                               conf_mode=net.conf_mode,\n                                               head_type='regression')",
              "symbol": "mast3r_head_factory"
            }
          ]
        },
        {
          "title": "Decode geometry from the patch grids",
          "credit": "MASt3R inherits the DUSt3R DPT adapter. Its sources include CroCo (© 2022–present NAVER Corporation, CC BY-NC-SA 4.0) and MultiMAE (© EPFL VILAB, CC BY-NC 4.0), with NAVER’s changes to token selection and grid alignment.",
          "references": [
            {
              "label": "CroCo source license",
              "url": "https://github.com/naver/croco/blob/master/LICENSE"
            },
            {
              "label": "Original MultiMAE adapter",
              "url": "https://github.com/EPFL-VILAB/MultiMAE/blob/main/multimae/output_adapters.py#L661-L690"
            },
            {
              "label": "MultiMAE license",
              "url": "https://github.com/EPFL-VILAB/MultiMAE/blob/main/LICENSE"
            }
          ],
          "note": "The DPT branch predicts XYZ and confidence from the intermediate token layers. Inside the inherited DPT implementation, token sequences become spatial grids and the refinement layers fuse coarse and fine features. The descriptor branch also receives the encoder and final decoder tokens.",
          "blocks": [
            {
              "file": "mast3r/catmlp_dpt_head.py",
              "start": 73,
              "end": 81,
              "repo": "naver/mast3r",
              "commit": "f5209afc300cec36239a7ac992263f36847bbba0",
              "text": "    def forward(self, decout, img_shape):\n        # pass through the heads\n        pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))\n\n        # recover encoder and decoder outputs\n        enc_output, dec_output = decout[0], decout[-1]\n        cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate\n        H, W = img_shape\n        B, S, D = cat_output.shape",
              "symbol": "Cat_MLP_LocalFeatures_DPT_Pts3d.forward"
            },
            {
              "file": "dust3r/heads/dpt_head.py",
              "start": 43,
              "end": 54,
              "repo": "naver/dust3r",
              "commit": "3cc8c88c413bb9e34c41db0e0eef99c2ee010b12",
              "text": "        # Hook decoder onto 4 layers from specified ViT layers\n        layers = [encoder_tokens[hook] for hook in self.hooks]\n\n        # Extract only task-relevant tokens and ignore global tokens.\n        layers = [self.adapt_tokens(l) for l in layers]\n\n        # Reshape tokens to spatial representation\n        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]\n\n        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]\n        # Project layers to chosen feature dim\n        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]",
              "symbol": "DPTOutputAdapter_fix.forward"
            },
            {
              "file": "dust3r/heads/dpt_head.py",
              "start": 56,
              "end": 65,
              "repo": "naver/dust3r",
              "commit": "3cc8c88c413bb9e34c41db0e0eef99c2ee010b12",
              "text": "        # Fuse layers using refinement stages\n        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]\n        path_3 = self.scratch.refinenet3(path_4, layers[2])\n        path_2 = self.scratch.refinenet2(path_3, layers[1])\n        path_1 = self.scratch.refinenet1(path_2, layers[0])\n\n        # Output head\n        out = self.head(path_1)\n\n        return out",
              "symbol": "DPTOutputAdapter_fix.forward"
            }
          ]
        },
        {
          "title": "Expand the descriptor for every pixel",
          "note": "The MLP receives the concatenated encoder and decoder channels. It emits a patch’s worth of outputs for each token. Pixel shuffle rearranges them into full-resolution descriptors and descriptor confidence, which are joined with the DPT outputs.",
          "blocks": [
            {
              "file": "mast3r/catmlp_dpt_head.py",
              "start": 63,
              "end": 71,
              "repo": "naver/mast3r",
              "commit": "f5209afc300cec36239a7ac992263f36847bbba0",
              "text": "        self.desc_mode = net.desc_mode\n        self.has_conf = has_conf\n        self.two_confs = net.two_confs  # independent confs for 3D regr and descs\n        self.desc_conf_mode = net.desc_conf_mode\n        idim = net.enc_embed_dim + net.dec_embed_dim\n\n        self.head_local_features = Mlp(in_features=idim,\n                                       hidden_features=int(hidden_dim_factor * idim),\n                                       out_features=(self.local_feat_dim + self.two_confs) * self.patch_size**2)",
              "symbol": "Cat_MLP_LocalFeatures_DPT_Pts3d.__init__"
            },
            {
              "file": "mast3r/catmlp_dpt_head.py",
              "start": 83,
              "end": 89,
              "repo": "naver/mast3r",
              "commit": "f5209afc300cec36239a7ac992263f36847bbba0",
              "text": "        # extract local_features\n        local_features = self.head_local_features(cat_output)  # B,S,D\n        local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)\n        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W\n\n        # post process 3D pts, descriptors and confidences\n        out = torch.cat([pts3d, local_features], dim=1)",
              "symbol": "Cat_MLP_LocalFeatures_DPT_Pts3d.forward"
            }
          ]
        },
        {
          "title": "Separate coordinates, confidence and descriptors",
          "note": "The first three channels encode geometry, followed by its confidence. The next 24 channels are normalized into descriptors. The metric checkpoint uses an independent confidence channel for descriptor matching.",
          "blocks": [
            {
              "file": "mast3r/catmlp_dpt_head.py",
              "start": 19,
              "end": 24,
              "repo": "naver/mast3r",
              "commit": "f5209afc300cec36239a7ac992263f36847bbba0",
              "text": "def reg_desc(desc, mode):\n    if 'norm' in mode:\n        desc = desc / desc.norm(dim=-1, keepdim=True)\n    else:\n        raise ValueError(f\"Unknown desc mode {mode}\")\n    return desc",
              "symbol": "reg_desc"
            },
            {
              "file": "mast3r/catmlp_dpt_head.py",
              "start": 30,
              "end": 41,
              "repo": "naver/mast3r",
              "commit": "f5209afc300cec36239a7ac992263f36847bbba0",
              "text": "    fmap = out.permute(0, 2, 3, 1)  # B,H,W,D\n    res = dict(pts3d=reg_dense_depth(fmap[..., 0:3], mode=depth_mode))\n    if conf_mode is not None:\n        res['conf'] = reg_dense_conf(fmap[..., 3], mode=conf_mode)\n    if desc_dim is not None:\n        start = 3 + int(conf_mode is not None)\n        res['desc'] = reg_desc(fmap[..., start:start + desc_dim], mode=desc_mode)\n        if two_confs:\n            res['desc_conf'] = reg_dense_conf(fmap[..., start + desc_dim], mode=desc_conf_mode)\n        else:\n            res['desc_conf'] = res['conf'].clone()\n    return res",
              "symbol": "postprocess"
            }
          ]
        },
        {
          "title": "Find reciprocal matches in descriptor space",
          "note": "The authors’ example removes the batch axis and uses dot-product search in both directions. The matcher iterates forward and backward nearest-neighbor lookups, then keeps unique converged pixel pairs. This step consumes the network’s output.",
          "blocks": [
            {
              "file": "README.md",
              "start": 263,
              "end": 270,
              "repo": "naver/mast3r",
              "commit": "f5209afc300cec36239a7ac992263f36847bbba0",
              "text": "    view1, pred1 = output['view1'], output['pred1']\n    view2, pred2 = output['view2'], output['pred2']\n\n    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()\n\n    # find 2D-2D matches between the two images\n    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,\n                                                   device=device, dist='dot', block_size=2**13)",
              "symbol": "README.md"
            },
            {
              "file": "mast3r/fast_nn.py",
              "start": 152,
              "end": 160,
              "repo": "naver/mast3r",
              "commit": "f5209afc300cec36239a7ac992263f36847bbba0",
              "text": "    while notyet.any():\n        _, xy2[notyet] = to_numpy(tree2.query(pts1[xy1[notyet]], **matcher_kw))\n        if not ret_basin:\n            notyet &= (old_xy2 != xy2)  # remove points that have converged\n\n        _, xy1[notyet] = to_numpy(tree1.query(pts2[xy2[notyet]], **matcher_kw))\n        if ret_basin:\n            basin[old_xy1[notyet]] = xy1[notyet]\n        notyet &= (old_xy1 != xy1)  # remove points that have converged",
              "symbol": "fast_reciprocal_NNs"
            },
            {
              "file": "mast3r/fast_nn.py",
              "start": 181,
              "end": 188,
              "repo": "naver/mast3r",
              "commit": "f5209afc300cec36239a7ac992263f36847bbba0",
              "text": "    else:\n        converged = ~notyet  # converged correspondences\n\n    # keep only unique correspondences, and sort on xy1\n    xy1, xy2 = merge_corres(xy1[converged], xy2[converged], (H1, W1), (H2, W2), ret_xy=ret_xy)\n    if ret_basin:\n        return xy1, xy2, basin\n    return xy1, xy2",
              "symbol": "fast_reciprocal_NNs"
            }
          ]
        }
      ]
    },
    "vggt": {
      "id": "vggt",
      "repo": "facebookresearch/vggt",
      "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
      "licenseFile": "LICENSE.txt",
      "inputs": [
        "images · [B, S, 3, H, W] or [S, 3, H, W] — S RGB views in [0, 1] on a common patch-compatible image grid.",
        "query_points · optional [B, N, 2] or [N, 2] — the (x, y) pixel coordinates of N points in the first input image."
      ],
      "outputs": [
        "world_points · [B, S, H, W, 3] — directly regressed points in the first camera’s frame, at relative scale. world_points_conf gives [B, S, H, W] scores.",
        "depth · [B, S, H, W, 1] — positive per-view depth with a separate [B, S, H, W] depth_conf map.",
        "pose_enc · [B, S, 9] — translation, quaternion and two fields of view per camera. The pose utility converts these into world-to-camera extrinsics and pixel intrinsics.",
        "track · [B, S, N, 2] — optional 2D trajectories in image pixels. vis and conf each have shape [B, S, N]."
      ],
      "steps": [
        {
          "title": "Convert RGB images to tokens",
          "note": "VGGT batches the input views, normalizes the RGB channels, then runs the DINOv2 patch encoder on each image. A 14 × 14 patch contributes one 1,024-channel feature. One camera token and four register tokens are prepended to each view.",
          "blocks": [
            {
              "file": "vggt/models/vggt.py",
              "start": 54,
              "end": 63,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "        # If without batch dimension, add it\n        if len(images.shape) == 4:\n            images = images.unsqueeze(0)\n            \n        if query_points is not None and len(query_points.shape) == 2:\n            query_points = query_points.unsqueeze(0)\n\n        aggregated_tokens_list, patch_start_idx = self.aggregator(images)\n\n        predictions = {}",
              "symbol": "VGGT.forward"
            },
            {
              "file": "vggt/models/aggregator.py",
              "start": 199,
              "end": 212,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "        B, S, C_in, H, W = images.shape\n\n        if C_in != 3:\n            raise ValueError(f\"Expected 3 input channels, got {C_in}\")\n\n        # Normalize images and reshape for patch embed\n        images = (images - self._resnet_mean) / self._resnet_std\n\n        # Reshape to [B*S, C, H, W] for patch embedding\n        images = images.view(B * S, C_in, H, W)\n        patch_tokens = self.patch_embed(images)\n\n        if isinstance(patch_tokens, dict):\n            patch_tokens = patch_tokens[\"x_norm_patchtokens\"]",
              "symbol": "Aggregator.forward"
            },
            {
              "file": "vggt/models/aggregator.py",
              "start": 214,
              "end": 221,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "        _, P, C = patch_tokens.shape\n\n        # Expand camera and register tokens to match batch size and sequence length\n        camera_token = slice_expand_and_flatten(self.camera_token, B, S)\n        register_token = slice_expand_and_flatten(self.register_token, B, S)\n\n        # Concatenate special tokens with patch tokens\n        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)",
              "symbol": "Aggregator.forward"
            }
          ]
        },
        {
          "title": "Alternate within-frame and global attention",
          "note": "One frame block sees P tokens from a single image. A global block sees S × P tokens from the complete scene. The default stack repeats this pair 24 times. At layers 4, 11, 17 and 23, frame and global features are concatenated into 2,048 channels for the heads.",
          "blocks": [
            {
              "file": "vggt/models/aggregator.py",
              "start": 241,
              "end": 250,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "        for _ in range(self.aa_block_num):\n            for attn_type in self.aa_order:\n                if attn_type == \"frame\":\n                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(\n                        tokens, B, S, P, C, frame_idx, pos=pos\n                    )\n                elif attn_type == \"global\":\n                    tokens, global_idx, global_intermediates = self._process_global_attention(\n                        tokens, B, S, P, C, global_idx, pos=pos\n                    )",
              "symbol": "Aggregator.forward"
            },
            {
              "file": "vggt/models/aggregator.py",
              "start": 291,
              "end": 299,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):\n        \"\"\"\n        Process global attention blocks. We keep tokens in shape (B, S*P, C).\n        \"\"\"\n        if tokens.shape != (B, S * P, C):\n            tokens = tokens.view(B, S, P, C).view(B, S * P, C)\n\n        if pos is not None and pos.shape != (B, S * P, 2):\n            pos = pos.view(B, S, P, 2).view(B, S * P, 2)",
              "symbol": "Aggregator._process_global_attention"
            },
            {
              "file": "vggt/models/aggregator.py",
              "start": 254,
              "end": 261,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "            for i in range(len(frame_intermediates)):\n                layer_idx = len(output_list)\n                if layer_idx in self.cached_layer_indices:\n                    # concat frame and global intermediates, [B x S x P x 2C]\n                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)\n                    output_list.append(concat_inter)\n                else:\n                    output_list.append(None)",
              "symbol": "Aggregator.forward"
            }
          ]
        },
        {
          "title": "Route shared features to camera, depth, points and tracks",
          "note": "The constructor allocates separate readouts. The point DPT head emits four channels for XYZ plus confidence. The depth DPT head emits two for depth plus confidence. Each dense head discards the special tokens and reshapes patch features onto a 2D grid.",
          "blocks": [
            {
              "file": "vggt/models/vggt.py",
              "start": 17,
              "end": 27,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "class VGGT(nn.Module, PyTorchModelHubMixin):\n    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,\n                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):\n        super().__init__()\n\n        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)\n\n        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None\n        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation=\"inv_log\", conf_activation=\"expp1\") if enable_point else None\n        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation=\"exp\", conf_activation=\"expp1\") if enable_depth else None\n        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None",
              "symbol": "VGGT"
            },
            {
              "file": "vggt/heads/dpt_head.py",
              "start": 205,
              "end": 216,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "        for layer_idx in self.intermediate_layer_idx:\n            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]\n\n            # Select frames if processing a chunk\n            if frames_start_idx is not None and frames_end_idx is not None:\n                x = x[:, frames_start_idx:frames_end_idx]\n\n            x = x.reshape(B * S, -1, x.shape[-1])\n\n            x = self.norm(x)\n\n            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))",
              "symbol": "DPTHead._forward_impl"
            },
            {
              "file": "vggt/heads/dpt_head.py",
              "start": 218,
              "end": 228,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "            x = self.projects[dpt_idx](x)\n            if self.pos_embed:\n                x = self._apply_pos_embed(x, W, H)\n            x = self.resize_layers[dpt_idx](x)\n\n            out.append(x)\n            dpt_idx += 1\n\n        # Fuse features from multiple layers.\n        out = self.scratch_forward(out)\n        # Interpolate fused output to match target image resolution.",
              "symbol": "DPTHead._forward_impl"
            }
          ]
        },
        {
          "title": "Read depth and a pointmap directly",
          "note": "DPT fuses four feature scales and upsamples to the image resolution. Its output convolution and activation separate predictions from confidence. The point head directly regresses scene XYZ. The depth head provides an independent estimate that can also be unprojected with the recovered cameras.",
          "blocks": [
            {
              "file": "vggt/heads/dpt_head.py",
              "start": 226,
              "end": 234,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "        # Fuse features from multiple layers.\n        out = self.scratch_forward(out)\n        # Interpolate fused output to match target image resolution.\n        out = custom_interpolate(\n            out,\n            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),\n            mode=\"bilinear\",\n            align_corners=True,\n        )",
              "symbol": "DPTHead._forward_impl"
            },
            {
              "file": "vggt/heads/dpt_head.py",
              "start": 242,
              "end": 247,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "        out = self.scratch.output_conv2(out)\n        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)\n\n        preds = preds.view(B, S, *preds.shape[1:])\n        conf = conf.view(B, S, *conf.shape[1:])\n        return preds, conf",
              "symbol": "DPTHead._forward_impl"
            },
            {
              "file": "vggt/models/vggt.py",
              "start": 71,
              "end": 83,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "            if self.depth_head is not None:\n                depth, depth_conf = self.depth_head(\n                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx\n                )\n                predictions[\"depth\"] = depth\n                predictions[\"depth_conf\"] = depth_conf\n\n            if self.point_head is not None:\n                pts3d, pts3d_conf = self.point_head(\n                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx\n                )\n                predictions[\"world_points\"] = pts3d\n                predictions[\"world_points_conf\"] = pts3d_conf",
              "symbol": "VGGT.forward"
            }
          ]
        },
        {
          "title": "Recover a camera from its token",
          "note": "The camera head reads token zero of the final layer for each view and iteratively adds a delta to a nine-value pose estimate. The utility separates translation, quaternion and horizontal/vertical field of view to construct the camera matrices.",
          "blocks": [
            {
              "file": "vggt/heads/camera_head.py",
              "start": 85,
              "end": 93,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "        # Use tokens from the last block for camera prediction.\n        tokens = aggregated_tokens_list[-1]\n\n        # Extract the camera tokens\n        pose_tokens = tokens[:, :, 0]\n        pose_tokens = self.token_norm(pose_tokens)\n\n        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)\n        return pred_pose_enc_list",
              "symbol": "CameraHead.forward"
            },
            {
              "file": "vggt/heads/camera_head.py",
              "start": 126,
              "end": 139,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "            pose_tokens_modulated = self.trunk(pose_tokens_modulated)\n            # Compute the delta update for the pose encoding.\n            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))\n\n            if pred_pose_enc is None:\n                pred_pose_enc = pred_pose_enc_delta\n            else:\n                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta\n\n            # Apply final activation functions for translation, quaternion, and field-of-view.\n            activated_pose = activate_pose(\n                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act\n            )\n            pred_pose_enc_list.append(activated_pose)",
              "symbol": "CameraHead.trunk_fn"
            },
            {
              "file": "vggt/utils/pose_enc.py",
              "start": 102,
              "end": 114,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "    if pose_encoding_type == \"absT_quaR_FoV\":\n        T = pose_encoding[..., :3]\n        quat = pose_encoding[..., 3:7]\n        fov_h = pose_encoding[..., 7]\n        fov_w = pose_encoding[..., 8]\n\n        R = quat_to_mat(quat)\n        extrinsics = torch.cat([R, T[..., None]], dim=-1)\n\n        if build_intrinsics:\n            H, W = image_size_hw\n            fy = (H / 2.0) / torch.tan(fov_h / 2.0)\n            fx = (W / 2.0) / torch.tan(fov_w / 2.0)",
              "symbol": "pose_encoding_to_extri_intri"
            }
          ]
        },
        {
          "title": "Track query pixels across the input views",
          "note": "With queries present, the tracking head decodes half-resolution feature maps. Its tracker samples each query’s feature in the first view and refines its coordinates across views. The model returns 2D tracks, visibility and confidence separately from the scene pointmap.",
          "credit": "VGGT credits VGGSfM and CoTracker for the tracker design, © Meta Platforms, Inc. and affiliates. The VGGSfM repository distributes its earlier implementation under CC BY-NC 4.0.",
          "references": [
            {
              "label": "Earlier VGGSfM implementation",
              "url": "https://github.com/facebookresearch/vggsfm/blob/main/vggsfm/models/track_modules/base_track_predictor.py#L92-L103"
            },
            {
              "label": "VGGSfM license",
              "url": "https://github.com/facebookresearch/vggsfm/blob/main/LICENSE.txt"
            },
            {
              "label": "CoTracker license",
              "url": "https://github.com/facebookresearch/co-tracker/blob/main/LICENSE.md"
            }
          ],
          "blocks": [
            {
              "file": "vggt/heads/track_head.py",
              "start": 91,
              "end": 104,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "        B, S, _, H, W = images.shape\n\n        # Extract features from tokens\n        # feature_maps has shape (B, S, C, H//2, W//2) due to down_ratio=2\n        feature_maps = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)\n\n        # Use default iterations if not specified\n        if iters is None:\n            iters = self.iters\n\n        # Perform tracking using the extracted features\n        coord_preds, vis_scores, conf_scores = self.tracker(query_points=query_points, fmaps=feature_maps, iters=iters)\n\n        return coord_preds, vis_scores, conf_scores",
              "symbol": "TrackHead.forward"
            },
            {
              "file": "vggt/heads/track_modules/base_track_predictor.py",
              "start": 104,
              "end": 114,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "        query_points = query_points / float(self.stride)\n\n        # Init with coords as the query points\n        # It means the search will start from the position of query points at the reference frames\n        coords = query_points.clone().reshape(B, 1, N, 2).repeat(1, S, 1, 1)\n\n        # Sample/extract the features of the query points in the query frame\n        query_track_feat = sample_features4d(fmaps[:, 0], coords[:, 0])\n\n        # init track feats by query feats\n        track_feats = query_track_feat.unsqueeze(1).repeat(1, S, 1, 1)  # B, S, N, C",
              "symbol": "BaseTrackerPredictor.forward"
            },
            {
              "file": "vggt/models/vggt.py",
              "start": 85,
              "end": 96,
              "repo": "facebookresearch/vggt",
              "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
              "text": "        if self.track_head is not None and query_points is not None:\n            track_list, vis, conf = self.track_head(\n                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points\n            )\n            predictions[\"track\"] = track_list[-1]  # track of the last iteration\n            predictions[\"vis\"] = vis\n            predictions[\"conf\"] = conf\n\n        if not self.training:\n            predictions[\"images\"] = images  # store the images for visualization during inference\n\n        return predictions",
              "symbol": "VGGT.forward"
            }
          ]
        }
      ]
    },
    "da3": {
      "id": "da3",
      "repo": "ByteDance-Seed/Depth-Anything-3",
      "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
      "licenseFile": "LICENSE",
      "inputs": [
        "x · [B, V, 3, H, W] — normalized RGB for the raw DA3-Large network. The public image processor resizes or crops to multiples of 14 and applies ImageNet normalization.",
        "extrinsics, intrinsics · [B, V, 4, 4], [B, V, 3, 3] — optional world-to-camera poses and pinhole calibration, encoded into camera tokens when supplied."
      ],
      "outputs": [
        "depth, depth_conf · [B, V, H, W] each — camera-axis depth and confidence in the raw Main output. RGB-only Main predicts relative scale.",
        "extrinsics, intrinsics · [B, V, 3, 4], [B, V, 3, 3] — world-to-camera and calibration matrices from the default raw camera decoder. The internal ray tensors are removed during camera decoding.",
        "World points · [M, 3] — the downstream GLB exporter unprojects valid depth samples and concatenates them across views. M counts retained pixels."
      ],
      "steps": [
        {
          "title": "Patchify RGB and supply camera features",
          "note": "DA3-Large takes V views together. Provided cameras pass through cam_enc. The vision transformer temporarily folds views into the batch to embed each 14×14 RGB patch, adds a prefix token and position encoding, then restores the view dimension.",
          "blocks": [
            {
              "file": "src/depth_anything_3/model/da3.py",
              "start": 125,
              "end": 134,
              "repo": "ByteDance-Seed/Depth-Anything-3",
              "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
              "text": "        # Extract features using backbone\n        if extrinsics is not None:\n            with torch.autocast(device_type=x.device.type, enabled=False):\n                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])\n        else:\n            cam_token = None\n\n        feats, aux_feats = self.backbone(\n            x, cam_token=cam_token, export_feat_layers=export_feat_layers, ref_view_strategy=ref_view_strategy\n        )",
              "symbol": "DepthAnything3Net.forward"
            },
            {
              "file": "src/depth_anything_3/model/dinov2/vision_transformer.py",
              "start": 261,
              "end": 269,
              "repo": "ByteDance-Seed/Depth-Anything-3",
              "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
              "text": "    def prepare_tokens_with_masks(self, x, masks=None, cls_token=None, **kwargs):\n        B, S, nc, w, h = x.shape\n        x = rearrange(x, \"b s c h w -> (b s) c h w\")\n        x = self.patch_embed(x)\n        if masks is not None:\n            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)\n        cls_token = self.prepare_cls_token(B, S)\n        x = torch.cat((cls_token, x), dim=1)\n        x = x + self.interpolate_pos_encoding(x, w, h)",
              "symbol": "DinoVisionTransformer.prepare_tokens_with_masks"
            }
          ]
        },
        {
          "title": "Mix views in selected transformer layers",
          "note": "The Large recipe taps four layers of a 24-block, 1,024-wide ViT. After the configured threshold, odd-indexed blocks attend across all views. Local attention treats views as separate batch items. Global attention joins their patch sequences, so features from different images can interact.",
          "blocks": [
            {
              "file": "src/depth_anything_3/configs/da3-large.yaml",
              "start": 12,
              "end": 17,
              "repo": "ByteDance-Seed/Depth-Anything-3",
              "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
              "text": "  name: vitl\n  out_layers: [11, 15, 19, 23]\n  alt_start: 8\n  qknorm_start: 8\n  rope_start: 8\n  cat_token: True",
              "symbol": "da3-large.yaml"
            },
            {
              "file": "src/depth_anything_3/model/dinov2/vision_transformer.py",
              "start": 333,
              "end": 342,
              "repo": "ByteDance-Seed/Depth-Anything-3",
              "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
              "text": "            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:\n                x = self.process_attention(\n                    x, blk, \"global\", pos=g_pos, attn_mask=kwargs.get(\"attn_mask\", None)\n                )\n            else:\n                x = self.process_attention(x, blk, \"local\", pos=l_pos)\n                local_x = x\n\n            if i in blocks_to_take:\n                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x",
              "symbol": "DinoVisionTransformer._get_intermediate_layers_not_chunked"
            },
            {
              "file": "src/depth_anything_3/model/dinov2/vision_transformer.py",
              "start": 351,
              "end": 360,
              "repo": "ByteDance-Seed/Depth-Anything-3",
              "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
              "text": "    def process_attention(self, x, block, attn_type=\"global\", pos=None, attn_mask=None):\n        b, s, n = x.shape[:3]\n        if attn_type == \"local\":\n            x = rearrange(x, \"b s n c -> (b s) n c\")\n            if pos is not None:\n                pos = rearrange(pos, \"b s n c -> (b s) n c\")\n        elif attn_type == \"global\":\n            x = rearrange(x, \"b s n c -> b (s n) c\")\n            if pos is not None:\n                pos = rearrange(pos, \"b s n c -> b (s n) c\")",
              "symbol": "DinoVisionTransformer.process_attention"
            }
          ]
        },
        {
          "title": "Reassemble patch features into depth and rays",
          "note": "DualDPT reshapes patch sequences into 2D feature maps at four scales. The main fusion path decodes positive depth and confidence at the image resolution. The auxiliary path produces six ray values and a confidence on a coarser grid.",
          "blocks": [
            {
              "file": "src/depth_anything_3/model/dualdpt.py",
              "start": 215,
              "end": 227,
              "repo": "ByteDance-Seed/Depth-Anything-3",
              "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
              "text": "        B, _, C = feats[0].shape\n        ph, pw = H // self.patch_size, W // self.patch_size\n        resized_feats = []\n        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):\n            x = feats[take_idx][:, patch_start_idx:]\n            x = self.norm(x)\n            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)  # [B*S, C, ph, pw]\n\n            x = self.projects[stage_idx](x)\n            if self.pos_embed:\n                x = self._add_pos_embed(x, W, H)\n            x = self.resize_layers[stage_idx](x)  # align scales\n            resized_feats.append(x)",
              "symbol": "DualDPT._forward_impl"
            },
            {
              "file": "src/depth_anything_3/model/dualdpt.py",
              "start": 244,
              "end": 247,
              "repo": "ByteDance-Seed/Depth-Anything-3",
              "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
              "text": "        main_logits = self.scratch.output_conv2(fused_main)\n        fmap = main_logits.permute(0, 2, 3, 1)\n        main_pred = self._apply_activation_single(fmap[..., :-1], self.activation)\n        main_conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)",
              "symbol": "DualDPT._forward_impl"
            },
            {
              "file": "src/depth_anything_3/model/dualdpt.py",
              "start": 255,
              "end": 264,
              "repo": "ByteDance-Seed/Depth-Anything-3",
              "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
              "text": "        last_aux_logits = self.scratch.output_conv2_aux[-1](last_aux)\n        fmap_last = last_aux_logits.permute(0, 2, 3, 1)\n        aux_pred = self._apply_activation_single(fmap_last[..., :-1], \"linear\")\n        aux_conf = self._apply_activation_single(fmap_last[..., -1], self.conf_activation)\n        return {\n            self.head_main: main_pred.squeeze(-1),\n            f\"{self.head_main}_conf\": main_conf,\n            self.head_aux: aux_pred,\n            f\"{self.head_aux}_conf\": aux_conf,\n        }",
              "symbol": "DualDPT._forward_impl"
            }
          ]
        },
        {
          "title": "Read the camera matrices",
          "note": "The default camera decoder reads the last layer’s camera tokens, decodes pose and calibration, and stores the inverse pose as world-to-camera extrinsics. With use_ray_pose enabled, the alternate branch fits cameras from the predicted ray field and its confidence.",
          "blocks": [
            {
              "file": "src/depth_anything_3/model/da3.py",
              "start": 139,
              "end": 144,
              "repo": "ByteDance-Seed/Depth-Anything-3",
              "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
              "text": "        with torch.autocast(device_type=x.device.type, enabled=False):\n            output = self._process_depth_head(feats, H, W)\n            if use_ray_pose:\n                output = self._process_ray_pose_estimation(output, H, W)\n            else:\n                output = self._process_camera_estimation(feats, H, W, output)",
              "symbol": "DepthAnything3Net.forward"
            },
            {
              "file": "src/depth_anything_3/model/da3.py",
              "start": 215,
              "end": 226,
              "repo": "ByteDance-Seed/Depth-Anything-3",
              "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
              "text": "        if self.cam_dec is not None:\n            pose_enc = self.cam_dec(feats[-1][1])\n            # Remove ray information as it's not needed for pose estimation\n            if \"ray\" in output:\n                del output.ray\n            if \"ray_conf\" in output:\n                del output.ray_conf\n\n            # Convert pose encoding to extrinsics and intrinsics\n            c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))\n            output.extrinsics = affine_inverse(c2w)\n            output.intrinsics = ixt",
              "symbol": "DepthAnything3Net._process_camera_estimation"
            },
            {
              "file": "src/depth_anything_3/model/da3.py",
              "start": 185,
              "end": 191,
              "repo": "ByteDance-Seed/Depth-Anything-3",
              "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
              "text": "        if \"ray\" in output and \"ray_conf\" in output:\n            pred_extrinsic, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(\n                output.ray,\n                output.ray_conf,\n                output.ray.shape[-3],\n                output.ray.shape[-2],\n            )",
              "symbol": "DepthAnything3Net._process_ray_pose_estimation"
            }
          ]
        },
        {
          "title": "Apply metric alignment in the Nested variant",
          "note": "For metric output, the separate Nested model runs its any-view and monocular metric networks on the same images. A least-squares fit on selected depth pixels supplies one scalar for depth and camera translation. The Main model skips this second network.",
          "blocks": [
            {
              "file": "src/depth_anything_3/model/da3.py",
              "start": 361,
              "end": 370,
              "repo": "ByteDance-Seed/Depth-Anything-3",
              "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
              "text": "        # Get predictions from both branches\n        output = self.da3(\n            x, extrinsics, intrinsics, export_feat_layers=export_feat_layers, infer_gs=infer_gs, use_ray_pose=use_ray_pose, ref_view_strategy=ref_view_strategy\n        )\n        metric_output = self.da3_metric(x)\n\n        # Apply metric scaling and alignment\n        output = self._apply_metric_scaling(output, metric_output)\n        output = self._apply_depth_alignment(output, metric_output)\n        output = self._handle_sky_regions(output, metric_output)",
              "symbol": "NestedDepthAnything3Net.forward"
            },
            {
              "file": "src/depth_anything_3/model/da3.py",
              "start": 405,
              "end": 414,
              "repo": "ByteDance-Seed/Depth-Anything-3",
              "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
              "text": "        # Compute scale factor using least squares\n        valid_depth = output.depth[align_mask]\n        valid_metric_depth = metric_output.depth[align_mask]\n        scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)\n\n        # Apply scaling to depth and extrinsics\n        output.depth *= scale_factor\n        output.extrinsics[:, :, :3, 3] *= scale_factor\n        output.is_metric = 1\n        output.scale_factor = scale_factor.item()",
              "symbol": "NestedDepthAnything3Net._apply_depth_alignment"
            }
          ]
        },
        {
          "title": "Unproject depth into the scene for export",
          "note": "The GLB exporter takes each retained homogeneous pixel [u,v,1], applies inverse intrinsics, and multiplies the resulting camera ray by camera-axis depth. It transforms that camera point with the inverse extrinsic matrix and pairs it with the same pixel’s RGB.",
          "blocks": [
            {
              "file": "src/depth_anything_3/utils/export/glb.py",
              "start": 238,
              "end": 251,
              "repo": "ByteDance-Seed/Depth-Anything-3",
              "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
              "text": "        d_flat = d.reshape(-1)\n        vidx = np.flatnonzero(valid.reshape(-1))\n\n        K_inv = np.linalg.inv(K[i])  # (3,3)\n        c2w = np.linalg.inv(_as_homogeneous44(ext_w2c[i]))  # (4,4)\n\n        rays = K_inv @ pix[vidx].T  # (3,M)\n        Xc = rays * d_flat[vidx][None, :]  # (3,M)\n        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])\n        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M,3)\n\n        cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)  # (M,3)\n\n        pts_all.append(Xw)",
              "symbol": "_depths_to_world_points_with_colors"
            }
          ]
        }
      ]
    },
    "mapanything": {
      "id": "mapanything",
      "repo": "facebookresearch/map-anything",
      "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
      "licenseFile": "LICENSE",
      "inputs": [
        "views[i]['img'] · [B, 3, H, W] — each of V view dictionaries carries a normalized image and its data_norm_type.",
        "ray_directions_cam, depth_along_ray · [B, H, W, 3], [B, H, W, 1] — optional raw-forward geometry. In infer, each camera-axis depth_z map requires intrinsics or ray_directions for that view.",
        "camera_pose_quats, camera_pose_trans · [B, 4], [B, 3] — optional camera-to-world pose inputs. When passing camera_poses to infer, include view zero’s pose. is_metric_scale flags metric inputs."
      ],
      "outputs": [
        "pts3d_cam, pts3d · [B, H, W, 3] each per view — local-camera XYZ and shared-world XYZ after learned metric scaling.",
        "ray_directions, depth_along_ray · [B, H, W, 3], [B, H, W, 1] per view — a unit camera ray and the metric distance along it.",
        "cam_quats, cam_trans, metric_scaling_factor · [B, 4], [B, 3], [B, 1] — raw-forward orientation, scaled translation and scale factor. Per-pixel confidence and non_ambiguous_mask identify how the model rates each point.",
        "depth_z, intrinsics, camera_poses · [B, H, W, 1], [B, 3, 3], [B, 4, 4] per view — the infer wrapper adds these fields during postprocessing."
      ],
      "steps": [
        {
          "title": "Run one RGB encoder over the view list",
          "note": "This traces the original v1 recipe. It chooses a DINOv2-Large image encoder, a 24-layer alternating-attention transformer and the DPT, pose and scale heads. RGB images concatenate along the batch axis for encoding, then split back into per-view features.",
          "blocks": [
            {
              "file": "configs/model/mapanything_v1.yaml",
              "start": 1,
              "end": 6,
              "repo": "facebookresearch/map-anything",
              "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
              "text": "defaults:\n  - default\n  - encoder: dinov2_large\n  - info_sharing: aat_ifr_24_layers\n  - pred_head: dpt_pose_scale\n  - task: images_only",
              "symbol": "mapanything_v1.yaml"
            },
            {
              "file": "mapanything/models/mapanything/model.py",
              "start": 669,
              "end": 679,
              "repo": "facebookresearch/map-anything",
              "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
              "text": "        num_views = len(views)\n        data_norm_type = views[0][\"data_norm_type\"][0]\n        imgs_list = [view[\"img\"] for view in views]\n        all_imgs_across_views = torch.cat(imgs_list, dim=0)\n        encoder_input = ViTEncoderInput(\n            image=all_imgs_across_views, data_norm_type=data_norm_type\n        )\n        encoder_output = self.encoder(encoder_input)\n        all_encoder_features_across_views = encoder_output.features.chunk(\n            num_views, dim=0\n        )",
              "symbol": "MapAnything._encode_n_views"
            }
          ]
        },
        {
          "title": "Fuse available geometry into each view",
          "note": "The infer wrapper can turn camera-axis depth plus a camera ray into distance along that ray. Inside the network, a dedicated encoder maps supplied ray directions into features. A per-sample mask gates their addition to visual features. The depth and pose branches similarly contribute when inputs are present.",
          "blocks": [
            {
              "file": "mapanything/utils/inference.py",
              "start": 235,
              "end": 243,
              "repo": "facebookresearch/map-anything",
              "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
              "text": "        # Step 2: Convert depth_z to depth_along_ray\n        if \"depth_z\" in view:\n            depth_z = view[\"depth_z\"]\n            ray_directions = processed_view[\"ray_directions\"]\n            ray_directions_unit_plane = ray_directions / ray_directions[..., 2:3]\n            pts3d_cam = depth_z[..., None] * ray_directions_unit_plane\n            depth_along_ray = torch.norm(pts3d_cam, dim=-1, keepdim=True)\n            processed_view[\"depth_along_ray\"] = depth_along_ray\n            del processed_view[\"depth_z\"]",
              "symbol": "preprocess_input_views_for_inference"
            },
            {
              "file": "mapanything/models/mapanything/model.py",
              "start": 855,
              "end": 867,
              "repo": "facebookresearch/map-anything",
              "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
              "text": "        # Encode the ray directions\n        ray_dirs_features_across_views = self.ray_dirs_encoder(\n            ViTEncoderNonImageInput(data=ray_dirs)\n        ).features\n\n        # Fuse the ray direction features with the other encoder features (zero out the features where the ray direction input mask is False)\n        ray_dirs_features_across_views = (\n            ray_dirs_features_across_views\n            * per_sample_ray_dirs_input_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)\n        )\n        all_encoder_features_across_views = (\n            all_encoder_features_across_views + ray_dirs_features_across_views\n        )",
              "symbol": "MapAnything._encode_and_fuse_ray_dirs"
            }
          ]
        },
        {
          "title": "Share scene information and estimate one scale",
          "note": "The v1 fusion module alternates attention within and across views. Its configuration distinguishes reference and source views and exposes intermediate layers for dense decoding. A learned scene-scale token travels through this same transformer alongside the visual features.",
          "blocks": [
            {
              "file": "configs/model/info_sharing/aat_ifr_24_layers.yaml",
              "start": 10,
              "end": 20,
              "repo": "facebookresearch/map-anything",
              "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
              "text": "  name: \"aat_24_layers_ifr\"\n  # Indices of the intermediate features to be shared (indices start from 0)\n  indices: [11, 17]\n  # Normalize intermediate features\n  norm_intermediate: True\n  # Size string\n  size: \"24_layers\"\n  # Depth\n  depth: 24\n  # Distinguish Reference and Non-Reference Views\n  distinguish_ref_and_non_ref_views: True",
              "symbol": "aat_ifr_24_layers.yaml"
            },
            {
              "file": "mapanything/models/mapanything/model.py",
              "start": 1583,
              "end": 1596,
              "repo": "facebookresearch/map-anything",
              "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
              "text": "        # Expand the scale token to match the batch size\n        input_scale_token = (\n            self.scale_token.unsqueeze(0)\n            .unsqueeze(-1)\n            .repeat(batch_size_per_view, 1, 1)\n        )  # (B, C, 1)\n\n        # Combine all images into view-centric representation\n        # Output is a list containing the encoded features for all N views after information sharing.\n        info_sharing_input = MultiViewTransformerInput(\n            features=all_encoder_features_across_views,\n            additional_input_tokens_per_view=all_encoder_registers_across_views,\n            additional_input_tokens=input_scale_token,\n        )",
              "symbol": "MapAnything.forward"
            },
            {
              "file": "mapanything/models/mapanything/model.py",
              "start": 1601,
              "end": 1605,
              "repo": "facebookresearch/map-anything",
              "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
              "text": "        elif self.info_sharing_return_type == \"intermediate_features\":\n            (\n                final_info_sharing_multi_view_feat,\n                intermediate_info_sharing_multi_view_feat,\n            ) = self.info_sharing(info_sharing_input)",
              "symbol": "MapAnything.forward"
            }
          ]
        },
        {
          "title": "Decode dense geometry, pose and scale",
          "note": "The DPT head reads features from four levels. Its adapter decodes unit rays, ray depth, confidence and a mask whose true values mark non-ambiguous pixels. The pose head reads the final spatial features. A third head reads the shared token and outputs a scalar that applies to the whole scene.",
          "blocks": [
            {
              "file": "configs/model/pred_head/dpt_pose_scale.yaml",
              "start": 4,
              "end": 16,
              "repo": "facebookresearch/map-anything",
              "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
              "text": "type: \"dpt+pose\"\nfeature_head:\n  feature_dim: 256\n  hooks: [0, 1, 2, 3]\n  checkpoint_gradient: ${model.pred_head.gradient_checkpointing}\nregressor_head:\n  output_dim: ${model.pred_head.adaptor_config.input_dim}\n  checkpoint_gradient: ${model.pred_head.gradient_checkpointing}\npose_head:\n  num_resconv_block: 2\n  rot_representation_dim: 4\nscale_head:\n  output_dim: 1",
              "symbol": "dpt_pose_scale.yaml"
            },
            {
              "file": "mapanything/models/mapanything/model.py",
              "start": 1364,
              "end": 1376,
              "repo": "facebookresearch/map-anything",
              "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
              "text": "        elif self.pred_head_type in [\"dpt\", \"dpt+pose\"]:\n            dense_head_outputs = self.dense_head(\n                PredictionHeadLayeredInput(\n                    list_features=dense_head_inputs,\n                    target_output_shape=img_shape,\n                )\n            )\n            dense_final_outputs = self.dense_adaptor(\n                AdaptorInput(\n                    adaptor_feature=dense_head_outputs.decoded_channels,\n                    output_shape_hw=img_shape,\n                )\n            )",
              "symbol": "MapAnything.downstream_dense_head"
            },
            {
              "file": "mapanything/models/mapanything/model.py",
              "start": 1516,
              "end": 1526,
              "repo": "facebookresearch/map-anything",
              "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
              "text": "        # Scale prediction is lightweight, so we can run it in one go\n        scale_head_output = self.scale_head(\n            PredictionHeadTokenInput(last_feature=scale_head_inputs)\n        )\n        scale_final_output = self.scale_adaptor(\n            AdaptorInput(\n                adaptor_feature=scale_head_output.decoded_channels,\n                output_shape_hw=img_shape,\n            )\n        )\n        scale_final_output = scale_final_output.value.squeeze(-1)  # (B, 1, 1) -> (B, 1)",
              "symbol": "MapAnything.downstream_head"
            }
          ]
        },
        {
          "title": "Construct local and world pointmaps",
          "note": "The raw dense representation splits into three ray channels and one distance. The pose splits into translation and quaternion. In the geometry helper, multiplying a unit ray by its distance gives camera XYZ. A homogeneous camera-to-world transform moves each point into the same scene frame.",
          "blocks": [
            {
              "file": "mapanything/models/mapanything/model.py",
              "start": 1753,
              "end": 1764,
              "repo": "facebookresearch/map-anything",
              "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
              "text": "                # Reshape output dense rep to (B * V, H, W, C)\n                output_dense_rep = dense_final_outputs.value.permute(\n                    0, 2, 3, 1\n                ).contiguous()\n                # Get the predicted ray directions and depths along rays\n                output_ray_directions, output_depth_along_ray = output_dense_rep.split(\n                    [3, 1], dim=-1\n                )\n                # Get the predicted camera translations and quaternions\n                output_cam_translations, output_cam_quats = (\n                    pose_final_outputs.value.split([3, 4], dim=-1)\n                )",
              "symbol": "MapAnything.forward"
            },
            {
              "file": "mapanything/utils/geometry.py",
              "start": 887,
              "end": 901,
              "repo": "facebookresearch/map-anything",
              "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
              "text": "    # Convert quaternions to rotation matrices (B x 3 x 3)\n    rot_mat = quaternion_to_rotation_matrix(pose_quats)\n\n    # Get pose matrix (B x 4 x 4)\n    pose_mat = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)\n    pose_mat[:, :3, :3] = rot_mat\n    pose_mat[:, :3, 3] = pose_trans\n\n    # Compute 3D points in local camera frame\n    pts3d_local = depth_along_ray * ray_directions\n\n    # Compute 3D points in world frame\n    pts3d_homo = torch.cat([pts3d_local, torch.ones_like(pts3d_local[..., :1])], dim=-1)\n    pts3d_world = ein.einsum(pose_mat, pts3d_homo, \"b i k, b h w k -> b h w i\")\n    pts3d_world = pts3d_world[..., :3]",
              "symbol": "convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap"
            }
          ]
        },
        {
          "title": "Return metric quantities for each view",
          "note": "The result dictionary applies the same learned scale to world points, camera-local points, distances and translations. Directions and rotations retain their original units. The higher-level infer wrapper also derives camera-axis depth and pinhole intrinsics from these outputs.",
          "blocks": [
            {
              "file": "mapanything/models/mapanything/model.py",
              "start": 1791,
              "end": 1804,
              "repo": "facebookresearch/map-anything",
              "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
              "text": "                    res.append(\n                        {\n                            \"pts3d\": output_pts3d_per_view[i]\n                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),\n                            \"pts3d_cam\": output_pts3d_cam_per_view[i]\n                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),\n                            \"ray_directions\": output_ray_directions_per_view[i],\n                            \"depth_along_ray\": output_depth_along_ray_per_view[i]\n                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),\n                            \"cam_trans\": output_cam_translations_per_view[i]\n                            * scale_final_output,\n                            \"cam_quats\": output_cam_quats_per_view[i],\n                            \"metric_scaling_factor\": scale_final_output,\n                        }",
              "symbol": "MapAnything.forward"
            },
            {
              "file": "mapanything/utils/inference.py",
              "start": 351,
              "end": 360,
              "repo": "facebookresearch/map-anything",
              "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
              "text": "        # 2. Add Z depth if we have camera frame points\n        if \"pts3d_cam\" in processed_output:\n            processed_output[\"depth_z\"] = processed_output[\"pts3d_cam\"][..., 2:3]\n\n        # 3. Recover pinhole camera intrinsics from ray directions if available\n        if \"ray_directions\" in processed_output:\n            intrinsics = recover_pinhole_intrinsics_from_ray_directions(\n                processed_output[\"ray_directions\"]\n            )\n            processed_output[\"intrinsics\"] = intrinsics",
              "symbol": "postprocess_model_outputs_for_inference"
            }
          ]
        }
      ]
    },
    "pi4": {
      "id": "pi4",
      "repo": "yyfz/Pi4",
      "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
      "licenseFile": "LICENSE",
      "inputs": [
        "images · [B, T, 3, H, W] — floating RGB in [0,1] on CUDA. H and W must be positive multiples of 16.",
        "query_frame · integer or [B] integers — optional source frame index per example. Tracking follows its pixel grid across at least two frames."
      ],
      "outputs": [
        "local_points, confidence · [B, T, H, W, 3], [B, T, H, W] — a point per pixel in its own camera frame at native scene scale, and its confidence.",
        "camera_poses, metric_scale · [B, T, 4, 4], [B] — camera-to-world transforms at native scale and one positive multiplier per example.",
        "tracks_2d, tracks_3d · [B, T, H, W, 2], [B, T, H, W, 3] — optional tracks indexed by query-frame pixels. The 2D values are target-image pixel coordinates. The 3D values are already in the metric world frame.",
        "dynamic_mask, visibility, tracking_confidence · per-pixel, per-frame probabilities when a query frame is supplied."
      ],
      "steps": [
        {
          "title": "Encode 16×16 RGB patches",
          "note": "Pi4 normalizes RGB, flattens batch and time together, and encodes each image independently. A stride-16 convolution projects patches to 1,024 channels. The encoder concatenates features from its twelfth and eighteenth transformer blocks and projects them back to 1,024 channels.",
          "blocks": [
            {
              "file": "pi4/models/pi4.py",
              "start": 79,
              "end": 90,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "        images = images.float()\n        mean = images.new_tensor((0.485, 0.456, 0.406)).reshape(1, 1, 3, 1, 1)\n        deviation = images.new_tensor((0.229, 0.224, 0.225)).reshape(1, 1, 3, 1, 1)\n        normalized = (images - mean) / deviation\n        positions = patch_positions(\n            batch_size * frames, height // 16, width // 16, images.device\n        )\n        with torch.autocast(\"cuda\", dtype=torch.bfloat16):\n            tokens = self.encoder(normalized.flatten(0, 1))\n            geometry, initial = self.geometry_decoder(\n                tokens, positions, batch_size, frames\n            )",
              "symbol": "Pi4.forward"
            },
            {
              "file": "pi4/models/encoder.py",
              "start": 12,
              "end": 18,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "        self.patch_embed = nn.Conv2d(3, 1024, 16, stride=16)\n        self.prefix_tokens = nn.Parameter(torch.empty(1, 5, 1024))\n        self.blocks = nn.ModuleList(\n            Block(1024, 16, epsilon=1e-5, scaled=True) for _ in range(18)\n        )\n        self.norm = nn.LayerNorm(1024, eps=1e-5)\n        self.projection = nn.Linear(2048, 1024)",
              "symbol": "Encoder.__init__"
            },
            {
              "file": "pi4/models/encoder.py",
              "start": 45,
              "end": 50,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "        intermediate = []\n        for index, block in enumerate(self.blocks):\n            features = block(features, rotary=rotary)\n            if index in (11, 17):\n                intermediate.append(self.norm(features)[:, 5:])\n        return self.projection(torch.cat(intermediate, dim=-1))",
              "symbol": "Encoder.forward"
            }
          ]
        },
        {
          "title": "Loop one frame block and one global block",
          "note": "Five learned register tokens join the patch tokens for each view. Sixteen iterations reuse the same two transformer blocks with step-dependent gates. Reshaping to [B×T, tokens, 1024] keeps attention inside a frame. Reshaping to [B, T×tokens, 1024] connects all frames.",
          "blocks": [
            {
              "file": "pi4/models/decoder.py",
              "start": 19,
              "end": 27,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "class GeometryDecoder(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.register_tokens = nn.Parameter(torch.empty(1, 1, 5, 1024))\n        self.frame_block = Block(1024, 16, normalize=True, scaled=True)\n        self.global_block = Block(1024, 16, normalize=True, scaled=True)\n        self.readout = Block(1024, 16, normalize=True, scaled=True)\n        self.frame_gates = TimeGates(16)\n        self.global_gates = TimeGates(16)",
              "symbol": "GeometryDecoder"
            },
            {
              "file": "pi4/models/decoder.py",
              "start": 39,
              "end": 54,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "        features = initial\n        token_count = features.shape[1]\n        for index in range(16):\n            features = self.frame_block(\n                features.reshape(batch_size * frames, token_count, 1024),\n                positions,\n                gates=self.frame_gates(index, batch_size * frames),\n            )\n            features = self.global_block(\n                features.reshape(batch_size, frames * token_count, 1024),\n                gates=self.global_gates(index, batch_size),\n            )\n        features = self.readout(\n            features.reshape(batch_size * frames, token_count, 1024), positions\n        )\n        return features, initial",
              "symbol": "GeometryDecoder.forward"
            }
          ]
        },
        {
          "title": "Decode and refine local XYZ",
          "note": "A convolutional pyramid recovers two camera-plane ray coordinates and log depth per pixel. The sparse refiner updates log depth using learned residuals. Exponentiating gives z, and concatenating the two ray coordinates times z with z itself gives camera-local XYZ.",
          "blocks": [
            {
              "file": "pi4/models/heads/point.py",
              "start": 32,
              "end": 44,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "class PointHead(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.points = ConvPyramid((512, 256, 128, 64, 32), 3)\n        self.confidence = ConfidenceHead()\n\n    def forward(self, tokens, height, width):\n        pyramid = feature_pyramid(tokens.float(), height, width, 5)\n        coordinates, prefix = self.points(pyramid)\n        logits = self.confidence(prefix, pyramid)[:, 0]\n        coordinates = coordinates.permute(0, 2, 3, 1)\n        rays, log_depth = coordinates[..., :2], coordinates[..., 2]\n        return rays, log_depth, logits, pyramid[0]",
              "symbol": "PointHead"
            },
            {
              "file": "pi4/models/refiner.py",
              "start": 163,
              "end": 169,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "            shape = torch.Size(\n                (batch_size, height, width, int(bins.max().item()) + 1, 3)\n            )\n            residual = self.network(features, coordinates, shape, condition)\n            log_depth = log_depth + residual.float().reshape_as(log_depth)\n        depth = log_depth.exp()[..., None]\n        return torch.cat((rays * depth, depth), dim=-1)",
              "symbol": "DepthRefiner.forward"
            },
            {
              "file": "pi4/models/pi4.py",
              "start": 94,
              "end": 103,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "            for frame_tokens in geometry[:, 5:].split(1):\n                rays, log_depth, confidence, feature = self.point_head(\n                    frame_tokens, height, width\n                )\n                coarse_depth = log_depth.exp().clamp_min(1e-6).log()\n                points.append(self.depth_refiner(rays, coarse_depth, feature))\n                confidences.append(confidence)\n            local_points = torch.cat(points).reshape(\n                batch_size, frames, height, width, 3\n            )",
              "symbol": "Pi4.forward"
            }
          ]
        },
        {
          "title": "Read cameras and the metric multiplier",
          "note": "The camera head fits a rotation and translation for each view. A separate metric head pools geometry across the whole set and exponentiates its scalar. Pi4 returns local points and camera poses at native scale. To place a point in metric world coordinates, apply the pose and multiply by that scalar.",
          "blocks": [
            {
              "file": "pi4/models/heads/camera.py",
              "start": 41,
              "end": 53,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "        raw_pose = self.pose(features.float())\n        rotation = raw_pose[:, 3:].reshape(-1, 3, 3)\n        left, _, right = torch.svd(F.normalize(rotation, dim=-1).transpose(-2, -1))\n        determinant = torch.det(right @ left.transpose(-2, -1))\n        rotation = torch.cat(\n            (right[:, :, :-1], right[:, :, -1:] * determinant[:, None, None]), dim=-1\n        ) @ left.transpose(-2, -1)\n        pose = torch.zeros(\n            raw_pose.shape[0], 4, 4, device=tokens.device, dtype=torch.float32\n        )\n        pose[:, :3, :3] = rotation\n        pose[:, :3, 3] = raw_pose[:, :3]\n        pose[:, 3, 3] = 1",
              "symbol": "CameraHead.forward"
            },
            {
              "file": "pi4/models/heads/metric.py",
              "start": 16,
              "end": 22,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "    def forward(self, tokens, positions, batch_size):\n        context = self.projects_y(tokens.reshape(batch_size, -1, 1024).float())\n        positions = positions.reshape(batch_size, -1, 2)\n        features = self.projects_x(self.token.expand(batch_size, -1, -1))\n        for block in self.blocks:\n            features = block(features, context, positions)\n        return self.output(features).reshape(batch_size).exp()",
              "symbol": "MetricHead.forward"
            },
            {
              "file": "pi4/models/pi4.py",
              "start": 114,
              "end": 119,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "        prediction = dict(\n            local_points=local_points,\n            camera_poses=cameras,\n            metric_scale=metric_scale,\n            confidence=confidence_logits.sigmoid(),\n        )",
              "symbol": "Pi4.forward"
            }
          ]
        },
        {
          "title": "Track a query frame through the other views",
          "note": "Supplying query_frame starts a separate eight-iteration motion decoder. Geometry features modulate its recurrent updates. The tracker predicts target pixel coordinates and visibility for the query-frame grid, while a separate head predicts moving-object masks.",
          "blocks": [
            {
              "file": "pi4/models/decoder.py",
              "start": 83,
              "end": 93,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "    def forward(self, initial, geometry, positions, batch_size, frames):\n        token_count = initial.shape[1]\n        features = initial\n        for index in range(8):\n            frame_geometry = geometry.reshape(batch_size * frames, token_count, 1024)\n            features = self.frame_block(\n                features.reshape_as(frame_geometry),\n                positions,\n                gates=self.frame_gates(index, batch_size * frames),\n                modulation=self.frame_condition(frame_geometry, index),\n            )",
              "symbol": "MotionDecoder.forward"
            },
            {
              "file": "pi4/models/pi4.py",
              "start": 121,
              "end": 131,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "        if queries is not None:\n            with torch.autocast(\"cuda\", dtype=torch.bfloat16):\n                motion = self.motion_decoder(\n                    initial, geometry, positions, batch_size, frames\n                )[:, 5:]\n                del initial, geometry\n                tracks, visibility, track_confidence = self.tracker(\n                    motion, normalized, queries\n                )\n                mask = self.motion_mask(motion, normalized.flatten(0, 1))\n            prediction.update(",
              "symbol": "Pi4.forward"
            },
            {
              "file": "pi4/models/pi4.py",
              "start": 132,
              "end": 141,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "                tracks_2d=tracks,\n                visibility=visibility.sigmoid(),\n                tracking_confidence=track_confidence.sigmoid(),\n                dynamic_mask=mask.reshape(batch_size, frames, height, width).sigmoid(),\n                query_frame=queries,\n            )\n            prediction[\"tracks_3d\"] = self.lift_tracks(\n                tracks, local_points, cameras, metric_scale\n            )\n        return prediction",
              "symbol": "Pi4.forward"
            }
          ]
        },
        {
          "title": "Lift each 2D correspondence into metric 3D",
          "note": "For each target frame, lift_tracks converts the tracked pixel to normalized grid coordinates and samples that frame’s local pointmap. It rotates and translates the sample with the target camera pose, then applies metric_scale. Out-of-bounds target pixels produce NaNs.",
          "blocks": [
            {
              "file": "pi4/models/pi4.py",
              "start": 143,
              "end": 154,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "    @staticmethod\n    def lift_tracks(tracks, local_points, camera_poses, metric_scale):\n        batch_size, frames, height, width, _ = local_points.shape\n        grid = (tracks + 0.5) * tracks.new_tensor((2 / width, 2 / height)) - 1\n        sampled = F.grid_sample(\n            local_points.flatten(0, 1).permute(0, 3, 1, 2),\n            grid.flatten(0, 1),\n            align_corners=False,\n        )\n        sampled = sampled.permute(0, 2, 3, 1).reshape(\n            batch_size, frames, height, width, 3\n        )",
              "symbol": "Pi4.lift_tracks"
            },
            {
              "file": "pi4/models/pi4.py",
              "start": 155,
              "end": 165,
              "repo": "yyfz/Pi4",
              "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
              "text": "        world = torch.einsum(\"btij,bthwj->bthwi\", camera_poses[..., :3, :3], sampled)\n        world = (world + camera_poses[..., :3, 3][:, :, None, None]) * metric_scale[\n            :, None, None, None, None\n        ]\n        inside = (\n            (tracks[..., 0] >= 0)\n            & (tracks[..., 0] \u003c= width - 1)\n            & (tracks[..., 1] >= 0)\n            & (tracks[..., 1] \u003c= height - 1)\n        )\n        return world.masked_fill(~inside[..., None], float(\"nan\"))",
              "symbol": "Pi4.lift_tracks"
            }
          ]
        }
      ]
    }
  },
  "licenses": {
    "nv-tlabs/dvlt": {
      "file": "LICENSE",
      "commit": "d9f1c0c481eea8b6a399b507b30c9f484e99fd69",
      "notices": [
        {
          "name": "LICENSE",
          "text": "Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n\nThis repository is primarily licensed under the Apache License, Version 2.0.\nPortions of the repository incorporate or adapt third-party code and remain\nsubject to the third-party copyright and license notices preserved in the\nrelevant source files and in THIRD_PARTY_LICENSES.md. The VGGT-derived files\nidentified in THIRD_PARTY_LICENSES.md are distributed under the VGGT License;\nsee LICENSES/VGGT-LICENSE.txt.\n\nThe model weights (the nvidia/dvlt checkpoint) are not covered by this license.\nThey are released separately under the NVIDIA License (non-commercial; research\nand evaluation use only); see LICENSES/NVIDIA-LICENSE.txt.\n\n                                 Apache License\n                           Version 2.0, January 2004\n                        http://www.apache.org/licenses/\n\n   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n\n   1. Definitions.\n\n      \"License\" shall mean the terms and conditions for use, reproduction,\n      and distribution as defined by Sections 1 through 9 of this document.\n\n      \"Licensor\" shall mean the copyright owner or entity authorized by\n      the copyright owner that is granting the License.\n\n      \"Legal Entity\" shall mean the union of the acting entity and all\n      other entities that control, are controlled by, or are under common\n      control with that entity. For the purposes of this definition,\n      \"control\" means (i) the power, direct or indirect, to cause the\n      direction or management of such entity, whether by contract or\n      otherwise, or (ii) ownership of fifty percent (50%) or more of the\n      outstanding shares, or (iii) beneficial ownership of such entity.\n\n      \"You\" (or \"Your\") shall mean an individual or Legal Entity\n      exercising permissions granted by this License.\n\n      \"Source\" form shall mean the preferred form for making modifications,\n      including but not limited to software source code, documentation\n      source, and configuration files.\n\n      \"Object\" form shall mean any form resulting from mechanical\n      transformation or translation of a Source form, including but\n      not limited to compiled object code, generated documentation,\n      and conversions to other media types.\n\n      \"Work\" shall mean the work of authorship, whether in Source or\n      Object form, made available under the License, as indicated by a\n      copyright notice that is included in or attached to the work\n      (an example is provided in the Appendix below).\n\n      \"Derivative Works\" shall mean any work, whether in Source or Object\n      form, that is based on (or derived from) the Work and for which the\n      editorial revisions, annotations, elaborations, or other modifications\n      represent, as a whole, an original work of authorship. For the purposes\n      of this License, Derivative Works shall not include works that remain\n      separable from, or merely link (or bind by name) to the interfaces of,\n      the Work and Derivative Works thereof.\n\n      \"Contribution\" shall mean any work of authorship, including\n      the original version of the Work and any modifications or additions\n      to that Work or Derivative Works thereof, that is intentionally\n      submitted to Licensor for inclusion in the Work by the copyright owner\n      or by an individual or Legal Entity authorized to submit on behalf of\n      the copyright owner. For the purposes of this definition, \"submitted\"\n      means any form of electronic, verbal, or written communication sent\n      to the Licensor or its representatives, including but not limited to\n      communication on electronic mailing lists, source code control systems,\n      and issue tracking systems that are managed by, or on behalf of, the\n      Licensor for the purpose of discussing and improving the Work, but\n      excluding communication that is conspicuously marked or otherwise\n      designated in writing by the copyright owner as \"Not a Contribution.\"\n\n      \"Contributor\" shall mean Licensor and any individual or Legal Entity\n      on behalf of whom a Contribution has been received by Licensor and\n      subsequently incorporated within the Work.\n\n   2. Grant of Copyright License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      copyright license to reproduce, prepare Derivative Works of,\n      publicly display, publicly perform, sublicense, and distribute the\n      Work and such Derivative Works in Source or Object form.\n\n   3. Grant of Patent License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      (except as stated in this section) patent license to make, have made,\n      use, offer to sell, sell, import, and otherwise transfer the Work,\n      where such license applies only to those patent claims licensable\n      by such Contributor that are necessarily infringed by their\n      Contribution(s) alone or by combination of their Contribution(s)\n      with the Work to which such Contribution(s) was submitted. If You\n      institute patent litigation against any entity (including a\n      cross-claim or counterclaim in a lawsuit) alleging that the Work\n      or a Contribution incorporated within the Work constitutes direct\n      or contributory patent infringement, then any patent licenses\n      granted to You under this License for that Work shall terminate\n      as of the date such litigation is filed.\n\n   4. Redistribution. You may reproduce and distribute copies of the\n      Work or Derivative Works thereof in any medium, with or without\n      modifications, and in Source or Object form, provided that You\n      meet the following conditions:\n\n      (a) You must give any other recipients of the Work or\n          Derivative Works a copy of this License; and\n\n      (b) You must cause any modified files to carry prominent notices\n          stating that You changed the files; and\n\n      (c) You must retain, in the Source form of any Derivative Works\n          that You distribute, all copyright, patent, trademark, and\n          attribution notices from the Source form of the Work,\n          excluding those notices that do not pertain to any part of\n          the Derivative Works; and\n\n      (d) If the Work includes a \"NOTICE\" text file as part of its\n          distribution, then any Derivative Works that You distribute must\n          include a readable copy of the attribution notices contained\n          within such NOTICE file, excluding those notices that do not\n          pertain to any part of the Derivative Works, in at least one\n          of the following places: within a NOTICE text file distributed\n          as part of the Derivative Works; within the Source form or\n          documentation, if provided along with the Derivative Works; or,\n          within a display generated by the Derivative Works, if and\n          wherever such third-party notices normally appear. The contents\n          of the NOTICE file are for informational purposes only and\n          do not modify the License. You may add Your own attribution\n          notices within Derivative Works that You distribute, alongside\n          or as an addendum to the NOTICE text from the Work, provided\n          that such additional attribution notices cannot be construed\n          as modifying the License.\n\n      You may add Your own copyright statement to Your modifications and\n      may provide additional or different license terms and conditions\n      for use, reproduction, or distribution of Your modifications, or\n      for any such Derivative Works as a whole, provided Your use,\n      reproduction, and distribution of the Work otherwise complies with\n      the conditions stated in this License.\n\n   5. Submission of Contributions. Unless You explicitly state otherwise,\n      any Contribution intentionally submitted for inclusion in the Work\n      by You to the Licensor shall be under the terms and conditions of\n      this License, without any additional terms or conditions.\n      Notwithstanding the above, nothing herein shall supersede or modify\n      the terms of any separate license agreement you may have executed\n      with Licensor regarding such Contributions.\n\n   6. Trademarks. This License does not grant permission to use the trade\n      names, trademarks, service marks, or product names of the Licensor,\n      except as required for reasonable and customary use in describing the\n      origin of the Work and reproducing the content of the NOTICE file.\n\n   7. Disclaimer of Warranty. Unless required by applicable law or\n      agreed to in writing, Licensor provides the Work (and each\n      Contributor provides its Contributions) on an \"AS IS\" BASIS,\n      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or\n      implied, including, without limitation, any warranties or conditions\n      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A\n      PARTICULAR PURPOSE. You are solely responsible for determining the\n      appropriateness of using or redistributing the Work and assume any\n      risks associated with Your exercise of permissions under this License.\n\n   8. Limitation of Liability. In no event and under no legal theory,\n      whether in tort (including negligence), contract, or otherwise,\n      unless required by applicable law (such as deliberate and grossly\n      negligent acts) or agreed to in writing, shall any Contributor be\n      liable to You for damages, including any direct, indirect, special,\n      incidental, or consequential damages of any character arising as a\n      result of this License or out of the use or inability to use the\n      Work (including but not limited to damages for loss of goodwill,\n      work stoppage, computer failure or malfunction, or any and all\n      other commercial damages or losses), even if such Contributor\n      has been advised of the possibility of such damages.\n\n   9. Accepting Warranty or Additional Liability. While redistributing\n      the Work or Derivative Works thereof, You may choose to offer,\n      and charge a fee for, acceptance of support, warranty, indemnity,\n      or other liability obligations and/or rights consistent with this\n      License. However, in accepting such obligations, You may act only\n      on Your own behalf and on Your sole responsibility, not on behalf\n      of any other Contributor, and only if You agree to indemnify,\n      defend, and hold each Contributor harmless for any liability\n      incurred by, or claims asserted against, such Contributor by reason\n      of your accepting any such warranty or additional liability.\n\n   END OF TERMS AND CONDITIONS\n\n   APPENDIX: How to apply the Apache License to your work.\n\n      To apply the Apache License to your work, attach the following\n      boilerplate notice, with the fields enclosed by brackets \"[]\"\n      replaced with your own identifying information. (Don't include\n      the brackets!)  The text should be enclosed in the appropriate\n      comment syntax for the file format. We also recommend that a\n      file or class name and description of purpose be included on the\n      same \"printed page\" as the copyright notice for easier\n      identification within third-party archives.\n\n   Copyright [yyyy] [name of copyright owner]\n\n   Licensed under the Apache License, Version 2.0 (the \"License\");\n   you may not use this file except in compliance with the License.\n   You may obtain a copy of the License at\n\n       http://www.apache.org/licenses/LICENSE-2.0\n\n   Unless required by applicable law or agreed to in writing, software\n   distributed under the License is distributed on an \"AS IS\" BASIS,\n   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n   See the License for the specific language governing permissions and\n   limitations under the License.\n"
        },
        {
          "name": "THIRD_PARTY_LICENSES.md · MoGe",
          "text": "# MoGe\n\nSource: https://github.com/microsoft/MoGe\n\n**Adapted files:**\n- `src/dvlt/model_components/pose_encoding.py` — `create_uv_grid` is a direct\n  port of MoGe's `normalized_view_plane_uv` helper\n  ([`moge/utils/geometry_torch.py`](https://github.com/microsoft/MoGe/blob/main/moge/utils/geometry_torch.py)).\n\n**Upstream copyright:** Copyright (c) Microsoft Corporation.\n\n```\nMIT License\n\nCopyright (c) Microsoft Corporation.\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the \"Software\"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n```"
        },
        {
          "name": "src/dvlt/model/dvlt/blocks.py · source header",
          "text": "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0"
        },
        {
          "name": "src/dvlt/model/dvlt/heads.py · source header",
          "text": "# SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.\n# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n#\n# ResidualConvBlock and DecoderHead (conv mode) are adapted from MoGe\n# (https://github.com/microsoft/MoGe), distributed by Microsoft under the MIT\n# License. See THIRD_PARTY_LICENSES.md for the full license text."
        }
      ],
      "url": "https://github.com/nv-tlabs/dvlt/blob/d9f1c0c481eea8b6a399b507b30c9f484e99fd69/LICENSE"
    },
    "facebookresearch/vggt-omega": {
      "file": "LICENSE",
      "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
      "notices": [
        {
          "name": "LICENSE",
          "text": "FAIR Noncommercial Research License\nv1 Last Updated: October 16, 2024\n\n“Acceptable Use Policy” means the FAIR Acceptable Use Policy, applicable to Research Materials, that is incorporated into this Agreement.\n\n“Agreement” means the terms and conditions for use, reproduction, distribution and modification of the Research Materials set forth herein.\n\n\n“Documentation” means the specifications, manuals and documentation accompanying\nResearch Materials distributed by Meta.\n\n\n“Licensee” or “you” means you, or your employer or any other person or entity (if you are entering into this Agreement on such person or entity’s behalf), of the age required under applicable laws, rules or regulations to provide legal consent and that has legal authority to bind your employer or such other person or entity if you are entering in this Agreement on their behalf.\n\n\n“Meta” or “we” means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your principal place of business is in the EEA or Switzerland) and Meta Platforms, Inc. (if you are located outside of the EEA or Switzerland).\n\n“Noncommercial Research Uses” means noncommercial research use cases related to research, development, education, processing, or analysis and in each case, is not primarily intended for commercial advantage or monetary compensation to you or others.\n\n“Research Materials” means, collectively, Documentation and the models, software and algorithms, including machine-learning model code, trained model weights, inference-enabling code, training-enabling code, fine-tuning enabling code, demonstration materials and other elements of the foregoing distributed by Meta and made available under this Agreement.\n\nBy clicking “I Accept” below or by using or distributing any portion or element of the Research Materials, you agree to be bound by this Agreement.\n\n\n1. License Rights and Redistribution.\n\n    a. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under Meta’s intellectual property or other rights owned by Meta embodied in the Research Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Research Materials.\n\n    b. Redistribution and Use.\n        i. You will not use the Research Materials or any outputs or results of the Research Materials in connection with any commercial uses or for any uses other than Noncommercial Research Uses;\n\n        ii. Distribution of Research Materials, and any derivative works thereof, are subject to the terms of this Agreement. If you distribute or make the Research Materials, or any derivative works thereof, available to a third party, you may only do so under the terms of this Agreement. You shall also provide a copy of this Agreement to such third party.\n\n        iii.  If you submit for publication the results of research you perform on, using, or otherwise in connection with Research Materials, you must acknowledge the use of Research Materials in your publication.\n\n        iv. Your use of the Research Materials must comply with applicable laws and regulations (including Trade Control Laws) and adhere to the FAIR Acceptable Use Policy, which is hereby incorporated by reference into this Agreement.\n\n2. User Support. Your Noncommercial Research Use of the Research Materials is done at your own discretion; Meta does not process any information nor provide any service in relation to such use.  Meta is under no obligation to provide any support services for the Research Materials. Any support provided is “as is”, “with all faults”, and without warranty of any kind.\n\n3. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE RESEARCH MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE RESEARCH MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE RESEARCH MATERIALS AND ANY OUTPUT AND RESULTS.\n\n4. Limitation of Liability. IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY DIRECT OR INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.\n\n5. Intellectual Property.\n\n    a. Subject to Meta’s ownership of Research Materials and derivatives made by or for Meta, with respect to any derivative works and modifications of the Research Materials that are made by you, as between you and Meta, you are and will be the owner of such derivative works and modifications.\n\n    b. If you institute litigation or other proceedings against Meta or any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Research Materials, outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless Meta from and against any claim by any third party arising out of or related to your use or distribution of the Research Materials.\n\n6. Term and Termination. The term of this Agreement will commence upon your acceptance of this Agreement or access to the Research Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of the Research Materials. Sections 5, 6 and 9 shall survive the termination of this Agreement.\n\n7. Governing Law and Jurisdiction. This Agreement will be governed and construed under the laws of the State of California without regard to choice of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement. The courts of California shall have exclusive jurisdiction of any dispute arising out of this Agreement.\n\n8. Modifications and Amendments. Meta may modify this Agreement from time to time; provided that they are similar in spirit to the current version of the Agreement, but may differ in detail to address new problems or concerns. All such changes will be effective immediately. Your continued use of the Research Materials after any modification to this Agreement constitutes your agreement to such modification. Except as provided in this Agreement, no modification or addition to any provision of this Agreement will be binding unless it is in writing and signed by an authorized representative of both you and Meta.\n\n\nFAIR Acceptable Use Policy\n\nThe Fundamental AI Research (FAIR) team at Meta seeks to further understanding of new and existing research domains with the mission of advancing the state-of-the-art in artificial intelligence through open research for the benefit of all.\n\nAs part of this mission, Meta makes certain research materials available for noncommercial research use. Meta is committed to promoting the safe and responsible use of such research materials.\n\nProhibited Uses\n\nYou agree you will not use, or allow others to use, Research Materials to:\n\n1. Violate the law or others’ rights, including to:\n    a. Engage in, promote, generate, contribute to, encourage, plan, incite, or further illegal or unlawful activity or content, such as:\n        i. Violence or terrorism\n\n        ii. Exploitation or harm to children, including the solicitation, creation, acquisition, or dissemination of child exploitative content or failure to report Child Sexual Abuse Material\n\n        iii. Human trafficking, exploitation, and sexual violence\n\n        iv. The illegal distribution of information or materials to minors, including obscene materials, or failure to employ legally required age-gating in connection with such information or materials.\n\n        v. Sexual solicitation\n\n        vi.Any other criminal activity\n\n    b. Engage in, promote, incite, or facilitate the harassment, abuse, threatening, or bullying of individuals or groups of individuals\n\n    c. Engage in, promote, incite, or facilitate discrimination or other unlawful or harmful conduct in the provision of employment, employment benefits, credit, housing, other economic benefits, or other essential goods and services\n\n    d. Engage in the unauthorized or unlicensed practice of any profession including, but not limited to, financial, legal, medical/health, or related professional practices\n\n    e. Collect, process, disclose, generate, or infer health, demographic, or other sensitive personal or private information about individuals without rights and consents required by applicable laws\n\n    f. Engage in or facilitate any action or generate any content that infringes, misappropriates, or otherwise violates any third-party rights, including the outputs or results of any technology using FAIR research materials\n\n    g. Create, generate, or facilitate the creation of malicious code, malware, computer viruses or do anything else that could disable, overburden, interfere with or impair the proper working, integrity, operation or appearance of a website or computer system\n\n2. Engage in, promote, incite, facilitate, or assist in the planning or development of activities that present a risk of death or bodily harm to individuals, including use of research artifacts related to the following:\n\n    a. Military, warfare, nuclear industries or applications, espionage, use for materials or activities that are subject to the International Traffic Arms Regulations (ITAR) maintained by the United States Department of State\n\n    b. Guns and illegal weapons (including weapon development)\n\n    c. Illegal drugs and regulated/controlled substances\n\n    d. Operation of critical infrastructure, transportation technologies, or heavy machinery\n\n    e. Self-harm or harm to others, including suicide, cutting, and eating disorders\n\n    f. Any content intended to incite or promote violence, abuse, or any infliction of bodily harm to an individual\n\n3. Intentionally deceive or mislead others, including use of FAIR Research Materials related to the following:\n\n    a. Generating, promoting, or furthering fraud or the creation or promotion of disinformation\n\n    b. Generating, promoting, or furthering defamatory content, including the creation of defamatory statements, images, or other content\n\n    c. Generating, promoting, or further distributing spam\n\n    d. Impersonating another individual without consent, authorization, or legal right\n\n    e. Representing that outputs of FAIR research materials or outputs from technology using FAIR research materials are human-generated\n\n    f. Generating or facilitating false online engagement, including fake reviews and other means of fake online engagement\n\n4. Fail to appropriately disclose to end users any known dangers of your Research Materials.\n\nPlease report any violation of this Policy or other problems that could lead to a violation of this Policy by submitting a report here [https://docs.google.com/forms/d/e/1FAIpQLSeb11cryAopJ7LNrC4nxEUXrHY26hfkXQMf_uH-oFgA3WlYZQ/viewform].\n"
        },
        {
          "name": "demo_gradio.py · source header",
          "text": "# Copyright (c) Meta Platforms, Inc. and affiliates.\n# All rights reserved.\n#\n# This source code is licensed under the license found in the\n# LICENSE file in the root directory of this source tree."
        },
        {
          "name": "vggt_omega/models/heads/dense_head.py · source header",
          "text": "# Copyright (c) Meta Platforms, Inc. and affiliates.\n# All rights reserved.\n#\n# This source code is licensed under the license found in the\n# LICENSE file in the root directory of this source tree.\n\n# Inspired by https://github.com/DepthAnything/Depth-Anything-V2"
        }
      ],
      "url": "https://github.com/facebookresearch/vggt-omega/blob/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/LICENSE"
    },
    "yyfz/Pi3": {
      "file": "LICENSE",
      "commit": "a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99",
      "notices": [
        {
          "name": "LICENSE",
          "text": "Copyright (c) 2025, the authors\nAll rights reserved.\n\nRedistribution and use in source and binary forms, with or without\nmodification, are permitted provided that the following conditions are met:\n\n1. Redistributions of source code must retain the above copyright notice, this\n   list of conditions and the following disclaimer.\n\n2. Redistributions in binary form must reproduce the above copyright notice,\n   this list of conditions and the following disclaimer in the documentation\n   and/or other materials provided with the distribution.\n\n3. Neither the name of the copyright holder nor the names of its\n   contributors may be used to endorse or promote products derived from\n   this software without specific prior written permission.\n\nTHIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\"\nAND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE\nIMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE\nDISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE\nFOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL\nDAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR\nSERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER\nCAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,\nOR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE\nOF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
        }
      ],
      "url": "https://github.com/yyfz/Pi3/blob/a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99/LICENSE"
    },
    "naver/dust3r": {
      "file": "LICENSE",
      "commit": "4c24a6ebf04809f2cfe59915e51779c8984aaa40",
      "notices": [
        {
          "name": "LICENSE",
          "text": "DUSt3R, Copyright (c) 2024-present Naver Corporation, is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license.\n\nA summary of the CC BY-NC-SA 4.0 license is located here:\n\thttps://creativecommons.org/licenses/by-nc-sa/4.0/\n\nThe CC BY-NC-SA 4.0 license is located here:\n\thttps://creativecommons.org/licenses/by-nc-sa/4.0/legalcode\n"
        },
        {
          "name": "NOTICE",
          "text": "DUSt3R\nCopyright 2024-present NAVER Corp.\n\nThis project contains subcomponents with separate copyright notices and license terms. \nYour use of the source code for these subcomponents is subject to the terms and conditions of the following licenses.\n\n====\n\nnaver/croco\nhttps://github.com/naver/croco/\n\nCreative Commons Attribution-NonCommercial-ShareAlike 4.0\n"
        },
        {
          "name": "dust3r/demo.py · source header",
          "text": "# Copyright (C) 2024-present Naver Corporation. All rights reserved.\n# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).\n#\n# --------------------------------------------------------\n# gradio demo\n# --------------------------------------------------------"
        },
        {
          "name": "dust3r/heads/dpt_head.py · source header",
          "text": "# Copyright (C) 2024-present Naver Corporation. All rights reserved.\n# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).\n#\n# --------------------------------------------------------\n# dpt head implementation for DUST3R\n# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;\n# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True\n# the forward function also takes as input a dictionnary img_info with key \"height\" and \"width\"\n# for PixelwiseTask, the output will be of dimension B x num_channels x H x W\n# --------------------------------------------------------"
        },
        {
          "name": "dust3r/heads/postprocess.py · source header",
          "text": "# Copyright (C) 2024-present Naver Corporation. All rights reserved.\n# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).\n#\n# --------------------------------------------------------\n# post process function for all heads: extract 3D points/confidence from output\n# --------------------------------------------------------"
        },
        {
          "name": "dust3r/model.py · source header",
          "text": "# Copyright (C) 2024-present Naver Corporation. All rights reserved.\n# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).\n#\n# --------------------------------------------------------\n# DUSt3R model class\n# --------------------------------------------------------"
        },
        {
          "name": "dust3r/utils/image.py · source header",
          "text": "# Copyright (C) 2024-present Naver Corporation. All rights reserved.\n# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).\n#\n# --------------------------------------------------------\n# utilitary functions about images (loading/converting...)\n# --------------------------------------------------------"
        }
      ],
      "url": "https://github.com/naver/dust3r/blob/4c24a6ebf04809f2cfe59915e51779c8984aaa40/LICENSE"
    },
    "naver/mast3r": {
      "file": "LICENSE",
      "commit": "f5209afc300cec36239a7ac992263f36847bbba0",
      "notices": [
        {
          "name": "LICENSE",
          "text": "MASt3R, Copyright (c) 2024-present Naver Corporation, is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license.\n\nA summary of the CC BY-NC-SA 4.0 license is located here:\n\thttps://creativecommons.org/licenses/by-nc-sa/4.0/\n\nThe CC BY-NC-SA 4.0 license is located here:\n\thttps://creativecommons.org/licenses/by-nc-sa/4.0/legalcode\n"
        },
        {
          "name": "NOTICE",
          "text": "MASt3R\nCopyright 2024-present NAVER Corp.\n\nThis project contains subcomponents with separate copyright notices and license terms. \nYour use of the source code for these subcomponents is subject to the terms and conditions of the following licenses.\n\n====\n\nnaver/dust3r\nhttps://github.com/naver/dust3r/\n\nCreative Commons Attribution-NonCommercial-ShareAlike 4.0\n\n====\n\nnaver/croco\nhttps://github.com/naver/croco/\n\nCreative Commons Attribution-NonCommercial-ShareAlike 4.0\n\n====\n\npytorch/pytorch\nhttps://github.com/pytorch/pytorch\n\nFrom PyTorch:\n\nCopyright (c) 2016-     Facebook, Inc            (Adam Paszke)\nCopyright (c) 2014-     Facebook, Inc            (Soumith Chintala)\nCopyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)\nCopyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)\nCopyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)\nCopyright (c) 2011-2013 NYU                      (Clement Farabet)\nCopyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)\nCopyright (c) 2006      Idiap Research Institute (Samy Bengio)\nCopyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)\n\nFrom Caffe2:\n\nCopyright (c) 2016-present, Facebook Inc. All rights reserved.\n\nAll contributions by Facebook:\nCopyright (c) 2016 Facebook Inc.\n\nAll contributions by Google:\nCopyright (c) 2015 Google Inc.\nAll rights reserved.\n\nAll contributions by Yangqing Jia:\nCopyright (c) 2015 Yangqing Jia\nAll rights reserved.\n\nAll contributions by Kakao Brain:\nCopyright 2019-2020 Kakao Brain\n\nAll contributions by Cruise LLC:\nCopyright (c) 2022 Cruise LLC.\nAll rights reserved.\n\nAll contributions from Caffe:\nCopyright(c) 2013, 2014, 2015, the respective contributors\nAll rights reserved.\n\nAll other contributions:\nCopyright(c) 2015, 2016 the respective contributors\nAll rights reserved.\n\nCaffe2 uses a copyright model similar to Caffe: each contributor holds\ncopyright over their contributions to Caffe2. The project versioning records\nall such contribution and copyright details. If a contributor wants to further\nmark their specific copyright on a particular contribution, they should\nindicate their copyright solely in the commit message of the change when it is\ncommitted.\n\nAll rights reserved.\n\nRedistribution and use in source and binary forms, with or without\nmodification, are permitted provided that the following conditions are met:\n\n1. Redistributions of source code must retain the above copyright\n   notice, this list of conditions and the following disclaimer.\n\n2. Redistributions in binary form must reproduce the above copyright\n   notice, this list of conditions and the following disclaimer in the\n   documentation and/or other materials provided with the distribution.\n\n3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America\n   and IDIAP Research Institute nor the names of its contributors may be\n   used to endorse or promote products derived from this software without\n   specific prior written permission.\n\nTHIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\"\nAND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE\nIMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE\nARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE\nLIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR\nCONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF\nSUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS\nINTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN\nCONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)\nARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE\nPOSSIBILITY OF SUCH DAMAGE.\n\n====\ngtolias/how\nhttps://github.com/gtolias/how\n\nMIT License https://github.com/gtolias/how/blob/master/LICENSE"
        },
        {
          "name": "mast3r/catmlp_dpt_head.py · source header",
          "text": "# Copyright (C) 2024-present Naver Corporation. All rights reserved.\n# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).\n#\n# --------------------------------------------------------\n# MASt3R heads\n# --------------------------------------------------------"
        },
        {
          "name": "mast3r/fast_nn.py · source header",
          "text": "# Copyright (C) 2024-present Naver Corporation. All rights reserved.\n# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).\n#\n# --------------------------------------------------------\n# MASt3R Fast Nearest Neighbor\n# --------------------------------------------------------"
        },
        {
          "name": "mast3r/model.py · source header",
          "text": "# Copyright (C) 2024-present Naver Corporation. All rights reserved.\n# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).\n#\n# --------------------------------------------------------\n# MASt3R model class\n# --------------------------------------------------------"
        }
      ],
      "url": "https://github.com/naver/mast3r/blob/f5209afc300cec36239a7ac992263f36847bbba0/LICENSE"
    },
    "facebookresearch/vggt": {
      "file": "LICENSE.txt",
      "commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
      "notices": [
        {
          "name": "LICENSE.txt",
          "text": "VGGT License\n\nv1 Last Updated: July 29, 2025\n\n“Acceptable Use Policy” means the Acceptable Use Policy, applicable to Research Materials, that is incorporated into this Agreement.\n\n“Agreement” means the terms and conditions for use, reproduction, distribution and modification of the Research Materials set forth herein.\n\n\n“Documentation” means the specifications, manuals and documentation accompanying \nResearch Materials distributed by Meta.\n\n\n“Licensee” or “you” means you, or your employer or any other person or entity (if you are entering into this Agreement on such person or entity’s behalf), of the age required under applicable laws, rules or regulations to provide legal consent and that has legal authority to bind your employer or such other person or entity if you are entering in this Agreement on their behalf.\n\n“Meta” or “we” means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your principal place of business is in the EEA or Switzerland) and Meta Platforms, Inc. (if you are located outside of the EEA or Switzerland).\n“Research Materials” means, collectively, Documentation and the models, software and algorithms, including machine-learning model code, trained model weights, inference-enabling code, training-enabling code, fine-tuning enabling code, demonstration materials and other elements of the foregoing distributed by Meta and made available under this Agreement.\n\nBy clicking “I Accept” below or by using or distributing any portion or element of the Research Materials, you agree to be bound by this Agreement.\n\n\n1. License Rights and Redistribution.\n\n\na. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under Meta’s intellectual property or other rights owned by Meta embodied in the Research Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Research Materials.  \n \nb. Redistribution and Use.  \n\n\ni. Distribution of Research Materials, and any derivative works thereof, are subject to the terms of this Agreement. If you distribute or make the Research Materials, or any derivative works thereof, available to a third party, you may only do so under the terms of this Agreement. You shall also provide a copy of this Agreement to such third party.\n\n\nii.  If you submit for publication the results of research you perform on, using, or otherwise in connection with Research Materials, you must acknowledge the use of Research Materials in your publication.\n\n\niii. Your use of the Research Materials must comply with applicable laws and regulations (including Trade Control Laws) and adhere to the Acceptable Use Policy, which is hereby incorporated by reference into this Agreement.\n2. User Support. Your  use of the Research Materials is done at your own discretion; Meta does not process any information nor provide any service in relation to such use.  Meta is under no obligation to provide any support services for the Research Materials. Any support provided is “as is”, “with all faults”, and without warranty of any kind.\n\n\n3. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE RESEARCH MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE RESEARCH MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE RESEARCH MATERIALS AND ANY OUTPUT AND RESULTS.\n \n4. Limitation of Liability. IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY DIRECT OR INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.\n \n5. Intellectual Property.\n\n\na. Subject to Meta’s ownership of Research Materials and derivatives made by or for Meta, with respect to any derivative works and modifications of the Research Materials that are made by you, as between you and Meta, you are and will be the owner of such derivative works and modifications.\n \nb. If you institute litigation or other proceedings against Meta or any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Research Materials, outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless Meta from and against any claim by any third party arising out of or related to your use or distribution of the Research Materials.\n \n6. Term and Termination. The term of this Agreement will commence upon your acceptance of this Agreement or access to the Research Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of the Research Materials. Sections 5, 6 and 9 shall survive the termination of this Agreement. \n \n7. Governing Law and Jurisdiction. This Agreement will be governed and construed under the laws of the State of California without regard to choice of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement. The courts of California shall have exclusive jurisdiction of any dispute arising out of this Agreement. \n\n\n8. Modifications and Amendments. Meta may modify this Agreement from time to time; provided that they are similar in spirit to the current version of the Agreement, but may differ in detail to address new problems or concerns. All such changes will be effective immediately. Your continued use of the Research Materials after any modification to this Agreement constitutes your agreement to such modification. Except as provided in this Agreement, no modification or addition to any provision of this Agreement will be binding unless it is in writing and signed by an authorized representative of both you and Meta.\n\n\nAcceptable Use Policy \n\nMeta seeks to further understanding of new and existing research domains with the mission of advancing the state-of-the-art in artificial intelligence through open research for the benefit of all. \n\nAs part of this mission, Meta makes certain research materials available for use in accordance with this Agreement (including the Acceptable Use Policy). Meta is committed to promoting the safe and responsible use of such research materials.  \n\nProhibited Uses\n\nYou agree you will not use, or allow others to use, Research Materials to:\n\n Violate the law or others’ rights, including to:\nEngage in, promote, generate, contribute to, encourage, plan, incite, or further illegal or unlawful activity or content, such as:\nViolence or terrorism\nExploitation or harm to children, including the solicitation, creation, acquisition, or dissemination of child exploitative content or failure to report Child Sexual Abuse Material\nHuman trafficking, exploitation, and sexual violence\nThe illegal distribution of information or materials to minors, including obscene materials, or failure to employ legally required age-gating in connection with such information or materials.\nSexual solicitation\nAny other criminal activity\n\nEngage in, promote, incite, or facilitate the harassment, abuse, threatening, or bullying of individuals or groups of individuals\n\nEngage in, promote, incite, or facilitate discrimination or other unlawful or harmful conduct in the provision of employment, employment benefits, credit, housing, other economic benefits, or other essential goods and services\n\nEngage in the unauthorized or unlicensed practice of any profession including, but not limited to, financial, legal, medical/health, or related professional practices\n\nCollect, process, disclose, generate, or infer health, demographic, or other sensitive personal or private information about individuals without rights and consents required by applicable laws\n\nEngage in or facilitate any action or generate any content that infringes, misappropriates, or otherwise violates any third-party rights, including the outputs or results of any technology using Research Materials\n\nCreate, generate, or facilitate the creation of malicious code, malware, computer viruses or do anything else that could disable, overburden, interfere with or impair the proper working, integrity, operation or appearance of a website or computer system\n\n2. Engage in, promote, incite, facilitate, or assist in the planning or development of activities that present a risk of death or bodily harm to individuals, including use of research artifacts related to the following:\n\nMilitary, warfare, nuclear industries or applications, espionage, use for materials or activities that are subject to the International Traffic Arms Regulations (ITAR) maintained by the United States Department of State\n\nGuns and illegal weapons (including weapon development)\n\nIllegal drugs and regulated/controlled substances\nOperation of critical infrastructure, transportation technologies, or heavy machinery\n\nSelf-harm or harm to others, including suicide, cutting, and eating disorders\nAny content intended to incite or promote violence, abuse, or any infliction of bodily harm to an individual\n\n3. Intentionally deceive or mislead others, including use of Research Materials related to the following:\n\n Generating, promoting, or furthering fraud or the creation or promotion of disinformation\n Generating, promoting, or furthering defamatory content, including the creation of defamatory statements, images, or other content\n\nGenerating, promoting, or further distributing spam\n\n Impersonating another individual without consent, authorization, or legal right\n\nRepresenting that outputs of research materials or outputs from technology using Research Materials are human-generated\n\nGenerating or facilitating false online engagement, including fake reviews and other means of fake online engagement\n\n4. Fail to appropriately disclose to end users any known dangers of your Research Materials.\n"
        },
        {
          "name": "vggt/heads/camera_head.py · source header",
          "text": "# Copyright (c) Meta Platforms, Inc. and affiliates.\n# All rights reserved.\n#\n# This source code is licensed under the license found in the\n# LICENSE file in the root directory of this source tree."
        },
        {
          "name": "vggt/heads/dpt_head.py · source header",
          "text": "# Copyright (c) Meta Platforms, Inc. and affiliates.\n# All rights reserved.\n#\n# This source code is licensed under the license found in the\n# LICENSE file in the root directory of this source tree.\n\n\n# Inspired by https://github.com/DepthAnything/Depth-Anything-V2"
        }
      ],
      "url": "https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/LICENSE.txt"
    },
    "ByteDance-Seed/Depth-Anything-3": {
      "file": "LICENSE",
      "commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
      "notices": [
        {
          "name": "LICENSE",
          "text": "                                 Apache License\n                           Version 2.0, January 2004\n                        http://www.apache.org/licenses/\n\n   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n\n   1. Definitions.\n\n      \"License\" shall mean the terms and conditions for use, reproduction,\n      and distribution as defined by Sections 1 through 9 of this document.\n\n      \"Licensor\" shall mean the copyright owner or entity authorized by\n      the copyright owner that is granting the License.\n\n      \"Legal Entity\" shall mean the union of the acting entity and all\n      other entities that control, are controlled by, or are under common\n      control with that entity. For the purposes of this definition,\n      \"control\" means (i) the power, direct or indirect, to cause the\n      direction or management of such entity, whether by contract or\n      otherwise, or (ii) ownership of fifty percent (50%) or more of the\n      outstanding shares, or (iii) beneficial ownership of such entity.\n\n      \"You\" (or \"Your\") shall mean an individual or Legal Entity\n      exercising permissions granted by this License.\n\n      \"Source\" form shall mean the preferred form for making modifications,\n      including but not limited to software source code, documentation\n      source, and configuration files.\n\n      \"Object\" form shall mean any form resulting from mechanical\n      transformation or translation of a Source form, including but\n      not limited to compiled object code, generated documentation,\n      and conversions to other media types.\n\n      \"Work\" shall mean the work of authorship, whether in Source or\n      Object form, made available under the License, as indicated by a\n      copyright notice that is included in or attached to the work\n      (an example is provided in the Appendix below).\n\n      \"Derivative Works\" shall mean any work, whether in Source or Object\n      form, that is based on (or derived from) the Work and for which the\n      editorial revisions, annotations, elaborations, or other modifications\n      represent, as a whole, an original work of authorship. For the purposes\n      of this License, Derivative Works shall not include works that remain\n      separable from, or merely link (or bind by name) to the interfaces of,\n      the Work and Derivative Works thereof.\n\n      \"Contribution\" shall mean any work of authorship, including\n      the original version of the Work and any modifications or additions\n      to that Work or Derivative Works thereof, that is intentionally\n      submitted to Licensor for inclusion in the Work by the copyright owner\n      or by an individual or Legal Entity authorized to submit on behalf of\n      the copyright owner. For the purposes of this definition, \"submitted\"\n      means any form of electronic, verbal, or written communication sent\n      to the Licensor or its representatives, including but not limited to\n      communication on electronic mailing lists, source code control systems,\n      and issue tracking systems that are managed by, or on behalf of, the\n      Licensor for the purpose of discussing and improving the Work, but\n      excluding communication that is conspicuously marked or otherwise\n      designated in writing by the copyright owner as \"Not a Contribution.\"\n\n      \"Contributor\" shall mean Licensor and any individual or Legal Entity\n      on behalf of whom a Contribution has been received by Licensor and\n      subsequently incorporated within the Work.\n\n   2. Grant of Copyright License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      copyright license to reproduce, prepare Derivative Works of,\n      publicly display, publicly perform, sublicense, and distribute the\n      Work and such Derivative Works in Source or Object form.\n\n   3. Grant of Patent License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      (except as stated in this section) patent license to make, have made,\n      use, offer to sell, sell, import, and otherwise transfer the Work,\n      where such license applies only to those patent claims licensable\n      by such Contributor that are necessarily infringed by their\n      Contribution(s) alone or by combination of their Contribution(s)\n      with the Work to which such Contribution(s) was submitted. If You\n      institute patent litigation against any entity (including a\n      cross-claim or counterclaim in a lawsuit) alleging that the Work\n      or a Contribution incorporated within the Work constitutes direct\n      or contributory patent infringement, then any patent licenses\n      granted to You under this License for that Work shall terminate\n      as of the date such litigation is filed.\n\n   4. Redistribution. You may reproduce and distribute copies of the\n      Work or Derivative Works thereof in any medium, with or without\n      modifications, and in Source or Object form, provided that You\n      meet the following conditions:\n\n      (a) You must give any other recipients of the Work or\n          Derivative Works a copy of this License; and\n\n      (b) You must cause any modified files to carry prominent notices\n          stating that You changed the files; and\n\n      (c) You must retain, in the Source form of any Derivative Works\n          that You distribute, all copyright, patent, trademark, and\n          attribution notices from the Source form of the Work,\n          excluding those notices that do not pertain to any part of\n          the Derivative Works; and\n\n      (d) If the Work includes a \"NOTICE\" text file as part of its\n          distribution, then any Derivative Works that You distribute must\n          include a readable copy of the attribution notices contained\n          within such NOTICE file, excluding those notices that do not\n          pertain to any part of the Derivative Works, in at least one\n          of the following places: within a NOTICE text file distributed\n          as part of the Derivative Works; within the Source form or\n          documentation, if provided along with the Derivative Works; or,\n          within a display generated by the Derivative Works, if and\n          wherever such third-party notices normally appear. The contents\n          of the NOTICE file are for informational purposes only and\n          do not modify the License. You may add Your own attribution\n          notices within Derivative Works that You distribute, alongside\n          or as an addendum to the NOTICE text from the Work, provided\n          that such additional attribution notices cannot be construed\n          as modifying the License.\n\n      You may add Your own copyright statement to Your modifications and\n      may provide additional or different license terms and conditions\n      for use, reproduction, or distribution of Your modifications, or\n      for any such Derivative Works as a whole, provided Your use,\n      reproduction, and distribution of the Work otherwise complies with\n      the conditions stated in this License.\n\n   5. Submission of Contributions. Unless You explicitly state otherwise,\n      any Contribution intentionally submitted for inclusion in the Work\n      by You to the Licensor shall be under the terms and conditions of\n      this License, without any additional terms or conditions.\n      Notwithstanding the above, nothing herein shall supersede or modify\n      the terms of any separate license agreement you may have executed\n      with Licensor regarding such Contributions.\n\n   6. Trademarks. This License does not grant permission to use the trade\n      names, trademarks, service marks, or product names of the Licensor,\n      except as required for reasonable and customary use in describing the\n      origin of the Work and reproducing the content of the NOTICE file.\n\n   7. Disclaimer of Warranty. Unless required by applicable law or\n      agreed to in writing, Licensor provides the Work (and each\n      Contributor provides its Contributions) on an \"AS IS\" BASIS,\n      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or\n      implied, including, without limitation, any warranties or conditions\n      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A\n      PARTICULAR PURPOSE. You are solely responsible for determining the\n      appropriateness of using or redistributing the Work and assume any\n      risks associated with Your exercise of permissions under this License.\n\n   8. Limitation of Liability. In no event and under no legal theory,\n      whether in tort (including negligence), contract, or otherwise,\n      unless required by applicable law (such as deliberate and grossly\n      negligent acts) or agreed to in writing, shall any Contributor be\n      liable to You for damages, including any direct, indirect, special,\n      incidental, or consequential damages of any character arising as a\n      result of this License or out of the use or inability to use the\n      Work (including but not limited to damages for loss of goodwill,\n      work stoppage, computer failure or malfunction, or any and all\n      other commercial damages or losses), even if such Contributor\n      has been advised of the possibility of such damages.\n\n   9. Accepting Warranty or Additional Liability. While redistributing\n      the Work or Derivative Works thereof, You may choose to offer,\n      and charge a fee for, acceptance of support, warranty, indemnity,\n      or other liability obligations and/or rights consistent with this\n      License. However, in accepting such obligations, You may act only\n      on Your own behalf and on Your sole responsibility, not on behalf\n      of any other Contributor, and only if You agree to indemnify,\n      defend, and hold each Contributor harmless for any liability\n      incurred by, or claims asserted against, such Contributor by reason\n      of your accepting any such warranty or additional liability.\n\n   END OF TERMS AND CONDITIONS\n\n   APPENDIX: How to apply the Apache License to your work.\n\n      To apply the Apache License to your work, attach the following\n      boilerplate notice, with the fields enclosed by brackets \"[]\"\n      replaced with your own identifying information. (Don't include\n      the brackets!)  The text should be enclosed in the appropriate\n      comment syntax for the file format. We also recommend that a\n      file or class name and description of purpose be included on\n      the same \"printed page\" as the copyright notice for easier\n      identification within third-party archives.\n\n   Copyright 2025 The Depth Anything 3 Team\n\n   Licensed under the Apache License, Version 2.0 (the \"License\");\n   you may not use this file except in compliance with the License.\n   You may obtain a copy of the License at\n\n       http://www.apache.org/licenses/LICENSE-2.0\n\n   Unless required by applicable law or agreed to in writing, software\n   distributed under the License is distributed on an \"AS IS\" BASIS,\n   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n   See the License for the specific language governing permissions and\n   limitations under the License.\n"
        },
        {
          "name": "src/depth_anything_3/model/da3.py · source header",
          "text": "# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates\n#\n# Licensed under the Apache License, Version 2.0 (the \"License\");\n# you may not use this file except in compliance with the License.\n# You may obtain a copy of the License at\n#\n#   http://www.apache.org/licenses/LICENSE-2.0\n#\n# Unless required by applicable law or agreed to in writing, software\n# distributed under the License is distributed on an \"AS IS\" BASIS,\n# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n# See the License for the specific language governing permissions and\n# limitations under the License."
        },
        {
          "name": "src/depth_anything_3/model/dinov2/vision_transformer.py · source header",
          "text": "# Copyright (c) Meta Platforms, Inc. and affiliates.\n#\n# This source code is licensed under the Apache License, Version 2.0\n# found in the LICENSE file in the root directory of this source tree.\n\n# References:\n#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py\n#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py"
        },
        {
          "name": "src/depth_anything_3/model/dualdpt.py · source header",
          "text": "# flake8: noqa E501\n# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates\n#\n# Licensed under the Apache License, Version 2.0 (the \"License\");\n# you may not use this file except in compliance with the License.\n# You may obtain a copy of the License at\n#\n#   http://www.apache.org/licenses/LICENSE-2.0\n#\n# Unless required by applicable law or agreed to in writing, software\n# distributed under the License is distributed on an \"AS IS\" BASIS,\n# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n# See the License for the specific language governing permissions and\n# limitations under the License."
        }
      ],
      "url": "https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/LICENSE"
    },
    "facebookresearch/map-anything": {
      "file": "LICENSE",
      "commit": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
      "notices": [
        {
          "name": "LICENSE",
          "text": "                                 Apache License\n                           Version 2.0, January 2004\n                        http://www.apache.org/licenses/\n\n   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n\n   1. Definitions.\n\n      \"License\" shall mean the terms and conditions for use, reproduction,\n      and distribution as defined by Sections 1 through 9 of this document.\n\n      \"Licensor\" shall mean the copyright owner or entity authorized by\n      the copyright owner that is granting the License.\n\n      \"Legal Entity\" shall mean the union of the acting entity and all\n      other entities that control, are controlled by, or are under common\n      control with that entity. For the purposes of this definition,\n      \"control\" means (i) the power, direct or indirect, to cause the\n      direction or management of such entity, whether by contract or\n      otherwise, or (ii) ownership of fifty percent (50%) or more of the\n      outstanding shares, or (iii) beneficial ownership of such entity.\n\n      \"You\" (or \"Your\") shall mean an individual or Legal Entity\n      exercising permissions granted by this License.\n\n      \"Source\" form shall mean the preferred form for making modifications,\n      including but not limited to software source code, documentation\n      source, and configuration files.\n\n      \"Object\" form shall mean any form resulting from mechanical\n      transformation or translation of a Source form, including but\n      not limited to compiled object code, generated documentation,\n      and conversions to other media types.\n\n      \"Work\" shall mean the work of authorship, whether in Source or\n      Object form, made available under the License, as indicated by a\n      copyright notice that is included in or attached to the work\n      (an example is provided in the Appendix below).\n\n      \"Derivative Works\" shall mean any work, whether in Source or Object\n      form, that is based on (or derived from) the Work and for which the\n      editorial revisions, annotations, elaborations, or other modifications\n      represent, as a whole, an original work of authorship. For the purposes\n      of this License, Derivative Works shall not include works that remain\n      separable from, or merely link (or bind by name) to the interfaces of,\n      the Work and Derivative Works thereof.\n\n      \"Contribution\" shall mean any work of authorship, including\n      the original version of the Work and any modifications or additions\n      to that Work or Derivative Works thereof, that is intentionally\n      submitted to Licensor for inclusion in the Work by the copyright owner\n      or by an individual or Legal Entity authorized to submit on behalf of\n      the copyright owner. For the purposes of this definition, \"submitted\"\n      means any form of electronic, verbal, or written communication sent\n      to the Licensor or its representatives, including but not limited to\n      communication on electronic mailing lists, source code control systems,\n      and issue tracking systems that are managed by, or on behalf of, the\n      Licensor for the purpose of discussing and improving the Work, but\n      excluding communication that is conspicuously marked or otherwise\n      designated in writing by the copyright owner as \"Not a Contribution.\"\n\n      \"Contributor\" shall mean Licensor and any individual or Legal Entity\n      on behalf of whom a Contribution has been received by Licensor and\n      subsequently incorporated within the Work.\n\n   2. Grant of Copyright License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      copyright license to reproduce, prepare Derivative Works of,\n      publicly display, publicly perform, sublicense, and distribute the\n      Work and such Derivative Works in Source or Object form.\n\n   3. Grant of Patent License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      (except as stated in this section) patent license to make, have made,\n      use, offer to sell, sell, import, and otherwise transfer the Work,\n      where such license applies only to those patent claims licensable\n      by such Contributor that are necessarily infringed by their\n      Contribution(s) alone or by combination of their Contribution(s)\n      with the Work to which such Contribution(s) was submitted. If You\n      institute patent litigation against any entity (including a\n      cross-claim or counterclaim in a lawsuit) alleging that the Work\n      or a Contribution incorporated within the Work constitutes direct\n      or contributory patent infringement, then any patent licenses\n      granted to You under this License for that Work shall terminate\n      as of the date such litigation is filed.\n\n   4. Redistribution. You may reproduce and distribute copies of the\n      Work or Derivative Works thereof in any medium, with or without\n      modifications, and in Source or Object form, provided that You\n      meet the following conditions:\n\n      (a) You must give any other recipients of the Work or\n          Derivative Works a copy of this License; and\n\n      (b) You must cause any modified files to carry prominent notices\n          stating that You changed the files; and\n\n      (c) You must retain, in the Source form of any Derivative Works\n          that You distribute, all copyright, patent, trademark, and\n          attribution notices from the Source form of the Work,\n          excluding those notices that do not pertain to any part of\n          the Derivative Works; and\n\n      (d) If the Work includes a \"NOTICE\" text file as part of its\n          distribution, then any Derivative Works that You distribute must\n          include a readable copy of the attribution notices contained\n          within such NOTICE file, excluding those notices that do not\n          pertain to any part of the Derivative Works, in at least one\n          of the following places: within a NOTICE text file distributed\n          as part of the Derivative Works; within the Source form or\n          documentation, if provided along with the Derivative Works; or,\n          within a display generated by the Derivative Works, if and\n          wherever such third-party notices normally appear. The contents\n          of the NOTICE file are for informational purposes only and\n          do not modify the License. You may add Your own attribution\n          notices within Derivative Works that You distribute, alongside\n          or as an addendum to the NOTICE text from the Work, provided\n          that such additional attribution notices cannot be construed\n          as modifying the License.\n\n      You may add Your own copyright statement to Your modifications and\n      may provide additional or different license terms and conditions\n      for use, reproduction, or distribution of Your modifications, or\n      for any such Derivative Works as a whole, provided Your use,\n      reproduction, and distribution of the Work otherwise complies with\n      the conditions stated in this License.\n\n   5. Submission of Contributions. Unless You explicitly state otherwise,\n      any Contribution intentionally submitted for inclusion in the Work\n      by You to the Licensor shall be under the terms and conditions of\n      this License, without any additional terms or conditions.\n      Notwithstanding the above, nothing herein shall supersede or modify\n      the terms of any separate license agreement you may have executed\n      with Licensor regarding such Contributions.\n\n   6. Trademarks. This License does not grant permission to use the trade\n      names, trademarks, service marks, or product names of the Licensor,\n      except as required for reasonable and customary use in describing the\n      origin of the Work and reproducing the content of the NOTICE file.\n\n   7. Disclaimer of Warranty. Unless required by applicable law or\n      agreed to in writing, Licensor provides the Work (and each\n      Contributor provides its Contributions) on an \"AS IS\" BASIS,\n      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or\n      implied, including, without limitation, any warranties or conditions\n      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A\n      PARTICULAR PURPOSE. You are solely responsible for determining the\n      appropriateness of using or redistributing the Work and assume any\n      risks associated with Your exercise of permissions under this License.\n\n   8. Limitation of Liability. In no event and under no legal theory,\n      whether in tort (including negligence), contract, or otherwise,\n      unless required by applicable law (such as deliberate and grossly\n      negligent acts) or agreed to in writing, shall any Contributor be\n      liable to You for damages, including any direct, indirect, special,\n      incidental, or consequential damages of any character arising as a\n      result of this License or out of the use or inability to use the\n      Work (including but not limited to damages for loss of goodwill,\n      work stoppage, computer failure or malfunction, or any and all\n      other commercial damages or losses), even if such Contributor\n      has been advised of the possibility of such damages.\n\n   9. Accepting Warranty or Additional Liability. While redistributing\n      the Work or Derivative Works thereof, You may choose to offer,\n      and charge a fee for, acceptance of support, warranty, indemnity,\n      or other liability obligations and/or rights consistent with this\n      License. However, in accepting such obligations, You may act only\n      on Your own behalf and on Your sole responsibility, not on behalf\n      of any other Contributor, and only if You agree to indemnify,\n      defend, and hold each Contributor harmless for any liability\n      incurred by, or claims asserted against, such Contributor by reason\n      of your accepting any such warranty or additional liability.\n\n   END OF TERMS AND CONDITIONS\n\n   APPENDIX: How to apply the Apache License to your work.\n\n      To apply the Apache License to your work, attach the following\n      boilerplate notice, with the fields enclosed by brackets \"[]\"\n      replaced with your own identifying information. (Don't include\n      the brackets!)  The text should be enclosed in the appropriate\n      comment syntax for the file format. We also recommend that a\n      file or class name and description of purpose be included on the\n      same \"printed page\" as the copyright notice for easier\n      identification within third-party archives.\n\n   Copyright 2025 Meta\n\n   Licensed under the Apache License, Version 2.0 (the \"License\");\n   you may not use this file except in compliance with the License.\n   You may obtain a copy of the License at\n\n       http://www.apache.org/licenses/LICENSE-2.0\n\n   Unless required by applicable law or agreed to in writing, software\n   distributed under the License is distributed on an \"AS IS\" BASIS,\n   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n   See the License for the specific language governing permissions and\n   limitations under the License.\n"
        },
        {
          "name": "mapanything/models/mapanything/model.py · source header",
          "text": "# Copyright (c) Meta Platforms, Inc. and affiliates.\n#\n# This source code is licensed under the Apache License, Version 2.0\n# found in the LICENSE file in the root directory of this source tree."
        }
      ],
      "url": "https://github.com/facebookresearch/map-anything/blob/3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9/LICENSE"
    },
    "yyfz/Pi4": {
      "file": "LICENSE",
      "commit": "32dfbea9ce6268ea17446208a804710df9475c32",
      "notices": [
        {
          "name": "LICENSE",
          "text": "Pi4 source code\nCopyright (c) 2026, the Pi4 authors\n\nOriginal Pi4 source code authored by the Pi4 authors is licensed under the\nCreative Commons Attribution-NonCommercial 4.0 International license\n(CC BY-NC 4.0). Commercial use of this original code is not permitted under\nthis license. Redistribution and adaptations must comply with the attribution\nand noncommercial conditions below.\n\nThis license applies only to original Pi4 contributions. Third-party materials\nretain their upstream licenses; see NOTICE and licenses/. Pi4 model weights are\ncovered separately by LICENSE_MODEL.\n\nThe full, unmodified CC BY-NC 4.0 license text follows.\n\nAttribution-NonCommercial 4.0 International\n\n=======================================================================\n\nCreative Commons Corporation (\"Creative Commons\") is not a law firm and\ndoes not provide legal services or legal advice. Distribution of\nCreative Commons public licenses does not create a lawyer-client or\nother relationship. Creative Commons makes its licenses and related\ninformation available on an \"as-is\" basis. Creative Commons gives no\nwarranties regarding its licenses, any material licensed under their\nterms and conditions, or any related information. Creative Commons\ndisclaims all liability for damages resulting from their use to the\nfullest extent possible.\n\nUsing Creative Commons Public Licenses\n\nCreative Commons public licenses provide a standard set of terms and\nconditions that creators and other rights holders may use to share\noriginal works of authorship and other material subject to copyright\nand certain other rights specified in the public license below. The\nfollowing considerations are for informational purposes only, are not\nexhaustive, and do not form part of our licenses.\n\n     Considerations for licensors: Our public licenses are\n     intended for use by those authorized to give the public\n     permission to use material in ways otherwise restricted by\n     copyright and certain other rights. Our licenses are\n     irrevocable. Licensors should read and understand the terms\n     and conditions of the license they choose before applying it.\n     Licensors should also secure all rights necessary before\n     applying our licenses so that the public can reuse the\n     material as expected. Licensors should clearly mark any\n     material not subject to the license. This includes other CC-\n     licensed material, or material used under an exception or\n     limitation to copyright. More considerations for licensors:\n    wiki.creativecommons.org/Considerations_for_licensors\n\n     Considerations for the public: By using one of our public\n     licenses, a licensor grants the public permission to use the\n     licensed material under specified terms and conditions. If\n     the licensor's permission is not necessary for any reason--for\n     example, because of any applicable exception or limitation to\n     copyright--then that use is not regulated by the license. Our\n     licenses grant only permissions under copyright and certain\n     other rights that a licensor has authority to grant. Use of\n     the licensed material may still be restricted for other\n     reasons, including because others have copyright or other\n     rights in the material. A licensor may make special requests,\n     such as asking that all changes be marked or described.\n     Although not required by our licenses, you are encouraged to\n     respect those requests where reasonable. More considerations\n     for the public:\n    wiki.creativecommons.org/Considerations_for_licensees\n\n=======================================================================\n\nCreative Commons Attribution-NonCommercial 4.0 International Public\nLicense\n\nBy exercising the Licensed Rights (defined below), You accept and agree\nto be bound by the terms and conditions of this Creative Commons\nAttribution-NonCommercial 4.0 International Public License (\"Public\nLicense\"). To the extent this Public License may be interpreted as a\ncontract, You are granted the Licensed Rights in consideration of Your\nacceptance of these terms and conditions, and the Licensor grants You\nsuch rights in consideration of benefits the Licensor receives from\nmaking the Licensed Material available under these terms and\nconditions.\n\n\nSection 1 -- Definitions.\n\n  a. Adapted Material means material subject to Copyright and Similar\n     Rights that is derived from or based upon the Licensed Material\n     and in which the Licensed Material is translated, altered,\n     arranged, transformed, or otherwise modified in a manner requiring\n     permission under the Copyright and Similar Rights held by the\n     Licensor. For purposes of this Public License, where the Licensed\n     Material is a musical work, performance, or sound recording,\n     Adapted Material is always produced where the Licensed Material is\n     synched in timed relation with a moving image.\n\n  b. Adapter's License means the license You apply to Your Copyright\n     and Similar Rights in Your contributions to Adapted Material in\n     accordance with the terms and conditions of this Public License.\n\n  c. Copyright and Similar Rights means copyright and/or similar rights\n     closely related to copyright including, without limitation,\n     performance, broadcast, sound recording, and Sui Generis Database\n     Rights, without regard to how the rights are labeled or\n     categorized. For purposes of this Public License, the rights\n     specified in Section 2(b)(1)-(2) are not Copyright and Similar\n     Rights.\n  d. Effective Technological Measures means those measures that, in the\n     absence of proper authority, may not be circumvented under laws\n     fulfilling obligations under Article 11 of the WIPO Copyright\n     Treaty adopted on December 20, 1996, and/or similar international\n     agreements.\n\n  e. Exceptions and Limitations means fair use, fair dealing, and/or\n     any other exception or limitation to Copyright and Similar Rights\n     that applies to Your use of the Licensed Material.\n\n  f. Licensed Material means the artistic or literary work, database,\n     or other material to which the Licensor applied this Public\n     License.\n\n  g. Licensed Rights means the rights granted to You subject to the\n     terms and conditions of this Public License, which are limited to\n     all Copyright and Similar Rights that apply to Your use of the\n     Licensed Material and that the Licensor has authority to license.\n\n  h. Licensor means the individual(s) or entity(ies) granting rights\n     under this Public License.\n\n  i. NonCommercial means not primarily intended for or directed towards\n     commercial advantage or monetary compensation. For purposes of\n     this Public License, the exchange of the Licensed Material for\n     other material subject to Copyright and Similar Rights by digital\n     file-sharing or similar means is NonCommercial provided there is\n     no payment of monetary compensation in connection with the\n     exchange.\n\n  j. Share means to provide material to the public by any means or\n     process that requires permission under the Licensed Rights, such\n     as reproduction, public display, public performance, distribution,\n     dissemination, communication, or importation, and to make material\n     available to the public including in ways that members of the\n     public may access the material from a place and at a time\n     individually chosen by them.\n\n  k. Sui Generis Database Rights means rights other than copyright\n     resulting from Directive 96/9/EC of the European Parliament and of\n     the Council of 11 March 1996 on the legal protection of databases,\n     as amended and/or succeeded, as well as other essentially\n     equivalent rights anywhere in the world.\n\n  l. You means the individual or entity exercising the Licensed Rights\n     under this Public License. Your has a corresponding meaning.\n\n\nSection 2 -- Scope.\n\n  a. License grant.\n\n       1. Subject to the terms and conditions of this Public License,\n          the Licensor hereby grants You a worldwide, royalty-free,\n          non-sublicensable, non-exclusive, irrevocable license to\n          exercise the Licensed Rights in the Licensed Material to:\n\n            a. reproduce and Share the Licensed Material, in whole or\n               in part, for NonCommercial purposes only; and\n\n            b. produce, reproduce, and Share Adapted Material for\n               NonCommercial purposes only.\n\n       2. Exceptions and Limitations. For the avoidance of doubt, where\n          Exceptions and Limitations apply to Your use, this Public\n          License does not apply, and You do not need to comply with\n          its terms and conditions.\n\n       3. Term. The term of this Public License is specified in Section\n          6(a).\n\n       4. Media and formats; technical modifications allowed. The\n          Licensor authorizes You to exercise the Licensed Rights in\n          all media and formats whether now known or hereafter created,\n          and to make technical modifications necessary to do so. The\n          Licensor waives and/or agrees not to assert any right or\n          authority to forbid You from making technical modifications\n          necessary to exercise the Licensed Rights, including\n          technical modifications necessary to circumvent Effective\n          Technological Measures. For purposes of this Public License,\n          simply making modifications authorized by this Section 2(a)\n          (4) never produces Adapted Material.\n\n       5. Downstream recipients.\n\n            a. Offer from the Licensor -- Licensed Material. Every\n               recipient of the Licensed Material automatically\n               receives an offer from the Licensor to exercise the\n               Licensed Rights under the terms and conditions of this\n               Public License.\n\n            b. No downstream restrictions. You may not offer or impose\n               any additional or different terms or conditions on, or\n               apply any Effective Technological Measures to, the\n               Licensed Material if doing so restricts exercise of the\n               Licensed Rights by any recipient of the Licensed\n               Material.\n\n       6. No endorsement. Nothing in this Public License constitutes or\n          may be construed as permission to assert or imply that You\n          are, or that Your use of the Licensed Material is, connected\n          with, or sponsored, endorsed, or granted official status by,\n          the Licensor or others designated to receive attribution as\n          provided in Section 3(a)(1)(A)(i).\n\n  b. Other rights.\n\n       1. Moral rights, such as the right of integrity, are not\n          licensed under this Public License, nor are publicity,\n          privacy, and/or other similar personality rights; however, to\n          the extent possible, the Licensor waives and/or agrees not to\n          assert any such rights held by the Licensor to the limited\n          extent necessary to allow You to exercise the Licensed\n          Rights, but not otherwise.\n\n       2. Patent and trademark rights are not licensed under this\n          Public License.\n\n       3. To the extent possible, the Licensor waives any right to\n          collect royalties from You for the exercise of the Licensed\n          Rights, whether directly or through a collecting society\n          under any voluntary or waivable statutory or compulsory\n          licensing scheme. In all other cases the Licensor expressly\n          reserves any right to collect such royalties, including when\n          the Licensed Material is used other than for NonCommercial\n          purposes.\n\n\nSection 3 -- License Conditions.\n\nYour exercise of the Licensed Rights is expressly made subject to the\nfollowing conditions.\n\n  a. Attribution.\n\n       1. If You Share the Licensed Material (including in modified\n          form), You must:\n\n            a. retain the following if it is supplied by the Licensor\n               with the Licensed Material:\n\n                 i. identification of the creator(s) of the Licensed\n                    Material and any others designated to receive\n                    attribution, in any reasonable manner requested by\n                    the Licensor (including by pseudonym if\n                    designated);\n\n                ii. a copyright notice;\n\n               iii. a notice that refers to this Public License;\n\n                iv. a notice that refers to the disclaimer of\n                    warranties;\n\n                 v. a URI or hyperlink to the Licensed Material to the\n                    extent reasonably practicable;\n\n            b. indicate if You modified the Licensed Material and\n               retain an indication of any previous modifications; and\n\n            c. indicate the Licensed Material is licensed under this\n               Public License, and include the text of, or the URI or\n               hyperlink to, this Public License.\n\n       2. You may satisfy the conditions in Section 3(a)(1) in any\n          reasonable manner based on the medium, means, and context in\n          which You Share the Licensed Material. For example, it may be\n          reasonable to satisfy the conditions by providing a URI or\n          hyperlink to a resource that includes the required\n          information.\n\n       3. If requested by the Licensor, You must remove any of the\n          information required by Section 3(a)(1)(A) to the extent\n          reasonably practicable.\n\n       4. If You Share Adapted Material You produce, the Adapter's\n          License You apply must not prevent recipients of the Adapted\n          Material from complying with this Public License.\n\n\nSection 4 -- Sui Generis Database Rights.\n\nWhere the Licensed Rights include Sui Generis Database Rights that\napply to Your use of the Licensed Material:\n\n  a. for the avoidance of doubt, Section 2(a)(1) grants You the right\n     to extract, reuse, reproduce, and Share all or a substantial\n     portion of the contents of the database for NonCommercial purposes\n     only;\n\n  b. if You include all or a substantial portion of the database\n     contents in a database in which You have Sui Generis Database\n     Rights, then the database in which You have Sui Generis Database\n     Rights (but not its individual contents) is Adapted Material; and\n\n  c. You must comply with the conditions in Section 3(a) if You Share\n     all or a substantial portion of the contents of the database.\n\nFor the avoidance of doubt, this Section 4 supplements and does not\nreplace Your obligations under this Public License where the Licensed\nRights include other Copyright and Similar Rights.\n\n\nSection 5 -- Disclaimer of Warranties and Limitation of Liability.\n\n  a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE\n     EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS\n     AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF\n     ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,\n     IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,\n     WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR\n     PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,\n     ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT\n     KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT\n     ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.\n\n  b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE\n     TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,\n     NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,\n     INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,\n     COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR\n     USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN\n     ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR\n     DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR\n     IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.\n\n  c. The disclaimer of warranties and limitation of liability provided\n     above shall be interpreted in a manner that, to the extent\n     possible, most closely approximates an absolute disclaimer and\n     waiver of all liability.\n\n\nSection 6 -- Term and Termination.\n\n  a. This Public License applies for the term of the Copyright and\n     Similar Rights licensed here. However, if You fail to comply with\n     this Public License, then Your rights under this Public License\n     terminate automatically.\n\n  b. Where Your right to use the Licensed Material has terminated under\n     Section 6(a), it reinstates:\n\n       1. automatically as of the date the violation is cured, provided\n          it is cured within 30 days of Your discovery of the\n          violation; or\n\n       2. upon express reinstatement by the Licensor.\n\n     For the avoidance of doubt, this Section 6(b) does not affect any\n     right the Licensor may have to seek remedies for Your violations\n     of this Public License.\n\n  c. For the avoidance of doubt, the Licensor may also offer the\n     Licensed Material under separate terms or conditions or stop\n     distributing the Licensed Material at any time; however, doing so\n     will not terminate this Public License.\n\n  d. Sections 1, 5, 6, 7, and 8 survive termination of this Public\n     License.\n\n\nSection 7 -- Other Terms and Conditions.\n\n  a. The Licensor shall not be bound by any additional or different\n     terms or conditions communicated by You unless expressly agreed.\n\n  b. Any arrangements, understandings, or agreements regarding the\n     Licensed Material not stated herein are separate from and\n     independent of the terms and conditions of this Public License.\n\n\nSection 8 -- Interpretation.\n\n  a. For the avoidance of doubt, this Public License does not, and\n     shall not be interpreted to, reduce, limit, restrict, or impose\n     conditions on any use of the Licensed Material that could lawfully\n     be made without permission under this Public License.\n\n  b. To the extent possible, if any provision of this Public License is\n     deemed unenforceable, it shall be automatically reformed to the\n     minimum extent necessary to make it enforceable. If the provision\n     cannot be reformed, it shall be severed from this Public License\n     without affecting the enforceability of the remaining terms and\n     conditions.\n\n  c. No term or condition of this Public License will be waived and no\n     failure to comply consented to unless expressly agreed to by the\n     Licensor.\n\n  d. Nothing in this Public License constitutes or may be interpreted\n     as a limitation upon, or waiver of, any privileges and immunities\n     that apply to the Licensor or You, including from the legal\n     processes of any jurisdiction or authority.\n\n=======================================================================\n\nCreative Commons is not a party to its public\nlicenses. Notwithstanding, Creative Commons may elect to apply one of\nits public licenses to material it publishes and in those instances\nwill be considered the “Licensor.” The text of the Creative Commons\npublic licenses is dedicated to the public domain under the CC0 Public\nDomain Dedication. Except for the limited purpose of indicating that\nmaterial is shared under a Creative Commons public license or as\notherwise permitted by the Creative Commons policies published at\ncreativecommons.org/policies, Creative Commons does not authorize the\nuse of the trademark \"Creative Commons\" or any other trademark or logo\nof Creative Commons without its prior written consent including,\nwithout limitation, in connection with any unauthorized modifications\nto any of its public licenses or any other arrangements,\nunderstandings, or agreements concerning use of licensed material. For\nthe avoidance of doubt, this paragraph does not form part of the\npublic licenses.\n\nCreative Commons may be contacted at creativecommons.org.\n"
        },
        {
          "name": "NOTICE",
          "text": "Pi4 licensing\n\nOriginal Pi4 source code is licensed under CC BY-NC 4.0 (LICENSE).\nPi4 model weights are licensed under CC BY-NC 4.0 (LICENSE_MODEL), including\ncheckpoints distributed separately from this repository.\n\nThird-party materials retain their upstream licenses. Neither Pi4 license\noverrides those terms or grants rights to third-party materials. In particular,\nCoWTracker-derived materials remain subject to the FAIR Noncommercial Research\nLicense, which is not replaced by CC BY-NC 4.0.\nSeparate checkpoint distributions should include LICENSE_MODEL and applicable\nthird-party notices.\n\nPi4 uses or adapts the following upstream implementations and architectures:\n\nThe bundled viewer is a reduced migration of the project's point_cloud_viewer:\nquaternion trackball controls, page layout/styles, point shaders and sorting,\nscreen-space picking, binary-array decoding, motion geometry scopes, progressive\nframe display, current-frame highlights, and seeded dynamic-track selection.\nIt runs independently and does not import that project at runtime.\n\nDINOv3: facebookresearch/dinov3; image encoder and axial rotary encoding.\nLicense text: licenses/DINOV3.txt.\n\nMoGe: microsoft/MoGe; point decoding and sparse depth-refinement architecture.\nLicense text: licenses/MOGE.txt.\n\nCoWTracker: facebookresearch/cowtracker; RGB features, dense iterative warping,\ncross-view refinement, and confidence/visibility outputs.\nLicense text: licenses/COWTRACKER.txt.\n\nSAM 2: facebookresearch/sam2; Hiera-small and feature-pyramid image encoding.\nLicense text: licenses/SAM2.txt.\n\nDepth Anything V2: DepthAnything/Depth-Anything-V2; DPT feature-fusion blocks.\nLicense text: licenses/DEPTH_ANYTHING_V2.txt.\n\nPyTorch Image Models: huggingface/pytorch-image-models; tracking ViT architecture.\nApache-2.0 text: licenses/APACHE-2.0.txt.\n\nPi3: yyfz/Pi3; geometry-decoder and camera-readout architecture.\nLicense text: licenses/PI3.txt.\n\nFlexGEMM: JeffreyXiang/FlexGEMM; separately installed sparse-convolution backend.\nIts own distribution retains its license and notices.\n\nThree.js: mrdoob/three.js; bundled browser rendering and TrackballControls.\nLicense text: licenses/THREE.txt. Only these runtime files are retained from\nthe larger point-cloud viewer; no server-rendering or recording code is used.\n"
        },
        {
          "name": "licenses/APACHE-2.0.txt",
          "text": "                                 Apache License\n                           Version 2.0, January 2004\n                        http://www.apache.org/licenses/\n\n   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n\n   1. Definitions.\n\n      \"License\" shall mean the terms and conditions for use, reproduction,\n      and distribution as defined by Sections 1 through 9 of this document.\n\n      \"Licensor\" shall mean the copyright owner or entity authorized by\n      the copyright owner that is granting the License.\n\n      \"Legal Entity\" shall mean the union of the acting entity and all\n      other entities that control, are controlled by, or are under common\n      control with that entity. For the purposes of this definition,\n      \"control\" means (i) the power, direct or indirect, to cause the\n      direction or management of such entity, whether by contract or\n      otherwise, or (ii) ownership of fifty percent (50%) or more of the\n      outstanding shares, or (iii) beneficial ownership of such entity.\n\n      \"You\" (or \"Your\") shall mean an individual or Legal Entity\n      exercising permissions granted by this License.\n\n      \"Source\" form shall mean the preferred form for making modifications,\n      including but not limited to software source code, documentation\n      source, and configuration files.\n\n      \"Object\" form shall mean any form resulting from mechanical\n      transformation or translation of a Source form, including but\n      not limited to compiled object code, generated documentation,\n      and conversions to other media types.\n\n      \"Work\" shall mean the work of authorship, whether in Source or\n      Object form, made available under the License, as indicated by a\n      copyright notice that is included in or attached to the work\n      (an example is provided in the Appendix below).\n\n      \"Derivative Works\" shall mean any work, whether in Source or Object\n      form, that is based on (or derived from) the Work and for which the\n      editorial revisions, annotations, elaborations, or other modifications\n      represent, as a whole, an original work of authorship. For the purposes\n      of this License, Derivative Works shall not include works that remain\n      separable from, or merely link (or bind by name) to the interfaces of,\n      the Work and Derivative Works thereof.\n\n      \"Contribution\" shall mean any work of authorship, including\n      the original version of the Work and any modifications or additions\n      to that Work or Derivative Works thereof, that is intentionally\n      submitted to Licensor for inclusion in the Work by the copyright owner\n      or by an individual or Legal Entity authorized to submit on behalf of\n      the copyright owner. For the purposes of this definition, \"submitted\"\n      means any form of electronic, verbal, or written communication sent\n      to the Licensor or its representatives, including but not limited to\n      communication on electronic mailing lists, source code control systems,\n      and issue tracking systems that are managed by, or on behalf of, the\n      Licensor for the purpose of discussing and improving the Work, but\n      excluding communication that is conspicuously marked or otherwise\n      designated in writing by the copyright owner as \"Not a Contribution.\"\n\n      \"Contributor\" shall mean Licensor and any individual or Legal Entity\n      on behalf of whom a Contribution has been received by Licensor and\n      subsequently incorporated within the Work.\n\n   2. Grant of Copyright License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      copyright license to reproduce, prepare Derivative Works of,\n      publicly display, publicly perform, sublicense, and distribute the\n      Work and such Derivative Works in Source or Object form.\n\n   3. Grant of Patent License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      (except as stated in this section) patent license to make, have made,\n      use, offer to sell, sell, import, and otherwise transfer the Work,\n      where such license applies only to those patent claims licensable\n      by such Contributor that are necessarily infringed by their\n      Contribution(s) alone or by combination of their Contribution(s)\n      with the Work to which such Contribution(s) was submitted. If You\n      institute patent litigation against any entity (including a\n      cross-claim or counterclaim in a lawsuit) alleging that the Work\n      or a Contribution incorporated within the Work constitutes direct\n      or contributory patent infringement, then any patent licenses\n      granted to You under this License for that Work shall terminate\n      as of the date such litigation is filed.\n\n   4. Redistribution. You may reproduce and distribute copies of the\n      Work or Derivative Works thereof in any medium, with or without\n      modifications, and in Source or Object form, provided that You\n      meet the following conditions:\n\n      (a) You must give any other recipients of the Work or\n          Derivative Works a copy of this License; and\n\n      (b) You must cause any modified files to carry prominent notices\n          stating that You changed the files; and\n\n      (c) You must retain, in the Source form of any Derivative Works\n          that You distribute, all copyright, patent, trademark, and\n          attribution notices from the Source form of the Work,\n          excluding those notices that do not pertain to any part of\n          the Derivative Works; and\n\n      (d) If the Work includes a \"NOTICE\" text file as part of its\n          distribution, then any Derivative Works that You distribute must\n          include a readable copy of the attribution notices contained\n          within such NOTICE file, excluding those notices that do not\n          pertain to any part of the Derivative Works, in at least one\n          of the following places: within a NOTICE text file distributed\n          as part of the Derivative Works; within the Source form or\n          documentation, if provided along with the Derivative Works; or,\n          within a display generated by the Derivative Works, if and\n          wherever such third-party notices normally appear. The contents\n          of the NOTICE file are for informational purposes only and\n          do not modify the License. You may add Your own attribution\n          notices within Derivative Works that You distribute, alongside\n          or as an addendum to the NOTICE text from the Work, provided\n          that such additional attribution notices cannot be construed\n          as modifying the License.\n\n      You may add Your own copyright statement to Your modifications and\n      may provide additional or different license terms and conditions\n      for use, reproduction, or distribution of Your modifications, or\n      for any such Derivative Works as a whole, provided Your use,\n      reproduction, and distribution of the Work otherwise complies with\n      the conditions stated in this License.\n\n   5. Submission of Contributions. Unless You explicitly state otherwise,\n      any Contribution intentionally submitted for inclusion in the Work\n      by You to the Licensor shall be under the terms and conditions of\n      this License, without any additional terms or conditions.\n      Notwithstanding the above, nothing herein shall supersede or modify\n      the terms of any separate license agreement you may have executed\n      with Licensor regarding such Contributions.\n\n   6. Trademarks. This License does not grant permission to use the trade\n      names, trademarks, service marks, or product names of the Licensor,\n      except as required for reasonable and customary use in describing the\n      origin of the Work and reproducing the content of the NOTICE file.\n\n   7. Disclaimer of Warranty. Unless required by applicable law or\n      agreed to in writing, Licensor provides the Work (and each\n      Contributor provides its Contributions) on an \"AS IS\" BASIS,\n      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or\n      implied, including, without limitation, any warranties or conditions\n      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A\n      PARTICULAR PURPOSE. You are solely responsible for determining the\n      appropriateness of using or redistributing the Work and assume any\n      risks associated with Your exercise of permissions under this License.\n\n   8. Limitation of Liability. In no event and under no legal theory,\n      whether in tort (including negligence), contract, or otherwise,\n      unless required by applicable law (such as deliberate and grossly\n      negligent acts) or agreed to in writing, shall any Contributor be\n      liable to You for damages, including any direct, indirect, special,\n      incidental, or consequential damages of any character arising as a\n      result of this License or out of the use or inability to use the\n      Work (including but not limited to damages for loss of goodwill,\n      work stoppage, computer failure or malfunction, or any and all\n      other commercial damages or losses), even if such Contributor\n      has been advised of the possibility of such damages.\n\n   9. Accepting Warranty or Additional Liability. While redistributing\n      the Work or Derivative Works thereof, You may choose to offer,\n      and charge a fee for, acceptance of support, warranty, indemnity,\n      or other liability obligations and/or rights consistent with this\n      License. However, in accepting such obligations, You may act only\n      on Your own behalf and on Your sole responsibility, not on behalf\n      of any other Contributor, and only if You agree to indemnify,\n      defend, and hold each Contributor harmless for any liability\n      incurred by, or claims asserted against, such Contributor by reason\n      of your accepting any such warranty or additional liability.\n\n   END OF TERMS AND CONDITIONS\n\n   APPENDIX: How to apply the Apache License to your work.\n\n      To apply the Apache License to your work, attach the following\n      boilerplate notice, with the fields enclosed by brackets \"[]\"\n      replaced with your own identifying information. (Don't include\n      the brackets!)  The text should be enclosed in the appropriate\n      comment syntax for the file format. We also recommend that a\n      file or class name and description of purpose be included on the\n      same \"printed page\" as the copyright notice for easier\n      identification within third-party archives.\n\n   Copyright [yyyy] [name of copyright owner]\n\n   Licensed under the Apache License, Version 2.0 (the \"License\");\n   you may not use this file except in compliance with the License.\n   You may obtain a copy of the License at\n\n       http://www.apache.org/licenses/LICENSE-2.0\n\n   Unless required by applicable law or agreed to in writing, software\n   distributed under the License is distributed on an \"AS IS\" BASIS,\n   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n   See the License for the specific language governing permissions and\n   limitations under the License.\n"
        },
        {
          "name": "licenses/COWTRACKER.txt",
          "text": "FAIR Noncommercial Research License\nv1 Last Updated: December 22, 2025\n\n“Acceptable Use Policy” means the FAIR Acceptable Use Policy, applicable to Research Materials, that is incorporated into this Agreement.\n\n“Agreement” means the terms and conditions for use, reproduction, distribution and modification of the Research Materials set forth herein.\n\n\n“Documentation” means the specifications, manuals and documentation accompanying \nResearch Materials distributed by Meta.\n\n\n“Licensee” or “you” means you, or your employer or any other person or entity (if you are entering into this Agreement on such person or entity’s behalf), of the age required under applicable laws, rules or regulations to provide legal consent and that has legal authority to bind your employer or such other person or entity if you are entering in this Agreement on their behalf.\n\n\n“Meta” or “we” means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your principal place of business is in the EEA or Switzerland) and Meta Platforms, Inc. (if you are located outside of the EEA or Switzerland).\n\n“Noncommercial Research Uses” means noncommercial research use cases related to research, development, education, processing, or analysis and in each case, is not primarily intended for commercial advantage or monetary compensation to you or others.\n\n“Research Materials” means, collectively, Documentation and the models, software and algorithms, including machine-learning model code, trained model weights, inference-enabling code, training-enabling code, fine-tuning enabling code, demonstration materials and other elements of the foregoing distributed by Meta and made available under this Agreement.\n\nBy clicking “I Accept” below or by using or distributing any portion or element of the Research Materials, you agree to be bound by this Agreement.\n\n\n1. License Rights and Redistribution.\n\n\na. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under Meta’s intellectual property or other rights owned by Meta embodied in the Research Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Research Materials.  \n \nb. Redistribution and Use.  \n i. You will not use the Research Materials or any outputs or results of the Research Materials in connection with any commercial uses or for any uses other than Noncommercial Research Uses;\n\n\nii. Distribution of Research Materials, and any derivative works thereof, are subject to the terms of this Agreement. If you distribute or make the Research Materials, or any derivative works thereof, available to a third party, you may only do so under the terms of this Agreement. You shall also provide a copy of this Agreement to such third party.\n\n\niii.  If you submit for publication the results of research you perform on, using, or otherwise in connection with Research Materials, you must acknowledge the use of Research Materials in your publication.\n\n\niv. Your use of the Research Materials must comply with applicable laws and regulations (including Trade Control Laws) and adhere to the FAIR Acceptable Use Policy, which is hereby incorporated by reference into this Agreement.\n2. User Support. Your Noncommercial Research Use of the Research Materials is done at your own discretion; Meta does not process any information nor provide any service in relation to such use.  Meta is under no obligation to provide any support services for the Research Materials. Any support provided is “as is”, “with all faults”, and without warranty of any kind.\n\n\n3. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE RESEARCH MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE RESEARCH MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE RESEARCH MATERIALS AND ANY OUTPUT AND RESULTS.\n \n4. Limitation of Liability. IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY DIRECT OR INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.\n \n5. Intellectual Property.\n\n\na. Subject to Meta’s ownership of Research Materials and derivatives made by or for Meta, with respect to any derivative works and modifications of the Research Materials that are made by you, as between you and Meta, you are and will be the owner of such derivative works and modifications.\n \nb. If you institute litigation or other proceedings against Meta or any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Research Materials, outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless Meta from and against any claim by any third party arising out of or related to your use or distribution of the Research Materials.\n \n6. Term and Termination. The term of this Agreement will commence upon your acceptance of this Agreement or access to the Research Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of the Research Materials. Sections 3, 4 and 7 shall survive the termination of this Agreement. \n \n7. Governing Law and Jurisdiction. This Agreement will be governed and construed under the laws of the State of California without regard to choice of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement. The courts of California shall have exclusive jurisdiction of any dispute arising out of this Agreement. \n\n\n8. Modifications and Amendments. Meta may modify this Agreement from time to time; provided that they are similar in spirit to the current version of the Agreement, but may differ in detail to address new problems or concerns. All such changes will be effective immediately. Your continued use of the Research Materials after any modification to this Agreement constitutes your agreement to such modification. Except as provided in this Agreement, no modification or addition to any provision of this Agreement will be binding unless it is in writing and signed by an authorized representative of both you and Meta.\n\n\nFAIR Acceptable Use Policy \n\nThe Fundamental AI Research (FAIR) team at Meta seeks to further understanding of new and existing research domains with the mission of advancing the state-of-the-art in artificial intelligence through open research for the benefit of all. \n\nAs part of this mission, Meta makes certain research materials available for noncommercial research use. Meta is committed to promoting the safe and responsible use of such research materials.  \n\nProhibited Uses\n\nYou agree you will not use, or allow others to use, Research Materials to:\n\n Violate the law or others’ rights, including to:\nEngage in, promote, generate, contribute to, encourage, plan, incite, or further illegal or unlawful activity or content, such as:\nViolence or terrorism\nExploitation or harm to children, including the solicitation, creation, acquisition, or dissemination of child exploitative content or failure to report Child Sexual Abuse Material\nHuman trafficking, exploitation, and sexual violence\nThe illegal distribution of information or materials to minors, including obscene materials, or failure to employ legally required age-gating in connection with such information or materials.\nSexual solicitation\nAny other criminal activity\n\nEngage in, promote, incite, or facilitate the harassment, abuse, threatening, or bullying of individuals or groups of individuals\n\nEngage in, promote, incite, or facilitate discrimination or other unlawful or harmful conduct in the provision of employment, employment benefits, credit, housing, other economic benefits, or other essential goods and services\n\nEngage in the unauthorized or unlicensed practice of any profession including, but not limited to, financial, legal, medical/health, or related professional practices\n\nCollect, process, disclose, generate, or infer health, demographic, or other sensitive personal or private information about individuals without rights and consents required by applicable laws\n\nEngage in or facilitate any action or generate any content that infringes, misappropriates, or otherwise violates any third-party rights, including the outputs or results of any technology using FAIR research materials\n\nCreate, generate, or facilitate the creation of malicious code, malware, computer viruses or do anything else that could disable, overburden, interfere with or impair the proper working, integrity, operation or appearance of a website or computer system\n\n2. Engage in, promote, incite, facilitate, or assist in the planning or development of activities that present a risk of death or bodily harm to individuals, including use of research artifacts related to the following:\n\nMilitary, warfare, nuclear industries or applications, espionage, use for materials or activities that are subject to the International Traffic Arms Regulations (ITAR) maintained by the United States Department of State\n\nGuns and illegal weapons (including weapon development)\n\nIllegal drugs and regulated/controlled substances\n\nOperation of critical infrastructure, transportation technologies, or heavy machinery\n\nSelf-harm or harm to others, including suicide, cutting, and eating disorders\n\nAny content intended to incite or promote violence, abuse, or any infliction of bodily harm to an individual\n\n3. Intentionally deceive or mislead others, including use of FAIR Research Materials related to the following:\n\n Generating, promoting, or furthering fraud or the creation or promotion of disinformation\n\n Generating, promoting, or furthering defamatory content, including the creation of defamatory statements, images, or other content\n\nGenerating, promoting, or further distributing spam\n\n Impersonating another individual without consent, authorization, or legal right\n\nRepresenting that outputs of FAIR research materials or outputs from technology using FAIR research materials are human-generated\n\nGenerating or facilitating false online engagement, including fake reviews and other means of fake online engagement\n\n4. Fail to appropriately disclose to end users any known dangers of your Research Materials.\n\nPlease report any violation of this Policy or other problems that could lead to a violation of this Policy by submitting a report here [https://docs.google.com/forms/d/e/1FAIpQLSeb11cryAopJ7LNrC4nxEUXrHY26hfkXQMf_uH-oFgA3WlYZQ/viewform]. \n"
        },
        {
          "name": "licenses/DINOV3.txt",
          "text": "# DINOv3 License\n\n*Last Updated: August 19, 2025*\n\n**“Agreement”** means the terms and conditions for use, reproduction, distribution and modification of the DINO Materials set forth herein.\n\n**“DINO Materials”** means, collectively, Documentation and the models, software and algorithms, including machine-learning model code, trained model weights, inference-enabling code, training-enabling code, fine-tuning enabling code, and other elements of the foregoing distributed by Meta and made available under this Agreement.\n\n**“Documentation”** means the specifications, manuals and documentation accompanying\nDINO Materials distributed by Meta.\n\n**“Licensee”** or **“you”** means you, or your employer or any other person or entity (if you are entering into this Agreement on such person or entity’s behalf), of the age required under applicable laws, rules or regulations to provide legal consent and that has legal authority to bind your employer or such other person or entity if you are entering in this Agreement on their behalf.\n\n**“Meta”** or **“we”** means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your principal place of business is in the EEA or Switzerland) or Meta Platforms, Inc. (if you are located outside of the EEA or Switzerland).\n\n**“Sanctions”** means any economic or trade sanctions or restrictions administered or enforced by the United States (including the Office of Foreign Assets Control of the U.S. Department of the Treasury (“OFAC”), the U.S. Department of State and the U.S. Department of Commerce), the United Nations, the European Union, or the United Kingdom.\n\n**“Trade Controls”** means any of the following: Sanctions and applicable export and import controls.\n\nBy clicking “I Accept” below or by using or distributing any portion or element of the DINO Materials, you agree to be bound by this Agreement.\n\n## 1. License Rights and Redistribution.\n\na. \u003cins>Grant of Rights\u003c/ins>. You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under Meta’s intellectual property or other rights owned by Meta embodied in the DINO Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the DINO Materials.\n\nb. \u003cins>Redistribution and Use\u003c/ins>.\n\ni. Distribution of DINO Materials, and any derivative works thereof, are subject to the terms of this Agreement. If you distribute or make the DINO Materials, or any derivative works thereof, available to a third party, you may only do so under the terms of this Agreement and you shall provide a copy of this Agreement with any such DINO Materials.\n\nii.  If you submit for publication the results of research you perform on, using, or otherwise in connection with DINO Materials, you must acknowledge the use of DINO Materials in your publication.\n\niii. Your use of the DINO Materials must comply with applicable laws and regulations, including Trade Control Laws and applicable privacy and data protection laws.\n\niv. Your use of the DINO Materials will not involve or encourage others to reverse engineer, decompile or discover the underlying components of the DINO Materials.\n\nv. You are not the target of Trade Controls and your use of DINO Materials must comply with Trade Controls. You agree not to use, or permit others to use, DINO Materials for any activities subject to the International Traffic in Arms Regulations (ITAR) or end uses prohibited by Trade Controls, including those related to military or warfare purposes, nuclear industries or applications, espionage, or the development or use of guns or illegal weapons.\n\n## 2. User Support.\n\nYour use of the DINO Materials is done at your own discretion; Meta does not process any information nor provide any service in relation to such use.  Meta is under no obligation to provide any support services for the DINO Materials. Any support provided is “as is”, “with all faults”, and without warranty of any kind.\n\n## 3. Disclaimer of Warranty.\n\nUNLESS REQUIRED BY APPLICABLE LAW, THE DINO MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE DINO MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE DINO MATERIALS AND ANY OUTPUT AND RESULTS.\n\n## 4. Limitation of Liability.\n\nIN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY DIRECT OR INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.\n\n## 5. Intellectual Property.\n\na. Subject to Meta’s ownership of DINO Materials and derivatives made by or for Meta, with respect to any derivative works and modifications of the DINO Materials that are made by you, as between you and Meta, you are and will be the owner of such derivative works and modifications.\n\nb. If you institute litigation or other proceedings against Meta or any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the DINO Materials, outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless Meta from and against any claim by any third party arising out of or related to your use or distribution of the DINO Materials.\n\n## 6. Term and Termination.\n\nThe term of this Agreement will commence upon your acceptance of this Agreement or access to the DINO Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of the DINO Materials. Sections 3, 4 and 7 shall survive the termination of this Agreement.\n\n## 7. Governing Law and Jurisdiction.\n\nThis Agreement will be governed and construed under the laws of the State of California without regard to choice of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement. The courts of California shall have exclusive jurisdiction of any dispute arising out of this Agreement.\n\n## 8. Modifications and Amendments.\n\nMeta may modify this Agreement from time to time; provided that they are similar in spirit to the current version of the Agreement, but may differ in detail to address new problems or concerns. All such changes will be effective immediately. Your continued use of the DINO Materials after any modification to this Agreement constitutes your agreement to such modification. Except as provided in this Agreement, no modification or addition to any provision of this Agreement will be binding unless it is in writing and signed by an authorized representative of both you and Meta.\n"
        },
        {
          "name": "licenses/MOGE.txt",
          "text": "    MIT License\n\n    Copyright (c) Microsoft Corporation.\n\n    Permission is hereby granted, free of charge, to any person obtaining a copy\n    of this software and associated documentation files (the \"Software\"), to deal\n    in the Software without restriction, including without limitation the rights\n    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n    copies of the Software, and to permit persons to whom the Software is\n    furnished to do so, subject to the following conditions:\n\n    The above copyright notice and this permission notice shall be included in all\n    copies or substantial portions of the Software.\n\n    THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n    SOFTWARE\n\n\n                                 Apache License\n                           Version 2.0, January 2004\n                        http://www.apache.org/licenses/\n\n   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n\n   1. Definitions.\n\n      \"License\" shall mean the terms and conditions for use, reproduction,\n      and distribution as defined by Sections 1 through 9 of this document.\n\n      \"Licensor\" shall mean the copyright owner or entity authorized by\n      the copyright owner that is granting the License.\n\n      \"Legal Entity\" shall mean the union of the acting entity and all\n      other entities that control, are controlled by, or are under common\n      control with that entity. For the purposes of this definition,\n      \"control\" means (i) the power, direct or indirect, to cause the\n      direction or management of such entity, whether by contract or\n      otherwise, or (ii) ownership of fifty percent (50%) or more of the\n      outstanding shares, or (iii) beneficial ownership of such entity.\n\n      \"You\" (or \"Your\") shall mean an individual or Legal Entity\n      exercising permissions granted by this License.\n\n      \"Source\" form shall mean the preferred form for making modifications,\n      including but not limited to software source code, documentation\n      source, and configuration files.\n\n      \"Object\" form shall mean any form resulting from mechanical\n      transformation or translation of a Source form, including but\n      not limited to compiled object code, generated documentation,\n      and conversions to other media types.\n\n      \"Work\" shall mean the work of authorship, whether in Source or\n      Object form, made available under the License, as indicated by a\n      copyright notice that is included in or attached to the work\n      (an example is provided in the Appendix below).\n\n      \"Derivative Works\" shall mean any work, whether in Source or Object\n      form, that is based on (or derived from) the Work and for which the\n      editorial revisions, annotations, elaborations, or other modifications\n      represent, as a whole, an original work of authorship. For the purposes\n      of this License, Derivative Works shall not include works that remain\n      separable from, or merely link (or bind by name) to the interfaces of,\n      the Work and Derivative Works thereof.\n\n      \"Contribution\" shall mean any work of authorship, including\n      the original version of the Work and any modifications or additions\n      to that Work or Derivative Works thereof, that is intentionally\n      submitted to Licensor for inclusion in the Work by the copyright owner\n      or by an individual or Legal Entity authorized to submit on behalf of\n      the copyright owner. For the purposes of this definition, \"submitted\"\n      means any form of electronic, verbal, or written communication sent\n      to the Licensor or its representatives, including but not limited to\n      communication on electronic mailing lists, source code control systems,\n      and issue tracking systems that are managed by, or on behalf of, the\n      Licensor for the purpose of discussing and improving the Work, but\n      excluding communication that is conspicuously marked or otherwise\n      designated in writing by the copyright owner as \"Not a Contribution.\"\n\n      \"Contributor\" shall mean Licensor and any individual or Legal Entity\n      on behalf of whom a Contribution has been received by Licensor and\n      subsequently incorporated within the Work.\n\n   2. Grant of Copyright License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      copyright license to reproduce, prepare Derivative Works of,\n      publicly display, publicly perform, sublicense, and distribute the\n      Work and such Derivative Works in Source or Object form.\n\n   3. Grant of Patent License. Subject to the terms and conditions of\n      this License, each Contributor hereby grants to You a perpetual,\n      worldwide, non-exclusive, no-charge, royalty-free, irrevocable\n      (except as stated in this section) patent license to make, have made,\n      use, offer to sell, sell, import, and otherwise transfer the Work,\n      where such license applies only to those patent claims licensable\n      by such Contributor that are necessarily infringed by their\n      Contribution(s) alone or by combination of their Contribution(s)\n      with the Work to which such Contribution(s) was submitted. If You\n      institute patent litigation against any entity (including a\n      cross-claim or counterclaim in a lawsuit) alleging that the Work\n      or a Contribution incorporated within the Work constitutes direct\n      or contributory patent infringement, then any patent licenses\n      granted to You under this License for that Work shall terminate\n      as of the date such litigation is filed.\n\n   4. Redistribution. You may reproduce and distribute copies of the\n      Work or Derivative Works thereof in any medium, with or without\n      modifications, and in Source or Object form, provided that You\n      meet the following conditions:\n\n      (a) You must give any other recipients of the Work or\n          Derivative Works a copy of this License; and\n\n      (b) You must cause any modified files to carry prominent notices\n          stating that You changed the files; and\n\n      (c) You must retain, in the Source form of any Derivative Works\n          that You distribute, all copyright, patent, trademark, and\n          attribution notices from the Source form of the Work,\n          excluding those notices that do not pertain to any part of\n          the Derivative Works; and\n\n      (d) If the Work includes a \"NOTICE\" text file as part of its\n          distribution, then any Derivative Works that You distribute must\n          include a readable copy of the attribution notices contained\n          within such NOTICE file, excluding those notices that do not\n          pertain to any part of the Derivative Works, in at least one\n          of the following places: within a NOTICE text file distributed\n          as part of the Derivative Works; within the Source form or\n          documentation, if provided along with the Derivative Works; or,\n          within a display generated by the Derivative Works, if and\n          wherever such third-party notices normally appear. The contents\n          of the NOTICE file are for informational purposes only and\n          do not modify the License. You may add Your own attribution\n          notices within Derivative Works that You distribute, alongside\n          or as an addendum to the NOTICE text from the Work, provided\n          that such additional attribution notices cannot be construed\n          as modifying the License.\n\n      You may add Your own copyright statement to Your modifications and\n      may provide additional or different license terms and conditions\n      for use, reproduction, or distribution of Your modifications, or\n      for any such Derivative Works as a whole, provided Your use,\n      reproduction, and distribution of the Work otherwise complies with\n      the conditions stated in this License.\n\n   5. Submission of Contributions. Unless You explicitly state otherwise,\n      any Contribution intentionally submitted for inclusion in the Work\n      by You to the Licensor shall be under the terms and conditions of\n      this License, without any additional terms or conditions.\n      Notwithstanding the above, nothing herein shall supersede or modify\n      the terms of any separate license agreement you may have executed\n      with Licensor regarding such Contributions.\n\n   6. Trademarks. This License does not grant permission to use the trade\n      names, trademarks, service marks, or product names of the Licensor,\n      except as required for reasonable and customary use in describing the\n      origin of the Work and reproducing the content of the NOTICE file.\n\n   7. Disclaimer of Warranty. Unless required by applicable law or\n      agreed to in writing, Licensor provides the Work (and each\n      Contributor provides its Contributions) on an \"AS IS\" BASIS,\n      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or\n      implied, including, without limitation, any warranties or conditions\n      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A\n      PARTICULAR PURPOSE. You are solely responsible for determining the\n      appropriateness of using or redistributing the Work and assume any\n      risks associated with Your exercise of permissions under this License.\n\n   8. Limitation of Liability. In no event and under no legal theory,\n      whether in tort (including negligence), contract, or otherwise,\n      unless required by applicable law (such as deliberate and grossly\n      negligent acts) or agreed to in writing, shall any Contributor be\n      liable to You for damages, including any direct, indirect, special,\n      incidental, or consequential damages of any character arising as a\n      result of this License or out of the use or inability to use the\n      Work (including but not limited to damages for loss of goodwill,\n      work stoppage, computer failure or malfunction, or any and all\n      other commercial damages or losses), even if such Contributor\n      has been advised of the possibility of such damages.\n\n   9. Accepting Warranty or Additional Liability. While redistributing\n      the Work or Derivative Works thereof, You may choose to offer,\n      and charge a fee for, acceptance of support, warranty, indemnity,\n      or other liability obligations and/or rights consistent with this\n      License. However, in accepting such obligations, You may act only\n      on Your own behalf and on Your sole responsibility, not on behalf\n      of any other Contributor, and only if You agree to indemnify,\n      defend, and hold each Contributor harmless for any liability\n      incurred by, or claims asserted against, such Contributor by reason\n      of your accepting any such warranty or additional liability.\n\n   END OF TERMS AND CONDITIONS\n\n   APPENDIX: How to apply the Apache License to your work.\n\n      To apply the Apache License to your work, attach the following\n      boilerplate notice, with the fields enclosed by brackets \"[]\"\n      replaced with your own identifying information. (Don't include\n      the brackets!)  The text should be enclosed in the appropriate\n      comment syntax for the file format. We also recommend that a\n      file or class name and description of purpose be included on the\n      same \"printed page\" as the copyright notice for easier\n      identification within third-party archives.\n\n   Copyright [yyyy] [name of copyright owner]\n\n   Licensed under the Apache License, Version 2.0 (the \"License\");\n   you may not use this file except in compliance with the License.\n   You may obtain a copy of the License at\n\n       http://www.apache.org/licenses/LICENSE-2.0\n\n   Unless required by applicable law or agreed to in writing, software\n   distributed under the License is distributed on an \"AS IS\" BASIS,\n   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n   See the License for the specific language governing permissions and\n   limitations under the License.\n"
        },
        {
          "name": "licenses/PI3.txt",
          "text": "Copyright (c) 2025, the authors\nAll rights reserved.\n\nRedistribution and use in source and binary forms, with or without\nmodification, are permitted provided that the following conditions are met:\n\n1. Redistributions of source code must retain the above copyright notice, this\n   list of conditions and the following disclaimer.\n\n2. Redistributions in binary form must reproduce the above copyright notice,\n   this list of conditions and the following disclaimer in the documentation\n   and/or other materials provided with the distribution.\n\n3. Neither the name of the copyright holder nor the names of its\n   contributors may be used to endorse or promote products derived from\n   this software without specific prior written permission.\n\nTHIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\"\nAND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE\nIMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE\nDISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE\nFOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL\nDAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR\nSERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER\nCAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,\nOR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE\nOF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n"
        }
      ],
      "url": "https://github.com/yyfz/Pi4/blob/32dfbea9ce6268ea17446208a804710df9475c32/LICENSE"
    }
  }
};

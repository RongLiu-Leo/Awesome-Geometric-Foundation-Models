# Contributing a model

The catalog covers learned models that recover scene geometry or motion from images, video, or accompanying measurements. A useful entry tells readers what goes in, what comes out, and which design choice distinguishes the method. Link the authors’ paper and, when available, their project page and implementation.

1. Add one object to the array in [data.js](data.js). Pick a unique lowercase ID, using hyphens between words. The ID becomes the permanent link to the profile. Use the earliest public paper date (arXiv v1 for arXiv papers). Reuse a matching architecture group, or give a new group a descriptive name.
2. Run `node scripts/update-readme.cjs` from the repository root. It checks the entries and refreshes the paper lists in the GitHub README and the homepage HTML. Open `index.html` to preview the index, roadmap, and comparison. The website works without a build.
3. Send a pull request with the catalog, README, and homepage changes. Mention the paper or upstream section supporting the input and output descriptions.

The required fields are shown below, using an abbreviated version of the existing DUSt3R entry.

```js
{
  id: "dust3r",
  name: "DUSt3R",
  title: "DUSt3R: Geometric 3D Vision Made Easy",
  preprint: "2023-12-21",
  group: "Pairwise",
  summary: "Two uncalibrated views become dense pointmaps in the first camera’s frame.",
  paper: "https://arxiv.org/abs/2312.14132",
  inputs: "Two RGB images.",
  output: "A 3D point and confidence value for each pixel in each view."
}
```

Add `project` and `code` for official links. For a fuller profile, add `stages` as an ordered list of `[layer or operation, explanation]` pairs, then describe `exchange`, `scale`, `motion`, `strength`, or `limit` where the paper supports them. The site can show a model before it has custom artwork or a source walkthrough.

For a source walkthrough, follow an existing entry in `walk-data.js`. Pin excerpts to a commit and line range, identify the upstream repository for each excerpt, and include its license notices. Run `node scripts/update-readme.cjs --check` to check the catalog, README, and homepage without writing. With Chrome or Chromium installed, `node --test tests/catalog-growth.test.cjs` also exercises adding a model in the browser.

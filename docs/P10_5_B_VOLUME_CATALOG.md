# P10.5-B - Volume slice catalog and previews

## Scope

P10.5-B extends real 2D inference outputs with a volume-aware slice catalog when the input is a volumetric study such as MHA/MHD. The model still runs inference only on the selected slice. The service does not infer masks, landmarks, or measurements for every slice.

## Runtime behavior

- The input volume is loaded once by the existing real inference runtime.
- A PNG preview is generated for each slice from the already-loaded canonical volume.
- Existing assets remain unchanged: `input.png`, `overlay.png`, `mask.npy`, `confidence.npy`, and `mask-preview.png`.
- The selected/inferred slice also receives a slice-specific overlay asset named `slice-###-overlay.png`.
- Non-selected slices have `hasResults=false`, `overlayAsset=null`, and empty `measurementIds` / `landmarkIds`.

## Slice contract

Each slice entry is exposed under `plane.input.slices` in the canonical v2 response and under `plane.metadata.slices` in the legacy adapter for backend compatibility.

```json
{
  "index": 1,
  "displayIndex": 2,
  "previewAsset": {
    "assetName": "slice-001.png",
    "role": "slice-preview",
    "contentType": "image/png",
    "generated": true,
    "url": "/assets/{runId}/{plane}/slice-001.png"
  },
  "hasResults": true,
  "overlayAsset": {
    "assetName": "slice-001-overlay.png",
    "role": "slice-overlay",
    "contentType": "image/png",
    "generated": true,
    "url": "/assets/{runId}/{plane}/slice-001-overlay.png"
  },
  "measurementIds": ["axial-raw_50-area"],
  "landmarkIds": ["lm-mask-axial-raw-50-centroid"]
}
```

## Geometry metadata

`originMm`, `directionMatrix`, and `geometryComplete` are propagated from image headers when available. Default or incomplete header values are not promoted as complete geometry. This keeps volumetric traceability honest until P10.5-C/P9-A.3 can use full DICOM geometry.

## Sanitization

Public slice entries expose asset names and relative API URLs only. They do not expose source paths, output directories, temporary directories, hosts, tokens, or raw filesystem locations.


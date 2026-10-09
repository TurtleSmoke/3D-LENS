# Synthetic view generation

This folder builds the synthetic training set of 3D-LENS (Sec. 3.2 of the paper) from a real dataset.

| Step                | Script                                        | Input                 | Output                             |
|---------------------|-----------------------------------------------|-----------------------|------------------------------------|
| 1. Segmentation     | `segment.py`                                  | real images           | RGBA masks                         |
| 2. 3D lifting       | Hunyuan3D-2 (external)                        | RGBA masks            | textured meshes (`.glb`)           |
| 3. Pose calibration | `calibrate_pose.py`                           | meshes + masks        | `<mesh>.csv` (pose, IoU)           |
| 4. Novel views      | `render_views.py`                             | calibrated meshes     | foreground renders `_00` ... `_09` |
| 5. Background       | `composite_background.py`                     | renders + real images | composites + StyleID lists         |
| 6. Style transfer   | StyleID (external) + `styleid/run_styleid.py` | composites            | final synthetic views              |
| 7. Image lists      | `build_lists.sh`                              | dataset directory     | `train_all.txt`                    |

Install the generation dependencies with `uv sync --group generation`.
Steps 3 and 4 render offscreen with pyrender (`PYOPENGL_PLATFORM=egl`, a GPU is needed).
All scripts take a list of files for splitting across GPUs or jobs.

The example below is for AG-ReID, and `$DATA` is the directory holding the datasets.

## 1. Segmentation

YOLO11 instance segmentation (`yolo11l-seg.pt`, downloaded by Ultralytics on first use).

```bash
python generation/segment.py --root $DATA/AG-ReID --output $DATA/AG-ReID_segmented \
    --model yolo11l-seg.pt $DATA/AG-ReID/bounding_box_train/*.jpg
```

## 2. 3D lifting with Hunyuan3D-2

> For optimal results, it is recommended to use more recent versions of Hunyuan3D-2 (or other, better 3D lifting
> methods), which may improve quality and speed.

We use [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) at commit `404cf4c`, with the `tencent/Hunyuan3D-2`
weights, to lift every segmented image (RGBA PNG from step 1) to a textured mesh.

Save each textured mesh as `$DATA/AG-ReID_meshes/<relative path of the image>.glb`, for example
`AG-ReID_meshes/bounding_box_train/P0001T04041A0C0F121.glb`.

## 3. Pose calibration

Finds the camera yaw and pitch that maximize the IoU between the rendered silhouette and the mask (Eq. 1).

```bash
python generation/calibrate_pose.py --mesh-root $DATA/AG-ReID_meshes --segmented-root $DATA/AG-ReID_segmented \
    $DATA/AG-ReID_meshes/bounding_box_train/*.glb
```

## 4. Novel views

Meshes with IoU < 0.7 are discarded. Each remaining mesh gets 10 views, with the elevation shifted by 0 to 30 degrees
toward the other viewpoint (up for ground images, down for aerial images), azimuth shifted uniformly in [-30, 30]
degrees.

```bash
python generation/render_views.py --mesh-root $DATA/AG-ReID_meshes --output $DATA/AG-ReID_renders \
    $DATA/AG-ReID_meshes/bounding_box_train/*.glb
```

## 5. Background compositing

Each render is composited on the background of a random real image of the same camera type, whose person is removed
with [LaMa](https://github.com/advimman/lama) (`simple-lama-inpainting`).

```bash
python generation/composite_background.py --render-root $DATA/AG-ReID_renders \
    --segmented-root $DATA/AG-ReID_segmented --real-root $DATA/AG-ReID --output $DATA/AG-ReID_composite \
    $DATA/AG-ReID_renders/bounding_box_train/*.png
```

## 6. Style transfer with StyleID

We use [StyleID](https://github.com/jiwoogit/StyleID) at commit `d4ea655` with Stable Diffusion v1.4
(`models/ldm/stable-diffusion-v1/model.ckpt`). Replace its `run_styleid.py` with our `styleid/run_styleid.py`, which
adapts the original StyleID to person crops and optimizes the speed and memory usage for our use case.

```bash
cd StyleID && git checkout d4ea655 && cp /path/to/3D-LENS/generation/styleid/run_styleid.py .
python run_styleid.py --cnt $DATA/AG-ReID_composite/content_0.txt --sty $DATA/AG-ReID_composite/style_0.txt \
    --cnt_prefix $(realpath $DATA/AG-ReID_composite)/ --output_path $DATA/AG-ReID_3D-LENS \
    --gamma 0.9 --T 1.5 --precision full --without_init_adain
```

## 7. Image lists

```bash
bash generation/build_lists.sh AG-ReID $DATA/AG-ReID            # real images
bash generation/build_lists.sh AG-ReID $DATA/AG-ReID_3D-LENS    # synthetic views
```

## AG-ReID.v2 and MOO

- **AG-ReID.v2**. Run the same steps with `--root $DATA/AG-ReID.v2` (images under `AG-ReID.v2/train_all/`), and
  `build_lists.sh AG-ReID.v2` on `$DATA/AG-ReID.v2` and `$DATA/AG-ReID.v2_3D-LENS`.
- **MOO**. The images are already segmented renders, so steps 1, 5 and 6 are skipped. Lift the MOO images directly
  (step 2), calibrate with `--segmented-root $DATA/MOO`, render to `$DATA/MOO_3D-LENS` (step 4), then run
  `build_lists.sh MOO` on both directories. MOO views keep their black background, like the real MOO images.

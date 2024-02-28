#### Repository for: Motion History Images: a New Method for Tracking Microswimmers in 3D

The authors of the code in this repository are Hadi Albalkhi* and Max Riekeles*.
Contact: riekeles@tu-berlin.de

All software is licensed according to the [JPL Software License](https://github.com/JPLMLIA/OWLS-Autonomy/blob/master/LICENSE.md)


_* Both authors contributed to the code in equal shares._

## Prerequisites

- Python 3.8
- Python packages:
    - numpy
    - opencv2
    - matplotlib
- For generating the MHI refer to the [OWLS-repository](https://github.com/JPLMLIA/OWLS-Autonomy)
- Datasets described in the paper can be found here:
    - [Raw holograms frames of Bacillus Subtilis at different temperatures](https://doi.org/10.5061/dryad.ns1rn8pv6)
    - [Dataset for “Identifying and Characterizing Motile and Fluorescent Microorganisms in Microscopy Data Using Onboard Science Autonomy”](https://doi.org/10.48577/jpl.2KTVW5)

After the reconstructions of the holograms have been generated, a typical run of the code is:

1. Create maximum z-projections
2. Create MHI (with [OWLS-repository](https://github.com/JPLMLIA/OWLS-Autonomy)) from the maximum z-projections
3. Perform blob detection in the maximum z-projections at each time point
4. Apply region growing to the MHI (either manually or with automatic seedpoint selection)
5. Apply Z-layer selection
    1. apply DBSCAN if necessary.
    2. apply time filtering if necessary.
6. 3D Plot

## Max Z-Projection

Calculates the maximum projection through the Z stack of reconstructed holograms for the given range of timepoints.

### Usage

- the script assumes that `<reconstructions>`:
    - contains multiple folders for each timepoint of the recorded experiment.
    - each folder contains the reconstruction of the hologram at the corresponding timepoint.
    - the folder names are five-digit numbers  `xxxxx` (with leading zeros where needed) which represent the
      timestamps.
      The timestamps will be written to the resulting filenames and will be used in consequent processing steps (blob
      detection for instance).

- example run:
  ```
  python max_z_projections.py \
    --dataset ds1 --reconstructions /ds1/reconstructions \
    --output-dir /path/to/output \
    --z-start-plane -200 \
    --z-end-plane 200 \
    --end-timepoint 851
  ```
- results will be saved to `<output_dir>/` (or per default to `./<data_set>_max_z_projection_results` &Dagger;) and the
  max Z-projection filenames will be in the form:
  `max_z_projection_zs_<z_start_plane>_ze_<z_end_plane>_<z_plane_jump_steps>_<timestamp>.tif`

### Parameters Overview

| Argument                | Short | Required | Default Value | Description                                                                                |
|-------------------------|-------|----------|---------------|--------------------------------------------------------------------------------------------|
| `--reconstructions`     | `-r`  | Yes      | N/A           | Path to reconstructions of the holograms - contains folders of all timepoints              |
| `--dataset`             | `-d`  | Yes      | N/A           | Name of the dataset of the recorded experiment - used only for naming the resulting files. |
| `--output-dir`          | `-o`  | No       | &Dagger;      | Path to the directory where the results will be stored                                     |
| `--start-timepoint`     | `-ts` | No       | `1`           | A number representing the start timepoint of the range to be considered                    |
| `--end-timepoint`       | `-te` | Yes      | N/A           | A number representing the end timepoint of the range to be considered                      |
| `--z-start-plane`       | `-zs` | Yes      | N/A           | Z-plane start value                                                                        |
| `--z-end-plane`         | `-ze` | Yes      | N/A           | Z-plane end value                                                                          |
| `--z-plane-jump-steps`  | `-zj` | No       | `1`           | Z-plane jump steps                                                                         |
| `--zeros-padding-width` | -     | No       | `5`           | Zeros padding width of folder names of reconstructed timepoints                            |

## Blob Detection

This script applies OpenCV SimpleBlobDetector on provided max Z-projection images and save the results as NumPy
array files (`.npy`).

### Usage

- the script requires `<max_z_projections>` to contain the max Z-projections image files with a specific naming format
  as described in max Z-projection above:
  `some_file_name_xxxxx.tif`
  where `xxxxx` is a five-digit number (with leading zeros where needed) which represents the timestamp.
    - this is important because the script will extract the timestamp i.e. the value of the time frame out of the
      filename.
- example run:

  ```
  python blob_detection.py \
      --dataset ds1 \
      --max-z-projections /ds1/projections 
  ```
- results will be saved to `<output_dir>/` (or per default to `./<data_set>_blob_detection_results` &Dagger;) as NumPy
  array files (`.npy`).
  Each blob is itself an array of the form:

  `[x, y, timestamp, <diameter_of_detected_blob>]`

### Parameters Overview

| Argument              | Short | Required | Default Value | Description                                                                                     |
|-----------------------|-------|----------|---------------|-------------------------------------------------------------------------------------------------|
| `--max-z-projections` | `-p`  | Yes      | N/A           | Path to the max Z-projection files of all time points.                                          |
| `--dataset`           | `-d`  | Yes      | N/A           | Name of the dataset of the MHI, used only for naming the resulting track segment.               |
| `--output-dir`        | `-o`  | No       | &Dagger;      | Path to the directory where the results will be stored.                                         |
| `--min-area`          | -     | No       | `60`          | Change the minimum area (in pixels) of blobs, smaller blobs will be filtered out (default: 60). |
| `--max-area`          | -     | No       | `300`         | Change the maximum area (in pixels) of blobs, larger blobs will be filtered out (default: 300). |
| `--max-threshold`     | -     | No       | `255`         | Change the maxThreshold value for openCV SimpleBlobDetector (default: 255).                     |

## Region Growing - manual seed selection

Performs region growing on a given MHI from a manually selected seed point.

### Usage

- the script requires `<path_to_mhi>` to contain two files: `<data_set>_mhi.npy` and `<data_set>_mhi.png`
  For instance: when dataset is `ds1` then the files should be `ds1_mhi.npy`and `DS1_mhi.png`
- the seed can be set by clicking on the mhi or by providing the x and y coordinates from the command line.
- example run:

  ```
  python region_growing_manual_seed_selection.py \
      --dataset ds1 \
      --path-to-mhi /ds1/path/to/mhi \ 
      --track-nr 1 \
      --radius 5 \
      --threshold 5 \
      --n-values-to-ignore 3 \
      --seed-x 954 \
      --seed-y 110
  ```

    - After choosing the seedpoint you have to **press the space bar** key **to continue**.
    - If the seedpoint coordinates were provided from the command line, then just press the space bar key to continue.
      Any clicks will be ignored.
    - When the execution is finished the resulting track segment is show. To close the windows press the space bar
      again.
- results will be saved to `<output_dir>` (or per default to `./<data_set>_region_growing_results` &Dagger;)

### Parameters Overview

| Argument                | Short  | Required | Default Value | Description                                                                               |
|-------------------------|--------|----------|---------------|-------------------------------------------------------------------------------------------|
| `--path-to-mhi`         | `-mhi` | Yes      | N/A           | path to the motion history image (MHI) file.                                              |
| `--dataset`             | `-d`   | Yes      | N/A           | name of the dataset of the MHI, used only for naming the resulting track segment.         |
| `--output-dir`          | `-o`   | No       | &Dagger;      | path to the directory where the results will be stored.                                   |
| `--track-nr`            | `-t`   | No       | `1`           | track number, used only for naming the resulting track segment (default: 1).              |
| `--radius`              | `-r`   | No       | `5`           | search radius for neighboring pixels (default: 5).                                        |
| `--threshold`           | -      | No       | `5`           | the time difference threshold for comparing two pixels (default: 5).                      |
| `--rejection-threshold` | -      | No       | `1`           | number of rejection votes required for a pixel to be excluded from a region (default: 1). |
| `--n-values-to-ignore`  | `-n`   | No       | `3`           | used to filter out the n most prominent values in the MHI (default: 3).                   | `-zj` |  |
| `--seed-x`              | `-x`   | No       | N/A           | x-coordinate of the seed point for the region growing.                                    | -     ||
| `--seed-y`              | `-y`   | No       | N/A           | y-coordinate of the seed point for the region growing.                                    |

## Z-Layer Selection

The script uses the output of previous steps along with the reconstructed holograms of the recorded experiment to find
the best estimate of the z coordinate of each blob on a given track segment.

### Usage

- example run:

  ```
  python z_layer_selection.py  \
      --dataset ds1 \
      --mhi-npy-file /ds1/path/to/ds1_mhi.npy \
      --blobs-file /ds1/blob_detection_results/detected_blobs.npy \
      --reconstructions /ds1/path/to/holograms/reconstructions \
      --tracks /ds1/region_growing_results/path/to/2dtracks \
      --z-start-plane -170 \
      --z-end-plane 26
  ```
- results will be saved to `<output_dir>/` (or per default to `./<dataset>_z_layer_selection_results` &Dagger;) as NumPy
  array files (`.npy`)
  Each point is itself an array of the form:

  `[x, y, z, timestamp, <diameter_of_detected_blob>]`

### Parameters Overview

| Argument                | Short  | Required | Default Value | Description                                                                                                                                                  |
|-------------------------|--------|----------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--dataset`             | `-d`   | Yes      | N/A           | Name of the dataset of the MHI                                                                                                                               |
| `--mhi-npy-file`        | `-mhi` | Yes      | N/A           | Path to the motion history image (MHI) npy-file                                                                                                              |
| `--reconstructions`     | `-r`   | Yes      | N/A           | Path to reconstructions of the holograms - contains folders of all timepoints                                                                                |
| `--tracks`              | `-tr`  | Yes      | N/A           | Path to the 2D track segments, which are the results of region growing                                                                                       |
| `--blobs-file`          | `-b`   | Yes      | N/A           | Path to the detected blobs npy-file. The file is the result of blob_detection. The file contains the blobs of all timepoints regardless of any track segment |
| `--z-start-plane`       | `-zs`  | Yes      | N/A           | Z-plane start value**                                                                                                                                        |
| `--z-end-plane`         | `-ze`  | Yes      | N/A           | Z-plane end value**                                                                                                                                          |
| `--output-dir`          | `-o`   | No       | &Dagger;      | The directory to which the results will be stored                                                                                                            |
| `--mhi-time-tolerance`  | -      | No       | N/A           | MHI time tolerance - if not provided no filtering will take place.                                                                                           |
| `--min-diameter-size`   | -      | No       | `0`           | Blobs with smaller diameter size will not be considered                                                                                                      |
| `--zeros-padding-width` | -      | No       | `5`           | Zeros padding width of folder names of reconstructed timepoints                                                                                              |

** this should be the exact value used to calculate the may-Z projections. In future versions of this script, these
values will be read automatically.

## Region Growing - automatic seed selection

#### Usage

- example run:

```
python region_growing_automatic_seed_selection.py \
    --dataset ds1 \
    --mhi-npy-file /ds1/path/to/ds1_mhi.npy \
    --blobs-file /ds1/blob_detection_results/detected_blobs.npy \
    --output-dir /path/to/output \
    --radius 7 \
    --threshold 5 \
    --rejection-threshold 1 \
    --n-values-to-ignore 3 \
    --min-track-length 500
```

- results will be saved to `<output_dir>/` (or per default to `./<dataset>_region_growing_automatic_results` &Dagger;)

### Parameters Overview

| Argument                | Short  | Required | Default Value | Description                                                                                                                                                   |
|-------------------------|--------|----------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--mhi-npy-file`        | `-mhi` | Yes      | N/A           | Path to the motion history image (MHI) npy-file.                                                                                                              |
| `--blobs-file`          | `-b`   | Yes      | N/A           | Path to the detected blobs npy-file. The file is the result of blob_detection. The file contains the blobs of all timepoints regardless of any track segment. |
| `--output-dir`          | `-o`   | No       | &Dagger;      | The directory to which the results will be stored.                                                                                                            |
| `--dataset`             | `-d`   | Yes      | N/A           | Name of the dataset of the MHI - used only for naming the resulting track segment.                                                                            |
| `--radius`              | `-r`   | No       | `5`           | Search radius for neighboring pixels.                                                                                                                         |
| `--threshold`           | -      | No       | `5`           | The time difference threshold for comparing two pixels.                                                                                                       |
| `--rejection-threshold` | -      | No       | `1`           | Number of rejection votes for a pixel to be excluded from a region.                                                                                           |
| `--n-values-to-ignore`  | `-n`   | No       | `3`           | Used to filter out the most prominent n values in the MHI.                                                                                                    |
| `--min-track-length`    | -      | No       | `500`         | Tracks smaller than the minimum track length (in pixel count) will be ignored.                                                                                |

## 3d plot

The script plots the output of the Z-layer selection in 3D.

### Usage

In line `8`, set the variable `centers_max_z` to the path to the 3D track npy-file, which will be plotted.

## DBSCAN

The script performs clustering additionally in the z-axis with DBSCAN. 
It is useful when the output of the Z-layer selection potentially consists of several tracks. 
The results strongly depended on the distance metric and the maximum distance allowed for a point to 
be considered a part of a cluster. 
The parameter `min_samples` is the minimum number for a group of points - close enough to each other - to be considered a cluster.  
The clustered tracks are saved separately each in its own npy-file.

### Usage

- example run:

```
python dbscan.py \
    --track-3d ./some_3d_track.npy \
    --dbscan-eps 0.15 \
    --dbscan-min-samples 10
```

### Parameters Overview
| Argument               | Short   | Required | Default Value | Description                           |
|------------------------|---------|----------|---------------|---------------------------------------|
| `--track-3d`           | `-ts3d` | Yes      | N/A           | Path to the 3D track segment npy-file |
| `--dbscan-eps`         | `-eps`  | No       | `0.15`        | DBSCAN EPS parameter                  |
| `--dbscan-min-samples` | `-ms`   | No       | `10`          | DBSCAN min_samples parameter          |

## Time filtering

After applying the Z-layer selection, usually the output is very noisy and this script could be useful
for eliminating unrelated particles. 
This is done by comparing the time associated with each particle to the corresponding time value on
the MHI.

### Usage

- results will be saved to `<output_dir>/` (or per default to `./<dataset>_time_filtering_results` &Dagger;) as NumPy
  array files (`.npy`)
- example run:

```
python time_filtering.py \
    --dataset ds1 \
    --output-dir /path/to/output \
    --mhi-npy-file /path/to/ds1_mhi.npy \
    --track-2d /path/to/track_2d_nr_1_ds1.npy \
    --track-3d /path/to/track_3d_nr_1_ds1.npy \
    --mhi-time-tolerance 5
```

### Parameters Overview

| Argument               | Short   | Required | Default Value | Description                                                                        |
|------------------------|---------|----------|---------------|------------------------------------------------------------------------------------|
| `--dataset`            | `-d`    | Yes      | N/A           | Name of the dataset of the MHI - used only for naming the resulting track segment. |
| `--mhi-npy-file`       | `-mhi`  | Yes      | N/A           | Path to the motion history image (MHI) npy-file.                                   |
| `--track-2d`           | `-ts2d` | Yes      | N/A           | Path to the 2D track segment npy-file, which corresponds to the 3D track.          |
| `--track-3d`           | `-ts3d` | Yes      | N/A           | Path to the 3D track segment npy-file, which will be filtered.                     |
| `--mhi-time-tolerance` | -       | No       | `None`        | MHI time tolerance - if not provided no filtering will take place.                 |
| `--output-dir`         | `-o`    | No       | &Dagger;      | The directory to which the results will be stored.                                 |



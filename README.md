# PKLotAnnotator

Annotates the Blanderbuss PKLot dataset - https://www.kaggle.com/datasets/blanderbuss/parking-lot-dataset

1. Runs YOLOv8 on all images. Saves the images with predicted classes and bounding boxes with the `<imagepath>-pred.<ext>` suffix.
2. For each detection, compares the detected bounding box with the ground truth bounding box found in `<imagepath>.xml`. When a detection is found, counts as a positive presence; otherwise, missed detections count as non-detected presences. These results are saved as `<imagepath>-pred.xml`
3. For each XML, makes an equivalent `<imagepath>.npz` and `<imagepath>-pred.npz`. The first element of each vector in the matrix is the UNIX timestamp equivalent of `<imagepath>`. The other elements are the parking lot spots and their detections (1 if detected, 0 otherwise).
4. For each parking lot, aggregates the `.npz` into `<parking lot>/<parking lot>.npz` and `<parking lot>/<parking lot>-pred.npz`.
5. For each parking lot, compares the pred and ground truth aggregated `.npz` and reports the accuracy in `<parking lot>/accuracy.txt`.

## PKLot dataset supplementary information

I do not own this dataset. 

This dataset contains two parking lots, one of which has two vantage points (we thus consider this to be three parking lots). 
Each parking lot has different atmospheric/weather conditions, which we ignore. Each file name is the timestamp of the detection.

The `tree .` of this dataset is

```
tree . --filelimit 13
.
├── PUCPR
│   ├── Cloudy
│   │   ├── 2012-09-12  [102 entries exceeds filelimit, not opening dir]
│   │   ├── 2012-09-16  [290 entries exceeds filelimit, not opening dir]
│   │   ├── 2012-09-28  [264 entries exceeds filelimit, not opening dir]
│   │   ├── 2012-10-05  [20 entries exceeds filelimit, not opening dir]
│   │   ├── 2012-10-12  [308 entries exceeds filelimit, not opening dir]
│   │   ├── 2012-10-13  [310 entries exceeds filelimit, not opening dir]
│   │   ├── 2012-10-14  [298 entries exceeds filelimit, not opening dir]
│   │   ├── 2012-10-28  [312 entries exceeds filelimit, not opening dir]
│   │   ├── 2012-10-31  [270 entries exceeds filelimit, not opening dir]
│   │   ├── 2012-11-08  [324 entries exceeds filelimit, not opening dir]
│   │   └── 2012-11-11  [158 entries exceeds filelimit, not opening dir]
│   ├── Rainy
│   │   ├── 2012-09-16  [48 entries exceeds filelimit, not opening dir]
│   │   ├── 2012-09-21  [190 entries exceeds filelimit, not opening dir]
│   │   ├── 2012-10-11  [370 entries exceeds filelimit, not opening dir]
│   │   ├── 2012-10-23  [22 entries exceeds filelimit, not opening dir]
│   │   ├── 2012-10-25  [272 entries exceeds filelimit, not opening dir]
│   │   ├── 2012-10-26  [318 entries exceeds filelimit, not opening dir]
│   │   ├── 2012-11-09  [322 entries exceeds filelimit, not opening dir]
│   │   └── 2012-11-10  [120 entries exceeds filelimit, not opening dir]
│   └── Sunny  [24 entries exceeds filelimit, not opening dir]
├── UFPR04
│   ├── Cloudy  [15 entries exceeds filelimit, not opening dir]
│   ├── Rainy  [14 entries exceeds filelimit, not opening dir]
│   └── Sunny  [20 entries exceeds filelimit, not opening dir]
└── UFPR05
    ├── Cloudy  [19 entries exceeds filelimit, not opening dir]
    ├── Rainy
    │   ├── 2013-02-26
    │   │   ├── 2013-02-26_13_04_33.jpg
    │   │   ├── 2013-02-26_13_04_33.xml
    │   │   ├── 2013-02-26_13_19_33.jpg
    │   │   └── 2013-02-26_13_19_33.xml
    │   ├── 2013-03-05  [68 entries exceeds filelimit, not opening dir]
    │   ├── 2013-03-13  [98 entries exceeds filelimit, not opening dir]
    │   ├── 2013-03-16  [46 entries exceeds filelimit, not opening dir]
    │   ├── 2013-03-19  [28 entries exceeds filelimit, not opening dir]
    │   ├── 2013-03-20  [94 entries exceeds filelimit, not opening dir]
    │   ├── 2013-04-12  [52 entries exceeds filelimit, not opening dir]
    │   └── 2013-04-13  [62 entries exceeds filelimit, not opening dir]
    └── Sunny  [25 entries exceeds filelimit, not opening dir]
```
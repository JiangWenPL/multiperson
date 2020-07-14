**NOTE: All the datasets should be put under `./mmdetection/data/`. You would have a better understanding of the dataset
structure after reading our configuration files (e.g., `./mmdetection/configs/smpl/tune.py`)**

# Test Set Preparation
## Panoptic
We adopt the evaluation protocol defined by previous work. 
We would like to thank Andrei Zanfir for sharing the evaluation details from their [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zanfir_Monocular_3D_Pose_CVPR_2018_paper.pdf) and [NIPS 2018](https://papers.nips.cc/paper/8061-deep-network-for-the-integrated-3d-sensing-of-multiple-people-in-natural-images.pdf) papers.

You can find and download the following sequences from [Panoptic Studio Website](http://domedb.perception.cs.cmu.edu/).
We will need cameras *00_16* and *00_30*.
```
160422_ultimatum1
160422_haggling1
160906_pizza1
160422_mafia2
```

We recommend using the official [panoptic-toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) to [download](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/tree/master/scripts) and [extract](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/blob/master/scripts/extractAll.sh) the data.
Then, you can use the relevant script in `misc/preprocess_datasets'`to generate the annotation file.

This script takes three inputs: path to the panoptic dataset, the sequence name as we described previously, and the test frame ids which can
be downloaded from this [link](https://drive.google.com/file/d/1k212OS4X9DXtt_adMK5Oq8QtFOmAGHkX/view?usp=sharing). For example:
```bash
cd misc/preprocess_datasets/full
# Replace the paths to where your dataset and sequence files are stored. 
python panoptic.py /path/to/Panoptic/ 160422_ultimatum1 /path/to/test_sequences/160422_ultimatum1.txt
```  
After running the corresponding script for each sequence, you will find a folder named `processed` under the directory of 
you Panoptic dataset. Then you can run:
```bash
mkdir -p mmdetection/data/Panoptic
cd mmdetection/data/Panoptic
ln -s /path/to/processed/ ./processed
```

We also provide the pre-generated annotation files [here](https://drive.google.com/file/d/1OyJTn_1SaYVasb4zQcbGFhjfJpnP_sml/view?usp=sharing).
We highlight that the evaluation is on the 17 joints from Human3.6M. Since some of these joints do not appear on Panoptic (different set of joints), we first fit SMPL on the Panoptic 3D poses (with the code we provide in `misc/smplify-x`), and then compute the Human3.6M joints from the fitted mesh and replaced *only the missing* joints from the Panoptic annotations with them.

## MuPoTS-3D
You can download the MuPoTS-3D data from [the website of the dataset](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/) and download 
our [pre-generated annotation files](https://drive.google.com/file/d/1xXjzE-aN4Q6N1ODlF4EWu3VJfe75aduF/view?usp=sharing).

The file structure should look like this:

```bash
./mmdetection/
    ...
    ./data/
        ...
        ./mupots-3d/
            ./TS1/
            ./TS2/
            ....
            rcnn/
                ./all_sorted.pkl
```

# Training Set Preparation
## Human3.6M
The official Human3.6M dataset can be found [here](http://vision.imar.ro/human3.6m/description.php).
To extract the Human3.6M data you will need to manually install the [pycdf package of the spacepy library](https://pythonhosted.org/SpacePy/pycdf.html)
to process some of the original files. If you face difficulties with the installation, you can find more elaborate instructions [here](https://stackoverflow.com/questions/37232008/how-read-common-data-formatcdf-in-python).
After installing all the necessary packages you can run:
```bash
mkdir -p mmdetection/data/h36m/rcnn
cd misc/preprocess_datasets/full
# Replace the paths to where your dataset and sequence files are stored. 
python h36m.py /path/to/h36m/ ../../../mmdetection/data/h36m/rcnn --split=train
```
You can repeat the same process for `split=val_p1` and `split=val_p2`, which are used for evaluation.

We also provide code to fit SMPL on the 3D keypoints of the Human3.6M dataset. The code is in `misc/smplify-x`. 
You can do the fitting by running:
```bash
cd misc/smplify-x
# Replace the paths to where your dataset and sequence files are stored. 
python smplifyx/main.py --config cfg_files/fit_smpl.yaml --dataset_file=/path/to/pkl --model_folder=/path/to/smpl --prior_folder=/path/to/prior --output_folder=smpl_fits
```
The structure of the output will be
```
--smpl_fits/
    --0000000/
      --00.pkl
      --01.pkl
      ...
    ...
    --0000001/
      --00.pkl
      ...
    ...

```
After the fitting, you can merge the fits with the original annotation files by running:
```bash
python merge_pkl.py --input_pkl=/path/to/pkl --fits_dir=/path/to/fits --output_pkl=/path/to/out/pkl

```

## COCO
The COCO2014 dataset can be obtained from [the dataset website](http://cocodataset.org/#home). You can also download our [pre-generated annotation files](https://drive.google.com/file/d/1xd4TmldU_NdQ8VbGnPFYHu4W5U80cCPh/view?usp=sharing)
 and place it under `./coco/annotations/`. 
Alternatively, you can generate them yourself by running:
```bash
cd misc/preprocess_datasets/full
# Replace the paths to where your dataset and sequence files are stored. 
python coco.py ../../../mmdetection/data/coco ../../../mmdetection/data/coco/rcnn
```  

## PoseTrack
The posetrack2018 data can be downloaded from the [website of the dataset](https://posetrack.net/users/download.php). Simply extract the dataset
and link it to `./mmdetection/data/posetrack2018`. You can either download our [pre-generated annotation files](https://drive.google.com/file/d/1Im3-Tj9BgkPapx7VB-lHaZ9UEJ-htM94/view?usp=sharing) or run our preprocessing scripts:
```bash
cd mmdetection
mkdir -p ./data/posetrack2018/rcnn
cd ../misc/preprocess_datasets/full
python3 posetrack.py ../../../mmdetection/data/posetrack2018/annotations/train/ ../../../mmdetection/data/posetrack2018/rcnn/train.pkl
python3 posetrack.py ../../../mmdetection/data/posetrack2018/annotations/val/ ../../../mmdetection/data/posetrack2018/rcnn/val.pkl
```

## MPI-INF-3DHP
You can download and extract MPI-INF-3DHP from the [official website of the dataset](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/).
The dataset should be under `./mmdetection/data/` with the name `mpi_inf_3dhp`. 
Then you can download our [pre-generated annotation files](https://drive.google.com/file/d/1P2xAtTLkHJNUhoOVtVMHfioDVynfS6-I/view?usp=sharing) and put it under `./mmdetection/data/mpi_inf_3dhp/rcnn`.
Alternatively you can generate the annotation files by running:
```bash
cd mmdetection
mkdir -p ./data/mpi_inf_3dhp/rcnn
cd ../misc/preprocess_datasets/full
python3 mpi_inf_3dhp.py ../../../mmdetection/data/mpi_inf_3dhp/annotations/train/ ../../../mmdetection/data/mpi_inf_3dhp/rcnn/train.pkl
```

## MPII
You can download the compressed version of [MPII dataset](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz)
Similarly, you can place our [pre-generated annotation files](https://drive.google.com/file/d/1qvJEpyJ1kbitC_KFOsA6Yvm4sA4GTcXA/view?usp=sharing) to `./mmdetection/data/mpii/rcnn`.
Alternatively you can generate the annotation files by running:
```bash
cd mmdetection
mkdir -p ./data/mpii/rcnn
cd ../misc/preprocess_datasets/full
python3 mpii.py ../../../mmdetection/data/mpii/annotations/train/ ../../../mmdetection/data/mpii/rcnn/train.pkl
```

For the latter you will also need the preprocessed hdf5 file that you can download from [here](https://github.com/princeton-vl/pose-hg-demo/blob/master/annot/train.h5)

## Dataset preparation for pretraining
Please follow the same instructions as above. The scripts are located in `misc/preprocess_data/pretrain` instead of `misc/preprocess_data/full` and have the same interface.
The output files should be placed at `mmdetection/data/$DATASET_NAME/` as expected in the pretraining config file that is located at `mmdetection/configs/smpl/pretrain.py`.
Since pretraining involves processing and cropping the images, we do not provide pre-generated annotation files in this case.

For pretraining we additionally use LSP and LSP extended.

## LSP
You need to download the high resolution version of the dataset [LSP dataset original](http://sam.johnson.io/research/lsp_dataset_original.zip).
You can generate the cropped version together with the annotation files by running:
```bash
cd mmdetection
mkdir -p data/rcnn-pretrain/lsp
cd ../misc/preprocess_datasets/pretrain
python3 lsp.py /path/to/lsp ../../../mmdetection/data/rcnn-pretrain/lsp/train.pkl
```

## LSP Extended
We use the extended training set of LSP in its high resolution form (HR-LSPET).
You need to download the [high resolution images](http://datasets.d2.mpi-inf.mpg.de/hr-lspet/hr-lspet.zip).
You can generate the cropped version together with the annotation files by running:
```bash
cd mmdetection
mkdir -p data/rcnn-pretrain/lspet
cd ../misc/preprocess_datasets/pretrain
python3 lsp.py /path/to/lspet ../../../mmdetection/data/rcnn-pretrain/lspet/train.pkl
```

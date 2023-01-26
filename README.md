# For reviewers:

You
can [reproduce graphs and tables](#reproduce-graphs-and-tables), [download datasets and images](#download-datasets-and-images), [visualize attacked datasets](#visualize-attacked-datasets),
or [replicate experiments](#replicate-experiments): the randomized attack in the paper is named `random` and the
deliberate attack is named `arbitrary` below.

I have anonymized all directories. Home directory is `$USER_DIR`, and the working directory
is `$PROJECT_DIR/obfuscation`, i.e. all relative paths begin there unless there is an explicit `cd` command. Replace
these stand-in names with your own directories to run the commands.

# Reproduce graphs and tables

The scripts store the attack trend as `*_trend.csv` and sample bboxes as `*_bboxes.parquet` in `./data/random/results`
and `./data/arbitrary/results`. Knitting `./analysis/random.Rmd` and `./analysis/arbitrary.Rmd` will produce the
graphs (in `./analysis/rmd_imgs/`) and tables (in `./analysis/random.tex` and `./analysis/arbitrary.tex`). Hypotheses
and models are produced by knitting `./analysis/summary.Rmd` (to `./analysis/summary.tex`). Fully knitted respective
documents are
included in `./analysis`. You can use `renv::restore()`
to [reproduce the R environment](https://rstudio.github.io/renv/articles/renv.html).

# Download datasets and images

Two sample datasets and images are included with the code in `./sample`. To browse and download all datasets and
attacked images, please go to
the [anonymized google cloud storage bucket](https://console.cloud.google.com/storage/browser/icml-review)
named `icml-review` (it's public though may still need to sign in). You can browse the bucket online or even
download the 135 GB bucket entirely or partially
using [these instructions online](https://cloud.google.com/storage/docs/access-public-data).

On the bucket, the attacked images are stored in `./data/random/images` and `./data/arbitrary/images`, grouped by
attack iteration. The attacked datasets containing the predictions on those attacked images are saved
in `./data/random/datasets` and `./data/arbitrary/datasets`.

The experiments are broken down into 100 images per repetition: the 750 random repetitions are named
as `random_repeat_{repeat_number}_{attack_mode}_bbox_{model_name}`, and the 480 arbitrary repetitions are named
as `bbox_{perturb_size}_dist_{perturb_target_distance}_repeat_{repeat_number}_{attack_mode}_bbox_{model_name}`. For
downloading datasets, remember to download both the `*.py` and the directory,
e.g. `random_repeat_1_mislabel_bbox_cascade_rcnn_coco.py` and `./random_repeat_1_mislabel_bbox_cascade_rcnn_coco`.

# Visualize attacked datasets

Minimal code to visualize a dataset (on the original images):

1. Install packages (e.g. in a virtual environment)

   ```bash
   pip install --upgrade pip 
   pip install mmcv==1.6.2 fiftyone==0.17.2
   ```

2. Download COCO  ([troubleshoot as needed](https://voxel51.com/docs/fiftyone/getting_started/troubleshooting.html))

   ```python
   import fiftyone.zoo as foz
   dataset = foz.load_zoo_dataset("coco-2017", split="validation")  # only validation required
   ```

3. Import dataset

   ```python
   from pathlib import Path
   
   import fiftyone as fo
   from mmcv import Config
   
   # dataset_dir = Path("./sample/arbitrary/datasets")  # change as needed
   # cfg_name = Path("bbox_100_dist_10_repeat_1_vanish_arbitrary_yolo_v3_coco")  # change as needed

   dataset_dir = Path("./sample/random/datasets")  # change as needed
   cfg_name = Path("random_repeat_1_mislabel_bbox_cascade_rcnn_coco")  # change as needed
      
   cfg = Config.fromfile(dataset_dir / cfg_name.with_suffix(".py"))
   
   coco_dir = "~/fiftyone/coco-2017/validation/data"  # change as needed
   name = cfg.dataset_name  # change as needed
   
   if cfg.dataset_name not in fo.list_datasets():
       cfg_dir = dataset_dir / cfg_name.stem
       dataset = fo.Dataset.from_dir(
           dataset_dir=cfg_dir,
           dataset_type=fo.types.FiftyOneDataset,
           rel_dir=coco_dir,
           name=name)
   
   session = fo.launch_app()  # use `session = fo.launch_app(remote=True, port=5151)` to run remotely; change as needed
   session.wait()
   ```

   For remote sessions, connect with

   ```shell
   ssh -N -L 5151:127.0.0.1:5151 [<username>@]<hostname>
   ```

   Then point browser to `localhost:5151`. Select the dataset on the top bar. The attacked (`attacked`) and success
   images (`success_{attack_iteration}`) are in the `SAMPLES TAGS` section, perturb (`perturb_{attack_iteration}`) and
   target (`target_{attack_iteration}`) bboxes are in the `LABEL TAGS` section, and prediction
   results (`pgd_{attack_iteration}`) and arbitrary perturb bbox (`arbitrary_{attack_iteration}`; in arbitrary
   simulation only) are in the `LABELS` section.

   NB: The `__name__ == "__main__"` section in `./src/main/analyze.py` visualizes and re-analyses all given datasets.

# Replicate experiments

Adapt as needed. I use miniconda to install conda packages and bash to run shell commands on the internal HPC. The exact
commands to reproduce the experiments (including the random seeds) are included in `./slurm`

## Patch mmdetection

Already patched in `./mmdetection`:
Download [mmdetection 2.25.2](https://github.com/open-mmlab/mmdetection/releases/tag/v2.25.2) and rename
to `mmdetection`. Patch `mmdetection` (to retrieve class probabilities and remove data augmentation) by running

```bash
cd changelist
chmod +x move.sh
./move.sh
```

## Download COCO

Download COCO 2017 5k validation images using

```bash
cd $PROJECT_DIR

mkdir coco 
cd coco

wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip val2017.zip
unzip annotations_trainval2017.zip

rm val2017.zip
rm annotations_trainval2017.zip
```

## Install packages

Install conda and create new environment:

```bash
conda create --name coco python=3.8 -y 
conda activate coco  
```

Install pytorch, pyarrow (to analyse `.parquet` data), pytest ([to test](run-pytests)), tqdm (to run scripts), and
ipython (to code easier)

```bash
conda install pytorch=1.11 torchvision torchaudio cudatoolkit=10.2 pyarrow tqdm pytest ipython -c pytorch -c conda-forge -y  
```

Install fiftyone

```bash
pip install fiftyone==0.17.2
```

Check fiftyone

```python
import fiftyone as fo
```

I get mongod import error on internal HPC. My solution is to install RHEL addon
and/or [install standalone mongodb](install-standalone-mongodb), which is usually quicker and more reliable. You
may [troubleshoot as needed](https://voxel51.com/docs/fiftyone/getting_started/troubleshooting.html)

```bash
pip install fiftyone-db-rhel7
```

Install mmcv

```bash
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html
```

Install mmdet in developer mode

```bash
cd ../mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

Install openmim utility and download model checkpoints in `$PROJECT_DIR/mmdetection`

```bash
pip install openmim
mim install mmengine
mkdir -p checkpoints && cd checkpoints
mim download mmdet --config faster_rcnn_r50_fpn_1x_coco --dest .
mim download mmdet --config cascade_rcnn_r50_fpn_1x_coco --dest .
mim download mmdet --config retinanet_r50_fpn_1x_coco --dest .
mim download mmdet --config ssd512_coco --dest .
mim download mmdet --config yolov3_d53_mstrain-608_273e_coco --dest .
```

Check mmdet

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
model = init_detector(config_file, checkpoint_file, device=device)
inference_detector(model, 'demo/demo.jpg')
```

## Setup mongodb

Install standalone mongodb to run fiftyone in parallel, as a fiftyone could kill parallel fiftyone processes.

```bash
conda create --name db -y 
conda activate db
conda install -c conda-forge mongodb==5.0.0 -y
```

Check mongodb by:

1. launching mongod server

   ```bash
   conda activate db
   mkdir -p ~/db
   ulimit -n 4096 # max limit allowed on slurm
   mongod --dbpath ~/db --port 12345 
   ```

2. connecting with mongod shell

   ```bash
   conda activate db
   mongo --shell --port 12345
   ```

3. connecting with fiftyone

   ```bash
   conda activate db
   mongo --eval 'db.adminCommand({setFeatureCompatibilityVersion: "4.4"})' --port 12345 # enable compatibility with fiftyone
   conda activate coco
   export FIFTYONE_DATABASE_URI=mongodb://127.0.0.1:12345/
   ```

   launch python

   ```python
   import fiftyone as fo
   ```

4. shutdown mongod

   ```bash
   mongod --shutdown --dbpath ~/db
   ```

## Only the code

Adapt as needed:

```bash
curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash ./miniconda.sh -b -p $USER_DIR/miniconda3

eval "$($USER_DIR/miniconda3/bin/conda shell.bash hook)"
conda init

cd $PROJECT_DIR/obfuscation

mkdir coco 
cd coco

wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip val2017.zip
unzip annotations_trainval2017.zip

rm val2017.zip
rm annotations_trainval2017.zip

conda create --name coco python=3.8 -y
conda activate coco

conda install pytorch=1.11 torchvision torchaudio cudatoolkit=10.2 pyarrow tqdm pytest ipython -c pytorch -c conda-forge -y

pip install fiftyone==0.17.2
pip install fiftyone-db-rhel7

pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html

cd ../mmdetection
pip install -r requirements/build.txt
pip install -v -e . # or "python setup.py develop"

pip install openmim
mim install mmengine
mkdir -p checkpoints && cd checkpoints
mim download mmdet --config faster_rcnn_r50_fpn_1x_coco --dest .
mim download mmdet --config cascade_rcnn_r50_fpn_1x_coco --dest .
mim download mmdet --config retinanet_r50_fpn_1x_coco --dest .
mim download mmdet --config ssd512_coco --dest .
mim download mmdet --config yolov3_d53_mstrain-608_273e_coco --dest .

conda create --name db -y
conda activate db
conda install -c conda-forge mongodb==5.0.0 -y
```

## Run scripts

My internal HPC uses [slurm](https://slurm.schedmd.com/documentation.html). The entry point to the code
is `./src/run.py`. The scripts below simply batches the experiment.

1. (Already generated.) Generate attack configs to `./configs` and slurm scripts to `./slurm`:

   ```bash
   cd ./configs && chmod +x generate_config.sh 
   ./generate_config.sh faster_rcnn yolo_v3 retinanet ssd_512 cascade_rcnn  

   cd ./slurm && chmod +x generate_scripts.sh
   
   # random
   ./generate_scripts.sh -m faster_rcnn -m yolo_v3 -m retinanet -m ssd_512 -m cascade_rcnn -a vanish_bbox -a mislabel_bbox -a untarget_bbox -d random -l random -o result_dir="./data/random/results" -o dataset_dir="./data/random/datasets" -o log_dir="./data/random/logs" -o img_dir="./data/random/images" -o attack_samples=100 -o gt_samples=1000 -r 50
      
   # arbitrary
   chmod +x arbitrary_regression.sh && ./arbitrary_regression.sh
   ```

2. Run scripts using sbatch (change mongod `DB_ENV` and pytorch `TORCH_ENV` environment names in `./run_batch.sh`):

   ```bash
   cd $PROJECT_DIR/obfuscation/slurm && mkdir -p logs && sbatch ./random/run_batch.sh
   cd $PROJECT_DIR/obfuscation/slurm && mkdir -p logs && sbatch ./arbitrary/run_batch.sh
   ```

   Analysis results, datasets, images and logs will be saved in the corresponding directories passed to
   the `./generate_scripts.sh` command.

# Diagnostics

## Run pytests

```bash
conda activate coco
cd $PROJECT_DIR/obfuscation/
python -m pytest ./test
```

## Check slurm logs

```bash
cd $PROJECT_DIR/obfuscation/slurm/logs
```

# Credits

The code includes a patched [mmdetection 2.25.2](https://github.com/open-mmlab/mmdetection/releases/tag/v2.25.2)
directory, which contains an APACHE license.
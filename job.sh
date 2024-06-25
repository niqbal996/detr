srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=40 -p A100-IML --mem=80000 \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/detr:/home/iqbal/detr \
  --container-image=/netscratch/naeem/detrex-torch1.13-detectron2-sourced.sqsh \
  --mail-type=END --mail-user=naeem.iqbal@dfki.de --job-name=DETR_fb_real \
  --container-workdir=/home/iqbal/detr \
  --time=01-00:00 \
  bash train_detr.sh

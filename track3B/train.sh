mkdir -p validation/
mkdir -p tmp/

timestamp()
{
date +"%Y-%m-%d_%H-%M-%S"
}

TIME=$(timestamp)
ZIP="validation/val_$TIME.zip"
FNAME="model_$TIME"
CUDA_VISIBLE_DEVICES=1 python detection.py --store --store_model --num_workers 4 --name $FNAME
zip -j $ZIP $FNAME/*.json
mv "${FNAME}.pt" validation/
mv "${FNAME}_val.txt" validation/

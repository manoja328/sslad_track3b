mkdir -p validation/
mkdir -p tmp/

timestamp()
{
date +"%Y-%m-%d_%H-%M-%S"
}

TIME=$(timestamp)
ZIP="validation/offline_$TIME.zip"
FNAME="offline_$TIME"
CUDA_VISIBLE_DEVICES=2 python detection_offline.py --store --store_model --num_workers 4 --name $FNAME #--fake
zip -j $ZIP $FNAME/*.json
mv "${FNAME}.pt" validation/
mv "${FNAME}_val.txt" validation/

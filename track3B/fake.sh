mkdir -p fakevalidation/

timestamp()
{
date +"%Y-%m-%d_%H-%M-%S"
}

TIME=$(timestamp)
DIR="fakevalidation"
ZIP="${DIR}/val_$TIME.zip"
FNAME="fake_model_$TIME"
CUDA_VISIBLE_DEVICES=1 python detection.py --store --store_model --num_workers 4 --name $FNAME --fake
zip -j $ZIP $FNAME/*.json
mv "${FNAME}.pt" $DIR
mv "${FNAME}_val.txt" $DIR

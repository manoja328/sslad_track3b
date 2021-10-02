#test on test set and save file
mkdir -p tmp/
MODEL="validation/model_2021-10-01_13-02-31" #no extension here
FNAME=`echo $MODEL | cut -d"/" -f2`
FNAME="test_${FNAME}"
CUDA_VISIBLE_DEVICES=3 python detection.py --store --test_only --test --num_workers 4 --load_model "${MODEL}.pt" --name $FNAME

ZIP="test_$FNAME.zip"
zip -j $ZIP $FNAME/*.json





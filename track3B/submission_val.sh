#test on test set and save file
mkdir -p tmp/
#MODEL="validation/model_2021-10-02_23-16-47" #no extension here
MODEL="validation/model_2021-10-03_19-00-04"
FNAME=`echo $MODEL | cut -d"/" -f2`
FNAME="tvaltest_${FNAME}"
CUDA_VISIBLE_DEVICES=2 python detection.py --store --test_only  --num_workers 4 --load_model "${MODEL}.pt" --name $FNAME

ZIP="tvaltest_$FNAME.zip"
zip -j $ZIP $FNAME/*.json

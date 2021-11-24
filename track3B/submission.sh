#test on test set and save file
mkdir -p tmp/
#MODEL="validation/model_2021-10-02_23-16-47" #no extension here
MODEL="validation/model_2021-10-03_19-00-04"
FNAME=`echo $MODEL | cut -d"/" -f2`
FNAME="finaltest_${FNAME}"
CUDA_VISIBLE_DEVICES=1 python detection.py --store --test_only --test --num_workers 4 --load_model "${MODEL}.pt" --name $FNAME

ZIP="finaltest_$FNAME.zip"
zip -j $ZIP $FNAME/*.json

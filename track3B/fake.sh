mkdir -p fakevalidation/

timestamp()
{
date +"%Y-%m-%d_%H-%M-%S"
}

TIME=$(timestamp)
FNAME="fake_$TIME"
CUDA_VISIBLE_DEVICES=2 python detection.py --num_workers 4 --name $FNAME --fake
mv "${FNAME}_val.txt" fakevalidation/

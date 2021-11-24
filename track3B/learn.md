### Learn

The method used should allow for more than 4 tasks, without big changes in the network structure.
Any type of rehearsal memory can maximally hold 250 images. If separate objects are stored, they should come from the same 250 images.
Pretraining on ImageNet or Microsoft COCO is allowed. During training, no other data can be used.


The 4 tasks are based around 4 different domains:.

- Task 1: Daytime, *citystreet* and clear weather
- Task 2: Daytime, *highway* and clear/overcast weather
- Task 3: *Night*
- Task 4: Daytime, *rain*

**The IoU overlap** threshold for pedestrian, cyclist, tricycle is set to 0.5, and for car, truck, tram is set to 0.7.

- width: 1920 , hieght: 1080 pixels


### My Comments:

- test set submission only 5 times , validation submission virtually infinite
- seem to be some type of domain adaptation task , class is always 7 ( 6 + BG)
- maksRCNN or FPN giving better results
- how to use unlabelled data maybe good to use it as a SSL mechanism
- just SSL to train the backbone maybe using unlabelled images
- PU loss
- plot where best IOU , FPs, FNs , best classification acc etc.
  - use that work where they visualize what the error are ECCV
- looks like easily if you use large model can be beat

### Links: [coda link](https://competitions.codalab.org/competitions/33993#results)

- [avalanche/avalanche/training/plugins at master · VerwimpEli/avalanche (github.com)](https://github.com/VerwimpEli/avalanche/tree/master/avalanche/training/plugins)
- Looks at LWF and ICARL here
  - [avalanche/avalanche/training/strategies at master · VerwimpEli/avalanche (github.com)](https://github.com/VerwimpEli/avalanche/tree/master/avalanche/training/strategies)
- 


### Overview

Today’s ML models are well suited for solving problems given static data, but fail when the data distributions change over time. Continual learning aims to create models that can learn from changing data distributions, and those that can prevent catastrophic forgetting. In this challenge, the goal is to build towards a more realistic, real-world setup. This subtrack (3B) focuses on continual object detection. In this detection challenge, multiple non-centered objects are present. Object detection is a more real-world problem than classification, since we can't expect single, centered objects in real-world imagery. Continual object detection is still a challenge, we take a task based approach here. Four tasks are trained sequentially and the model is evaluated on the average mAP across all tasks. A more elaborate introduction to this setup can be found in [this](https://github.com/VerwimpEli/SSLAD_Track_3/blob/65ef2b25bd9923f55178cabb55dd1f2ca7e38d31/track3B/data_intro.ipynb) notebook.

### Data Description

SODA10M is a large-scale 2D object detection dataset for Autonomous Driving, which contains 10 million unlabeled images and 20k images fully-annotated with 6 representative categories (pedestrian, cyclist, car, truck, tram, tricycle). The unlabeled images are not used in this subtrack. For labeled set there are 5K images for training, 5K images for validation and 10K images for testing. To improve diversity, the images are collected every ten seconds per frame within 32 different cities under different weather conditions, periods and location scenes. For more details about this dataset, please refer to [arxiv report](https://arxiv.org/abs/2106.11118) and [dataset website](https://soda-2d.github.io/).

The 4 tasks are based around 4 different domains:.

- Task 1: Daytime, *citystreet* and clear weather
- Task 2: Daytime, *highway* and clear/overcast weather
- Task 3: *Night*
- Task 4: Daytime, *rain*

SODA10M dataset can be downloaded at the download page of [dataset website](https://soda-2d.github.io/). For this subtrack, objects are cut out of the frames of the scenes. (i.e. we assume a perfect bounding box oracle exists). This is done on the fly by the code provided. See below.

### Avalanche Framework

For this track, we require participants to use the [Avalanche](https://avalanche.continualai.org/) framework. Specifically designed for continual learning benchmarks, this framework will ensure an equal playing field across participants. You only have to implement your own strategy. Data processing, data loading, evalulation and everything else is taken care of by the framework. Files and instructions are on this [git repo](https://github.com/VerwimpEli/SSLAD_Track_3).

### General Rules

- To ensure fairness, the top 3 winners in each track are required to send the technical report.
- Each entry is required to be associated to a team and its affiliation (members of one team should register as one and the affiliation should be set in the team name).
- Using multiple accounts to increase the number of submissions is strictly prohibited.
- Results should follow the correct format and must be uploaded to the evaluation server through the CodaLab competition site. Detailed information about how results will be evaluated is represented on the evaluation page.
- The best entry of each team will be public in the leaderboard at all time.
- The organizer reserves the absolute right to disqualify entries which is incomplete or illegible, late entries or entries that violate the rules.

### Track 3B specific rules

- The method used shoud allow for more than 4 tasks, without big changes in the network structure.
- Any type of rehearsal memory can maximally hold 250 images. If separate objects are stored, they should come from the same 250 images.
- Pretraining on ImageNet or Microsoft COCO is allowed. During training, no other data can be used.

### Awards

Challenge participants with the most successful and innovative entries will be invited to present at this workshop and will receive awards. A 5,000 USD cash prize will be awarded to the top performers in each task and 2nd and 3rd places will be awarded with 2.500 USD.

### Contact Us

For more information, please contact us at **sslad2021@googlegroups.com**.



### Evaluation Metrics

For this task, we use Mean Average Precision(mAP) calculated as in [COCO API](http://github.com/pdollar/coco) among all categories as our evaluation metric and across all tasks; That is, the mean over the APs of pedestrian, cyclist, car, truck, tram and tricycle is calculated for each task. Then, these are averaged for the final result. The IoU overlap threshold for pedestrian, cyclist, tricycle is set to 0.5, and for car, truck, tram is set to 0.7.

### Submission Format

The .json results produced by Avalanche should be packed into a single zip file. An example zip file is available [here](https://drive.google.com/file/d/1VfV5O2_19fhrjnVInRpwV_FRi9UaaM3x/view?usp=sharing). There should be exactly 4 .json files in the zip folder, one for each task. Avalanche will create these for you.


# Prediction-Model---Baseball-Game Winner on Real Data


**Brief:**

## Abstract

**Author:** [Ali Hassan]

**Email:** [malihassanahmad@gmail.com]

## Table of Contents

- [Introduction](#br2)
- [The Data Set](#br3)
- [Analysis on Runs Scored](#br4)
- [Distribution Analysis](#br5)
- [Correlation Analysis](#br6)
- [Linear Regression Model](#br7)
- [Testing the Model on Unseen 2018 Data](#br8)
- [Analysis on Highly Correlated Elements](#br9)
- [Hypothesis Test: New York Yankees](#br10)
- [Conclusion](#br10)

<a name="br2"></a> 

## Prediction-Model---Baseball-Game Winner on Real Data

**Brief:**
Predicting the outcome of a Baseball Game in a complete regular season using Regression Model, and Normal Distributions, also analyzing the trends via resourceful graphs generated between resourceful elements from the selected Dataset.

Page 1

<a name="br3"></a> 

## Prediction-Model---Baseball-Game Winner on Real Data

**Brief:**

### The Data Set

The data used for this project came from *baseball-reference.com*. The initial dataset consisted of each team’s seasonal hitting statistics from 1990–2018 (excluding 1994–1995 due to MLB strike). Data was split into 2 groups: 1990–2017 and 2018 for testing.

Using yearly team hitting statistics, the columns are...

**Target Variable:** Total Runs Scored in the Season

Page 2

<a name="br4"></a> 

## Analysis on Runs Scored

Page 3

<a name="br5"></a> 

## Prediction-Model---Baseball-Game Winner on Real Data

### Distribution Analysis

Looking at the distributions of some of the columns (runs, on base percentage, slugging, on base x slugging), they seem to be normally distributed which is an overall good sign.

Page 4

<a name="br6"></a> 

## Prediction-Model---Baseball-Game Winner on Real Data

On the left, we see those incremental increases in OBP, and yearly plate appearances have a strong positive correlation with scoring more runs. This is intuitive because the more at-bats a team has during a game, the higher chance they will have to score runs.

On the other hand, looking at left on base and on base percentage is a little bit different. Their correlation with runs per year is still positive but it is weaker...

Linear Regression Model...

Page 5

<a name="br7"></a> 

## Prediction-Model---Baseball-Game Winner on Real Data

**The residuals seem to be normally distributed (great sign).**

**Testing the Model on Unseen 2018 Data**

Page 6

<a name="br8"></a> 

As seen above, the average error for each prediction is about 17 runs per season. Comparing this to the mean runs scored in 2018, we get an average error of about 2.45%.

### Analysis on Highly Correlated Elements from Data Set

Page 7

<a name="br9"></a> 

## Prediction-Model---Baseball-Game Winner on Real Data

### Testing on a Team

**New York Yankees Hypothesis Test**

The New York Yankees have won the most World Series Championships within the span of 1990–2018 (5).

H0: The mean runs for NYY = the mean runs for all of MLB

Ha: The mean runs for NYY > the mean runs for all of MLB

With a z score of .95, we fail to reject the null hypothesis that the New York Yankees significantly score more runs than the rest of the MLB.

Conclusion

Page 8

<a name="br10"></a> 

## Prediction-Model---Baseball-Game Winner on Real Data

After obtaining the data and doing my initial EDA, it seems that the most valuable hitting statistics for predicting runs scored per season are Plate Appearances, Left On Base, On Base Percentage, and On Base x Slugging Percentage. These variables together account for 96% of the variance in runs scored per season within the dataset. Going forward, I would like to test this model on the 2019 regular season as well as turn my focus toward the target variable of wins per season.

### Normal Distribution
#### Distribution of Runs
![Alt Text]([./images/my_image.png](https://github.com/theAliHassan/Prediction-Model---Baseball-Game/blob/main/distributionOFRuns.png))
Page 9

<a name="br11"></a> 

## Prediction-Model---Baseball-Game Winner on Real Data

**To be Notes:** Data is Highly Correlated and is Normal.





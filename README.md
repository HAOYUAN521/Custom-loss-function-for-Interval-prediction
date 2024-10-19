# Neural Network Custom Loss Function for Flexible and High-Quality Prediction Intervals
In this project, we created a new loss function to predict intervals. This custom loss function combined with a neural network can output high-quality prediction intervals. At the same time, it can meet the requirements of different scenarios by flexibly controlling the upper and lower bounds.

## Discription
### Introduction
In the realm of deep learning, interval prediction plays a crucial role in diverse applications. However, the rise of complex challenges in these domains requires more flexible and high-quality prediction intervals. To address this need, we present a novel custom loss function specifically designed for deep learning models. This function optimizes both prediction interval width and coverage, boasting superior flexibility compared to traditional approaches. Specifically, our loss function utilizes multiple penalty coefficients to precisely control the interval width. Higher coefficients favor wider intervals to guarantee coverage, while lower ones prioritize narrower intervals for improved prediction accuracy. This delicate balance allows the model to adapt to different situations and application
demands. Extensive experiments on ten public datasets across various domains reveal the impressive performance of this custom loss function. It consistently generates high-quality prediction intervals with remarkable generalization capabilities, readily adapting to diverse application scenarios.
### Achievement of Flexibility
For the same set of data, we give completely opposite weights to the upper and lower bounds. When the actual problem requires us to be more strict in estimating the lower bound, the weight given to the lower bound should be increased. In this case, the true value is closer to the lower bound.
![image](https://github.com/HAOYUAN521/Custom-loss-function-for-Interval-prediction/blob/main/Images/strict%20lower%20bound.png)
On the contrary, the true value is closer to the upper bound.
![image](https://github.com/HAOYUAN521/Custom-loss-function-for-Interval-prediction/blob/main/Images/strict%20upper%20bound.png)


## Code structure
 * Benchmarking.py
 * Simulation.py

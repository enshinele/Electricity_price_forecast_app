# Application of electricity price forecasting

By Hongfei Yan, Ziwei Cheng, Xiaoxuan Cai, Huiwen Zheng, Donghao Song, Yinxian Li, Yikai Kang, Yuan Huang
## Introduction
This is an application for electricity price forecasting. At the top of this web app, there is the logo of uni and the names of the members of our group. You can expand to view the names of the members by clicking the box on the left of Show the member.The current time in the middle will automatically read the current time when the app is executed. After our group used four different model training data and made a comparison, we finally chose to use MLP to predict the data one day later, and use LSTM to predict the data one hour later and one week later. You can click the Click to predict button at the bottom to obtain the predicted value corresponding to the current time in one hour, one day, and one week.

![deformable_detr](.figs/web.png)


## Main Results

<table>
<caption>Test results of hour ahead EPF using optimal models</caption>
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th style="text-align: center;">RMSE</th>
<th style="text-align: center;">R2</th>
<th style="text-align: center;">MAPE</th>
<th style="text-align: center;">MaxAPE</th>
<th style="text-align: center;">AWT</th>
<th style="text-align: center;">interval</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;">MLP</td>
<td style="text-align: center;"><span>3.17</span></td>
<td style="text-align: center;"><span>0.96</span></td>
<td style="text-align: center;">0.047</td>
<td style="text-align: center;">0.58</td>
<td style="text-align: center;"><span>0.93</span></td>
<td style="text-align: center;">[0.016, 0.064]</td>
</tr>
<tr class="even">
<td style="text-align: left;">LSTM</td>
<td style="text-align: center;">3.23</td>
<td style="text-align: center;"><span>0.96</span></td>
<td style="text-align: center;"><span>0.043</span></td>
<td style="text-align: center;"><span>0.56</span></td>
<td style="text-align: center;"><span>0.93</span></td>
<td style="text-align: center;"><span>[0.014, 0.054]</span></td>
</tr>
<tr class="odd">
<td style="text-align: left;">Seq2seq</td>
<td style="text-align: center;">3.74</td>
<td style="text-align: center;">0.94</td>
<td style="text-align: center;">0.053</td>
<td style="text-align: center;">0.56</td>
<td style="text-align: center;">0.89</td>
<td style="text-align: center;">[0.015, 0.061]</td>
</tr>
<tr class="even">
<td style="text-align: left;">CNN</td>
<td style="text-align: center;">8.50</td>
<td style="text-align: center;">0.47</td>
<td style="text-align: center;">0.132</td>
<td style="text-align: center;">0.87</td>
<td style="text-align: center;">0.43</td>
<td style="text-align: center;">[0.015, 0.057]</td>
</tr>
</tbody>
</table>


<table>
<caption>Test results of day ahead EPF using optimal models</caption>
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th style="text-align: center;">RMSE</th>
<th style="text-align: center;">R2</th>
<th style="text-align: center;">MAPE</th>
<th style="text-align: center;">MaxAPE</th>
<th style="text-align: center;">AWT</th>
<th style="text-align: center;">interval</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;">MLP</td>
<td style="text-align: center;">5.34</td>
<td style="text-align: center;">0.80</td>
<td style="text-align: center;">0.107</td>
<td style="text-align: center;">0.85</td>
<td style="text-align: center;"><span>0.61</span></td>
<td style="text-align: center;">[0.029, 0.178]</td>
</tr>
<tr class="even">
<td style="text-align: left;">LSTM</td>
<td style="text-align: center;"><span>4.78</span></td>
<td style="text-align: center;"><span>0.86</span></td>
<td style="text-align: center;"><span>0.097</span></td>
<td style="text-align: center;"><span>0.53</span></td>
<td style="text-align: center;"><span>0.61</span></td>
<td style="text-align: center;"><span>[0.027, 0.120]</span></td>
</tr>
<tr class="odd">
<td style="text-align: left;">Seq2seq</td>
<td style="text-align: center;">6.53</td>
<td style="text-align: center;">0.79</td>
<td style="text-align: center;">0.127</td>
<td style="text-align: center;">0.97</td>
<td style="text-align: center;">0.49</td>
<td style="text-align: center;">[0.032, 0.171]</td>
</tr>
<tr class="even">
<td style="text-align: left;">CNN</td>
<td style="text-align: center;">10.78</td>
<td style="text-align: center;">0.43</td>
<td style="text-align: center;">0.324</td>
<td style="text-align: center;">0.81</td>
<td style="text-align: center;">0.57</td>
<td style="text-align: center;">[0.034, 0.181]</td>
</tr>
</tbody>
</table>

<table>
<caption>Test results of week ahead EPF using optimal models</caption>
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th style="text-align: center;">RMSE</th>
<th style="text-align: center;">R2</th>
<th style="text-align: center;">MAPE</th>
<th style="text-align: center;">MaxAPE</th>
<th style="text-align: center;">AWT</th>
<th style="text-align: center;">interval</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;">MLP</td>
<td style="text-align: center;"><span>4.17</span></td>
<td style="text-align: center;">0.89</td>
<td style="text-align: center;"><span>0.103</span></td>
<td style="text-align: center;">1.56</td>
<td style="text-align: center;"><span>0.67</span></td>
<td style="text-align: center;"><span>[0.028, 0.151]</span></td>
</tr>
<tr class="even">
<td style="text-align: left;">LSTM</td>
<td style="text-align: center;">4.28</td>
<td style="text-align: center;"><span>0.90</span></td>
<td style="text-align: center;">0.133</td>
<td style="text-align: center;"><span>0.60</span></td>
<td style="text-align: center;">0.50</td>
<td style="text-align: center;">[0.051, 0.120]</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Seq2seq</td>
<td style="text-align: center;">7.96</td>
<td style="text-align: center;">0.63</td>
<td style="text-align: center;">0.181</td>
<td style="text-align: center;">1.88</td>
<td style="text-align: center;">0.48</td>
<td style="text-align: center;">[0.051, 0.219]</td>
</tr>
<tr class="even">
<td style="text-align: left;">CNN</td>
<td style="text-align: center;">9.89</td>
<td style="text-align: center;">0.26</td>
<td style="text-align: center;">0.327</td>
<td style="text-align: center;">1.18</td>
<td style="text-align: center;"><span>0.67</span></td>
<td style="text-align: center;">[0.064, 0.321]</td>
</tr>
</tbody>
</table>




*Note:*

In this project, we used four metrics: Mean Squared Error(MSE), Root
Mean Square Error (RMSE), coefficient of
determination(*R*2*s**c**o**r**e*) and Mean absolute
percentage error(MAPE) from the library sklearn and three self-defined
metrics: max absolute percentage error, middle interval(25% - 75%) of
total error rate and Accuracy with Tolerance(AWT)

## Installation

### Requirements

* Python>=3.7
  
* Docker>=20.10.8

* Docker compose>=1.29.2
## Usage

### Model deployment on server
* Clone the repository locally:
    ```bash
    git clone git@gitlab.ldv.ei.tum.de:ami2021/group10.git
    ```
* Create images and containers:
    ```bash
    docker-compose up
    ```


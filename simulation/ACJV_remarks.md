# ACJV remarks

Hi Haozhe,

I compared by code for the ACJV estimators and your code and found differences/issues with my code, your code and the paper.

The first thing is really simple: We both use .resample(‚D‘).sum() for summing up hf, hf^2, hf^3 and hf^4 for each day. The dataframe with the log returns from the Heston process hf has a datetime-index where we have only weekdays as anyone would expect. Using hf.resample(‚D‘).sum() we get

2017-05-12    0.000413
2017-05-13    0.000000
2017-05-14    0.000000
2017-05-15    0.000483
2017-05-16    0.000527
               ...   
2017-07-08    0.000000
2017-07-09    0.000000
2017-07-10    0.000516
2017-07-11    0.000471
2017-07-12    0.000501
Freq: D, Name: Path_1, Length: 62, dtype: float64

You see there for some dates (2017-05-13, 2017-05-14, 2017-07-08 and 2017-07-09) a cumulated log return of 0. This caught my attention because the chance of a stock making 0 return on a whole day is rather low. So I checked these days and they were not in the index of hf and indeed, all of these days are Saturdays and Sundays. So it makes sense that these days would not occur in hf. But somehow .resample(‚D‘) adds these days. This would not be a problem but if we further aggregate to lower frequencies for the ACJV estimator we use the rolling mean. And here it makes a difference if we have days with 0 log return or not. It changes the moments (for one path) from

[-3.82436244e-02  8.53155121e-03 -3.47822174e-06  1.77227553e-07 -4.41382655e-03  2.43486696e-03]

to 

[-6.23328824e-02  1.19743838e-02 -5.15385404e-06  2.49695469e-07 -3.93325726e-03  1.74142314e-03]

To fix this, I changed
```python
hf.resample(‚D‘).sum()
```
to
```python
hf.resample(‚D‘).sum(min_count=1).dropna()
```
Because when we have less that 1 value to sum up, Pandas puts a NaN for this day which we than remove with .dropna(). Similar for hf^2, hf^3 and hf^4.

---

The next thing is about the multiplication with sqrt(N) in the daily skewness estimator and the multiplication with N in the daily kurtosis estimator in the ACJV paper where N stands for the number of log returns per day. You already mentioned that the multiplication with these numbers seems not correct, I now used the data from the Heston process to get some actual numbers. For the skewness of the log returns from the first path I get -4.35001946 and for the kurtosis 65.6587160. These numbers are far away from your’s -3.93325726e-03 and 1.74142314e-03, so I guess that the multiplication here is wrong. When I remove this, I get for the skewness -0.489415427 and for the kurtosis 0.831122987 which are way closer to your numbers. I also used the skew() and kurt() functions from scipy.stats, they gave -0.1107... and 0.0699... which further confirms that the multiplication with sqrt(N) or N is wrong.

---

The third thing is about aggregating skewness and kurtosis vs aggregating the third and fourth moment. In your code you aggregate the third and fourth moment and only in the end you take the aggregated values and calculate third_moment/second_moment^1.5 to get a single skewness value (for kurtosis similar). I understood the paper in that way, that we calculate for each day a skewness and kurtosis and summing and averaging them over the whole time. 
```python
def realized_daily_skewness(r: pd.Series) -> pd.Series:
   return (r**3).resample('D').sum(min_count=1).dropna()/(realized_daily_variance(r)**(3/2))

def realized_daily_variance(r: pd.Series) -> pd.Series:
   return (r**2).resample('D').sum(min_count=1).dropna()

# … (Code for generating df_logreturn)
realized_daily_skewness(df_logreturn[column]).rolling(22).sum().dropna().mean()
```
With this approach I get the above mentioned values of skewness and kurtosis.

Which approach is now correct? Or should I use them all and later compare how well they work in the expansion methods?

Thanks and best wishes
Henry

---

Hi Henry,

Regarding the first point, since we aggregate the high-frequency data instead of averaging them, there should be no impact, right?

For the 2nd point, I agree.

For the 3rd point, based on the formula, I believe both formulas should lead to the same result. Would you agree on it? Basically, I would suggest to aggregate them first and then compute the skewness and kurtosis. The reason is that we can better benchmark the result against each other to see whether we get the identical outcomes.

Happy to discuss further^^

Best,
Haozhe

---

Hi Haozhe,

Well, there is an impact if the aggregation is the average. ACJV take the mean over the HF data to get lower frequencies. In their paper to get the estimate on weekly basis they average daily estimates. We extend this to average the last 22 days to get a monthly estimate but if we have weekends in the last 22 days (which we didn’t had in the HF data from the Heston process, but .resample(‚D‘) added) we get a wrong estimate. I fixed this by removing the weekends added by .resample(‚D‘) but another way would be to modify the .mean() function in such a way that it ignores weekends.

Both formulas lead to different results because skewness is not a linear function. I wrote a small example that demonstrates this issue:
```python
import pandas as pd

def skewness(x3, var):
   return x3 + var # a dummy implementation, linear
   # return x3 / var**(3/2) # a real implementation, non-linear

r = pd.DataFrame(range(1,5), columns=['r‘])

# daily variance
r['var'] = r['r']**2

# daily third moment
r['x^3'] = r['r']**3

# daily skewness
r['skew'] = skewness(r['x^3'], r['var‘])

print(r)

# 2nd + 3rd aggregated moment
second_moment = r['var'].rolling(4).sum().dropna().mean()
third_moment = r['x^3'].rolling(4).sum().dropna().mean()

# skewness
skewness_1 = skewness(third_moment, second_moment)

# skewness aggregation
skewness_2 = r['skew'].rolling(4).sum().dropna().mean()
print(skewness_1, skewness_2)
```
If we use the linear „skewness“ both implementations give in the same result of 130. If we use the real implementation skewness_1 has a value 
of 0.6086 and skewness_2 has a value of 4.

Best wishes
Henry
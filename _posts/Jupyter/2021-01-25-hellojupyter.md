---
title: "jupyter test 6"
search: true
categories:
 - Notebook
tags:
 - Notebook
last_modified_at: 2021-01-25 18:44
layout: jupyter
classes: wide
---

# Hello Jupyter
---

```python
import pandas as pd
import matplotlib.pyplot as plt
```


```python
columns = ['col1', 'col2', 'col3', 'col4']
data = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
```


```python
df = pd.DataFrame(data, columns = columns)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.plot(df)
plt.legend(df.columns)
plt.show()
```


    
<p align="center">
  <img src="/assets/img/jupyter/hellojupyter_4_0.png" alt="df"/>
</p>  


# IE517
## Homework #1 
##### (Due March 22)
## Abdullah Hanefi Önaldı
## Solving the TSP using Construction Heuristics and 2-Opt Improvement Heuristic

### Problem Description

In this homework, you are going to solve the TSP for three data sets. They are called eil51.dat, eil76.dat, and eil101.dat, and consist of 51, 76, and 101 customer locations, respectively. Each data set includes the x-coordinates and y-coordinates of customers. The distances between customer locations are measured via Euclidean distance rounded to two digits after the decimal point. You can also compute the optimal tour length by considering the sequence given in the xxxopt.dat files.

1. Solve each instance using the one-sided nearest neighbor heuristic starting at cities 10, 20, and 30. This means that you will obtain nine tours. Provide the tour length of each one using the table below.
2. Solve each instance using the two-sided nearest neighbor heuristic starting at cities 10, 20, and 30. This means that you will obtain nine tours. Provide the tour length of each one using the table below.
3. Solve each instance using the nearest insertion heuristic starting at cities 10, 20, and 30. This means that you will obtain nine tours. Provide the tour length of each one using the table below.
4. Solve each instance using the farthest insertion heuristic starting at cities 10, 20, and 30. This means that you will obtain nine tours. Provide the tour length of each one using the table below.
5. For each tour obtained so far, apply the 2-opt improvement heuristic, and give the tour length using the table below.

I would like to remind you the following points which you should consider when you submit your homework. It will consist of two parts: your code and report. First, your code must be clear and you should define the following using comment lines in the code: variables names and their purpose, function names and their purpose. For example, you should write "X is the location variable", "CompObj calculates the objective value", etc. Or, you can use a function name that is self explanatory e.g., ApplyMove.

In the report part, you have to mention which solution representation and neighborhood structure you used as well as other pertinent and tiny details worth pointing out. You can use the following table for the output of your solutions.

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>method</th>
      <th colspan="2" halign="left">1-Sided_NN</th>
      <th colspan="2" halign="left">2-Sided_NN</th>
      <th colspan="2" halign="left">Nearest_Insert</th>
      <th colspan="2" halign="left">Furthest_Insert</th>
    </tr>
    <tr>
      <th></th>
      <th>stage</th>
      <th>Initial</th>
      <th>After_2-opt</th>
      <th>Initial</th>
      <th>After_2-opt</th>
      <th>Initial</th>
      <th>After_2-opt</th>
      <th>Initial</th>
      <th>After_2-opt</th>
    </tr>
    <tr>
      <th>dataset</th>
      <th>initial_customer</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">eil76</th>
      <th>10</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>20</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>30</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">eil101</th>
      <th>10</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>20</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>30</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">eil51</th>
      <th>10</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>20</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>30</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

### Solution
#### Solution Representation

I used simple lists of customers to represent solutions to the TSP problem. The first customer in the list occurs at the end, and all other customers occur exactly once.

These solutions can be viewed in the excel file provided along with this report.

#### Data Structures

I used :
- `paths` : pandas dataframe with MultiIndex'es to store all paths
- `df` : pandas dataframe with MultiIndex'es to store the lengths of the paths
- `INSTANCES` : dictionary that contains all the following for each problem instance
  - contents of input files
  - contents of optimal solution files
  - `file` : the path of the dataset
  - `file_opt` : the path of the file containing optimal solution
  - `optimal_path` : the list of nodes visited in the optimal path
  - `distances` : pairwise distances of customers in a matrix
  - `optimal_length` : the length of the optimal solution

### Code

Let's start by importing functions/modules and defining several helper functions:

- `pandas` : library for managing tabular data
- `numpy` : numeric operations, linear algebra modules etc
- `squareform` : used for creating square distance matrices
- `pdist` : pairwise distance calculations
- `partial` : creation of partial functions
- `combinations` : given a number of lists, generates all possible combinations of elements by taking one from each list


- `calc_total_length` : calculates the total path length, given the pairwise distances of all customers, and the order they are visited
- `insertion_cost` : given the pairwise distances between customers, calculates the cost of inserting customer k between i and j
- `find_method` : given a method name, construct the partial functions that will solve the problem using the aforementioned method


```python
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
from functools import partial
from itertools import combinations


def calc_total_length(path, distances):
    return distances.lookup(path[:-1], path[1:]).sum()


def insertion_cost(distances, i, j, k):
    return distances.loc[i, k] + distances.loc[k, j] - distances.loc[i, j]


def find_method(method_name):
    method_dict = {
        '1-Sided_NN': partial(nearest_neighbor, num_sides=1),
        '2-Sided_NN': partial(nearest_neighbor, num_sides=2),
        'Nearest_Insert': partial(insertion, kind='nearest'),
        'Furthest_Insert': partial(insertion, kind='farthest'),
    }
    return method_dict[method_name]
```

The static variables storing problem instances, initial customer indices, methods, and stages as described in the problem description


```python
INSTANCES = {
    'eil76': {
        'file': 'data/eil76.dat',
        'file_opt': 'data/eil76opt.dat'
    },
    'eil101': {
        'file': 'data/eil101.dat',
        'file_opt': 'data/eil101opt.dat'
    },
    'eil51': {
        'file': 'data/eil51.dat',
        'file_opt': 'data/eil51opt.dat'
    },
}
INITIAL_CUSTOMERS = [10, 20, 30]
METHODS = ['1-Sided_NN', '2-Sided_NN', 'Nearest_Insert', 'Furthest_Insert']
STAGES = ['Initial', 'After_2-opt']
```

Create the Pandas DataFrame that will hold all the solutions


```python
def create_df(instances=INSTANCES,
              initial_customers=INITIAL_CUSTOMERS,
              methods=METHODS,
              stages=STAGES):
    indexes = [instances.keys(), initial_customers]
    row_index = pd.MultiIndex.from_product(
        indexes, names=['dataset', 'initial_customer'])

    indexes = [methods, stages]
    column_index = pd.MultiIndex.from_product(
        indexes, names=['method', 'stage'])

    df = pd.DataFrame(index=row_index, columns=column_index)

    return df


df = create_df()
paths = create_df()
```

Read the files of the given instances


```python
def read_files(instances=INSTANCES):
    for instance in instances.values():
        coords = pd.read_csv(
            instance['file'], header=None, index_col=0, delim_whitespace=True)

        instance['optimal_path'] = pd.read_csv(
            instance['file_opt'], header=None, squeeze=True)

        instance['distances'] = pd.DataFrame(
            squareform(pdist(coords)),
            columns=coords.index,
            index=coords.index)

        instance['optimal_length'] = calc_total_length(
            path=instance['optimal_path'], distances=instance['distances'])


read_files()
```

Create the path using the nearest neighbor heuristic given the number of sides to search, the pairwise distances, and the initial customer to start the search


```python
def nearest_neighbor(num_sides, distances, initial_node):
    distances = distances.copy()
    np.fill_diagonal(distances.values, np.nan)
    path = [initial_node]

    if num_sides is 1:
        current = initial_node
        for _ in range(distances.shape[0] - 1):
            next_ = distances[current].idxmin()
            path.append(next_)
            distances.loc[current, :] = np.nan
            current = next_

    elif num_sides is 2:
        head, tail = initial_node, distances[initial_node].idxmin()
        path.append(tail)
        distances.loc[:, 'head'] = np.nan
        distances.loc[:, 'tail'] = np.nan

        for _ in range(distances.shape[0] - 2):
            next_head, next_tail = distances[[head, tail]].idxmin()
            if distances.loc[head, next_head] > distances.loc[next_tail, tail]:
                path.insert(0, next_tail)
                distances.loc[tail, :] = np.inf
                tail = next_tail
            else:
                path.append(next_head)
                distances.loc[head, :] = np.inf
                head = next_head
    else:
        raise ValueError('nearest_neighbor is either one or two sided')

    path.append(path[0])
    return path
```

Create the path using the insertion heuristic given the kind (farthest or nearest), the pairwise distances, and the initial customer to start the search


```python
def insertion(kind, distances, initial_node):
    distances = distances.copy()
    np.fill_diagonal(distances.values, np.nan)

    if kind is 'nearest':
        closest = distances[initial_node].idxmin()
        path = [initial_node, closest, initial_node]

        distances['subtour'] = distances[[closest, initial_node]].min(axis=1)
        for _ in range(distances.shape[0] - 2):
            distances['subtour'].loc[path] = np.nan
            closest = distances['subtour'].idxmin()
            costs = [
                insertion_cost(distances, i, j, closest)
                for i, j in zip(path, path[1:])
            ]
            min_cost = np.argmin(costs) + 1
            path.insert(min_cost, closest)
            distances['subtour'] = distances[[closest, 'subtour']].min(axis=1)

    elif kind is 'farthest':
        fartest = distances[initial_node].idxmax()
        path = [initial_node, fartest, initial_node]

        distances['subtour'] = distances[[fartest, initial_node]].min(axis=1)
        for _ in range(distances.shape[0] - 2):
            distances['subtour'].loc[path] = np.nan
            fartest = distances['subtour'].idxmax()
            costs = [
                insertion_cost(distances, i, j, fartest)
                for i, j in zip(path, path[1:])
            ]
            min_cost = np.argmin(costs) + 1
            path.insert(min_cost, fartest)
            distances['subtour'] = distances[[fartest, 'subtour']].min(axis=1)
    else:
        ValueError('insertion is either nearest or farthest')

    return path
```

Improve a solution using two opt given pairwise distances and the path found in the solution


```python
def two_opt(distances, path):
    path = path.copy()
    while True:
        no_gain = True
        for start, end in combinations(range(1, len(path) - 2), r=2):
            if end - start is 1:
                continue
            c1 = path[start - 1]
            c2 = path[start]
            c3 = path[end]
            c4 = path[end + 1]

            gain = + distances[c1][c2] + distances[c3][c4] \
                   - distances[c1][c3] - distances[c2][c4]

            if gain > 1e-10:
                no_gain = False
                path[start:end + 1] = path[end:start - 1:-1]

        if no_gain:
            return path
```

Iterate over all the datasets, initial customers, and methods. Construct solutions and then improve them using two opt heuristic


```python
for instance in INSTANCES:
    df.loc[instance, 'optimal'] = INSTANCES[instance]['optimal_length']
    distances = INSTANCES[instance]['distances']
    for initial in INITIAL_CUSTOMERS:
        for method_name in METHODS:
            method = find_method(method_name)

            path = method(distances=distances, initial_node=initial)
            length = calc_total_length(path, distances)

            paths.loc[(instance, initial), (method_name, 'Initial')] = path
            df.loc[(instance, initial), (method_name, 'Initial')] = length

            better_path = two_opt(distances, path)
            length = calc_total_length(better_path, distances)

            df.loc[(instance, initial), (method_name, 'After_2-opt')] = length
            paths.loc[(instance, initial), (method_name,
                                            'After_2-opt')] = better_path
            
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

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>method</th>
      <th colspan="2" halign="left">1-Sided_NN</th>
      <th colspan="2" halign="left">2-Sided_NN</th>
      <th colspan="2" halign="left">Nearest_Insert</th>
      <th colspan="2" halign="left">Furthest_Insert</th>
      <th>optimal</th>
    </tr>
    <tr>
      <th></th>
      <th>stage</th>
      <th>Initial</th>
      <th>After_2-opt</th>
      <th>Initial</th>
      <th>After_2-opt</th>
      <th>Initial</th>
      <th>After_2-opt</th>
      <th>Initial</th>
      <th>After_2-opt</th>
      <th></th>
    </tr>
    <tr>
      <th>dataset</th>
      <th>initial_customer</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">eil76</th>
      <th>10</th>
      <td>640.533</td>
      <td>599.838</td>
      <td>704.103</td>
      <td>652.116</td>
      <td>636.201</td>
      <td>610.281</td>
      <td>599.215</td>
      <td>596.102</td>
      <td>545.387552</td>
    </tr>
    <tr>
      <th>20</th>
      <td>735.983</td>
      <td>650.89</td>
      <td>708.861</td>
      <td>663.681</td>
      <td>614.819</td>
      <td>608.437</td>
      <td>580.563</td>
      <td>580.563</td>
      <td>545.387552</td>
    </tr>
    <tr>
      <th>30</th>
      <td>730.285</td>
      <td>642.509</td>
      <td>711.812</td>
      <td>635.205</td>
      <td>626.494</td>
      <td>609.627</td>
      <td>579.946</td>
      <td>579.946</td>
      <td>545.387552</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">eil101</th>
      <th>10</th>
      <td>796.041</td>
      <td>712.081</td>
      <td>720.585</td>
      <td>696.957</td>
      <td>728.333</td>
      <td>714.068</td>
      <td>684.778</td>
      <td>684.033</td>
      <td>642.309536</td>
    </tr>
    <tr>
      <th>20</th>
      <td>800.708</td>
      <td>735.65</td>
      <td>800.014</td>
      <td>710.235</td>
      <td>735.845</td>
      <td>710.125</td>
      <td>692.276</td>
      <td>692.276</td>
      <td>642.309536</td>
    </tr>
    <tr>
      <th>30</th>
      <td>776.518</td>
      <td>699.699</td>
      <td>784.347</td>
      <td>742</td>
      <td>735.845</td>
      <td>717.476</td>
      <td>688.832</td>
      <td>682.889</td>
      <td>642.309536</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">eil51</th>
      <th>10</th>
      <td>558.849</td>
      <td>511.2</td>
      <td>496.688</td>
      <td>432.482</td>
      <td>490.181</td>
      <td>473.476</td>
      <td>444.555</td>
      <td>444.555</td>
      <td>429.983312</td>
    </tr>
    <tr>
      <th>20</th>
      <td>567.304</td>
      <td>526.656</td>
      <td>554.273</td>
      <td>506.732</td>
      <td>514.379</td>
      <td>478.556</td>
      <td>454.656</td>
      <td>454.656</td>
      <td>429.983312</td>
    </tr>
    <tr>
      <th>30</th>
      <td>520.018</td>
      <td>479.436</td>
      <td>527.628</td>
      <td>459.112</td>
      <td>490.181</td>
      <td>471.406</td>
      <td>458.272</td>
      <td>458.272</td>
      <td>429.983312</td>
    </tr>
  </tbody>
</table>
</div>



Divide the path lengths by the optimal length to better see the performance


```python
df = df.div(df['optimal'], axis='rows')
df.drop(columns=['optimal'], inplace=True)
df
```

    /Users/hanefi/code/jupyter/.venv/lib/python3.6/site-packages/pandas/core/generic.py:2530: PerformanceWarning: dropping on a non-lexsorted multi-index without a level parameter may impact performance.
      obj = obj._drop_axis(labels, axis, level=level, errors=errors)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>method</th>
      <th colspan="2" halign="left">1-Sided_NN</th>
      <th colspan="2" halign="left">2-Sided_NN</th>
      <th colspan="2" halign="left">Nearest_Insert</th>
      <th colspan="2" halign="left">Furthest_Insert</th>
    </tr>
    <tr>
      <th></th>
      <th>stage</th>
      <th>Initial</th>
      <th>After_2-opt</th>
      <th>Initial</th>
      <th>After_2-opt</th>
      <th>Initial</th>
      <th>After_2-opt</th>
      <th>Initial</th>
      <th>After_2-opt</th>
    </tr>
    <tr>
      <th>dataset</th>
      <th>initial_customer</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">eil76</th>
      <th>10</th>
      <td>1.17445</td>
      <td>1.09984</td>
      <td>1.29101</td>
      <td>1.19569</td>
      <td>1.16651</td>
      <td>1.11899</td>
      <td>1.0987</td>
      <td>1.09299</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.34947</td>
      <td>1.19345</td>
      <td>1.29974</td>
      <td>1.2169</td>
      <td>1.12731</td>
      <td>1.1156</td>
      <td>1.0645</td>
      <td>1.0645</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1.33902</td>
      <td>1.17808</td>
      <td>1.30515</td>
      <td>1.16469</td>
      <td>1.14871</td>
      <td>1.11779</td>
      <td>1.06336</td>
      <td>1.06336</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">eil101</th>
      <th>10</th>
      <td>1.23934</td>
      <td>1.10863</td>
      <td>1.12187</td>
      <td>1.08508</td>
      <td>1.13393</td>
      <td>1.11172</td>
      <td>1.06612</td>
      <td>1.06496</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.24661</td>
      <td>1.14532</td>
      <td>1.24553</td>
      <td>1.10575</td>
      <td>1.14562</td>
      <td>1.10558</td>
      <td>1.07779</td>
      <td>1.07779</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1.20895</td>
      <td>1.08935</td>
      <td>1.22114</td>
      <td>1.15521</td>
      <td>1.14562</td>
      <td>1.11702</td>
      <td>1.07243</td>
      <td>1.06318</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">eil51</th>
      <th>10</th>
      <td>1.2997</td>
      <td>1.18888</td>
      <td>1.15513</td>
      <td>1.00581</td>
      <td>1.14</td>
      <td>1.10115</td>
      <td>1.03389</td>
      <td>1.03389</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.31936</td>
      <td>1.22483</td>
      <td>1.28906</td>
      <td>1.17849</td>
      <td>1.19628</td>
      <td>1.11297</td>
      <td>1.05738</td>
      <td>1.05738</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1.20939</td>
      <td>1.11501</td>
      <td>1.22709</td>
      <td>1.06774</td>
      <td>1.14</td>
      <td>1.09634</td>
      <td>1.06579</td>
      <td>1.06579</td>
    </tr>
  </tbody>
</table>
</div>



Write the paths and performances to an excel file


```python
writer = pd.ExcelWriter('results.xlsx')
df.to_excel(writer, sheet_name='performance')
paths.to_excel(writer, sheet_name='paths')
writer.save()
```

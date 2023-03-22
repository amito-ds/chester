# MadCat

MadCat is a Python package for end-to-end machine learning, including all necessary steps and plots.
Given the data, it creates a comprehensive report that includes feature statistics, pre-model analysis,
model training, and post-model analysis.

# Installation

You can install MadCat using pip:

```
pip install MadCat
```

# Usage
More than 50 examples how to use the pacakge for solving different data science tasks: 
https://github.com/amito-ds/chester/blob/main/projects/projects.md
d
To use MadCat, you'll need a dataframe containing your features and a target column.
Here's an example using the Iris dataset:

```
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
dataset.rename(columns={'class': 'target'}, inplace=True)
df = dataset.sample(frac=1).reset_index(drop=True)

run_metadata_collector = full_run.run_madcat(
    user_classes.Data(df=df, target_column='target'), 
)
```

## Contributing

If you are interested in contributing to MadCat, please see our CONTRIBUTING guidelines.

## License

TCAP is released under the MIT License.

## Acknowledgements

The MadCat package was developed by Amit Osi. MadCat makes use of the following open-source libraries: NLTK, spaCy,
Gensim,
Sklearn.


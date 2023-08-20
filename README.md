# MadCat

> Streamline your data science journey from raw data to insightful analytics.

MadCat is a versatile tool designed to simplify and automate data science tasks, encapsulating everything from video and
audio data processing to advanced analytics.

## Highlights

- **Chester Module**: Simplifying complex data science tasks.
    - Streamlined **data pre-processing** for efficient model training.
    - **Model training and evaluation** made easy with intuitive controls.
    - Comprehensive **post-model analysis** to identify model strengths and weaknesses.

- **Diamond Module**: Comprehensive image processing solutions.
    - **Image description**: Extract insightful metadata from images.
    - **Object detection**: Accurately classify objects within images.
    - **Face detection**: Recognize and pinpoint faces in images.

- **UDI Module**: Turning audio data into actionable insights.
    - **Speech-to-text**: Transcribe audio sequences for text analysis.
    - Extract crucial features like **MFCC** for audio classification.
    - Enhanced integration capabilities with the **Chester** analytics suite.

- **Vis Module**: A holistic approach to video data analytics.
    - **Grayscale conversion** for enhanced video processing.
    - **Object tracking** across frames for motion analysis.
    - Adjust and inspect videos with the **zoom functionality**.

## Installation

```
pip install MadCat
```

## Chester's Main Functionality: Simplified Data Science Journey

Chester, as a cornerstone of MadCat, simplifies the process of data science, taking you from raw data to insightful model analytics with ease. Let's delve deeper into its main function:

### run()

The `run()` function is a comprehensive method designed to process and analyze your data seamlessly.

#### Parameters

- **data_spec (Data)**: The main data input.

- **feature_types (dict, optional)**: Dictionary specifying the types of features (e.g., categorical, numerical, etc.)

- **text_handler (TextHandler, optional)**: Manages text data transformations.

- **is_text_handler (bool, default=True)**: Flag to determine if text handler should be used.

- **text_summary (TextSummary, optional)**: Provides summary statistics and insights for text data.

- **time_series_handler (TimeSeriesHandler, optional)**: Manages time series data transformations.

- **is_time_series_handler (bool, default=True)**: Flag to determine if time series handler should be used.

- **feature_stats (FeatureStats, optional)**: Provides feature statistics.

- **is_feature_stats (bool, default=True)**: Flag to determine if feature statistics should be used.

- **text_feature_extraction (TextFeatureExtraction, optional)**: Deals with text data processing and feature extraction.

- **is_feature_extract (bool, default=True)**: Flag to determine if text feature extraction should be used.

- **is_pre_model (bool, default=True)**: Flag to determine if pre-modeling processes should be initiated.

- **is_model_training (bool, default=True)**: Flag to determine if model training should be initiated.

- **model_run (ModelRun, optional)**: Manages and controls the model's lifecycle from training to testing.

- **is_post_model (bool, default=True)**: Flag to determine if post-modeling processes should be initiated.

- **is_model_weaknesses (bool, default=True)**: Flag to determine if model weaknesses should be analyzed.

- **plot (bool, default=True)**: Flag to determine if plots should be generated.

- **max_stats_col_width (int, default=30)**: Maximum column width for statistics display.

Using Chester's `run()` function, you can integrate, preprocess, train, and analyze your models effortlessly.


## Usage

MadCat's primary goal is to uncomplicate complex data science challenges. To see it in action:

[50+ MadCat Usage Examples](https://github.com/amito-ds/chester/blob/main/projects/projects.md)

```python
import pandas as pd
from chester.run.full_run import run

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Rename target column and shuffle dataset
dataset.rename(columns={'class': 'target'}, inplace=True)
df = dataset.sample(frac=1).reset_index(drop=True)

# Run MadCat on the dataset
run_metadata_collector = run(
    data=df, 
    target_column='target'
)
```


## License

Released under the MIT License. 
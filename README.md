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

## Usage

MadCat's primary goal is to uncomplicate complex data science challenges. To see it in action:

[50+ MadCat Usage Examples](https://github.com/amito-ds/chester/blob/main/projects/projects.md)

```python
import pandas as pd
from MadCat import full_run, user_classes

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
dataset.rename(columns={'class': 'target'}, inplace=True)
df = dataset.sample(frac=1).reset_index(drop=True)

run_metadata_collector = full_run.run(
    user_classes.Data(df=df, target_column='target'),
)
```

## License

Released under the MIT License. 
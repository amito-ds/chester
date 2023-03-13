from flask import Flask, request, render_template
import io
import pandas as pd

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    # read the file
    file = request.files['file']
    file_contents = file.read()

    # read the file contents into a pandas dataframe
    df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))

    # apply the function f to the dataframe
    output = df
    print(output[0:10])

    # return the output as a dictionary
    return render_template('output.html', output=output)


if __name__ == '__main__':
    app.run(debug=True)

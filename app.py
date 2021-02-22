import re
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
import json
from slugify import slugify
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import uuid
import os
from flask import Flask, render_template, request, flash, redirect, url_for, session
import matplotlib
matplotlib.use('Agg')
plt.style.use('ggplot')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/uploads')
ALLOWED_EXTENSIONS = {'txt', 'xls', 'xlsx'}

app = Flask(__name__)
app.static_url_path = 'static'
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SECRET_KEY'] = 'qqqqqqqqqqqqqqq'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train')
def train():
    return render_template('train.html')


@app.route('/trainmodel', methods=['POST'])
def trainmodel():
    if request.method == 'POST':
        predictors = request.files['predictors']
        dependents = request.files['dependents']
        selected_dependents = request.form.getlist('dependent_labels')

        if not any(f in request.files for f in ('predictors', 'dependents')):
            flash('No file part')
            return redirect(url_for('trainmodel'))
        if predictors.filename == '':
            flash('No selected file')
            return redirect(url_for('trainmodel'))
        if not allowed_file(predictors.filename) or not allowed_file(dependents.filename):
            flash('not supported file ext.')
            return redirect(url_for('trainmodel'))

        # if file is supported than upload file and return hash
        hash = uuid.uuid4().hex[:8]
        hashpath = UPLOAD_FOLDER + '/' + hash + '/'
        os.makedirs(hashpath)

        predictors_ext = predictors.filename.rsplit('.', 1)[1].lower()
        dependents_ext = dependents.filename.rsplit('.', 1)[1].lower()
        dependents.save(os.path.join(hashpath, "dependents." + dependents_ext))
        predictors.save(os.path.join(hashpath, "predictors." + predictors_ext))

        # split data to predictors and dependents
        predictors = pd.read_excel(hashpath + 'predictors.xlsx')
        dependents = pd.read_excel(hashpath + 'dependents.xlsx')
        predictors.rename(columns={predictors.columns[0]: 'name'})
        dependents.rename(columns={dependents.columns[0]: 'name'})
        merge = pd.merge(predictors, dependents, on='name').set_index('name')

        X = merge.iloc[:, 0:len(predictors.columns) - 1]  # predictors
        Y = merge.iloc[:, len(predictors.columns) - 1:]  # dependents / labels
        params = pd.read_excel('static/models/scores.xlsx')

        result = {'hash': hash, 'predict_type': 1, 'dependents': []}
        for i in range(len(selected_dependents)):
            dependent_name = selected_dependents[i]
            best_param = params[params['dependent'] ==
                                dependent_name]['best_params'].values[0]
            best_param_dict = json.loads(best_param.replace("\'", "\""))
            y = Y[dependent_name]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=None)
            model = XGBRegressor(**best_param_dict)
            model.fit(X_train, y_train)
            y_predict = model.predict(X)
            mse = mean_squared_error(y, y_predict)
            print(str(dependent_name) + '-> MSE: %2.3f ' % (mse))
            feature_importance_plot(X.columns.values, model.feature_importances_,
                                    hashpath + dependent_name + '_feature.png')
            prediction_histogram(y_predict, hashpath +
                                 dependent_name + '_histogram.png')
            pd.DataFrame({'strains': Y.index, 'predicts': y_predict}).set_index('strains').to_excel(
                hashpath + dependent_name + '_predicts.xlsx')
            result['dependents'].append({'name': dependent_name, 'converted_name': convert_column_name(dependent_name), 'mse': mse, 'predicts': pd.DataFrame(
                {'strains': Y.index, 'predicts': y_predict}).set_index('strains').T.to_html(
                classes='table table-striped table-bordered customframe', header="true")})

        flash('ML model trained successfully.')
        return render_template('result.html', result=result)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':

        file = request.files['resistome']
        predict_type = request.form['predict_type']
        selected_dependents = request.form.getlist('dependent_labels')

        if 'resistome' not in request.files:
            flash('No file part')
            return redirect(url_for('index'))
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('index'))
        if not allowed_file(file.filename):
            flash('not supported file ext.')
            return redirect(url_for('index'))

        # if file is supported than upload file and return hash
        ext = file.filename.rsplit('.', 1)[1].lower()
        hash = uuid.uuid4().hex[:8]
        hashpath = UPLOAD_FOLDER + '/' + hash + '/'
        os.makedirs(hashpath)
        file.save(os.path.join(hashpath, "resistome." + ext))

        # split data to predictors and dependents
        predictors = pd.read_excel(hashpath + 'resistome.xlsx')
        dependents = pd.read_excel('static/metadata.xlsx')
        predictors.rename(columns={predictors.columns[0]: 'name'})
        dependents.rename(columns={dependents.columns[0]: 'name'})
        merge = pd.merge(predictors, dependents, on='name').set_index('name')

        X = merge.iloc[:, 0:len(predictors.columns) - 1]  # predictors
        Y = merge.iloc[:, len(predictors.columns) - 1:]  # dependents

        # predict selected dependents
        result = {'hash': hash, 'predict_type': predict_type, 'dependents': []}

        for i in range(len(selected_dependents)):
            dependent_name = selected_dependents[i]
            model_file = 'static/models/' + slugify(dependent_name) + '.ml'
            model = joblib.load(model_file)
            y = Y[dependent_name]
            y_predict = model.predict(X)
            mse = mean_squared_error(y, y_predict)
            print(str(dependent_name) + '-> MSE: %2.3f ' % (mse))
            feature_importance_plot(X.columns.values, model.feature_importances_,
                                    hashpath + dependent_name + '_feature.png')
            prediction_histogram(y_predict, hashpath +
                                 dependent_name + '_histogram.png')
            pd.DataFrame({'strains': Y.index, 'predicts': y_predict}).set_index('strains').to_excel(
                hashpath + dependent_name + '_predicts.xlsx')
            result['dependents'].append({'name': dependent_name, 'converted_name': convert_column_name(dependent_name), 'mse': mse, 'predicts': pd.DataFrame(
                {'strains': Y.index, 'predicts': y_predict}).set_index('strains').T.to_html(
                classes='table table-striped table-bordered customframe', header="true")})

        flash('Selected dependents predicted with pre-trained ML model successfully.')
        return render_template('result.html', result=result)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def feature_importance_plot(feature, importance, filename=None):
    data = pd.DataFrame({'feature': feature, 'importance': importance}).sort_values(by='importance', ascending=False)[
        0:10]
    ax = data.plot(kind='barh', x='feature', y='importance')
    if filename is not None:
        ax.figure.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close('all')


def prediction_histogram(y_predict, filename=None):
    plt.hist(y_predict, density=True, bins=30)
    plt.title("Histogram")
    plt.ylabel('Prediction')
    plt.xlabel('Data')
    if filename is not None:
        plt.savefig(filename)
    plt.clf()
    plt.cla()
    plt.close('all')


def convert_column_name(name):
    if name.find('-') == -1:
        x = re.findall(r'(.*)_(.*)', name)
        type = x[0][0] == 'r' and "Generation Time" or "Growth Yield"
        string = ' '.join([type, x[0][1]])
    else:
        x = re.findall(r'(.*)_(.*)-(.*)', name)
        type = x[0][0] == 'r' and "Generation Time" or "Growth Yield"
        string = ' '.join([type, x[0][1], '<br /> Concentration:', x[0][2]])
    return string


if __name__ == '__main__':
    app.run(debug=True)

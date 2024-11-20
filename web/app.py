from data_preprocessing import preprocess
from catboost import Pool
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from data_preprocessing import copy_train
import shap

app = Flask(__name__)
app.secret_key = 'abc'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

from tqdm import tqdm
from catboost import Pool, CatBoostClassifier

model = CatBoostClassifier().load_model('final/model_cb.cbm')
explainer = shap.Explainer(model)
features_df = pd.read_excel('описание.xlsx')


def explain_predictions(data, top=5):
    """
    Функция для объяснения вероятности предсказания модели CatBoost для нескольких строк.
    Параметры:
        - model: обученная модель CatBoostClassifier
        - data: DataFrame с данными для объяснения
    Возвращает:
        DataFrame с текстовыми объяснениями вкладов признаков в вероятность предсказания.
    """

    data_pool = Pool(data)

    shap_values = model.get_feature_importance(type="ShapValues", data=data_pool)

    feature_contributions = shap_values[:, :-1]
    predictions = shap_values[:, -1]

    explanations = []

    medians = data.median()

    for i in tqdm(range(len(data))):
        feature_info = [
            (feature_name, feature_value, feature_contributions[i, j])
            for j, (feature_name, feature_value) in enumerate(zip(data.columns, data.iloc[i]))
        ]

        if predictions[i] > 0.5:
            sorted_features = sorted([f for f in feature_info if f[2] > 0], key=lambda x: abs(x[2]), reverse=True)
        else:
            sorted_features = sorted([f for f in feature_info if f[2] < 0], key=lambda x: abs(x[2]), reverse=True)

        explanation = []

        for i, (feature_name, feature_value, contribution) in enumerate(sorted_features[:top]):
            threshold = medians[feature_name]
            sign = '+' if contribution > 0 else '-'
            explanation.append(
                f"{i + 1}) {features_df[features_df.колонка == feature_name].описание.values[0].strip()} значение {feature_value:.3f} {'>' if feature_value > threshold else '<='} чем медиана {threshold} -> {sign}{abs(contribution):.3f} к вероятности")

        explanations.append("\n".join(explanation))

    return pd.DataFrame({"interpretation": explanations})


# def explain_predictions(model, data, top=5):
#     """
#     Функция для объяснения вероятности предсказания модели CatBoost для нескольких строк.
#     Параметры:
#         - model: обученная модель CatBoostClassifier
#         - data: DataFrame с данными для объяснения
#     Возвращает:
#         DataFrame с текстовыми объяснениями вкладов признаков в вероятность предсказания.
#     """
#
#     data_pool = Pool(data)
#
#     shap_values = model.get_feature_importance(type="ShapValues", data=data_pool)
#
#     feature_contributions = shap_values[:, :-1]
#     predictions = shap_values[:, -1]
#
#     explanations = []
#
#     medians = data.median()
#
#     for i in range(len(data)):
#         feature_info = [
#             (feature_name, feature_value, feature_contributions[i, j])
#             for j, (feature_name, feature_value) in enumerate(zip(data.columns, data.iloc[i]))
#         ]
#
#         if predictions[i] > 0.5:
#             sorted_features = sorted([f for f in feature_info if f[2] > 0], key=lambda x: abs(x[2]), reverse=True)
#         else:
#             sorted_features = sorted([f for f in feature_info if f[2] < 0], key=lambda x: abs(x[2]), reverse=True)
#
#         explanation = []
#
#         for i, (feature_name, feature_value, contribution) in enumerate(sorted_features[:top]):
#             threshold = medians[feature_name]
#             sign = '+' if contribution > 0 else '-'
#             explanation.append(
#                 f"{i + 1}) {features_df[features_df.колонка == feature_name].описание.values[0].strip()} значение {feature_value:.3f} {'>' if feature_value > threshold else '<='} чем медиана {threshold} -> {sign}{abs(contribution):.3f} к вероятности")
#
#         explanations.append("\n".join(explanation))
#
#     return pd.DataFrame({"interpretation": explanations})
#
#
# # Пример использования
# expl = explain_predictions(model.cb_model, temp1)
# expl
def get_plot(data):
    shap_values = explainer(data)
    index = 0
    plt.figure()
    shap.waterfall_plot(shap_values[index], show=False)  # Prevents immediate display
    plt.savefig('static/shap.png', format='png', bbox_inches='tight', dpi=300)  # Adjust DPI for better quality
    plt.close()


@app.route('/show_histogram')
def show_histogram():
    path = request.args.get('path')
    scored, all_test = path.split('∫')
    df = pd.read_csv(scored)
    score_column = 'score'  # Replace with the actual column name for scores in your data

    # Generate histogram plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df[score_column], kde=True)
    plt.title("Score Distribution Histogram")
    plt.xlabel("Score")
    plt.ylabel("Frequency")

    # Save plot to a temporary buffer and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return render_template("histogram.html", image_base64=image_base64)


@app.route('/interpret_score', methods=['GET', 'POST'])
def interpret_score():
    row = -1
    k = 5
    if request.method == 'POST':
        print(request.form)
        if request.form['num_rows']!='':
            row=int(request.form['num_rows'])
        if request.form['num_features']!='':
            k=int(request.form['num_features'])

    path = request.args.get('path')
    scored, filename = path.split('∫')
    df = pd.read_csv(scored[:-4] + "©" + '.csv')
    expl = explain_predictions(df, top=k)
    df = pd.read_csv(scored)
    df['explanation'] = expl.interpretation
    fl = f"{scored[:-4]}interpret_top{k}.csv"
    df_show = df.copy()
    if row==-1:
        df_show['explanation'] = df_show['explanation'].apply(lambda row: row[:70] + "...")
    else:
        df_show['explanation'] = df_show['explanation'].apply(lambda row: row[:400] + "...")
    df_show = df_show.round(4)
    df.to_csv(fl, index=False)
    fl2 = f"{scored[:-4]}interpret_top{k}_rez.csv"
    if row!=-1:
        df_show = df_show.iloc[row-1 : row]
    df_show.to_csv(fl2, index=False)
    headers, data, total_columns, total_rows = process_csv(fl2)
    if row==-1:
        return render_template('score_interpretation.html', k=k, headers=headers, data=data, total_columns=total_columns,
                               total_rows=total_rows,
                               path=fl, flname=filename)
    else:
        get_plot(pd.read_csv(scored[:-4] + "©" + '.csv').iloc[row-1:row])
        import base64

        with open('static/shap.png', 'rb') as img:
            # Открываем файл изображения в бинарном режиме
            boxplot_base64 = base64.b64encode(img.read()).decode('utf-8')
        return render_template('score_interpretation.html', k=k, headers=headers, data=data,
                               total_columns=total_columns,
                               total_rows=total_rows,
                               path=fl, flname=filename,
                               boxplot_base64=boxplot_base64)


@app.route('/find_best_contractor')
def find_best_contractor():
    path = request.args.get('path')
    scored, all_test = path.split('∫')
    df_sc = pd.read_csv(scored)
    tested = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], all_test))
    tested['score'] = df_sc['score']
    dd = tested.groupby('contractor_id')['score'].mean()
    tt = dd.sort_values().to_frame()['score'][:10].round(5)
    best_contractor = tt
    headers = ['contractor_id', 'mean default probability']

    return render_template("best_contractor.html", contractor=best_contractor, headers=headers)


@app.route('/data_analysis')
def data_analysis():
    path = request.args.get('path')
    scored, all_test = path.split('∫')
    df_sc = pd.read_csv(scored)
    tested = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], all_test))
    tested['score'] = df_sc['score']
    dd = tested.groupby('building_id')['score'].mean()
    tt = dd.sort_values().to_frame()['score'][:3].round(5)
    ttt = tt[:0]
    ttt['Building_id'] = ""
    dd = tested.groupby('specialization_id')['score'].mean()
    tt2 = dd.sort_values().to_frame()['score'][:3].round(5)
    ttt2 = tt2[:0]
    ttt2['Specialization_id'] = ""
    dd = tested.groupby('project_id')['score'].mean()
    tt3 = dd.sort_values().to_frame()['score'][:3].round(5)
    ttt3 = tt3[:0]
    ttt3['Project_id'] = ""
    dd = tested.groupby('contract_id')['score'].mean()
    tt4 = dd.sort_values().to_frame()['score'][:3].round(5)
    ttt4 = tt4[:0]
    ttt4['Contract_id'] = ""
    best = pd.concat([ttt, tt, ttt2, tt2, ttt3, tt3, ttt4, tt4])
    headers = ['Category and id', 'Mean default probability']
    import base64

    with open('static/output.png', 'rb') as img:
        # Открываем файл изображения в бинарном режиме
        boxplot_base64 = base64.b64encode(img.read()).decode('utf-8')
    plt.close()

    return render_template("data_analysis.html", boxplot_base64=boxplot_base64, contractor=best, headers=headers)


@app.route('/graph_visualisation')
def graph_visualisation():
    path = request.args.get('path')
    scored, all_test = path.split('∫')
    df = pd.read_csv(scored)

    # Generate diagrams/graphics (for demonstration, using basic stats and plots)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    import base64

    with open('static/2024-11-15 08.44.09.jpg', 'rb') as img:
        # Открываем файл изображения в бинарном режиме
        boxplot_base64 = base64.b64encode(img.read()).decode('utf-8')
    plt.close()

    return render_template("graph_visualisation.html", boxplot_base64=boxplot_base64)


def process_csv(file_path, max_columns=7, max_rows=7):
    data = []
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)

        # Проверка количества столбцов
        if len(headers) > max_columns + 3:
            displayed_headers = headers[:max_columns] + ['...'] + headers[-3:]
        else:
            displayed_headers = headers

        # Подготовка данных
        for row in csvreader:
            if len(row) > max_columns + 3:
                displayed_row = row[:max_columns] + ['...'] + row[-3:]
            else:
                displayed_row = row
            data.append(displayed_row)

        # Проверка количества строк
        if len(data) > max_rows + 3:
            displayed_data = data[:max_rows] + [['...'] * len(displayed_headers)] + data[-3:]
        else:
            displayed_data = data

    return displayed_headers, displayed_data, len(headers), len(data)


# Создаем папку для загрузки, если она не существует
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/download')
def download_file():
    path = request.args.get('path')
    return send_file(path, as_attachment=True)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if len(request.form) != 0:
            filename = list(request.form.to_dict().keys())[0]
            preprocess(pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename)), copy_train,
                       os.path.join(app.config['UPLOAD_FOLDER'], "preds_" + filename))
            headers, data, total_columns, total_rows = process_csv(
                os.path.join(app.config['UPLOAD_FOLDER'], "preds_" + filename))
            return render_template('table.html', headers=headers, data=data, total_columns=total_columns,
                                   total_rows=total_rows,
                                   path=os.path.join(app.config['UPLOAD_FOLDER'], "preds_" + filename), flname=filename)
        if 'file' in request.files and request.files['file'] and allowed_file(request.files['file'].filename):
            file = request.files['file']
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded!')
            headers, data, total_columns, total_rows = process_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('upload.html', headers=headers, data=data, total_columns=total_columns,
                                   total_rows=total_rows, path=os.path.join(app.config['UPLOAD_FOLDER']), uploaded=True,
                                   Upload='Upload another', flname=filename)
        else:
            flash('Only .csv files are allowed.')
            return redirect(url_for('upload_file'))

    return render_template('upload.html', uploaded=False, Upload='Upload')


if __name__ == '__main__':
    app.run(debug=True, port=3333)

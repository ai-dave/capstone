from bioinformatics import na_read

import time
from datetime import datetime  
import pytz
import os, json
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for, redirect, flash

import logging
import pickle

app = Flask(__name__)

app.config['SECRET_KEY'] = 'df0331cefc6c2b9a5d0208a726a5d1c0fd37324feba25506'

cl = pickle.load(open('model-human.pkl','rb'))
cvr = pickle.load(open('CountVectorizer-human.pkl','rb'))

messages = []

genefamilies = {
    1: "G protein coupled receptors",
    2: "Tyrosine kinase",
    3: "Tyrosine phospotase",
    4: "Synthetase",
    5: "Synthase",
    6: "Ion channel",
    7: "Transcription factor"
}


@app.route('/home')
def home():
    return render_template('home.html', messages=messages)


# ...

@app.route('/query/', methods=('GET', 'POST'))
def query():

    if request.method == 'POST':
        annotation = request.form['annotation']
        sequence = request.form['sequence']

        if not annotation:
            flash('Annotation is required!')
        elif not sequence:
            flash('Sequence is required!')
        else:
            X = sequence.lower()
            y_pred_text = predict(sequence)

            my_datetime=datetime.fromtimestamp(time.time())
            datetime_local = my_datetime.astimezone(pytz.timezone('US/Pacific')).strftime('%Y-%m-%d %H:%M:%S %Z%z')

            messages.append({'annotation': annotation, 'length':  len(X), 'start': X[:30], 'end': X[-30:], 'prediction': y_pred_text, 'timestamp': datetime_local})
            return redirect(url_for('home'))

    return render_template('query.html')



@app.route('/hints')
def hints():
    filename = os.path.join(app.static_folder, 'data', 'test_data.json')

    with open(filename) as test_file:
        data = json.load(test_file)

    return render_template('hints.html', data=data)


@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/course')
def course():
    return render_template('course.html')



@app.route('/api/',methods=['POST'])
def api():
    data = request.get_json(force=True)

    sequence = data['sequence']
    annotation = data['annotation']
    y_pred_text = predict(sequence)

    response = {
        "prediction": y_pred_text,
        "annotation": annotation
    }
    return jsonify(response)



def predict(sequence):
    X = sequence.lower()
    X2 = na_read.make_kmer_sentence(6, sequence)

    X_embedding = cvr.transform([X2])
    y_pred = cl.predict(X_embedding)
    iy = int(y_pred[0])
    y_pred_text = genefamilies[iy] if (iy>0 and iy<=len(genefamilies)) else '---'

    app.logger.info(f'data: len(X): [{len(X)}] - [{X[:30]}]...[{X[-30:]}], [{X2[:30]}]...[{X2[-30:]}], predict: [{y_pred}]: [{y_pred_text}]')
    return y_pred_text




if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True)
        #app.run(port=8088, host = '0.0.0.0', debug=True)
    except:
        print("Server exited.")

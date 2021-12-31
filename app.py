from flask import Flask, request, url_for, render_template
from markupsafe import escape

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def init():
    app.logger.debug('Method: '+ request.method)
    if request.method == 'POST':
        tweet = request.form['tweet']
        return render_template('result.html', res=tweet)
    else:
        return render_template('index.html') 
from bottle import route, run, template, post, request, static_file
import MNIST_predict
import numpy


index_html = open("index.html").read()

@route('/')
def index():
    return template(index_html)

@post('/api')
def api():
    data = numpy.asarray(request.json, 'float32')
    if numpy.amax(data) != 0:
        result = MNIST_predict.predict(data / numpy.amax(data))
        return str(result.tolist())

@route('/js/<path:path>')
def js(path):
    return static_file(path, 'js')

run(host='localhost', port=8080)


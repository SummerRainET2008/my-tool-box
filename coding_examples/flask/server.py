from flask import Flask, jsonify, abort

app = Flask(__name__)

@app.route('/sim/tasks/<string:input>', methods=['GET'])
def get_tasks(input):
  result = {
    "input": input,
    "len": len(input),
  }
  return jsonify(result)

if __name__ == '__main__':
  # app.run(debug=True, host="192.168.1.201", port='5000')
  app.run(debug=True, host="0.0.0.0", port='5000')


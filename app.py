# app.py
from flask import Flask, render_template, request, jsonify
from llm import give_advice

app = Flask(__name__)

chat_history = []

def get_answer(question):
    # This function just returns a mirror of the question for demonstration purposes
    patient_data_dict={"a":1}
    memory = []
    logger = None
    answer = give_advice(question, patient_data_dict, memory, logger)
    return f"{answer}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form['user_input']
    chat_history.append({'user': True, 'message': user_input})
    answer = get_answer(user_input)
    chat_history.append({'user': False, 'message': answer})
    return jsonify({'user_input': user_input, 'answer': answer, 'chat_history': chat_history})

if __name__ == '__main__':
    app.run(debug=True)

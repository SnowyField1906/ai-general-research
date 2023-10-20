import re
import os
from flask import Flask, request, jsonify

app = Flask(__name__)
current_directory = os.path.dirname(os.path.realpath(__file__))
database_file_path = os.path.join(current_directory, "database.txt")

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if username and password:
        with open(database_file_path, 'r') as database:
            file_data = database.read()
            if re.search(f"username: {username}, password: {password}", file_data):
                return jsonify({'success': False}), 201

        with open(database_file_path, 'a') as database:
            database.write(f"username: {username}, password: {password}\n")
        
        return jsonify({'success': True}), 201
    else:
        return jsonify({'success': False}), 400

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if username and password:
        with open(database_file_path, 'r') as database:
            file_data = database.read()
            if re.search(f"username: {username}, password: {password}", file_data):
                return jsonify({'success': True}), 200
            else:
                return jsonify({'success': False}), 400
    else:
        return jsonify({'success': False}), 400

if __name__ == '__main__':
    app.run(debug=False)

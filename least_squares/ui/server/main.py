from flask import Flask, request, jsonify
import least_squares

from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/least-squares', methods=['POST'])
def calc():
    try:
        data = request.get_json()
        data_points = data.get('data_points')
        degree = data.get('degree')
        A, B = least_squares.points_to_matrix(data_points, degree)
        Q, R = least_squares.qr_decomposition(A)
        img = least_squares.plot_solution(A, B)
        return jsonify(
            {
                'A': A.tolist(),
                'B': B.tolist(),
                'Q': Q.tolist(),
                'R': R.tolist(),
                'x': least_squares.least_squares(A, B).tolist(),
                'img': img
            }
        ), 200

    except:
        return jsonify({'error': 'Something went wrong'}), 500
    
@app.route('/')
def index():
    return "Hello World!"

if __name__ == '__main__':
    app.run(debug=True)


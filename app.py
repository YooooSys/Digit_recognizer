from flask import Flask, Response, render_template, request, jsonify
import subprocess
import os

app = Flask(__name__)

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-python')
def run_python():
    user_ip = request.remote_addr

    def generate():
        # Get the client's IP address

        # Pass the IP address as an argument to the Python script
        process = subprocess.Popen(['python', 'run.py', user_ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        for line in iter(process.stdout.readline, ''):
            yield f"data: {line}\n\n"
            process.stdout.flush()  # Force flush the output

        for line in iter(process.stderr.readline, ''):
            yield f"data: [ERROR] {line}\n\n"
            process.stderr.flush()  # Force flush the output

    return Response(generate(), mimetype='text/event-stream')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file:
        # Get the client's IP address
        user_ip = request.remote_addr

        # Save the file to the user's directory
        file_path = os.path.join(UPLOAD_FOLDER, f"({user_ip}).jpg")
        file.save(file_path)
        return jsonify({'message': f'File uploaded successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True)
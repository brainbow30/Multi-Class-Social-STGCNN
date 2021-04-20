from flask import Flask, render_template, Response

import config
from videoCamera import VideoCamera

app = Flask(__name__)


def gen_frames(camera):
    while True:
        success, frame = camera.get_frame()  # read the camera frame
        if success:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(VideoCamera(config.frameSkip)), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, threaded=True)

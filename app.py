# adapted with permission from: https://github.com/enjeck/Blobby

from flask import Flask, request, json, send_file
from flask import send_from_directory
from skinGAN import Skin_generator

from io import BytesIO
from waitress import serve

app = Flask(__name__,static_url_path='')
Gen = Skin_generator("models/Minecraft_skins_G9.pt")

@app.route('/')
def home_page():
    return send_from_directory("static/", 'index.html')



@app.route('/skin', methods=['POST'])
def getSkin():
    global raw1
    rendered, raw = Gen.generate_skin()
    raw1 = raw
    img_io = BytesIO()
    rendered.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')


@app.route('/download', methods=['POST'])
def downloadSkin():

    img_io = BytesIO()
    raw1.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

# run app
serve(app, host="0.0.0.0", port=8080)
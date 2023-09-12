import torch
from torchvision import transforms
from PIL import Image
from os import listdir
import os
import random
import torch.nn.functional as F
import torch.nn as nn
from flask import Flask
import io
import base64 


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(256),
                                 transforms.ToTensor(),
                                 normalize])

class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.conv3 = nn.Conv2d(12, 18, kernel_size=5)
        self.conv4 = nn.Conv2d(18, 24, kernel_size=5)
        self.fc1 = nn.Linear(3456, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = x.view(-1,3456)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)


if os.path.isfile("meinNetz.pt"):
    model = torch.load("meinNetz.pt")
    model.cuda()
    model.eval()
else:
    raise FileExistsError("No model file found")

def eval(img):
    files = listdir('test/')
    f = random.choice(files)
    img = Image.open('test/' + f)
    img_eval_tensor = transforms(img)
    img_eval_tensor.unsqueeze_(0)
    data = img_eval_tensor.cuda()
    out = model(data)
    out = str(out.data.max(1, keepdim=True)[1].item()).replace("1", "Hund").replace("0", "Katze")

    return out, f

def base64_to_img(base64_obj):
    return Image.open(io.BytesIO(base64.b64decode(base64_obj.split(",",1)[0])))

from flask import Flask, request, jsonify

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/cat_or_dog", methods=["POST"])
def test():
    data = request.json
    img = base64_to_img(data["img"])
    out = eval(img)
    return jsonify({"out": out[0]})


if __name__ == '__main__':
    app.run(debug=True)

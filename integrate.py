import torch
import numpy as np
from PIL import Image
import glob
from torchvision import transforms

# if torch.cuda.is_available():
#     device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")


def integration(test_folder,in_type):

    class_c = ["black", "blue", "green", "orange", "red", "white", "yellow"]
    class_n = ["crew", "square", "turtle", "v-neck"]
    class_s = ["long", "short", "sleeveless"]
    class_b = ['buttons','no buttons']

    device = "cpu" # torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    if in_type == 'colors':
        model = torch.load('./models/model_c_94.pt')
        model.eval()
        class_names = class_c
        norm = 0.69370186
        std = 0.3048072
    elif in_type == 'sleeves':
        device = "cpu"
        model = torch.load('./models/model_sleeves_cpu_best.pt')
        model.eval()
        class_names = class_s
        norm = 0.69370186
        std = 0.3048072
    elif in_type == 'necklines':
        model = torch.load('./models/model_n_79.pt')
        model.eval()
        class_names = class_n
        norm = 0.68 #, 0.68, 0.68
        std = 0.283#, 0.283, 0.283
    elif in_type == 'buttons':
        device = "cpu"
        model = torch.load('./models/model_button_best.pt')
        model.eval()
        class_names = class_b
        norm = 0.69370186
        std = 0.3048072
    else:
        print("ERR: input type doesn't exist")
        return -1

    out = []
    for i, filename in enumerate(glob.glob(test_folder + '/*.jpg')):
        img = Image.open(filename)
        if in_type == 'buttons':
            img =img.resize((100,100))
        else:
            img = img.resize((100, 100))
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((norm,norm,norm), (std,std,std))])
        img = trans(img).to(device)
        output = model(img.unsqueeze(0), batch_norm=True)

        # if in_type == 'buttons':
        #     output = torch.sigmoid(output)
        if in_type == 'sleeves' or in_type == 'necklines':
            output = torch.softmax(output, dim=1)

        output = torch.Tensor.cpu(output)
        if in_type == 'buttons':
            if output.data.numpy() >= 0.5:
                label = 'buttons'
            else:
                label = 'no buttons'
        else:
            label = class_names[np.argmax(output.data.numpy())]
        value = float(torch.max(output).detach().numpy())
        if (in_type == 'colors' and value < 0.80) or (in_type == 'sleeves' and value < 0.8) or (in_type == 'necklines' and value < 0.65) or (in_type == 'buttons' and value < 0.70):
            value = 'NULL'
        else:
            value = round(value,3)
        out += [[label,value]]

    return out

# ans = integration('../test_folder','colors')
# for dab in ans:
#     print(dab)
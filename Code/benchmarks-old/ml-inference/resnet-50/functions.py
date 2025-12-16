import os
import torch
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer

model = None

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, os.path.pardir, os.path.pardir, os.path.pardir, 'benchmarks', 'ml-inference', 'resnet-50')

def function(obj):

    global model

    if model is None:

        #print("Model load", flush=True)
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        before = timer()
        #model = torch.load(os.path.join(INPUT_DIR, 'resnet-50.pt'))
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
# model = torch.load('resnext50_32x4d.pt')

        #print("Model done", flush=True)
        model.eval()
        #print("Model finished", flush=True)
        model.to('cuda')
        #print("Model finished to CUDA", flush=True)
        torch.cuda.get_device_properties('cuda')
        after = timer()

        # UNCOMMENT FOR swapping benchmark
        #print('model eval time:', after-before, flush=True)
        #print(after - before)

    start = timer()

    input_image = Image.open(os.path.join(INPUT_DIR, 'dog.jpg'))
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    input_batch = input_batch.to('cuda')


    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    top_catid = top_catid[0].item()
    top_prob = top_prob[0].item()

    print("RESULT", top_catid, top_prob, flush=True)

if __name__ == "__main__":

    for i in range(11):
        start = timer()
        function({})
        end = timer()
        print("Time:", end-start)

# 单张图片推理调用示例
import time
from torchvision.io import read_image
from torchvision.transforms import transforms


test_transform = transforms.Compose([               # image preprocess
    transforms.Resize(size=256),                    # resize the short edge to 256 with its original height/width ratio
    transforms.CenterCrop(size=(height, width)),    # (h, w)
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],                 # RGB
        std=[0.229, 0.224, 0.225]                   # RGB
    )
])


device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_image_full_path = "./scenes/seg_test/buildings/20057.jpg"     # your test image full path
image = read_image(test_image_full_path)                           # load the test image
print("raw image's shape = {}".format(image.shape))

model_path = "./scenes.pth"                                        # specify the model weights
loaded_model = torch.jit.load(model_path)                          # load pre-trained model
loaded_model.eval()                                                # configure it as inference mode

run_times = 500

t1 = time.time()
for _ in range(run_times):
    model_input = test_transform(image)
    model_input = model_input[None, :]
    model_input = model_input.to(device)
    logits, probs = loaded_model(model_input)
t2 = time.time()

probs = probs.detach().cpu().numpy()

print("Time cost for each inference (224x224x3 input size) = {} ms".format(1000 * (t2 - t1) / run_times))
print("probs = {}".format(probs))
print("probs.shape = {}".format(probs.shape))

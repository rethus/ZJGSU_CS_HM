import torch
import torch.nn as nn
from utils import transforms
import cv2

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(
            x
        )  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)
        return x


if __name__ == "__main__":
    model = PatchEmbed(img_size=128, patch_size=8)
    device = torch.device("cpu")
    model.to(device)

    img = cv2.imread(f"../data/coco2017/train2017/000000000086.jpg")
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    # 这里需要注意opencv独特图像存储方式
    trans = transforms.ToTensor()
    img = trans(img, img)
    output = model(img)
    print(output)

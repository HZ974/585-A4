import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()

        # Load the pretrained VGG-16 and use its features
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())

        # Encoder
        self.features_block1 = nn.Sequential(*features[:5])  # First pooling
        self.features_block2 = nn.Sequential(*features[5:10])  # Second pooling
        self.features_block3 = nn.Sequential(*features[10:17])  # Third pooling
        self.features_block4 = nn.Sequential(*features[17:24])  # Fourth pooling
        self.features_block5 = nn.Sequential(*features[24:])  # Fifth pooling

        # Modify the classifier part of VGG-16
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        # Decoder
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_final = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

        # Skip connections
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = x.to(device)

        pool3 = self.features_block3(self.features_block2(self.features_block1(x)))
        pool4 = self.features_block4(pool3)
        pool5 = self.features_block5(pool4)

        # Classifier
        score = self.classifier(pool5)

        # Decoder
        upscore2 = self.upscore2(score)
        score_pool4 = self.score_pool4(pool4)


        # sliced_score_upscore2 = torch.cat((upscore2[:, :, :12, :], upscore2[:, :, -12:, :]), dim=2)
        # final_sliced_score_upscore2 = torch.cat((sliced_score_upscore2[:, :, :, :16], sliced_score_upscore2[:, :, :, -16:]), dim=3)
        final_sliced_score_upscore2 = F.interpolate(upscore2, size=(24, 32), mode='bilinear', align_corners=False)

        upscore_pool4 = self.upscore_pool4(score_pool4 + final_sliced_score_upscore2)
        score_pool3 = self.score_pool3(pool3)

        # sliced_score_upscore_pool4 = torch.cat((upscore_pool4[:, :, :24, :], upscore_pool4[:, :, -24:, :]), dim=2)
        # final_sliced_score_upscore_pool4 = torch.cat((sliced_score_upscore_pool4[:, :, :, :32], sliced_score_upscore_pool4[:, :, :, -32:]), dim=3)
        final_sliced_score_upscore_pool4 = F.interpolate(upscore_pool4, size=(48, 64), mode='bilinear', align_corners=False)

        
        upscore_final = self.upscore_final(final_sliced_score_upscore_pool4 + score_pool3)

        sliced_upscore_final = torch.cat((upscore_final[:, :, :192, :], upscore_final[:, :, -192:, :]), dim=2)
        final_sliced_upscore_final = torch.cat((sliced_upscore_final[:, :, :, :256], sliced_upscore_final[:, :, :, -256:]), dim=3)

        return final_sliced_upscore_final


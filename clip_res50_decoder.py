class DecoderBottleneck(nn.Module):

    expansion = 4  # Changed from 4 to 2

    def __init__(self, inplanes, planes, stride=1):

        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes // self.expansion, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes // self.expansion)
        self.conv2 = nn.Conv2d(planes // self.expansion, planes // self.expansion, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes // self.expansion)
        self.conv3 = nn.Conv2d(planes // self.expansion, planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True) if stride > 1 else nn.Identity()

        self.stride = stride

        if stride > 1 or inplanes != planes:
            self.shortcut = nn.Sequential(OrderedDict([
                ("0", nn.Conv2d(inplanes, planes, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes)),
                ("2", nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True))

            ]))

        else:

            self.shortcut = nn.Identity()

    def forward(self, z: torch.Tensor):

        identity = self.shortcut(z)
        out = self.relu(self.bn1(self.conv1(z)))
        out = self.upsample(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = self.relu(out)

        return out

class ModifiedResNetDecoder(nn.Module):

    def __init__(self, layers, input_dim, output_dim=3, width=64):

        super().__init__()

        self.input_dim = input_dim

        self.output_dim = output_dim

        self._inplanes = width * 32  # starting from the end of encoder

        self.layer4 = self._make_layer(width * 16, layers[3], stride=1)
        self.layer3 = self._make_layer(width * 8, layers[2], stride=2)
        self.layer2 = self._make_layer(width * 4, layers[1], stride=2)
        self.layer1 = self._make_layer(width, layers[0], stride=1)

        # the 3-layer "stem" (in reverse)

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.ConvTranspose2d(width, width // 2, kernel_size=4, stride=2,padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.ConvTranspose2d(width // 2, width // 2, kernel_size=3,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv1 = nn.ConvTranspose2d(width // 2, output_dim, kernel_size=3, padding=1, bias=False)

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(DecoderBottleneck(self._inplanes, planes, stride))
        self._inplanes = planes
        for _ in range(1, blocks):
            layers.append(DecoderBottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, z):
        # x torch.Size([64, 2048, 16, 8])
        z = self.layer4(z)     ######## result:torch.Size([64, 1024, 16, 8])
        z = self.layer3(z)  ###torch.Size([64, 512, 32, 16])
        z = self.layer2(z)  ####torch.Size([64, 256, 64, 32])
        z = self.layer1(z)  ####torch.Size([64, 64, 64, 32])
        z = self.upsample(z) #torch.Size([64, 3, 256, 128])
        z = self.relu(self.bn3(self.conv3(z))) #torch.Size([64, 64, 128, 64])
        z = self.relu(self.bn2(self.conv2(z))) #torch.Size([64, 64, 128, 64])
        z = self.conv1(z)  #torch.Size([64, 3, 128, 64])
        #z = self.sigmoid(z)

        return z

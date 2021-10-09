import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageDraw

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Skin_generator:
    # initial setup, needs to be run once to load model
    def __init__(self,model_path, latent_vector_size=100):
        self.device = torch.device("cpu")
        self.nz = latent_vector_size
        self.nc = 3 # n channels
        self.ngf = 64 # image size 64x64
        self.netG = Generator(0, self.nz, self.ngf, self.nc).to(self.device)
        self.netG.load_state_dict(torch.load(model_path))
        self.netG.eval()

    # render skin from raw skin image
    def render_skin(self, skin):
        background = Image.new("RGB", (68, 36), (200, 200, 200))

        # front
        head = skin.crop((8, 8, 16, 16))
        background.paste(head, (4, 2, 12, 10))

        left_arm = skin.crop((36, 52, 40, 64))
        background.paste(left_arm, (12, 10, 16, 22))

        right_arm = skin.crop((44, 20, 48, 32))
        background.paste(right_arm, (0, 10, 4, 22))

        torso = skin.crop((20, 20, 28, 32))
        background.paste(torso, (4, 10, 12, 22))

        right_leg = skin.crop((4, 20, 8, 32))
        background.paste(right_leg, (4, 22, 8, 34))

        left_leg = skin.crop((20, 52, 24, 64))
        background.paste(left_leg, (8, 22, 12, 34))

        # back
        head = skin.crop((24, 8, 32, 16))
        background.paste(head, (24, 2, 32, 10))

        left_arm = skin.crop((44, 52, 48, 64))
        background.paste(left_arm, (32, 10, 36, 22))

        right_arm = skin.crop((52, 20, 56, 32))
        background.paste(right_arm, (20, 10, 24, 22))

        torso = skin.crop((32, 20, 40, 32))
        background.paste(torso, (24, 10, 32, 22))

        right_leg = skin.crop((12, 20, 16, 32))
        background.paste(right_leg, (24, 22, 28, 34))

        left_leg = skin.crop((28, 52, 32, 64))
        background.paste(left_leg, (28, 22, 32, 34))

        # left

        head = skin.crop((16, 8, 24, 16))
        background.paste(head, (42, 2, 50, 10))

        left_arm = skin.crop((40, 52, 44, 64))
        background.paste(left_arm, (44, 10, 48, 22))

        left_leg = skin.crop((24, 52, 28, 64))
        background.paste(left_leg, (44, 22, 48, 34))

        # right

        head = skin.crop((0, 8, 8, 16))
        background.paste(head, (60, 2, 68, 10))

        right_arm = skin.crop((40, 20, 44, 32))
        background.paste(right_arm, (62, 10, 66, 22))

        left_leg = skin.crop((0, 20, 4, 32))
        background.paste(left_leg, (62, 22, 66, 34))

        finished = background.resize((680, 360), resample=Image.NEAREST)

        return finished

    #generate new skin and render it
    def generate_skin(self):
        noise = torch.randn(1, self.nz, 1, 1, device=self.device)
        fake = vutils.make_grid(self.netG(noise).detach().cpu(), padding=2, normalize=True)
        tran1 = transforms.ToPILImage()
        pil_image_single = self.make_transparent(tran1(fake))
        img = self.render_skin(pil_image_single)
        return img, pil_image_single

    def make_transparent(self,skin):
        background = Image.new("RGBA", (64, 64), (255, 0, 0, 0))

        # front
        head = skin.crop((8, 8, 16, 16))
        background.paste(head, (8, 8, 16, 16))

        left_arm = skin.crop((36, 52, 40, 64))
        background.paste(left_arm, (36, 52, 40, 64))

        right_arm = skin.crop((44, 20, 48, 32))
        background.paste(right_arm, (44, 20, 48, 32))

        torso = skin.crop((20, 20, 28, 32))
        background.paste(torso, (20, 20, 28, 32))

        right_leg = skin.crop((4, 20, 8, 32))
        background.paste(right_leg, (4, 20, 8, 32))

        left_leg = skin.crop((20, 52, 24, 64))
        background.paste(left_leg, (20, 52, 24, 64))

        # back
        head = skin.crop((24, 8, 32, 16))
        background.paste(head, (24, 8, 32, 16))

        left_arm = skin.crop((44, 52, 48, 64))
        background.paste(left_arm, (44, 52, 48, 64))

        right_arm = skin.crop((52, 20, 56, 32))
        background.paste(right_arm, (52, 20, 56, 32))

        torso = skin.crop((32, 20, 40, 32))
        background.paste(torso, (32, 20, 40, 32))

        right_leg = skin.crop((12, 20, 16, 32))
        background.paste(right_leg, (12, 20, 16, 32))

        left_leg = skin.crop((28, 52, 32, 64))
        background.paste(left_leg, (28, 52, 32, 64))

        # left

        head = skin.crop((16, 8, 24, 16))
        background.paste(head, (16, 8, 24, 16))

        left_arm = skin.crop((40, 52, 44, 64))
        background.paste(left_arm, (40, 52, 44, 64))

        left_leg = skin.crop((24, 52, 28, 64))
        background.paste(left_leg, (24, 52, 28, 64))

        # right

        head = skin.crop((0, 8, 8, 16))
        background.paste(head, (0, 8, 8, 16))

        right_arm = skin.crop((40, 20, 44, 32))
        background.paste(right_arm, (40, 20, 44, 32))

        left_leg = skin.crop((0, 20, 4, 32))
        background.paste(left_leg, (0, 20, 4, 32))

        #tops and bottoms

        head_top = skin.crop((8, 0, 16, 8))
        background.paste(head_top, (8, 0, 16, 8))

        head_bottom = skin.crop((16, 0, 24, 8))
        background.paste(head_bottom, (16, 0, 24, 8))

        right_leg_top = skin.crop((4, 16, 8, 20))
        background.paste(right_leg_top, (4, 16, 8, 20))

        right_leg_bottom = skin.crop((8, 16, 12, 20))
        background.paste(right_leg_bottom, (8, 16, 12, 20))

        left_leg_bottom = skin.crop((24, 48, 28, 52))
        background.paste(left_leg_bottom,(24, 48, 28, 52))

        left_leg_top = skin.crop((20, 48, 24, 52) 	)
        background.paste(left_leg_top,(20, 48, 24, 52))

        torso_top = skin.crop((20, 16, 28, 20) )
        background.paste(torso_top ,(20, 16, 28, 20))

        torso_bottom = skin.crop((28, 16, 36, 20))
        background.paste(torso_bottom, (28, 16, 36, 20))

        left_arm_top = skin.crop((36, 48, 40, 52))
        background.paste(left_arm_top, (36, 48, 40, 52))

        left_arm_bottom = skin.crop((40, 48, 44, 52))
        background.paste(left_arm_bottom, (40, 48, 44, 52))

        right_arm_top = skin.crop((44, 16, 48, 20))
        background.paste(right_arm_top, (44, 16, 48, 20))

        right_arm_bottom = skin.crop((48, 16, 52, 20))
        background.paste(right_arm_bottom, (48, 16, 52, 20))

        return background


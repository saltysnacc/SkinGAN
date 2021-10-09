from skinGAN import Skin_generator

Gen = Skin_generator("models/Minecraft_skins_G9.pt")

rendered, raw_skin = Gen.generate_skin()

rendered.show()
raw_skin.show()
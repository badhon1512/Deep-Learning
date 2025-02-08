from pattern import Circle
from pattern import Spectrum
from pattern import Checker
from generator import ImageGenerator



c = Checker(250,25)
c.show()

s = Spectrum(250)
s.show()
print(s.output.shape)

cir = Circle(1024, 200, (512, 256))
cir.show()

label_path = './Labels.json'
file_path = './exercise_data/'
gen = ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=False, mirroring=False,
                             shuffle=False)

gen.show()
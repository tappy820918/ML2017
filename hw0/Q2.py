import sys
from PIL import Image


def main(*args):
	path = args[0][1]
	image = Image.open(path)
	rgb = list(image.getdata())
	rgb = [tuple(int(t/2) for t in x ) for x in rgb]
	new_image = image
	new_image.putdata(rgb)
	# new_image.show()
	new_image.save('Q2.png')
	

if __name__ == '__main__':
	main(sys.argv)
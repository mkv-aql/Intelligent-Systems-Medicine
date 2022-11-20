import Get_Image
import glob
import os

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Strg+F8 to toggle the breakpoint.


if __name__ == '__main__':
    file_path = "C:/Users/Makav/Desktop/ISM_2022w/test" #Simply change the file path to the correct location
    Get_Image.get_image(file_path)
    print(os.listdir(file_path))
    #print(Get_Image.images)
    #print(glob.glob("C:/Users/Makav/Desktop/ISM_2022w/test/*"))
    #print(glob.glob("C:/Users/Makav/Desktop/ISM_2022w/test/*.png"))

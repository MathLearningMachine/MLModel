# Trims dataset to 1000 per subfolder (per class)

import os

count = 0
for dir in os.listdir('.'):
    for file in os.listdir(dir)[1000:]:
        try:
            os.remove(dir + '/' + file)
        except:
            count = count + 1

print(count)
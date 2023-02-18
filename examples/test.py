import os
for root, dirs, files in os.walk("C:/Users/devid/OneDrive/Epoka MSc/3rd WS2223/CEN 593 Graduate Project/thesis"):
    for file in files:
        if file.endswith(".txt"):
             print(os.path.join(root, file))

import matplotlib.pyplot as plt
import os

filename = "FetchPush-v2_HER_REDQ.txt"
file = open(os.path.join("outputs", filename), "r")
lines = file.readlines()

goto_sentence = "test_reward: "

array = []
for line in lines:
    pos = line.find(goto_sentence)
    if pos != -1:
        pos += len(goto_sentence)
        array.append(float(line[pos:pos+8]))

x = range(len(array))
plt.plot(x, array)

plt.xlabel('Training epochs')
plt.ylabel('Score')
plt.title(filename[:-4].replace("_", " "))

plt.ylim(-50, 0)

# plt.show()

output_dir = "plots"
plt.savefig(os.path.join(output_dir, filename[:-4]+".png"),
            bbox_inches="tight", pad_inches=0.3)
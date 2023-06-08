import matplotlib.pyplot as plt
import os

filename = "FetchPush-v2_HER_REDQ.txt"
file = open(os.path.join("outputs", filename), "r")
lines = file.readlines()

goto_actor_loss = "loss/actor="
goto_critic1_loss = "loss/critics="
#goto_critic2_loss = "loss/critic2="

def extract_from_file(goto, length=4):
    array = []
    for line in lines:
        pos = line.find(goto)
        if pos != -1:
            pos += len(goto)
            array.append(float(line[pos:pos+length]))
    return array

array_actor_loss = extract_from_file(goto_actor_loss)
array_critic1_loss = extract_from_file(goto_critic1_loss)
#array_critic2_loss = extract_from_file(goto_critic2_loss)

x = range(len(array_actor_loss))
plt.plot(x, array_actor_loss, label="actor_loss")
plt.plot(x, array_critic1_loss, label="critic_loss")
#plt.plot(x, array_critic2_loss, label="critic2_loss")

plt.legend(loc="upper right")

plt.xlabel('Training epochs')
plt.ylabel('Loss')
plt.title(filename[:-4].replace("_", " "))

# plt.show()

output_dir = "losses"
plt.savefig(os.path.join(output_dir, filename[:-4]+".png"),
            bbox_inches="tight", pad_inches=0.3)
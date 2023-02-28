import matplotlib.pyplot as plt

file = open("demo/test", "r")
lines = file.readlines()

goto_sentence = "test_reward: "

array = []
for line in lines:
    pos = line.find(goto_sentence)
    if pos != -1:
        pos += len(goto_sentence)
        array.append(float(line[pos:pos+8]))

x = range(100)
plt.plot(x, array)

plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Training scores')

plt.show()
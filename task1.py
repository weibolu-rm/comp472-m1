import os
import matplotlib.pyplot as plt

base_path = "BBC/"

# getting file distribution
classes = {}
max_count = 0
for item in os.listdir(base_path):
    path = os.path.join(base_path, item)
    if(os.path.isdir(path)):
        print(item)
        count = len([file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))])
        print(count)
        if count > max_count:
            max_count = count
        classes.update({item : count})

print(classes)

plt.plot(list(classes.keys()), list(classes.values()), 'b')
plt.xlabel("Classes")
plt.ylabel("Count")
plt.savefig("BBC-distribution.pdf")


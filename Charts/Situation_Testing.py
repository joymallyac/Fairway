import numpy as np
import matplotlib.pyplot as plt

n_groups = 7
means_before = (11.26,4.36,8.45,12.27,1.32,10,1)
means_after = (0.77,.74,4.68,5.82,0.26,1.89,0)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.23
opacity = 1

rects1 = plt.bar(index, means_before, bar_width,
alpha=opacity,
color='orange',
label='Before')

rects2 = plt.bar(index  + bar_width, means_after, bar_width,
alpha=opacity,
color='dodgerblue',
label='After')

plt.xlabel('')
plt.ylabel('% of data points failing situation testing')
plt.title('')
plt.xticks(index + bar_width, ('Adult - Sex', 'Adult - Race', 'Compas - Sex', 'Compas - Race', 'Default Credit - Sex', 'Heart Health - Age', 'German - Sex'),rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig("Verification.pdf")
plt.show()
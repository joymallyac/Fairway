import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(num=None, figsize = (15,10), facecolor='w', edgecolor='k')

ax1 = fig.add_subplot(221)   #top left
n_groups = 7
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8

# recall data
means_Default = (.42, .42, .57, .57, .23, .73, .99)
means_Pre_processing = (.25, .34, .49, .56, .21, .84, .97)
means_Optimization = (.43, .42, .57, .59, .21, .83, .95)
means_Fairway = (.24, .38, .56, .54, .2, .76, .96)

rects1 = plt.bar(index, means_Default, bar_width,
alpha=opacity,
color='orange',
label='Baseline')

rects2 = plt.bar(index  + bar_width, means_Pre_processing, bar_width,
alpha=opacity,
color='dodgerblue',
label='Pre-processing(P)')

rects3 = plt.bar(index  + bar_width +  bar_width, means_Optimization, bar_width,
alpha=opacity,
color='limegreen',
label='Optimization(O)')

rects4 = plt.bar(index + bar_width +  bar_width +  bar_width , means_Fairway, bar_width,
alpha=opacity,
color='r',
label='Fairway(P+O)')

plt.xlabel('')
plt.ylabel('Recall')
plt.title('Change of Recall')
plt.xticks(index + bar_width, ('Adult - Sex', 'Adult - Race', 'Compas - Sex', 'Compas - Race', 'Default Credit - Sex', 'Heart Health - Age', 'German - Sex'),rotation=45)
plt.legend()

plt.tight_layout()

ax2 = fig.add_subplot(222)   #top right

## false alarm data
means_Default = (.05, .05, 0.30, 0.30, .02, .07, .56)
means_Pre_processing = (.01, 0.03, 0.21, 0.26, .02, .07, .61)
means_Optimization = (.06, 0.06, 0.31, 0.26, .03, .07, .60)
means_Fairway = (.01, 0.03, 0.25, 0.23, .02, .12, .62)

rects1 = plt.bar(index, means_Default, bar_width,
alpha=opacity,
color='orange',
label='')

rects2 = plt.bar(index  + bar_width, means_Pre_processing, bar_width,
alpha=opacity,
color='dodgerblue',
label='')

rects3 = plt.bar(index  + bar_width +  bar_width, means_Optimization, bar_width,
alpha=opacity,
color='limegreen',
label='')

rects4 = plt.bar(index + bar_width +  bar_width +  bar_width , means_Fairway, bar_width,
alpha=opacity,
color='r',
label='')

plt.xlabel('')
plt.ylabel('False alarm')
plt.title('Change of False Alarm')
plt.xticks(index + bar_width, ('Adult - Sex', 'Adult - Race', 'Compas - Sex', 'Compas - Race', 'Default Credit - Sex', 'Heart Health - Age', 'German - Sex'),rotation=45)


ax3 = fig.add_subplot(223)   #bottom left

## AOD data
means_Default = (.12,.03,.06,.02,.02,.04,.14)
means_Pre_processing = (.02,.02,.03,.05,.02,.06,.06)
means_Optimization = (.08,.04,.03,.03,.02,.06,.12)
means_Fairway = (.02,.02,.02,.04,.01,.05,.04)

rects1 = plt.bar(index, means_Default, bar_width,
alpha=opacity,
color='orange',
label='')

rects2 = plt.bar(index  + bar_width, means_Pre_processing, bar_width,
alpha=opacity,
color='dodgerblue',
label='')

rects3 = plt.bar(index  + bar_width +  bar_width, means_Optimization, bar_width,
alpha=opacity,
color='limegreen',
label='')

rects4 = plt.bar(index + bar_width +  bar_width +  bar_width , means_Fairway, bar_width,
alpha=opacity,
color='r',
label='')

plt.xlabel('')
plt.ylabel('AOD')
plt.title('Change of AOD')
plt.xticks(index + bar_width, ('Adult - Sex', 'Adult - Race', 'Compas - Sex', 'Compas - Race', 'Default Credit - Sex', 'Heart Health - Age', 'German - Sex'),rotation=45)


ax4 = fig.add_subplot(224)   #bottom right 
fig.subplots_adjust(hspace=.4)
fig.tight_layout(pad=3.0)


## EOD data
means_Default = (.30,.12,.32,.09,.06,.02,.05)
means_Pre_processing = (.01,.03,.21,.25,.04,.01,.08)
means_Optimization = (.23,.12,.15,.14,.05,.11,.06)
means_Fairway = (.03,.03,.21,.13,.02,.02,.04)


rects1 = plt.bar(index, means_Default, bar_width,
alpha=opacity,
color='orange',
label='')

rects2 = plt.bar(index  + bar_width, means_Pre_processing, bar_width,
alpha=opacity,
color='dodgerblue',
label='')

rects3 = plt.bar(index  + bar_width +  bar_width, means_Optimization, bar_width,
alpha=opacity,
color='limegreen',
label='')

rects4 = plt.bar(index + bar_width +  bar_width +  bar_width , means_Fairway, bar_width,
alpha=opacity,
color='r',
label='')

plt.xlabel('')
plt.ylabel('EOD')
plt.title('Change of EOD')
plt.xticks(index + bar_width, ('Adult - Sex', 'Adult - Race', 'Compas - Sex', 'Compas - Race', 'Default Credit - Sex', 'Heart Health - Age', 'German - Sex'),rotation=45)
# plt.legend()

plt.show()
fig.savefig("Combined.pdf")
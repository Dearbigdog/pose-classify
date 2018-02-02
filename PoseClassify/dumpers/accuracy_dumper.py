import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

statistics_accuracy_31=np.load('.\\statistics_accuracy_51.txt')
# statistics_accuracy_30=np.load('../statistics_accuracy_30.txt')
# statistics_accuracy_105=np.load('../statistics_accuracy_105.txt')
#
statistics_false_neg_31=np.load('.\\statistics_false_neg_51.txt')
statistics_false_pos_31=np.load('.\\statistics_false_pos_51.txt')

plt.figure()
plt.ylim((0,100))
plt.plot(range(0,len(statistics_accuracy_31)),statistics_accuracy_31, 'r--')

plt.ylabel('accuracy (%)')
plt.xlabel('iteration times')
plt.title('Accuracy \n Avg red={0:.2f}%'.format(np.average(statistics_accuracy_31)))

red_patch = mpatches.Patch(color='red', label='31 features')

plt.legend(handles=[red_patch])
plt.draw()

plt.figure()
# false positive & false negative
plt.ylim((0,100))
plt.plot(range(0,len(statistics_false_pos_31)),statistics_false_pos_31, 'b--')

plt.ylim((0,100))
plt.plot(range(0,len(statistics_false_neg_31)),statistics_false_neg_31, 'r--')

plt.ylabel('false predict (%)')
plt.xlabel('iteration times')
plt.title('False Predict \n false positive={0:.2f}%, false negative={1:.2f}%'.format(np.average(statistics_false_pos_31),np.average(statistics_false_neg_31)))

red_patch = mpatches.Patch(color='red', label='False pos')
blue_patch = mpatches.Patch(color='blue', label='False neg')
plt.legend(handles=[red_patch,blue_patch])
plt.draw()


plt.show()
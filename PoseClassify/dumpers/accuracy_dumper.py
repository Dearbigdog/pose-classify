import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

statistics_accuracy_43=np.load('../statistics_accuracy_43.txt')
statistics_accuracy_30=np.load('../statistics_accuracy_30.txt')
statistics_accuracy_105=np.load('../statistics_accuracy_105.txt')

statistics_false_neg_43=np.load('../statistics_false_neg_43.txt')
statistics_false_pos_43=np.load('../statistics_false_pos_43.txt')

plt.figure()
plt.ylim((0,100))
plt.plot(xrange(0,len(statistics_accuracy_43)),statistics_accuracy_43, 'g--')


plt.ylim((0,100))
plt.plot(xrange(0,len(statistics_accuracy_30)),statistics_accuracy_30, 'b--')

plt.ylim((0,100))
plt.plot(xrange(0,len(statistics_accuracy_105)),statistics_accuracy_105, 'r--')

plt.ylabel('accuracy (%)')
plt.xlabel('iteration times')
plt.title('Accuracy \n Avg green ={0:.2f}%,  Avg blue={1:.2f}%, Avg red={2:.2f}%'.format(np.average(statistics_accuracy_43),np.average(statistics_accuracy_30),np.average(statistics_accuracy_105)))

red_patch = mpatches.Patch(color='red', label='30 features')
blue_patch = mpatches.Patch(color='blue', label='105 features')
green_patch = mpatches.Patch(color='green', label='43 features')
plt.legend(handles=[red_patch,blue_patch,green_patch])
plt.draw()

plt.figure()
#false positive & false negative
plt.ylim((0,100))
plt.plot(xrange(0,len(statistics_false_pos_43)),statistics_false_pos_43, 'b--')

plt.ylim((0,100))
plt.plot(xrange(0,len(statistics_false_neg_43)),statistics_false_neg_43, 'r--')

plt.ylabel('false predict (%)')
plt.xlabel('iteration times')
plt.title('False Predict \n false positive={0:.2f}%, false negative={1:.2f}%'.format(np.average(statistics_false_pos_43),np.average(statistics_false_neg_43)))

red_patch = mpatches.Patch(color='red', label='False pos')
blue_patch = mpatches.Patch(color='blue', label='False neg')
plt.legend(handles=[red_patch,blue_patch])
plt.draw()


plt.show()
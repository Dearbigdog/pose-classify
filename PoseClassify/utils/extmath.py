import numpy as np

def vAngle(v1, v2, acute=True):
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return angle*180/np.pi

#x project on plane S whose normal is n
def planeProject(x, n):
    # unit n
    unitN=[i/np.linalg.norm(n) for i in n]
    temp=np.dot(x, unitN)
    xProjected = x -[temp*i for i in unitN]
    return xProjected
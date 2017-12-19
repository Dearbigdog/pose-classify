import numpy as np

def vector_angle(v1, v2, acute=True):
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return angle*180/np.pi

#x project on plane S whose normal is n
def plane_proj(x, n):
    # unit n
    unitN=[i/np.linalg.norm(n) for i in n]
    temp=np.dot(x, unitN)
    xProjected = x -[temp*i for i in unitN]
    return xProjected

def distanse_3d(x1,y1,z1,x2,y2,z2):
    d1=np.sqrt(np.power(x1-x2,2))
    d2=np.sqrt(np.power(y1-y2,2))
    d3=np.sqrt(np.power(z1-z2,2))
    return d1+d2+d3
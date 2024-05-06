import numpy as np
import torch
'''

Colors:
  red:    0
  white:  1
  blue:   2
  orange: 3
  green:  4
  yellow: 5

Input Channels for NN: The channels are in the same order of colors. This means first channel is for red and ... 


* Faces are numbered 0...5 as follows.
 *
 *  +-------+
 *  |\       \
 *  | \   0   \
 *  |  \_______\
 *  | 1|       |
 *  \  |   2   |
 *   \ |       |
 *    \|_______|
 *
 *  +-------+
 *  |       |\
 *  |   4   | \
 *  |       |  \
 *  |_______|3 |
 *  \       \  |
 *   \   5   \ |
 *    \_______\|
 *

corner locations:
  
 *                    
 *  3-------2         
 *  |\       \        
 *  | \       \       
 *  |  \0_____1\      
 *  7  |       |      
 *  \  |       |      
 *   \ |       |      
 *    \4_______5      
 *                    
 *                    
 *  3-------2         
 *  |       |\        
 *  |       | \       
 *  |       |  1      
 *  7_______6  |      
 *  \       \  |      
 *   \       \ |      
 *    \4______5|      


drawing purposes:

 
 *  6-------8
 *  |\       \
 *  | \       \
 *  |  \0_____2\
 *  24 |       |
 *  \  |       |
 *   \ |       |
 *    \18_____20
 *
 *
 *  6-------8
 *  |       |\
 *  |       | \
 *  |       |  2
 *  24____26|  |
 *  \       \  |
 *   \       \ |
 *    \18____20|

 
'''

class State:
  
  def __init__(self,):
    self.loc=[-1,-1,-1,-1,-1,-1,-1,-1]
    self.orientation=[-1,-1,-1,-1,-1,-1,-1,-1]

  def getFaceColor(self,face):
    cube = self.loc[face//3] 
    rot =  self.orientation[cube] 
    result= cube*3+(3+(face%3)-rot)%3

    thecolor=-1
    if result==0:
      thecolor=0
    elif result==1:
      thecolor=4
    elif result==2:
      thecolor=2
    elif result==3:
      thecolor=0
    elif result==4:
      thecolor=2
    elif result==5:
      thecolor=5
    elif result==6:
      thecolor=0
    elif result==7:
      thecolor=5
    elif result==8:
      thecolor=3
    elif result==9:
      thecolor=0
    elif result==10:
      thecolor=3
    elif result==11:
      thecolor=4
    elif result==12:
      thecolor=1
    elif result==13:
      thecolor=2
    elif result==14:
      thecolor=4
    elif result==15:
      thecolor=1
    elif result==16:
      thecolor=5
    elif result==17:
      thecolor=2
    elif result==18:
      thecolor=1
    elif result==19:
      thecolor=3
    elif result==20:
      thecolor=5
    elif result==21:
      thecolor=1
    elif result==22:
      thecolor=4
    elif result==23:
      thecolor=3
    else:
      raise ValueError("This value is not valid.")
    
    return thecolor
      

  def get_nn_input(self,one_hot=True):
    nn_input=np.ones((6,3,3))
    nn_input_one_hot_encoded=np.zeros((36,3,3))

    # color the center cubies
    # color all edge cubies black
    for i in range(6):
      
      # nn_input[i,1,1]=i 
      # nn_input[i,0,1]=nn_input[i,1,0]=nn_input[i,1,2]=nn_input[i,2,1]=i
      
      nn_input_one_hot_encoded[7*i,1,1]=1
      nn_input_one_hot_encoded[7*i,0,1]=nn_input_one_hot_encoded[7*i,1,0]=nn_input_one_hot_encoded[7*i,1,2]=nn_input_one_hot_encoded[7*i,2,1]=1

    
    for i in range(len(self.loc)):

      if i==0:
        # nn_input[0,2,0]=self.getFaceColor(0)
        # nn_input[2,0,0]=self.getFaceColor(2)
        # nn_input[4,0,2]=self.getFaceColor(1)

        nn_input_one_hot_encoded[self.getFaceColor(0),2,0]=1
        nn_input_one_hot_encoded[12+self.getFaceColor(2),0,0]=1
        nn_input_one_hot_encoded[24+self.getFaceColor(1),0,2]=1

      elif i==1:
        # nn_input[0,2,2]=self.getFaceColor(3)
        # nn_input[2,0,2]=self.getFaceColor(4)
        # nn_input[5,0,0]=self.getFaceColor(5)

        nn_input_one_hot_encoded[self.getFaceColor(3),2,2]=1
        nn_input_one_hot_encoded[12+self.getFaceColor(4),0,2]=1
        nn_input_one_hot_encoded[30+self.getFaceColor(5),0,0]=1

      elif i==2:
        # nn_input[0,0,2]=self.getFaceColor(6)
        # nn_input[5,0,2]=self.getFaceColor(7)
        # nn_input[3,0,0]=self.getFaceColor(8)

        nn_input_one_hot_encoded[self.getFaceColor(6),0,2]=1
        nn_input_one_hot_encoded[30+self.getFaceColor(7),0,2]=1
        nn_input_one_hot_encoded[18+self.getFaceColor(8),0,0]=1

      elif i==3:
        # nn_input[0,0,0]=self.getFaceColor(9)
        # nn_input[4,0,0]=self.getFaceColor(11)
        # nn_input[3,0,2]=self.getFaceColor(10)

        nn_input_one_hot_encoded[self.getFaceColor(9),0,0]=1
        nn_input_one_hot_encoded[24+self.getFaceColor(11),0,0]=1
        nn_input_one_hot_encoded[18+self.getFaceColor(10),0,2]=1

      elif i==4:
        # nn_input[1,0,0]=self.getFaceColor(12)
        # nn_input[2,2,0]=self.getFaceColor(13)
        # nn_input[4,2,2]=self.getFaceColor(14)

        nn_input_one_hot_encoded[6+self.getFaceColor(12),0,0]=1
        nn_input_one_hot_encoded[12+self.getFaceColor(13),2,0]=1
        nn_input_one_hot_encoded[24+self.getFaceColor(14),2,2]=1

      elif i==5:
        # nn_input[1,0,2]=self.getFaceColor(15)
        # nn_input[2,2,2]=self.getFaceColor(17)
        # nn_input[5,2,0]=self.getFaceColor(16)

        nn_input_one_hot_encoded[6+self.getFaceColor(15),0,2]=1
        nn_input_one_hot_encoded[12+self.getFaceColor(17),2,2]=1
        nn_input_one_hot_encoded[30+self.getFaceColor(16),2,0]=1

      elif i==6:
        # nn_input[1,2,2]=self.getFaceColor(18)
        # nn_input[5,2,2]=self.getFaceColor(20)
        # nn_input[3,2,0]=self.getFaceColor(19)

        nn_input_one_hot_encoded[6+self.getFaceColor(18),2,2]=1
        nn_input_one_hot_encoded[30+self.getFaceColor(20),2,2]=1
        nn_input_one_hot_encoded[18+self.getFaceColor(19),2,0]=1

      elif i==7:
        # nn_input[1,2,0]=self.getFaceColor(21)
        # nn_input[4,2,0]=self.getFaceColor(22)
        # nn_input[3,2,2]=self.getFaceColor(23)

        nn_input_one_hot_encoded[6+self.getFaceColor(21),2,0]=1
        nn_input_one_hot_encoded[24+self.getFaceColor(22),2,0]=1
        nn_input_one_hot_encoded[18+self.getFaceColor(23),2,2]=1
      else:
        pass
    
    if one_hot:
      return nn_input_one_hot_encoded
    else:
      return nn_input

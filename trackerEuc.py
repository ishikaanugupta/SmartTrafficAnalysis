#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math


# In[2]:


class EuclideanDistanceTracker:
    def __init__(self):
        self.center_points = {} #Store the center points of object
        self.id_count = 0
    
    def update(self,objects_rect):
        objects_bb_ids = [] #Objects boxes and ids
        
        for rect in objects_rect: #Get center point of the new object
            x,y,w,h = rect
            cx = x + x + w//2
            cy = y + y + h//2
            
            same_object_detected = False #To find out if that object was detected already
            for id,pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                
                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    #print(self.center_points)
                    objects_bb_ids.append([x,y,w,h,id])
                    same_object_detected = True
                    break
                    
            if same_object_detected is False: #Assigning new id to object
                self.center_points[self.id_count] = (cx, cy)
                objects_bb_ids.append([x,y,w,h,self.id_count])
                self.id_count+= 1
                
        new_center_points = {} #Clean the dictionary by center points to remove ids not in use
        for obj_bb_id in objects_bb_ids:
            _,_,_,_,object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
            
        self.center_points = new_center_points.copy()
        return objects_bb_ids


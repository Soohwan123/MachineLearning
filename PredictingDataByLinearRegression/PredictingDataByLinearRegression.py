#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf


# In[16]:


import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution() 


# In[17]:


xData = [1,2,3,4,5,6,7] # Declare X datas


# In[18]:


yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000] # Declare Y datas


# In[19]:


W = tf.Variable(tf.random.uniform([1], -100, 100)) #  Gradient


# In[20]:


b = tf.Variable(tf.random.uniform([1], -100, 100)) # y-intercept


# In[21]:


X = tf.placeholder(tf.float32)


# In[23]:


Y = tf.placeholder(tf.float32)


# In[22]:


H = W*X+b


# In[25]:


cost = tf.reduce_mean(tf.square(H-Y))


# In[26]:


a=tf.Variable(0.01) # declare the size of steps


# In[27]:


optimizer = tf.train.GradientDescentOptimizer(a) #Gradient descent Library


# In[28]:


train = optimizer.minimize(cost) # minimize the cost as possible as it can


# In[30]:


init = tf.global_variables_initializer() # reset all variables


# In[31]:


sess = tf.Session()


# In[32]:


sess.run(init)


# In[35]:


# Runs train
for i in range(5001):
    sess.run(train, feed_dict = {X: xData, Y: yData})
    if i % 500 == 0:
        print ( i, sess.run(cost, feed_dict={X: xData, Y:yData}), sess.run(W), sess.run(b)) # shows the progress every 500 times
        
print(sess.run(H, feed_dict={X: [8]})) # Makes the machine to predict the result when X is 8 after training


# In[ ]:





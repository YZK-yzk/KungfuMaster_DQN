import tensorflow as tf
import numpy as np
import gym
import time
import matplotlib.pyplot as plt


tf.set_random_seed(1)
np.random.seed(1)

class DQN():
    def __init__(self,nstate,naction):
        self.nstate=nstate
        self.naction=naction
        self.sess = tf.Session()
        self.memcnt=0
        self.BATCH_SIZE = 64
        self.LR = 0.001                      # learning rate  0.001
        self.EPSILON = 0.92                 # greedy policy   0.92
        self.GAMMA = 0.9999                   # reward discount  0.9999
        self.MEM_CAP = 10000                  #10000
        self.mem= np.zeros((self.MEM_CAP, self.nstate * 2 + 2))     # initialize memory
        self.updataT=1000                    #best 400  < 800 < 1000
        self.built_net()
        self.saver = tf.train.Saver()
        #self.saver.restore(self.sess, "./mode/mymodel.ckpt")  # 注意此处路径前添加"./"


    def built_net(self):
        self.s = tf.placeholder(tf.float64, [None,self.nstate])
        self.a = tf.placeholder(tf.int32, [None,])
        self.r = tf.placeholder(tf.float64, [None,])
        self.s_ = tf.placeholder(tf.float64, [None,self.nstate])

        with tf.variable_scope('q'):                                  # evaluation network
            l_eval = tf.layers.dense(self.s, 64, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1))
            self.q = tf.layers.dense(l_eval, self.naction, kernel_initializer=tf.random_normal_initializer(0, 0.1))

        with tf.variable_scope('q_next'):                                           # target network, not to train
            l_target = tf.layers.dense(self.s_, 64, tf.nn.relu, trainable=False)
            q_next = tf.layers.dense(l_target, self.naction, trainable=False)

        q_target = self.r + self.GAMMA * tf.reduce_max(q_next, axis=1)    #q_next:  shape=(None, naction),
        a_index=tf.stack([tf.range(self.BATCH_SIZE,dtype=tf.int32),self.a],axis=1)
        q_eval=tf.gather_nd(params=self.q,indices=a_index)
        loss=tf.losses.mean_squared_error(q_target,q_eval)
        self.train=tf.train.AdamOptimizer(self.LR).minimize(loss)
        #  q现实target_net- Q估计
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self,status):
        status=np.reshape(status,(1,self.nstate))
        if  self.memcnt>self.MEM_CAP and  np.random.uniform(0.0,1.0)<self.EPSILON:
            action=np.argmax( self.sess.run(self.q,feed_dict={self.s:status}))
        else:
            action=np.random.randint(0,self.naction)
        return action

    def learn(self):
        if(self.memcnt%self.updataT==0):
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
            self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

        rand_indexs=np.random.choice(self.MEM_CAP,self.BATCH_SIZE,replace=False)
        temp=self.mem[rand_indexs]
        bs = temp[:,0:self.nstate]#.reshape(self.BATCH_SIZE,NSTATUS)
        ba = temp[:,self.nstate]
        br = temp[:,self.nstate+1]
        bs_ = temp[:,self.nstate+2:]#.reshape(self.BATCH_SIZE,NSTATUS)
        self.sess.run(self.train, feed_dict={self.s:bs,self.a:ba,self.r:br,self.s_:bs_})


    def storeExp(self,s,a,r,s_):
        self.mem[self.memcnt%self.MEM_CAP]=np.hstack([s,a,r,s_])
        self.memcnt+=1


    def run(self,numsteps):
        cnt_win =0
        all_marks=[]
        for i in range(numsteps):
            if(self.memcnt>self.MEM_CAP):
                print("episode:", i)
            s=env.reset()
            all_r = 0.0
            while(True):
                a=self.choose_action(s/255.0)
                s_,r,done,_=env.step(a)
                if (self.EPSILON>0.93 and self.memcnt > self.MEM_CAP):
                    env.render()
                    # time.sleep(0.05)
                all_r+=r
                self.storeExp(s/255.0,a,r,s_/255.0)
                if(self.memcnt>self.MEM_CAP):
                    self.learn()
                if (done):
                    all_marks.append(all_r);
                    print(all_r)
                    # if (all_r > 10000):
                    #     self.saver.save(self.sess, "mode1/mymode.ckpt")
                    #     #self.EPSILON+=0.01
                    # elif(all_r>8000):
                    #     self.saver.save(self.sess, "mode2/mymode.ckpt")
                    #     #self.EPSILON += 0.005
                    # elif(all_r>6000):
                    #     #self.EPSILON += 0.003
                    #     self.saver.save(self.sess, "mode3/mymode.ckpt")
                    break
                s=s_

        plt.plot(all_marks)
        plt.show()



env = gym.make("KungFuMaster-ram-v0")
env = env.unwrapped
dqn=DQN(128,env.action_space.n)
dqn.run(100)

dqn.saver.save(dqn.sess, "mode0/mymodel.ckpt")

#Name : Jade Kevin Bestami
#Course: Senior Project
#Cryptology with adversarial neural networks



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

#logging
tf.logging.set_verbosity(tf.logging.INFO)

N = input("length of message and key? (default: 16)") or 16
EPOCHS = input("# of epochs (default: 50,000)") or 50000
BS = input("batch size? (default: 4096)") or 4096
LR = input("learning rate? (default: 0.0008)") or 0.0008
logPath = input("logging file? (default: no logging)") or "none"

N = int(N)
EPOCHS = int(EPOCHS)
BS = int(BS)
LR = float(LR)

if (logPath != "none"):
    writer = tf.summary.FileWriter(logPath)


def main():
    with tf.Session() as sess:

         net = Network(sess, msgLength=N, epochs=EPOCHS,
                              batchSize=BS, learningRate=LR, logPath=logPath)

         if(logPath!="none"):
            writer.add_graph(sess.graph)
         net.train()


class Network(object):

    #"constructor"
    def __init__(self, session, msgLength, batchSize,
                 epochs, learningRate, logPath):
        """
        :parameter
            session: TensorFlow session
            msgLength: The length of the input message to encrypt.
            batchSize: Minibatch size for each adversarial training
            epochs: Number of epochs in the adversarial training
            learningRate: Learning Rate for Adam Optimizer
        """

        self.session = session
        self.msgLength = msgLength
        self.key_len = self.msgLength
        self.N = self.msgLength
        self.batchSize = batchSize
        self.epochs = epochs
        self.learningRate = learningRate
        self.logPath = logPath

        self.build_model()

########################################################################################################################


    def build_model(self):

        """
        Function that builds and initialize Alice, Bob and Eve
        :post the model is built
        """

        self.curEpoch=0
        self.msg = tf.placeholder(tf.float32, shape=[None, self.msgLength], name="MSG")
        self.key = tf.placeholder(tf.float32, shape=[None, self.key_len],name="KEY")


        with tf.variable_scope("Alice"):
            # concatenate the message and the key and set the result as Alice's input
            self.alice_input = tf.concat(values=[self.msg, self.key],axis=1,name='concatMsgKey')

            with tf.variable_scope("AliceFC"):
                # initialize weights of FC layer
                self.w_alice = tf.get_variable("alice_w", shape=[2 * self.N, 2 * self.N],initializer=tf.contrib.layers.xavier_initializer())

                if (self.logPath != "none"):
                    tf.summary.histogram("AliceFCWeights", self.w_alice)  # logging

                # matrix multiply weights and inputs and apply sigmoid activation function
                self.alice_hidden = tf.nn.sigmoid(tf.matmul(self.alice_input, self.w_alice,name='matmulInputsWeights'),name='applySigmoid')
                self.alice_hidden = tf.expand_dims(self.alice_hidden, 2) #we want rank3 tensor for conv1d

            with tf.variable_scope("AliceConv"):
                self.alice_conv1 = tf.nn.relu(conv1D(self.alice_hidden, [4, 1, 2], stride=1, name='aliceConv1'))
                self.alice_conv2 = tf.nn.relu(conv1D(self.alice_conv1, [2, 2, 4], stride=2, name='aliceConv2'))
                self.alice_conv3 = tf.nn.relu(conv1D(self.alice_conv2, [1, 4, 4], stride=1, name='aliceConv3'))
                self.alice_conv4 = tf.nn.tanh(conv1D(self.alice_conv3, [1, 4, 1], stride=1, name='aliceConv4'), name='tanh')

            self.alice_output = tf.squeeze(self.alice_conv4)


        with tf.variable_scope("Bob"):

            # concat Alice's output with the key and set result as input
            self.bob_input = tf.concat([self.alice_output, self.key], 1, name='concatAliceOutKey')

            with tf.variable_scope("BobFC"):
                #init weights of FC layer
                self.w_bob = tf.get_variable("bob_w", shape=[2 * self.N, 2 * self.N],initializer=tf.contrib.layers.xavier_initializer())

                if (self.logPath != "none"):
                    tf.summary.histogram("BobFCWeights", self.w_bob)  # logging

               # matrix multiply weights and inputs and apply sigmoid activation function
                self.bob_hidden = tf.nn.sigmoid(tf.matmul(self.bob_input, self.w_bob,name='matmulInputsWeights'),name='applySigmoid')
                self.bob_hidden = tf.expand_dims(self.bob_hidden, 2)

            with tf.variable_scope("BobConv"):
                self.bob_conv1 = tf.nn.relu(conv1D(self.bob_hidden, [4, 1, 2], stride=1, name='bobConv1'))
                self.bob_conv2 = tf.nn.relu(conv1D(self.bob_conv1, [2, 2, 4], stride=2, name='bobConv2'))
                self.bob_conv3 = tf.nn.relu(conv1D(self.bob_conv2, [1, 4, 4], stride=1, name='bobConv3'))
                self.bob_conv4 = tf.nn.tanh(conv1D(self.bob_conv3, [1, 4, 1], stride=1, name='bobConv4'), name='tanh')

                self.bob_output = tf.squeeze(self.bob_conv4)


        with tf.variable_scope("Eve"):
            #Eve's input is Alice's output: the ciphertext. W/O the key!
            self.eve_input = self.alice_output

            with tf.variable_scope("EveFC1"):

                #init weights of first FC layer of Eve, note difference in shape
                self.w_eve1 = tf.get_variable("eve_w1", shape=[self.N, 2 * self.N],initializer=tf.contrib.layers.xavier_initializer())

                if (self.logPath != "none"):
                    tf.summary.histogram("EveFC1Weights", self.w_bob)  # logging

                self.eve_hidden1 = tf.nn.sigmoid(tf.matmul(self.eve_input, self.w_eve1))

            with tf.variable_scope("EveFC2"):
                # init weights of second FC layer of Eve
                self.w_eve2 = tf.get_variable("eve_w2", shape=[2 * self.N, 2 * self.N],initializer=tf.contrib.layers.xavier_initializer())
                self.eve_hidden2 = tf.nn.sigmoid(tf.matmul(self.eve_hidden1, self.w_eve2))
                self.eve_hidden2 = tf.expand_dims(self.eve_hidden2, 2)

            with tf.variable_scope("EveConv"):
                self.eve_conv1 = tf.nn.relu(conv1D(self.eve_hidden2, [4, 1, 2], stride=1, name='eveConv1'))
                self.eve_conv2 = tf.nn.relu(conv1D(self.eve_conv1, [2, 2, 4], stride=2, name='eveConv2'))
                self.eve_conv3 = tf.nn.relu(conv1D(self.eve_conv2, [1, 4, 4], stride=1, name='eveConv3'))
                self.eve_conv4 = tf.nn.tanh(conv1D(self.eve_conv3, [1, 4, 1], stride=1, name='eveConv4'), name='tanh')

            self.eve_output = tf.squeeze(self.eve_conv4)



    def train(self):

        #Eve's decryption error, how far off her guess is from plaintext, (L1)
        self.err_eve = tf.reduce_mean(tf.abs(self.msg - self.eve_output))

        #logging
        if (self.logPath != "none"):
            tf.summary.scalar("EveLoss", self.err_eve)

        #Bob's decryption error, (L1)
        self.err_bob = tf.reduce_mean(tf.abs(self.msg - self.bob_output))

        # logging
        if (self.logPath != "none"):
            tf.summary.scalar("BobLoss", self.err_bob)

        #Alice and Bob loss function
        self.loss_bob = self.err_bob + ((N/2 - self.err_eve) ** 2.)/(N/2)
        
        self.t_vars = tf.trainable_variables()
        self.alice_or_bob_vars = [var for var in self.t_vars if 'alice_' in var.name or 'bob_' in var.name]
        self.eve_vars = [var for var in self.t_vars if 'eve_' in var.name]

        self.bob_optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss_bob, var_list=self.alice_or_bob_vars)
        self.eve_optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.err_eve, var_list=self.eve_vars)

        self.bob_errors, self.eve_errors = [], []

        #initialize variables for training
        tf.global_variables_initializer().run()

        #train
        for i in range(self.epochs):

            print ('Training Alice and Bob, Epoch:', i + 1)
            bob_loss, _ = self.trainHelper('bob')
            self.bob_errors.append(bob_loss)
            print('Alice and Bob Error: ', bob_loss)


            print ('Training Eve, Epoch:', i + 1)
            _, eve_loss = self.trainHelper('eve')
            self.eve_errors.append(eve_loss)
            print('Eve error: ', eve_loss)

        self.plotDecryptionErrors()


    def trainHelper(self, network):
        """
        Helper function for training Alice and Bob or Eve
        :parameter
        network: "eve" or "bob"
        iterations: 1 for Alice and Bob
                    2 for Eve, to give her a computational edge
        :post
        the appropriate network is trained for 1 or 2 cycles
        """

        bob_error, eve_error = 1., 1.

        bs = self.batchSize

        if network == 'eve':
            bs == bs*2

        msg, key = generateData(bs, self.msgLength)

        self.curEpoch=self.curEpoch+1

        if (self.logPath != "none"):
            merge = tf.summary.merge_all() #logging

        if network == "bob":
            if (self.logPath != "none")&( self.curEpoch%101==0): #logging
                summary, _, decrypt_err = self.session.run([merge, self.bob_optimizer, self.err_bob], feed_dict={self.msg: msg, self.key: key})
                writer.add_summary(summary)
            else:
                _, decrypt_err = self.session.run([self.bob_optimizer, self.err_bob], feed_dict={self.msg: msg, self.key: key})

            bob_error = min(bob_error, decrypt_err)

        elif network == "eve":
            _, decrypt_err = self.session.run([self.eve_optimizer, self.err_eve], feed_dict={self.msg: msg, self.key: key})
            eve_error = min(eve_error, decrypt_err)

        return bob_error, eve_error



    def plotDecryptionErrors(self):

        sns.set_style("whitegrid")
        plt.xlabel('Epoch')
        plt.ylabel('Lowest Decryption error achieved')
        plt.title('N=16 BS=4096 LR=0.0008 Integer')
        plt.legend(['bob', 'eve'])

        plt.plot(self.bob_errors)
        plt.plot(self.eve_errors)
        plt.show()


def generateData(bs=BS, N=N):

    """
    Function to generate random messages and keys
    :parameter
    bs: Batch size
    N: length of message
    """

    return (np.random.uniform(0, 1, size=(bs, N))*2-1), (np.random.uniform(0, 1, size=(bs, N))*2-1)


def conv1D(input, filterShape, stride, name="conv1d"):

    """
    Helper function generates and return a convolutional layer of certain
    shape and stride
    :param input_:
    :param filter_shape:
    :param stride:
    :param name:
    """

    with tf.variable_scope(name):
        w = tf.get_variable("w_conv", shape=filterShape, initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv1d(input, w, stride, padding='SAME')

        return conv

if __name__ == '__main__':
    main()



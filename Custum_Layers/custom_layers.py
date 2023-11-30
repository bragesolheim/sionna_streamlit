import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imports import *

class NeuralDemapper(Layer): # Inherits from Keras Layer

    def __init__(self):
        super().__init__()

        # The three dense layers that form the custom trainable neural network-based demapper
        self.dense_1 = Dense(64, 'relu')
        self.dense_2 = Dense(64, 'relu')
        self.dense_3 = Dense(NUM_BITS_PER_SYMBOL, None) # The last layer has no activation and therefore outputs logits, i.e., LLRs

    def call(self, y):

        # y : complex-valued with shape [batch size, block length]
        # y is first mapped to a real-valued tensor with shape
        #  [batch size, block length, 2]
        # where the last dimension consists of the real and imaginary components
        # The dense layers operate on the last dimension, and treat the inner dimensions as batch dimensions, i.e.,
        # all the received symbols are independently processed.
        nn_input = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1)
        z = self.dense_1(nn_input)
        z = self.dense_2(z)
        z = self.dense_3(z) # [batch size, number of symbols per block, number of bits per symbol]
        llr = tf.reshape(z, [tf.shape(y)[0], -1]) # [batch size, number of bits per block]
        return llr
    
class End2EndSystem(Model): # Inherits from Keras Model

    def __init__(self, training):

        super().__init__() # Must call the Keras model initializer

        self.constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=True) # Constellation is trainable
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = NeuralDemapper() # Intantiate the NeuralDemapper custom layer as any other
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) # Loss function

        self.training = training

    @tf.function(jit_compile=True) # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):

        # no channel coding used; we set coderate=1.0
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                coderate=1.0)
        bits = self.binary_source([batch_size, 1200]) # Blocklength set to 1200 bits
        x = self.mapper(bits)
        y = self.awgn_channel([x, no])
        llr = self.demapper(y)  # Call the NeuralDemapper custom layer as any other
        if self.training:
            loss = self.bce(bits, llr)
            return loss
        else:
            return bits, llr
        
EBN0_DB_MIN = 10.0
EBN0_DB_MAX = 20.0
NUM_BITS_PER_SYMBOL = 4
BATCH_SIZE = 1000


###############################
# Baseline
###############################

class Baseline(Model): # Inherits from Keras Model

    def __init__(self):

        super().__init__() # Must call the Keras model initializer

        self.constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sn.utils.BinarySource()
        self.awgn_channel = sn.channel.AWGN()

    @tf.function # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):

        # no channel coding used; we set coderate=1.0
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                coderate=1.0)
        bits = self.binary_source([batch_size, 1200]) # Blocklength set to 1200 bits
        x = self.mapper(bits)
        y = self.awgn_channel([x, no])
        llr = self.demapper([y,no])
        return bits, llr

###############################
# Benchmarking
###############################

def simulate(baseline, model, ebno_dbs, batch_size, 
                 num_target_block_errors_baseline=500, num_target_block_errors_model=500, 
                 max_mc_iter_baseline=100, max_mc_iter_model=100, 
                 soft_estimates_baseline=True, soft_estimates_model=True):

    ber_plots = sn.utils.PlotBER("Neural Demapper")
    ber_plots.simulate(baseline,
                    ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                    batch_size=BATCH_SIZE,
                    num_target_block_errors=100, # simulate until 100 block errors occured
                    legend="Baseline",
                    soft_estimates=True,
                    max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples))
                    );
    ber_plots.simulate(model,
                    ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                    batch_size=BATCH_SIZE,
                    num_target_block_errors=100, # simulate until 100 block errors occured
                    legend="Untrained model",
                    soft_estimates=True,
                    max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                    );

    fig = plt.gcf()
    return fig

baseline = Baseline()
model = End2EndSystem(False)



###############################

if st.button('Simulate'):
    fig = simulate(baseline, model, EBN0_DB_MIN, EBN0_DB_MAX, BATCH_SIZE)
    st.pyplot(fig)


    
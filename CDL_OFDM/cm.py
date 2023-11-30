import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imports import *

   
st.sidebar.header("Simulation Parameters")
EBN0_DB_MIN = st.sidebar.number_input('Eb/N0 Min [dB]', value=-8.0)
EBN0_DB_MAX = st.sidebar.number_input('Eb/N0 Max [dB]', value=3.0)
BATCH_SIZE = st.sidebar.number_input('Batch Size', value=1000, step=100)

# OFDM System Parameters
st.sidebar.title("OFDM System Parameters")
NUM_UT = st.sidebar.number_input("Number of User Terminals", value=1, step=1)
NUM_BS = st.sidebar.number_input("Number of Base Stations", value=1, step=1)
NUM_UT_ANTENNAS = st.sidebar.number_input("Number of Antennas per User Terminal", value=1, step=1)
NUM_BS_ANTENNAS = st.sidebar.number_input("Number of Antennas per Base Station", value=4, step=1)
NUM_STREAMS_PER_TX = NUM_UT_ANTENNAS
RX_TX_ASSOCIATION = np.array([[1]])
STREAM_MANAGEMENT = sn.mimo.StreamManagement(RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX)
CARRIER_FREQUENCY = 2.6e9 # Carrier frequency in Hz.
                          # This is needed here to define the antenna element spacing.

UT_ARRAY = sn.channel.tr38901.Antenna(  polarization="single",
                                        polarization_type="V",
                                        antenna_pattern="38.901",
                                        carrier_frequency=CARRIER_FREQUENCY)
UT_ARRAY.show();

BS_ARRAY = sn.channel.tr38901.AntennaArray( num_rows=1,
                                            num_cols=int(NUM_BS_ANTENNAS/2),
                                            polarization="dual",
                                            polarization_type="cross",
                                            antenna_pattern="38.901", # Try 'omni'
                                            carrier_frequency=CARRIER_FREQUENCY)

NUM_BITS_PER_SYMBOL = 2
CODERATE = 0.5
DELAY_SPREAD = 100e-9 # Nominal delay spread in [s]. Please see the CDL documentation
                      # about how to choose this value.

DIRECTION = "uplink"  # The `direction` determines if the UT or BS is transmitting.
                      # In the `uplink`, the UT is transmitting.

CDL_MODEL = "C"       # Suitable values are ["A", "B", "C", "D", "E"]

SPEED = 10.0          # UT speed [m/s]. BSs are always assumed to be fixed.
                     # The direction of travel will chosen randomly within the x-y plane.

# Configure a channel impulse reponse (CIR) generator for the CDL model.
CDL = sn.channel.tr38901.CDL(CDL_MODEL,
                             DELAY_SPREAD,
                             CARRIER_FREQUENCY,
                             UT_ARRAY,
                             BS_ARRAY,
                             DIRECTION,
                             min_speed=SPEED)

class OFDMSystem(Model): # Inherits from Keras Model

    def __init__(self, perfect_csi):
        super().__init__() # Must call the Keras model initializer

        self.perfect_csi = perfect_csi

        n = int(RESOURCE_GRID.num_data_symbols*NUM_BITS_PER_SYMBOL) # Number of coded bits
        k = int(n*CODERATE) # Number of information bits
        self.k = k

        # The binary source will create batches of information bits
        self.binary_source = sn.utils.BinarySource()

        # The encoder maps information bits to coded bits
        self.encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)

        # The mapper maps blocks of information bits to constellation symbols
        self.mapper = sn.mapping.Mapper("qam", NUM_BITS_PER_SYMBOL)

        # The resource grid mapper maps symbols onto an OFDM resource grid
        self.rg_mapper = sn.ofdm.ResourceGridMapper(RESOURCE_GRID)

        # Frequency domain channel
        self.channel = sn.channel.OFDMChannel(CDL, RESOURCE_GRID, add_awgn=True, normalize_channel=True, return_channel=True)

        # The LS channel estimator will provide channel estimates and error variances
        self.ls_est = sn.ofdm.LSChannelEstimator(RESOURCE_GRID, interpolation_type="nn")

        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.lmmse_equ = sn.ofdm.LMMSEEqualizer(RESOURCE_GRID, STREAM_MANAGEMENT)

        # The demapper produces LLR for all coded bits
        self.demapper = sn.mapping.Demapper("app", "qam", NUM_BITS_PER_SYMBOL)

        # The decoder provides hard-decisions on the information bits
        self.decoder = sn.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)

    @tf.function # Graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        no = sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=NUM_BITS_PER_SYMBOL, coderate=CODERATE, resource_grid=RESOURCE_GRID)

        # Transmitter
        bits = self.binary_source([batch_size, NUM_UT, RESOURCE_GRID.num_streams_per_tx, self.k])
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        x_rg = self.rg_mapper(x)

        # Channel
        y, h_freq = self.channel([x_rg, no])

        # Receiver
        if self.perfect_csi:
            h_hat, err_var = h_freq, 0.
        else:
            h_hat, err_var = self.ls_est ([y, no])
        x_hat, no_eff = self.lmmse_equ([y, h_hat, err_var, no])
        llr = self.demapper([x_hat, no_eff])
        bits_hat = self.decoder(llr)

        return bits, bits_hat
 

# OFDM Resource Grid parameters
num_ofdm_symbols = int(st.sidebar.number_input('Number of OFDM Symbols', min_value=1, max_value=20, value=14))
fft_size = int(st.sidebar.number_input('FFT Size', min_value=1, value=76))
subcarrier_spacing = st.sidebar.number_input('Subcarrier Spacing (Hz)', value=30000)
cyclic_prefix_length = int(st.sidebar.number_input('Cyclic Prefix Length', min_value=1, value=6))
#pilot_ofdm_symbol_indices = st.sidebar.text_input('Pilot OFDM Symbol Indices', value='2, 11')

RESOURCE_GRID = sn.ofdm.ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                                    fft_size=fft_size,
                                    subcarrier_spacing=subcarrier_spacing,
                                    num_tx=NUM_UT,
                                    num_streams_per_tx=NUM_STREAMS_PER_TX,
                                    cyclic_prefix_length=cyclic_prefix_length,
                                    pilot_pattern="kronecker",
                                    pilot_ofdm_symbol_indices=[2,11])
model_ls = OFDMSystem(False)
model_pcsi = OFDMSystem(True)

ber_plots = sn.utils.PlotBER("OFDM over 3GPP CDL")

def simulate_model_ls(model_ls, ebno_dbs, batch_size, num_target_block_errors=500, max_mc_iter=100, soft_estimates=True):
    ber_plots.simulate(model_ls,
                    ebno_dbs=ebno_dbs,
                    batch_size=batch_size,
                    num_target_block_errors=num_target_block_errors, # simulate until 500 block errors occured
                    legend="LS Channel Estimation",
                    soft_estimates=soft_estimates,
                    max_mc_iter=max_mc_iter, # run 100 Monte-Carlo simulations (each with batch_size samples)
                    );
    fig = plt.gcf()
    return fig


def simulate_model_pcsi(model_pcsi, ebno_dbs, batch_size, num_target_block_errors=500, max_mc_iter=100, soft_estimates=True):
    ber_plots.simulate(model_pcsi,
                    ebno_dbs=ebno_dbs,
                    batch_size=batch_size,
                    num_target_block_errors=num_target_block_errors, # simulate until 500 block errors occured
                    legend="Perfect CSI",
                    soft_estimates=soft_estimates,
                    max_mc_iter=max_mc_iter, # run 100 Monte-Carlo simulations (each with batch_size samples)
                    );
    fig = plt.gcf()
    return fig

if st.button('Generate OFDM Resource Grid'):
    RESOURCE_GRID.show()
    fig_resource_grid = plt.gcf()
    st.pyplot(fig_resource_grid)
    plt.clf()  # Clear the plot

    # Now show the pilot pattern
    RESOURCE_GRID.pilot_pattern.show()  # Assuming this is the correct method
    fig_pilot_pattern = plt.gcf()
    st.pyplot(fig_pilot_pattern)


if st.button("Simulate model with LS Channel Estimation"):
    fig = simulate_model_ls(model_ls, np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20), BATCH_SIZE)
    st.pyplot(fig)

if st.button("Simulate model with Perfect CSI"):
    fig = simulate_model_pcsi(model_ls, np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20), BATCH_SIZE)
    st.pyplot(fig)
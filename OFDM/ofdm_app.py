import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imports import *

st.sidebar.header("Simulation Parameters")
# Mapping of bits per symbol to modulation names
modulation_mapping = {
    2: "QPSK (2bits/symbol)",
    4: "16-QAM (4bits/symbol)",
    6: "64-QAM (6bits/symbol)",
    8: "256-QAM (8bits/symbol)"
}

# Displaying the modulation name in the select box
NUM_BITS_PER_SYMBOL = st.sidebar.selectbox(
        'Modulation Scheme',
        options=list(modulation_mapping.keys()),
        format_func=lambda x: modulation_mapping[x] 
    )

CODERATE = st.sidebar.slider(
    'Code Rate (k/n)',
    min_value=0.1,
    max_value=1.0,
    value=0.5,  # Default value of the slider
    step=0.01,
    format="%.2f"  # Format to display the float with two decimal places
)
EBN0_DB_MIN = st.sidebar.number_input('Eb/N0 Min [dB]', value=-8.0)
EBN0_DB_MAX = st.sidebar.number_input('Eb/N0 Max [dB]', value=3.0)
BATCH_SIZE = st.sidebar.number_input('Batch Size', value=128, step=100)
NUM_UT = st.sidebar.number_input("Number of User Terminals", value=1, step=1)
NUM_UT_ANTENNAS = st.sidebar.number_input("Number of Antennas per User Terminal", value=1, step=1)
NUM_BS_ANTENNAS = st.sidebar.number_input("Number of Antennas per Base Station", value=4, step=1)

NUM_STREAMS_PER_TX = NUM_UT_ANTENNAS

RX_TX_ASSOCIATION = np.array([[1]])

STREAM_MANAGEMENT = sn.mimo.StreamManagement(RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX)

RESOURCE_GRID = sn.ofdm.ResourceGrid( num_ofdm_symbols=14,
                                      fft_size=76,
                                      subcarrier_spacing=30e3,
                                      num_tx=NUM_UT,
                                      num_streams_per_tx=NUM_STREAMS_PER_TX,
                                      cyclic_prefix_length=6,
                                      pilot_pattern="kronecker",
                                      pilot_ofdm_symbol_indices=[2,11])


CARRIER_FREQUENCY = 2.6e9 # Carrier frequency in Hz.
                          # This is needed here to define the antenna element spacing.

UT_ARRAY = sn.channel.tr38901.Antenna(  polarization="single",
                                        polarization_type="V",
                                        antenna_pattern="38.901",
                                        carrier_frequency=CARRIER_FREQUENCY)


BS_ARRAY = sn.channel.tr38901.AntennaArray( num_rows=1,
                                            num_cols=int(NUM_BS_ANTENNAS/2),
                                            polarization="dual",
                                            polarization_type="cross",
                                            antenna_pattern="38.901", # Try 'omni'
                                            carrier_frequency=CARRIER_FREQUENCY)


DELAY_SPREAD = 100e-9 # Nominal delay spread in [s]. Please see the CDL documentation
                      # about how to choose this value.

DIRECTION = "uplink"  # The `direction` determines if the UT or BS is transmitting.
                      # In the `uplink`, the UT is transmitting.

CDL_MODEL = "A"       # Suitable values are ["A", "B", "C", "D", "E"] 

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

# https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part4.html#Training-the-Neural-Receiver
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

    # https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part4.html#Training-the-Neural-Receiver
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

def run_simulation(ebno_db_min, ebno_db_max, batch_size):
    ber_plots = sn.utils.PlotBER("OFDM over 3GPP CDL")

    model_ls = OFDMSystem(False)
    ber_plots.simulate(model_ls,
                       ebno_dbs=np.linspace(ebno_db_min, ebno_db_max, 20),
                       batch_size=batch_size,
                       num_target_block_errors=100,
                       legend="LS Estimation",
                       soft_estimates=True,
                       max_mc_iter=100)

    model_pcsi = OFDMSystem(True)
    ber_plots.simulate(model_pcsi,
                       ebno_dbs=np.linspace(ebno_db_min, ebno_db_max, 20),
                       batch_size=batch_size,
                       num_target_block_errors=100,
                       legend="Perfect CSI",
                       soft_estimates=True,
                       max_mc_iter=100)

    fig = plt.gcf()
    return fig


model_ls = OFDMSystem(False)
model_pcsi = OFDMSystem(True)

# Button to run simulation
if st.button("Run Simulation"):
    fig = run_simulation(EBN0_DB_MIN, EBN0_DB_MAX, BATCH_SIZE)
    st.pyplot(fig)


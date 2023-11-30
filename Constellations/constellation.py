import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imports import *


# Mapping of bits per symbol to modulation names
modulation_mapping = {
    2: "QPSK (2bits/symbol)",
    4: "16-QAM (4bits/symbol)",
    6: "64-QAM (6bits/symbol)",
    8: "256-QAM (8bits/symbol)"
    # Add more mappings as needed
}

# Sidebar configuration
# with st.sidebar:
#     st.title("Simulation Configuration")
#     num_bits_per_symbol = st.selectbox(
#         'Modulation Scheme',
#         options=list(modulation_mapping.keys()),  # The keys are the number of bits
#         format_func=lambda x: modulation_mapping[x]  # Display the modulation name in the select box
#     )



# Display the plot in Streamlit
st.title("Create a QAM Constellation")
st.sidebar.title("Simulation Configuration")
#NUM_BITS_PER_SYMBOL = st.number_input("Number of bits per symbol", min_value=1, max_value=8, value=2, step=2)   # Number of bits per symbol
NUM_BITS_PER_SYMBOL = st.sidebar.selectbox(
        'Modulation Scheme',
        options=list(modulation_mapping.keys()),  # The keys are the number of bits
        format_func=lambda x: modulation_mapping[x]  # Display the modulation name in the select box
    )

constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)


# Create a BytesIO buffer to capture the plot
buf = io.BytesIO()
constellation.show()  # This will generate the plot
plt.savefig(buf, format='png')  # Save the plot to the buffer
buf.seek(0)  # Rewind the buffer to the beginning

st.image(buf, caption='Constellation Diagram')


# Initialize the Mapper and Demapper
mapper = sn.mapping.Mapper(constellation=constellation)
demapper = sn.mapping.Demapper("app", constellation=constellation)

st.markdown("### Mapper and Demapper")
input_data = st.text_input("Enter data to be mapped", "1010")

# Convert the input data to a Tensorflow tensor and reshape it to the desired shape
binary_data = tf.constant([int(b) for b in input_data], dtype=tf.int32)
binary_data = tf.reshape(binary_data, [1, -1])

# Check if the input length is appropriate
if binary_data.shape[1] % constellation.num_bits_per_symbol == 0:
    mapped_output = mapper(binary_data)

    # Convert the output tensor to a numpy array for display
    mapped_output_array = mapped_output.numpy()

    st.write("Mapped Output:", mapped_output_array)
else:
    st.write("Input length is not a multiple of the number of bits per symbol")
    st.stop()

# Initialize Sionna components
binary_source = sn.utils.BinarySource()
awgn_channel = sn.channel.AWGN()
mapper = sn.mapping.Mapper(constellation=constellation)
demapper = sn.mapping.Demapper("app", constellation=constellation)

min_exp = 3
max_exp = 10 

# Sidebar for user inputs
ebno_db = st.sidebar.slider("`Eb/N0` (dB)", min_value=0.0, max_value=30.0, value=10.0, step=0.1, help="The `Eb/No` value (=rate-adjusted SNR) in dB.")
batch_size = st.sidebar.slider("Batch Size", min_value=32, max_value=1024, value=64, step=64, help="How many examples are processed by Sionna in parallel")
block_length_options = [2**i for i in range(min_exp, max_exp+1)]
block_length = st.sidebar.selectbox(
        'Block Length',
        options=block_length_options,
        help="The number of bits per transmitted message block"
    )




#block_length = 2**exponent

# Convert Eb/N0 to noise variance
no = sn.utils.ebnodb2no(ebno_db=ebno_db,
                        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                        coderate=1.0)  # Coderate set to 1 for uncoded transmission


# Generate bits
bits = binary_source([batch_size, block_length])

# Map bits
x = mapper(bits)

# Add AWGN
y = awgn_channel([x, no])

# Demap
llr = demapper([y, no])

with st.expander("Shape of tensors"):
    st.write(f"Shape of bits: {bits.shape}")
    st.write(f"Shape of x: {x.shape}")
    st.write(f"Shape of y: {y.shape}")
    st.write(f"Shape of llr: {llr.shape}")

# Select how many samples to print
num_samples = 8  # This can be adjusted or made into a user input
num_symbols = int(num_samples / NUM_BITS_PER_SYMBOL)


st.markdown(f"##### First {num_samples} transmitted bits", help="The bits are the input to the mapper. The mapper converts the bits to symbols, which are then transmitted over the channel."   )
bits_to_display = bits[0,:num_samples].numpy().tolist() # Converting the tensor to a numpy array and then to a list for display
bits_df = pd.DataFrame([bits_to_display], columns=[f"Bit {i+1}" for i in range(num_samples)]) 
st.dataframe(bits_df, hide_index=True)


# Coluumns
col1, col2 = st.columns(2)

with col1:
    x.df = pd.DataFrame([np.round(x[0,:num_symbols], 2)], columns=[f"Symbol {i+1}" for i in range(num_symbols)])
    st.write(f"First {num_symbols} transmitted symbols:", np.round(x[0,:num_symbols], 2))

with col2:
    st.write(f"First {num_symbols} received symbols:", np.round(y[0,:num_symbols], 2))


st.markdown(f"##### First {num_samples} demapped LLRs", help="The LLRs are the log-likelihood ratios, which are the log of the ratio of the probability of the bit being 1 to the probability of the bit being 0. The LLRs are used by the decoder to make decisions about the transmitted bits. High LLR values: bit is likely to be 1. Low LLR values: bit is likely to be 0.")

llr_df = pd.DataFrame([np.round(llr[0,:num_samples], 2)], columns=[f"LLR {i+1}" for i in range(num_samples)])
cco = st.dataframe(llr_df, hide_index=True)

plt.figure(figsize=(8,8))
plt.axes().set_aspect(1)
plt.grid(True)
plt.title('Channel output')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')

real_parts = tf.math.real(y).numpy() if hasattr(y, 'numpy') else y.eval()
imag_parts = tf.math.imag(y).numpy() if hasattr(y, 'numpy') else y.eval()

plt.scatter(real_parts, imag_parts, color='blue')

# Use tight_layout to ensure the full plot is visible
plt.tight_layout()

# Streamlit function to display the plot
st.pyplot(plt)

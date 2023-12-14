# Streamlit based Sionna simulation web app
This project consits of two Streamlit based web apps, both utilizing the Sionna simulation library. The first app is an interactive QAM constellation simulator, and the second is an OFDM uplink transmission simulator in the Frequency Domain. 

## Features: Interactive QAM Constellation Simulator
- Modulation Sceheme Configurations: Select different modulation schemes to observe how they affect signal transmission.
- Eb/No Configurations: Dynamically change the 'Energy per Bit to Noise Power Spectral Density Ratio' to simulate different Signal to Noise scenarios.
- Adjustable parameters like batch size, block length and number of samples to display. 
- Real-time Constellation Diagram: Visualize the modulation constellation in real-time, reflecting the adjustments in simulation parameters.
- Data mapping and demapping: Observe how the data is mapped to the constellation points, and how it is demapped back to the original data.
- AWGN channel: Explore the effects of the AWGN channel on the transmitted signal.
- Tensor Shape Visualization: Inspect the effects of shape of the tensors involved in the simulation.

## Features: OFDM Uplink Transmission Simulator
- Modulation Sceheme Configurations: Select different modulation schemes to observe how they affect signal transmission.
- Code Rate Configurations: Select different code rates to observe how they affect signal transmission.
- Eb/No Configurations: Dynamically change the 'Energy per Bit to Noise Power Spectral Density Ratio' to simulate different Signal to Noise scenarios.
- Batch Size Configurations: Dynamically change the batch size to simulate different batch sizes.
- User Terminal Configurations: Dynamically change the number of user terminals to simulate different number of user terminals.
- Antenna Configurations: Dynamically change the number of antennas for UT and BS to simulate different scenarios.


# Installation
## Clone the repository
```sh
git clone https://github.com/bragesolheim/sionna_streamlit.git
```
## Navigate to the project directory
```sh
cd sionna_streamlit_app
```
## Install dependencies
```sh
pip install -r requirements.txt
```
## Run the Constellation Simulator
```sh
streamlit run Constellations/constellation_app.py
```

## Run the OFDM Uplink Simulator
```sh
streamlit run OFDM/ofdm_app.py
```

## Additional Setup for M1/M2 Mac Users

The Sionna 6G Physical-Layer Simulation interface requires additional setup steps for users running on M1/M2 Macs due to the ARM architecture. Follow these instructions to ensure compatibility:

1. **Install Homebrew**: If not already installed, you can install Homebrew, by running the following command in your terminal:
```sh
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

```
2. **Install LLVM**: Install LLVM by running the following command in your terminal:
```sh
   brew install llvm@16
```
3. **Set the LLVM path**: Set the LLVM path by running the following command in your terminal:
```sh
   export DRJIT_LIBLLVM_PATH="/opt/homebrew/opt/llvm@16/lib/libLLVM.dylib"
```

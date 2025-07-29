# QDNN-CFF

DNN models used for the extraction of CFFs in "Compton Form Factor Extraction using Quantum Deep Neural Networks"

CDNN: The CDNN architecture consists of multiple linear layers with ReLU activation functions, designed to process the kinematic input variables and output the predicted CFFs. It utilizes a custom mean-squared error loss function that evaluates the discrepancy between its predictions and true cross-section data. The training process employs an Adam optimizer to minimize this loss, with standard backpropagation and gradient clipping techniques.

BQDNN: The Basic QDNN is modeled on the CDNN and employs a PennyLane quantum circuit with AngleEmbedding and StronglyEntanglingLayers for quantum data processing, integrated within a PyTorch neural network that handles preprocessing and postprocessing of data. The model is trained the a custom mean squared error loss function, and the training loop includes an RMSprop optimizer and gradient clipping for stability.

FQDNN: The Full QDNN incorporates quantum-specific advantages, dynamically adjusting the depth of its quantum circuit during training. The quantum component, built with PennyLane, starts with an initial shallow depth and progressively increases its complexity every few epochs, allowing the model to adapt and potentially learn more intricate patterns as training progresses. Key features also include an input normalization layer to scale data for optimal quantum embedding and a specific weight initialization strategy. The model is trained with an RMSprop optimizer, and its performance is evaluated using the same custom physical model-based loss function as the other networks.

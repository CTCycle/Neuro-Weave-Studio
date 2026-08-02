# Export Pipeline

Last updated: 2026-07-30

Project JSON export is always available for a valid schema. Source export requires graph and framework compatibility validation. Keras and TensorFlow-through-Keras use the Keras functional API adapter. PyTorch generates an `nn.Module`. Binary `.keras` export constructs an initialized, untrained Keras model through a restricted direct adapter. `.h5`, weights, trained artifacts, ONNX, and cross-framework conversion are not claimed by the MVP.

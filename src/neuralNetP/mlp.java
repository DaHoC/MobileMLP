/**
 *
 * @author Jan Hendriks
 */

package neuralNetP;

import java.lang.Math.*;
import java.util.Vector;

class mlp {

    // Using nested inner classes to avoid some inheritance issues
    public class layer {

        // Using nested inner classes to avoid some inheritance issues
        public class neuron {

            private int neuronId; // unique id, e.g. for convenience or PRNG order
            private float weightedSum; // Store the calculated net for each neuron here
            private float output; // Contains the output of the neuron object
            private float delta; // The delta of the delta rule, not the weight change, assigned to every neuron

            // Constructor for neuron object, it only assigns an identifier to each neuron
            private neuron (int id) {
//                System.out.println( (id == 0) ? " > Creating BIAS neuron in current layer" : " > Creating neuron " + id);
                this.neuronId = id; // Set the current last neuron number in MLP
            }

            // This is the call to the calculations in the neuron object, to make sure the calculations are only issued one per neuron (mlp) input
            private void calculateWeightedSumAndOutput() {
                this.calculateWeightedSum();
                this.calculateOutput();
            }

            // The calculates the weighted sum for the current neuron object
            private void calculateWeightedSum() {
                if (this.neuronId == 0) {
                    this.weightedSum = 1; // If BIAS neuron
                } else {
                    if (layerId == 0) { // If input layer, the weighted sum and output is merely the input itself, since the input layer acts only as buffer
                        this.weightedSum = inputValues[this.neuronId-1];
                    } else {
                        // Prepare to walk through all neurons in the above layer
                        neuron[] aboveLayerNeurons = ((layer)layers.elementAt(layerId-1)).layerNeurons;
                        // Initialize the weighted sum with the bias weight from the bias to the current neuron
                        float weightedSumAccumulator = biasWeights.getWeightFromLayerNeuronToNeuronId(0, this.neuronId); // negative threshold, init net with Bias weight
//                        System.out.format(" 1      * W(  0>%3d) = %+3.5f * %+3.5f%n", this.neuronId, (double)1, weightedSumAccumulator);
                        for (int i = 0; i < aboveLayerNeurons.length; i++) {
                            // This is the weight from one above neuron to the current neuron object
                            float weightToCurrentNeuronFromAboveNeuron = ((weightsFromLayer)weights.elementAt(layerId-1)).getWeightFromLayerNeuronToNeuronId(i, this.neuronId);
                            // This is the output of one neuron in the above layer of the current layer
                            float outputOfNeuronInAboveLayer = aboveLayerNeurons[i].getOutput();
//                            System.out.format(" O(%3d) * W(%3d>%3d) = %+3.5f * %+3.5f%n", aboveLayerNeurons[i].neuronId , aboveLayerNeurons[i].neuronId, this.neuronId, weightToCurrentNeuronFromAboveNeuron, outputOfNeuronInAboveLayer);
                            // This calculates one summand (w_ij * o_i) for the weighted sum and adds it to the previous summands
                            weightedSumAccumulator += weightToCurrentNeuronFromAboveNeuron * outputOfNeuronInAboveLayer;
                        }
                        // After walking through all neurons in the above layer (and the bias) and adding up the summands, we have the weighted sum
                        this.weightedSum = weightedSumAccumulator;
                    }
//                    System.out.format("Calculated net for neuron %6d: %+3.5f%n", this.neuronId, this.weightedSum);
                }
            }

            // This calculates what the current neuron object outputs
            private void calculateOutput() {
                if (layerId == 0) {
                    this.output = this.weightedSum; // If input layer, the weighted sum and output is merely the input itself (same for BIAS)
                } else {
                    // Otherwise calculate the value of the specified transferfunction at the weighted sum, e.g. output = transferfunction(weightedsum)
                    switch(transferFunction) {
                        case 0:
                            this.output = (float)neuralNetP.mlpCanvas.tanh(this.weightedSum);
                            break;
                        case 1:
                            this.output = 1 / (1 + (float)(neuralNetP.mlpCanvas.pow(Math.E,-(this.weightedSum))));
                            break;
                        default:
                        case 2:
                            this.output = this.weightedSum;
                    }
                }
//                System.out.format("Calculated output for neuron %3d: %+3.5f%n", this.neuronId, this.output);
            }

            // Returns the value of the derivations of the given layer transfer functions at z
            private float getDerivationAt(float z) {
                switch(transferFunction) {
                    case 0:
                        float tanhTempResult = (float)neuralNetP.mlpCanvas.tanh(z);
                        return 1 - (float)neuralNetP.mlpCanvas.pow(tanhTempResult,2); // 1-tanh^2(z)
                    case 1:
                        float fermiTempResult = 1 / (1 + (float)neuralNetP.mlpCanvas.pow(Math.E, -z) );
                        return fermiTempResult*(1-fermiTempResult); // (f(1-f))(z)
                    default:
                    case 2:
                        return 1; // id(x) = x => id'(x) = 1
                }
            }

            // Calculates the delta of the delta rule, not the weight change
            private void calculateDelta() {
                // Test for output layer
                if (layerId == (layerCount-1)) {
                    // Exclude the bias neuron in the output layer
                    if (this.neuronId != 0) {
                        // Calculate delta for the output neurons (teacher y(m) - output y(m)) * f'(net)
                        this.delta = (teacherOutput[(this.neuronId-getFirstNeuronId())] - this.getOutput()) * this.getDerivationAt(this.getWeightedSum());
                    }
                } else if (layerId != 0) { // failsafe to exclude the input layer
                    // Prepare to walk through the neurons in the below layer
                    layer belowLayer = (layer)layers.elementAt(layerId+1);
                    float deltaAccumulator = 0;
                    int neuronNumberInLayer = this.neuronId - ((layer)layers.elementAt(layerId)).firstNeuronIdInLayer;
                    // Walk through neurons in the below layer
                    for (int i = 0; i < belowLayer.getLayerSize(); i++) {
                        // Sums up the (weights*deltas) from this neuron to the neurons in the below layers
                        deltaAccumulator += ((weightsFromLayer)weights.elementAt(layerId)).getWeightFromLayerNeuronToNeuronId(neuronNumberInLayer, belowLayer.layerNeurons[i].neuronId) * belowLayer.layerNeurons[i].getDelta();
//                        System.out.format(" delta += W(%3d>%3d) * delta (%3d) = %+3.5f * %+3.5f%n",this.neuronId, belowLayer.layerNeurons[i].neuronId, belowLayer.layerNeurons[i].neuronId, getWeightFromTo(this, belowLayer.layerNeurons[i]), belowLayer.layerNeurons[i].getDelta());
                    }
                    this.delta = deltaAccumulator * this.getDerivationAt(this.getWeightedSum());
                }

//                System.out.format(" Delta %+3.5f at layer %d%n", this.delta, this.getLayerId());
//                System.out.format(" f'(%+3.5f) = %+3.5f%n", this.getWeightedSum(), this.getDerivationAt(this.getWeightedSum()));
//                System.out.format("Delta of neuron %3d in layer %3d: %+3.5f%n", this.neuronId, layerId, this.delta);
            }

            // Returns the id of the layer in which the neuron resides
            public short getLayerId() {
                return layerId;
            }

            // Returns the weighted sum (=net) for the neuron
            public float getWeightedSum() {
                return this.weightedSum;
            }

            // Returns the output for the neuron
            public float getOutput() {
                return this.output;
            }

            // Returns the delta for the neuron
            public float getDelta() {
                return this.delta;
            }
        } // end of nested neuron class

        private neuron[] layerNeurons; // The neurons per layer excluding the bias neuron
        private short layerId; // layerId == 0 means input (=first) layer
        private short transferFunction; // The transfer function per layer, 0 = tanh, 1 = sigmoid/fermi, 2 = identity
        private int firstNeuronIdInLayer; // The id of the first neuron in the layer (excluding the bias neuron with id always=0 in every layer)
        private double learningRate; // The learning rate per layer, also denoted eta

        // Constructor for first layer (no above layer exists) -> not allowed to be called directly, use mlp.addLayer() instead!
        private layer(int layerSize, short transf, double eta, short layerId) {
            System.out.println( ((layerId == 0) ? " > Creating input layer of size " + layerSize : " > Creating layer " + layerId + " of size " + layerSize)+ " with " + transf + " transfer function");
            this.layerId = layerId;
            this.transferFunction = transf;
            this.learningRate = eta;
            this.layerNeurons = new neuron[layerSize];
            this.firstNeuronIdInLayer = lastNeuronNumber;
            for (int i = 0; i < layerSize; i++) {
                this.layerNeurons[i] = new neuron(lastNeuronNumber++);
            }
        }

        // Returns the id of the first neuron in the layer (excluding bias neuron with id 0 in every layer)
        private int getFirstNeuronId() {
            return this.firstNeuronIdInLayer;
        }

        // Returns the size excluding the bias neuron
        public int getLayerSize() {
            return this.layerNeurons.length;
        }

        // Collects and returns the output of all neurons in the layer
        private float[] getLayerOutput() {
            System.out.println("Returning output for layer " + this.layerId);
            float[] outputCollector = new float[this.getLayerSize()];
            for (int j = 0; j < this.getLayerSize(); j++) {
                outputCollector[j] = this.layerNeurons[j].getOutput();
                System.out.println("Returning output for layer neuron " + this.layerNeurons[j].neuronId + " (" + this.layerNeurons[j].getOutput() + ")");
            }
            return outputCollector;
        }

    } // end of nested layer class
/*
    // This represents the available choices of the transfer function per layer
    public enum Transferfunction {
        id,
        fermi,
        tanh
    }
*/
    private float[] inputValues; // x_1 .. x_N
    private float[] teacherOutput; // The teacher output teacher y_1 .. teacher y_M

    private int lastNeuronNumber = 1; // neurons of all layers have unique consecutive numbers (for id and PRNG)
    private weightsFromLayer biasWeights; // Extra weights-layer object or the BIAS weights
    private weightsFromLayer biasWeightDifferences; // Extra weights-layer object or the BIAS weight differences
    private Vector weights; // The weight entries per layer
    private Vector weightDifferences; // For calculated weight changes
    private short layerCount = 0;
    private Vector layers;

    // Init with an input pattern and amount of input and output neurons
    public mlp () {
        System.out.println(" > Initializing MLP");
        this.weights = new Vector();
        this.weightDifferences = new Vector();
        this.layers = new Vector();
    }

    /**
     * Initializes the weights randomly in a given range, e.g. -0.5,0.5
     * You can check the calculated weights with showMlpNeuronWeights()
     * @TODO: add function for double precision (easy)
     */
    public void initWeightsWithRandomValues(java.util.Random prng, float min, float max) {
        float range = max-min; // First get the range
        // First init BIAS to all neurons except the ones in the input layer
        int idOfFirstNeuronInHiddenLayer = ((layer)this.layers.elementAt(1)).firstNeuronIdInLayer;
        for (int n = idOfFirstNeuronInHiddenLayer; n < this.lastNeuronNumber; n++) {
            this.biasWeights.setWeightFromLayerNeuronToNeuronId(0, n, (prng.nextFloat()*range)+min); //  (prng.nextDouble()*range)+min nextDouble() returns numbers in [0..1), to scale it correctly do the following
        }
        // Next init all layer to layer+1 weights
        for (int l = 0; l < this.layerCount-1; l++) { // For each layer
            for (int neuronCount = 0; neuronCount < ((layer)this.layers.elementAt(l)).getLayerSize(); neuronCount++) { // For each neuron in the current layer
                // alternative: neuronCount < this.weights.elementAt(l).getNeuronCountInLayer()
                int maxNeuronIdInLayer = ((layer)this.layers.elementAt(l+1)).getLayerSize() + ((layer)this.layers.elementAt(l+1)).firstNeuronIdInLayer;
                for (int toNeuron = ((layer)this.layers.elementAt(l+1)).firstNeuronIdInLayer; toNeuron < maxNeuronIdInLayer; toNeuron++) { // for each neuron in the below layer
//                    System.out.println(" Layer "+l+", "+neuronCount+"-th neuron -> layer "+l+1+ " neuron id "+toNeuron);
                    ((weightsFromLayer)this.weights.elementAt(l)).setWeightFromLayerNeuronToNeuronId(neuronCount, toNeuron, (prng.nextFloat()*range)+min);
                }
            }
        }
    }

    // Sets input and initializes the MLP calculations (no lazy evaluation b/c all values will be referenced at some time)
    public void propagateInput(float[] inputVector) {
        this.inputValues = inputVector;
        // Init calculation here, propagate input (pass input forward through MLP)
        // Walk through all layers from input to output layer
        for (int currentLayer = 0; currentLayer < this.layerCount; currentLayer++) {
            // Walk through the neurons in the layer (except the bias neuron)
            for (int n = 0; n < ((layer)this.layers.elementAt(currentLayer)).getLayerSize(); n++) {
                // This calls the actual calculation of the weighted sum and the output per neuron in the layer
                ((layer)this.layers.elementAt(currentLayer)).layerNeurons[n].calculateWeightedSumAndOutput();
            }
        }
    }

    // Create one(!) input layer (has to be the first layer to be created)
    public void addLayer(int layerSize) {
        if (this.layerCount > 0) {
            System.err.println(" > Only one input layer allowed!");
        } else {
            // Create an extra weights-layer object or the BIAS weights
            this.biasWeights = new weightsFromLayer(1);
            this.biasWeightDifferences = new weightsFromLayer(1);
            // Init the input layer in a convenient way
            this.addLayer(layerSize,(short)0,0);
        }
    }

    // Create a hidden or output layer
    public void addLayer(int layerSize, short trans, float eta) {
        this.weights.addElement(new weightsFromLayer(layerSize));
        this.weightDifferences.addElement(new weightsFromLayer(layerSize));
        this.layers.addElement(new layer(layerSize,trans,eta, this.layerCount++));
    }

/**
 * @TODO: alternative to BP-algorithm below:
 *  Genetic algorithm (aka. evolutionary algorithm) using random weights for use with e.g. heaviside function (BP useless with this transferfunction because of inconsistent / useless derivation):
 *  evaluate the best weights by calculating (and continue mating) the weights that returned the lowest error
 */

    // This is the backpropagation of delta learning algorithm
    public void backpropagationOfDelta() {
        // Walk through layers backwards, starting at output layer
        for (int b = this.layerCount-1; b > 0; b--) { // There are layerCount-1 layers in the mlp
            // Walk through all neurons in the layer
            for (int n = 0; n < ((layer)this.layers.elementAt(b)).getLayerSize(); n++) {
                // Let each neuron calculate its delta
                ((layer)this.layers.elementAt(b)).layerNeurons[n].calculateDelta();
                // Prepare to walk through the above layer neurons
                layer aboveLayer = (layer)this.layers.elementAt(b-1);

                // First consider the bias neuron in the above layer
                // This is the actual weight change calculation (diff w_0j = eta_j * delta_j * o_0 = eta_j * delta_j)
                this.biasWeightDifferences.setWeightFromLayerNeuronToNeuronId(0, ((layer)this.layers.elementAt(b)).layerNeurons[n].neuronId, (float)((layer)this.layers.elementAt(b)).learningRate * (float)((layer)this.layers.elementAt(b)).layerNeurons[n].getDelta());
//                System.out.format(" > Weight change for weight W(%3d>%3d): eta(%3d) * delta(%3d) * o(%3d) = %+3.5f * %+3.5f * %+3.5f = %+3.5f%n",0, this.layers[b].layerNeurons[n].neuronId, this.layers[b].layerNeurons[n].neuronId, this.layers[b].layerNeurons[n].neuronId, 0, this.layers[b].learningRate, this.layers[b].layerNeurons[n].getDelta(), (double)1, this.differenceWeightMatrix[weightPosition[0]][weightPosition[1]]);

                // Consider all other neurons in the above layer
                for (int aboveLayerNeurons = 0; aboveLayerNeurons < aboveLayer.getLayerSize(); aboveLayerNeurons++) {
                    // This is the actual weight change calculation (diff w_ij = eta_j * delta_j * o_i)
                    ((weightsFromLayer)this.weightDifferences.elementAt(b-1)).setWeightFromLayerNeuronToNeuronId(aboveLayerNeurons, ((layer)this.layers.elementAt(b)).layerNeurons[n].neuronId, (float)((layer)this.layers.elementAt(b)).learningRate * (float)((layer)this.layers.elementAt(b)).layerNeurons[n].getDelta() * (float)aboveLayer.layerNeurons[aboveLayerNeurons].getOutput());
//                    System.out.format(" > Weight change for weight W(%3d>%3d): eta(%3d) * delta(%3d) * o(%3d) = %+3.5f * %+3.5f * %+3.5f = %+3.5f%n",aboveLayer.layerNeurons[aboveLayerNeurons].neuronId, this.layers[b].layerNeurons[n].neuronId, this.layers[b].layerNeurons[n].neuronId, this.layers[b].layerNeurons[n].neuronId, aboveLayer.layerNeurons[aboveLayerNeurons].neuronId, this.layers[b].learningRate, this.layers[b].layerNeurons[n].getDelta(), aboveLayer.layerNeurons[aboveLayerNeurons].getOutput(), this.differenceWeightMatrix[weightPosition[0]][weightPosition[1]]);
                }
            }
        }
    }

    // This is the update step, it adds the calculated weight changes (=difference weights) to the current weights
    public void updateWeights() {
        // First walk through BIAS values
        int idOfFirstNeuronInHiddenLayer = ((layer)this.layers.elementAt(1)).firstNeuronIdInLayer;
        float change = 0f;
        for (int n = idOfFirstNeuronInHiddenLayer; n < this.lastNeuronNumber; n++) {
            // Add the original weight to the weight change
            change = this.biasWeights.getWeightFromLayerNeuronToNeuronId(0, n) + this.biasWeightDifferences.getWeightFromLayerNeuronToNeuronId(0, n);
            // Apply the new weight
            this.biasWeights.setWeightFromLayerNeuronToNeuronId(0, n, change);
        }
        int maxNeuronIdInLayer;
        // Next walk through all layer to layer+1 weights
        for (int l = 0; l < this.layerCount-1; l++) { // For each layer
            for (int neuronCount = 0; neuronCount < ((layer)this.layers.elementAt(l)).getLayerSize(); neuronCount++) { // For each neuron in the current layer
                // alternative: neuronCount < this.weights.elementAt(l).getNeuronCountInLayer()
                idOfFirstNeuronInHiddenLayer = ((layer)this.layers.elementAt(l+1)).firstNeuronIdInLayer;
                maxNeuronIdInLayer = idOfFirstNeuronInHiddenLayer + ((layer)this.layers.elementAt(l+1)).getLayerSize();
                for (int toNeuron = idOfFirstNeuronInHiddenLayer; toNeuron < maxNeuronIdInLayer; toNeuron++) { // for each neuron in the below layer
//                    System.out.println(" Layer "+l+", "+neuronCount+"-th neuron -> layer "+l+1+ " neuron id "+toNeuron);
                    // Add the original weight to the weight change
                    change = ((weightsFromLayer)this.weights.elementAt(l)).getWeightFromLayerNeuronToNeuronId(neuronCount, toNeuron) + ((weightsFromLayer)this.weightDifferences.elementAt(l)).getWeightFromLayerNeuronToNeuronId(neuronCount, toNeuron);
                    // Apply the new weight
                    ((weightsFromLayer)this.weights.elementAt(l)).setWeightFromLayerNeuronToNeuronId(neuronCount, toNeuron, change);
                }
            }
        }
    }

    // Sets the teacher output vector for the mlp
    public void setTeacherOutputVector(float[] teacherOutput) {
        this.teacherOutput = teacherOutput;
    }

    // Calculates and returns the individual error = 1/2 sum (teacher y - net y)^2 for one pattern
    public float getErrorForOnePattern() {
        float error = 0;
        if (this.layerCount > 0) { // Failsafe if e.g. only input layer exists
            float[] mlpOutput = this.getOutput();
            for (int y = 0; y < mlpOutput.length; y++) {
                error +=
                        neuralNetP.mlpCanvas.pow(this.teacherOutput[y] - mlpOutput[y],2); // sum (teacher y-y)^2
            }
        }
        return (error/2); // Error = 1/2 sum (^y-y)^2
    }

    // Returns the output of the mlp in an array of size M
    public float[] getOutput() {
        if (this.layerCount > 0)
            return ((layer)this.layers.elementAt(this.layerCount-1)).getLayerOutput(); // The last (=output) layer has the id layerCount-1
        else
            return new float[0]; // Failsafe if e.g. only input layer exists
    }

    // Prints the mlp's output
    public void printOutput() {
        // Inits with the size of the output layer (M)
        float[] myNeuralNetOutput = this.getOutput();
        System.out.println("Output:");
        for (int j = 0; j < myNeuralNetOutput.length; j++) {
//            System.out.format(" %+3.5f",myNeuralNetOutput[j]);
        }
        System.out.println();
    }

    // Outputs the pattern input
    public void printPatternInput() {
        System.out.println("Input:");
        for (int i = 0; i < this.inputValues.length; i++) {
//            System.out.format(" %+3.5f",this.inputValues[i]);
        }
        System.out.println();
    }
/*
    // Outputs the weights in the mlp with neuron numbers
    public void showMlpNeuronWeights() {
        System.out.println("Weights:");
        // Walk through layers
        for (int layer = 0; layer < this.layerCount-1; layer++) {
            layer currentLayer = layers.elementAt(layer);
            layer belowLayer = layers.elementAt(layer+1);
            // Walk through neurons
            for (int belowNeuron = 0; belowNeuron < belowLayer.getLayerSize(); belowNeuron++) {
                for (int neuron = -1; neuron < currentLayer.getLayerSize(); neuron++) {
                    if (neuron == -1) // This is for the bias neuron
                        System.out.format(" W(%3d>%3d) = %+3.5f%n", 0, belowLayer.layerNeurons[belowNeuron].neuronId, currentLayer.getWeightFromTo(currentLayer.BIAS, belowLayer.layerNeurons[belowNeuron]));
                    else // for all other neurons in the layer
                        System.out.format(" W(%3d>%3d) = %+3.5f%n", currentLayer.layerNeurons[neuron].neuronId, belowLayer.layerNeurons[belowNeuron].neuronId, currentLayer.getWeightFromTo(currentLayer.layerNeurons[neuron], belowLayer.layerNeurons[belowNeuron]));
                }
            }
        }
    }

    // Outputs the weight changes in the mlp with neuron numbers
    public void showMlpNeuronWeightChanges() {
        System.out.println("Weight Changes:");
        int[] weightPosition;
        // Walk through layers
        for (int layer = 0; layer < this.layerCount-1; layer++) {
            layer currentLayer = layers.elementAt(layer);
            layer belowLayer = layers.elementAt(layer+1);
            // Walk through neurons
            for (int belowNeuron = 0; belowNeuron < belowLayer.getLayerSize(); belowNeuron++) {
                for (int neuron = -1; neuron < currentLayer.getLayerSize(); neuron++) {
                    if (neuron == -1) {// This is for the bias neuron
                        weightPosition = currentLayer.getWeightMatrixPositionForNeuronWeightFromTo(currentLayer.BIAS, belowLayer.layerNeurons[belowNeuron]);
                        System.out.format(" Diff W(%3d>%3d) = %+3.5f%n", 0, belowLayer.layerNeurons[belowNeuron].neuronId, this.differenceWeightMatrix[weightPosition[0]][weightPosition[1]]);
                    } else {// for all other neurons in the layer
                        weightPosition = currentLayer.getWeightMatrixPositionForNeuronWeightFromTo(currentLayer.layerNeurons[neuron], belowLayer.layerNeurons[belowNeuron]);
                        System.out.format(" Diff W(%3d>%3d) = %+3.5f%n", currentLayer.layerNeurons[neuron].neuronId, belowLayer.layerNeurons[belowNeuron].neuronId, this.differenceWeightMatrix[weightPosition[0]][weightPosition[1]]);
                    }
                }
            }
        }
    }

    // Outputs the weights in the mlp
    public void printWeightMatrix(int X, int Y) {
        System.out.println("WeightMatrix:");
        for (int i = 0; i < X; i++) {
            for (int j = 0; j < Y; j++) {
                System.out.format(" %+3.5f",this.weightMatrix[i][j]);
            }
            System.out.println();
        }
    }

    // Outputs the weight differences calculated by backprop
    public void printDifferencesWeightMatrix(int X, int Y) {
        System.out.println("Differences WeightMatrix:");
        for (int i = 0; i < X; i++) {
            for (int j = 0; j < Y; j++) {
                System.out.format(" %+3.5f",this.differenceWeightMatrix[i][j]);
            }
            System.out.println();
        }
    }
*/
} // end of class mlp

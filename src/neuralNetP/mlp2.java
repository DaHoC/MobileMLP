/**
 *
 * @author Jan Hendriks
 */

package neuralNetP;

import java.lang.Math.*;

class mlp2 {

    // Using nested inner classes to avoid some inheritance issues
    public class layer {

        // Using nested inner classes to avoid some inheritance issues
        public class neuron {

            // Store the calculated net for each neuron here, first parameter is the current pattern
            private float weightedSum; // Store the calculated net for each neuron here, first parameter is the current pattern
            private int neuronId; // unique id, e.g. for PRNG order
            private float output; // Output of the neuron
            private float delta; // The delta of the delta rule, not the weight change

            private neuron (int id) {
                System.out.println( (id == 0) ? " > Creating BIAS neuron in current layer" : " > Creating neuron " + id);
                this.neuronId = id; // Set the current last neuron number in MLP
            }

            private void calculateWeightedSumAndOutput() {
                this.calculateWeightedSum();
                this.calculateOutput();
            }

            private void calculateWeightedSum() {
                if (this.neuronId == 0) {
                    this.weightedSum = 1; // If BIAS neuron
                } else {
                    if (layerId == 0) { // If input layer, the weighted sum and output is merely the input itself
                        this.weightedSum = inputValues[this.neuronId-1];
                    } else {
                        // Walk through all neurons in the above layer
                        neuron[] aboveLayerNeurons = layers[(layerId-1)].layerNeurons;
                        float weightedSumAccumulator = getWeightFromTo(layers[(layerId-1)].BIAS,this); // negative threshold, init net with Bias weight
//                        System.out.format(" 1      * W(  0>%3d) = %+3.5f * %+3.5f%n", this.neuronId, (float)1, weightedSumAccumulator);
//                        System.out.println(" (1 * " + weightedSumAccumulator + ")");
                        for (int i = 0; i < aboveLayerNeurons.length; i++) {
                            float weightToCurrentNeuronFromAboveNeuron = getWeightFromTo(aboveLayerNeurons[i],this);
                            float outputOfNeuronInAboveLayer = aboveLayerNeurons[i].getOutput();
//                            System.out.format(" O(%3d) * W(%3d>%3d) = %+3.5f * %+3.5f%n", aboveLayerNeurons[i].neuronId , aboveLayerNeurons[i].neuronId, this.neuronId, weightToCurrentNeuronFromAboveNeuron, outputOfNeuronInAboveLayer);
                            weightedSumAccumulator += weightToCurrentNeuronFromAboveNeuron * outputOfNeuronInAboveLayer;
/// @todo: removeme                            weightedSumAccumulator += weightMatrix[getRowOffsetInWeightMatrix()+i+1][(this.neuronId-getFirstNeuronId())] * aboveLayerNeurons[i].getOutput();
//                            System.out.println(" (" + aboveLayerNeurons[i].getOutput() + " * " + getWeightFromTo(aboveLayerNeurons[i],this) + ")");
                        }
                        this.weightedSum = weightedSumAccumulator;
                    }
//                    System.out.format("Calculated net for neuron %6d: %+3.5f%n", this.neuronId, this.weightedSum);
                }
            }

            private void calculateOutput() {
                if (layerId == 0) {
                    this.output = this.weightedSum; // If input layer, the weighted sum and output is merely the input itself (same for BIAS)
                } else {
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

            // The derivations of the given layer transfer function at z
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

            // This is the delta of the delta rule, not the weight change
            private void calculateDelta() {
                // Test for output layer
                if (layerId == (layerCount-1)) {
                    // (teacher y(m) - output y(m)) * f'(net)
                    this.delta = (teacherOutput[(this.neuronId-getFirstNeuronId())] - this.getOutput()) * this.getDerivationAt(this.getWeightedSum());
                } else if (layerId != 0) { // failsafe to exclude the input layer
                    float deltaAccumulator = 0;
                    // Walk through neurons in the below layer
                    for (int i = 0; i < layers[layerId+1].getLayerSize(); i++) {
                        // this sums up the weighted sum (=net)
                        deltaAccumulator += getWeightFromTo(this,layers[layerId+1].layerNeurons[i]) * layers[layerId+1].layerNeurons[i].getDelta();
//                        System.out.println(" delta+= w_"+this.neuronId+"_"+layers[layerId+1].layerNeurons[i].neuronId + " * delta("+ layers[layerId+1].layerNeurons[i].neuronId +") = " + getWeightFromTo(this,layers[layerId+1].layerNeurons[i]) + " * " + layers[layerId+1].layerNeurons[i].getDelta());
                    }
                    this.delta = deltaAccumulator * this.getDerivationAt(this.getWeightedSum());
//                    System.out.format(" f'(%+3.5f) = %+3.5f%n", this.getWeightedSum(), this.getDerivationAt(this.getWeightedSum()));
                }
//                System.out.format("Delta of neuron %3d in layer %3d: %+3.5f%n", this.neuronId, layerId, this.delta);
            }

            public short getLayerId() {
                return layerId;
            }

            public float getWeightedSum() {
                return this.weightedSum;
            }

            public float getOutput() {
                return this.output;
            }

            public float getDelta() {
                return this.delta;
            }
        }

        private neuron[] layerNeurons;
        private neuron BIAS;
        private short layerId; // layerId == 0 means input (=first) layer
        private short transferFunction; // 0 = tanh, 1 = fermi, 2 = id
        private int weightRowOffset; // row offset in the weight matrix in current layer
        private int firstNeuronIdInLayer;
        private float learningRate;

        // Constructor for first layer (no above layer exists) -> not allowed to be called directly, use mlp.addLayer() instead!
        private layer(int layerSize, short transf, float eta, short layerId) {
            System.out.println( ((layerId == 0) ? " > Creating input layer of size " + layerSize : " > Creating layer " + layerId + " of size " + layerSize)+ " with " + transf + " transfer function");
            this.layerId = layerId;
            this.transferFunction = transf;
            this.learningRate = eta;
            this.BIAS = new neuron(0); // BIAS neuron (each layer gets one, b/c neurons always affiliate to a layer)
            this.layerNeurons = new neuron[layerSize];
            this.firstNeuronIdInLayer = lastNeuronNumber;
            for (int i = 0; i < layerSize; i++) {
                this.layerNeurons[i] = new neuron(lastNeuronNumber++);
            }
            this.calculateRowOffsetInWeightMatrix();

        }

        // Converts w_ij to corresponding matrix position and returns x,y in an array
        private int[] getWeightMatrixPositionForNeuronWeightFromTo(neuron fromNeuron, neuron toNeuron) {
            int[] weightMatrixPosition = new int[2];
            if (fromNeuron.neuronId == 0) { // because from neuron 0 = from BIAS neuron, ambigious for each layer
                weightMatrixPosition[0] = layers[(toNeuron.getLayerId() - 1)].getRowOffsetInWeightMatrix();
            } else {
                weightMatrixPosition[0] = layers[fromNeuron.getLayerId()].getRowOffsetInWeightMatrix() + 1 + (fromNeuron.neuronId - layers[fromNeuron.getLayerId()].getFirstNeuronId());
            }
            weightMatrixPosition[1] = (toNeuron.neuronId - layers[toNeuron.getLayerId()].getFirstNeuronId());
            return weightMatrixPosition;
        }

        // Convert w_i_j to matrix row & column and return weight w_i_j from weightMatrix
        private float getWeightFromTo(neuron fromNeuron, neuron toNeuron) {
            // First calculate the position of the weight w_i_j in the weight matrix
            int[] weightMatrixPosition = this.getWeightMatrixPositionForNeuronWeightFromTo(fromNeuron, toNeuron);
            // Now return the weight
            return weightMatrix[weightMatrixPosition[0]][weightMatrixPosition[1]];
        }

        // Calculates the row offset in the weight matrix for each layer (recursive)
        private void calculateRowOffsetInWeightMatrix() {
            if (this.layerId < 1) {
                this.weightRowOffset = 0;
            } else {
                this.weightRowOffset = layers[(this.layerId-1)].getLayerSize() + 1 + layers[(this.layerId-1)].getRowOffsetInWeightMatrix();
           }
        }

        private int getRowOffsetInWeightMatrix() {
            return this.weightRowOffset;
        }

        private int getFirstNeuronId() {
            return this.firstNeuronIdInLayer;
        }

        // @WARNING: size including a bias neuron!
        public int getLayerSize() {
            return this.layerNeurons.length;
        }

        private float[] getLayerOutput() {
            float[] outputCollector = new float[this.getLayerSize()];
            for (int j = 0; j < this.getLayerSize(); j++) {
                outputCollector[j] = this.layerNeurons[j].getOutput();
            }
            return outputCollector;
        }

    }
/*
    public enum Transferfunction {
        id,
        fermi,
        tanh
    }
*/
    // 2-dim array = matrix (nearly in hinton diagram style)
    private float[][] weightMatrix;
    private float[][] differenceWeightMatrix; // For calculated weight changes
    private float[] inputValues; // x_1 .. x_N
    private float[] outputValues; // y_1 .. y_M
    private float[] teacherOutput;

    private int lastNeuronNumber = 1; // neurons of all layers have unique consecutive numbers (for id and PRNG)

    private short layerCount = 0;
    private layer[] layers = new layer[255]; // Assuming <255 layers (short)

    // Init with an input pattern and amount of input and output neurons
    public mlp2 (float[][] weightMatrixFromFile) {
        System.out.println(" > Initializing MLP");
        this.weightMatrix = weightMatrixFromFile;
        this.initDifferenceWeightMatrix();
    }

    // Sets input and initializes the MLP calculations (no lazy evaluation b/c all values will be referenced at some time)
    public void propagateInput(float[] inputVector) {
        this.inputValues = inputVector;
        // init calculation here, propagate input (pass input forward through MLP)
        for (int currentLayer = 0; currentLayer < this.layerCount; currentLayer++) {
            for (int n = 0; n < this.layers[currentLayer].getLayerSize(); n++) {
                this.layers[currentLayer].layerNeurons[n].calculateWeightedSumAndOutput();
            }
        }
    }

    // For input layer
    public void addLayer(int layerSize) {
        if (this.layerCount > 0) {
            System.err.println(" > Only one input layer allowed!");
        } else {
            this.addLayer(layerSize,(short)2,0); // 2 = transferfunction id
        }
    }

    // For hidden & output layers
    public void addLayer(int layerSize, short trans, float eta) {
        this.layers[this.layerCount] = new layer(layerSize,trans,eta, this.layerCount++);
    }

    // Init the difference weight matrix
    private void initDifferenceWeightMatrix() {
        this.differenceWeightMatrix = new float[this.weightMatrix.length][this.weightMatrix[0].length]; // Set same dimensions
/* unneeded
        for (int x = 0; x < this.weightMatrix.length; x++) {
            for (int y = 0; y < this.weightMatrix[0].length; y++) {
                this.differenceWeightMatrix[x][y] = 0;
            }
        }
 */
    }

    public void backpropagationOfDelta() {
        // Walk through layers backwards, starting at output layer
        for (int b = this.layerCount-1; b > 0; b--) {
            // Walk through all neurons in the layer and calculate delta
            for (int n = 0; n < this.layers[b].getLayerSize(); n++) {
                this.layers[b].layerNeurons[n].calculateDelta();
                layer aboveLayer = this.layers[b-1];
                for (int aboveLayerNeurons = 0; aboveLayerNeurons < aboveLayer.getLayerSize(); aboveLayerNeurons++) {
                    int[] weightPosition = aboveLayer.getWeightMatrixPositionForNeuronWeightFromTo(aboveLayer.layerNeurons[aboveLayerNeurons],this.layers[b].layerNeurons[n]);
                    // diff w_ij = eta_j * delta_j * o_i
                    this.differenceWeightMatrix[weightPosition[0]][weightPosition[1]] = this.layers[b].learningRate * this.layers[b].layerNeurons[n].getDelta() * aboveLayer.layerNeurons[aboveLayerNeurons].getOutput();
                }
            }
        }
    }

    // This is the update step, it adds difference weights to the current weights
    public void updateWeights() {
        for (int x = 0; x < this.weightMatrix.length; x++) {
            for (int y = 0; y < this.weightMatrix[0].length; y++) {
                this.weightMatrix[x][y] += this.differenceWeightMatrix[x][y];
            }
        }
    }

    public void setTeacherOutputVector(float[] teacherOutput) {
        this.teacherOutput = teacherOutput;
    }

    // Returns the error = 1/2 sum (teacher y - net y)^2 for one pattern
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

    public float[] getOutput() {
        if (this.layerCount > 0)
            return this.layers[this.layerCount-1].getLayerOutput();
        else
            return new float[0]; // Failsafe if e.g. only input layer exists
    }

    public void printOutput() {
        // Inits with the size of the output layer (M)
        float[] myNeuralNetOutput = this.getOutput();
        System.out.println("Output:");
        for (int j = 0; j < myNeuralNetOutput.length; j++) {
//            System.out.format(" %+3.5f",myNeuralNetOutput[j]);
        }
        System.out.println();
    }

    // Just for debugging
    public void printPatternInput() {
        System.out.println("Input:");
        for (int i = 0; i < this.inputValues.length; i++) {
//            System.out.format(" %+3.5f",this.inputValues[i]);
        }
    }

    // Just for debugging
    public void printWeightMatrix(int X, int Y) {
        System.out.println("WeightMatrix:");
        for (int i = 0; i < X; i++) {
            for (int j = 0; j < Y; j++) {
//                System.out.format(" %+3.5f",this.weightMatrix[i][j]);
            }
            System.out.println();
        }
    }

    // Just for debugging
    public void printDifferencesWeightMatrix(int X, int Y) {
        System.out.println("Differences WeightMatrix:");
        for (int i = 0; i < X; i++) {
            for (int j = 0; j < Y; j++) {
//                System.out.format(" %+3.5f",this.differenceWeightMatrix[i][j]);
            }
            System.out.println();
        }
    }

} // end of class mlp

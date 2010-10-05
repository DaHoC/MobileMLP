/**
 * This class contains a space- and time-efficient implementation for the weight entries
 * (a sparse matrix turned out to be way too memory-consuming [at least for embedded devices] and too time-consuming [according to the profiler] )
 */

package neuralNetP;
import java.util.Hashtable;

/**
 * Efficient and memory-saving implementation of weights between the neurons of one layer (per object) to other neurons anywhere in the mlp
 * @WARNING: many implicit casts, e.g. from Float to float, Integer to int and vice versa
 * @author Jan Hendriks
 */
public class weightsFromLayer {

    /**
     *  @WARNING: This deprecated data structure is for use with J2ME, use Hashmap in standard environment
     *  @SuppressWarnings("UseOfObsoleteCollectionType")
     */
    private Hashtable[] mapToNeuronWeights; // For each [FromNeuronI] own Hashtable
    private int neuronCount;

    /**
     * Constructor
     * @param _neuronCount how many neurons the current layer contains
     */
    weightsFromLayer(int _neuronCount) {
        this.neuronCount = _neuronCount;
        /** @WARNING: unchecked (weak) operations */
        mapToNeuronWeights = new Hashtable[this.neuronCount];
        // Init Hashtable
        for (int fromNeuron = 0; fromNeuron < this.neuronCount; fromNeuron++) {
            mapToNeuronWeights[fromNeuron] = new Hashtable();
        }
    }

    void setWeightFromLayerNeuronToNeuronId(int layerNeuron, int neuronId, float weight) {
        if (layerNeuron >= this.neuronCount) {
            System.out.println("Warning: layer size exceeded, adding array entry (there is an error in the code - probably the neuronCount parameter of weightsFromLayer() constructor is wrong)!");
            for (int j = this.neuronCount; j < layerNeuron; j++) {
                mapToNeuronWeights[j] = new Hashtable();
            }
            this.neuronCount = (layerNeuron-1);
        }
        mapToNeuronWeights[layerNeuron].put(new Integer(neuronId), new Float(weight));
    }

    float getWeightFromLayerNeuronToNeuronId(int layerNeuron, int neuronId) {
        // If the key is not in the hashtable return zero weight
        if ( (!mapToNeuronWeights[layerNeuron].containsKey(new Integer(neuronId))) || (layerNeuron >= this.neuronCount) )
            return 0f;
        Float ret = (Float)(mapToNeuronWeights[layerNeuron].get(new Integer(neuronId)));
        return ret.floatValue();
    }

}

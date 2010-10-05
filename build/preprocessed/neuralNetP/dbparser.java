/**
 * @author gawe design
 */
package neuralNetP;

import java.util.Hashtable;

public class dbparser {

    // Separator strings for CSV
    private static final String lineSeparator = "\n";
    private static final String itemSeparator = ";";
//    private Vector dbArray =
    private Hashtable places = new Hashtable();
    private Hashtable states = new Hashtable();

    // Constructor, takes the string read from the file and constructs an array out of it
    public dbparser(String dbString) {
        // First split per line (=per row)
        int index = 0;
        int lastIndex = 0;
        while ((index = dbString.indexOf(lineSeparator, lastIndex)) >= 0) {
            this.extractItemsFromLine(dbString.substring(lastIndex, index));
            lastIndex = index + lineSeparator.length();
        }
        // Add the last line
        this.extractItemsFromLine(dbString.substring(lastIndex, dbString.length()));
    }

    // The CSV String has the form:
    // A   ;Augsburg;(Bay)
    private void extractItemsFromLine(String line) {
        int firstIndex = line.indexOf(itemSeparator);
        String index = line.substring(0, firstIndex).trim().toLowerCase();
        System.out.println("Key: " + index);
        String place;
        String state = "";
        int secondIndex = line.indexOf(itemSeparator, firstIndex + 1);
        if (secondIndex > 0) {
            place = line.substring(firstIndex + 1, secondIndex).trim();
            System.out.println("Place: " + place);
            int thirdIndex = line.indexOf(itemSeparator, secondIndex + 1);
            if (thirdIndex > 0)
                state = line.substring(secondIndex + 1, thirdIndex).trim();
            else
                state = line.substring(secondIndex + 1, line.length()).trim();
            System.out.println("State: " + state);
        } else { // Special badges or no state given
            place = line.substring(firstIndex + 1, line.length()).trim();
            System.out.println("Place: " + place);
        }
        places.put(index, place);
        states.put(index, state);
    }

    // Retrieves the place from the given the badge
    public String getPlaceOfBadge(String badge) {
        badge = badge.trim().toLowerCase();
        return (this.places.containsKey(badge) ? this.places.get(badge).toString() : "");
    }

    // Retrieves the state from the given the badge
    public String getStateOfBadge(String badge) {
        badge = badge.trim().toLowerCase();
        return (this.states.containsKey(badge) ? this.states.get(badge).toString() : "");
    }
}

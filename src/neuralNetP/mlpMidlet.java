/**
 * @author gawe design
 */
package neuralNetP;

import java.io.IOException;
import java.io.InputStream;
import javax.microedition.lcdui.Form;
import java.util.Random.*;
import javax.microedition.lcdui.Command;
import javax.microedition.lcdui.CommandListener;
import javax.microedition.lcdui.Display;
import javax.microedition.lcdui.Displayable;
import javax.microedition.lcdui.Graphics;
import javax.microedition.midlet.*;

public class mlpMidlet extends MIDlet implements CommandListener {

    // Midlet-specific
    private Display display;
    private Graphics graphics;
    private Form form;
    private simpleImageCanvas canvas;
    private dbparser weights;
    private static final String SPLASH_IMAGE = "/images/gawesplash.png";
    private static final String WEIGHTS = "/database/weights.csv";
    private static final String TEACHERS = "/database/teacher.csv";
    private final Command exitCommand = new Command("Exit", Command.EXIT, 1);
    private final Command pause = new Command("Pause", Command.OK, 2);
//    private TextField badge = new TextField("-?-", "", 3, TextField.ANY);

    public mlpMidlet() {
        this.display = Display.getDisplay(this);
    }

    public void startApp() throws MIDletStateChangeException {
        if (display == null) {
            display = Display.getDisplay(this);
        }
        this.showSplash();
        initMIDlet();
    }

    // The entry point to the game canvas and the actual program
    protected void initMIDlet() {
        mlpCanvas c = new mlpCanvas(this);
        c.addCommand(exitCommand);
        c.setCommandListener(this);
        Display.getDisplay(this).setCurrent(c);
    }

    private dbparser loadDatabase(String database) {
        // Reading resources of JAR file
        InputStream inStream = this.getClass().getResourceAsStream(database);
        if (inStream == null) {
            return null;
//            throw new Error("Failed to load database '" + database + "': Not found!");
        }
        byte[] buf = new byte[14*1000]; // csv file is 13979 Bytes
        if (inStream != null) {
            try {
                int total = 0;
                while (true) {
                    int numRead = inStream.read(buf, total, buf.length - total);
                    if (numRead <= 0) {
                        break;
                    }
                    total += numRead;
                }
                byte[] bufferWithCorrectLength = new byte[total];
                System.arraycopy(buf, 0, bufferWithCorrectLength, 0, total);
                /// @TODO
                return new dbparser(new String(bufferWithCorrectLength));
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                try {
                    inStream.close();
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
            }
        }
        /// @TODO: Correct fallback mechanism
        return null;
    }

    private void showSplash() {
        canvas = new simpleImageCanvas(SPLASH_IMAGE);
        display = Display.getDisplay(this);
        canvas.setFullScreenMode(true);
        display.setCurrent(canvas);
        waitMilliseconds(1500); // Wait some time
    }

    private static void waitMilliseconds(long ms) {
        try {
            Thread.sleep(ms); // ms, e.g for pause to avoid cpu starvation
        } catch (Exception ex) {
            System.err.print("Thread backgrounding failed!");
        }
    }

    public void commandAction(Command c, Displayable d) {
        if (c == exitCommand) {
            try {
                destroyApp(false);
            } catch (MIDletStateChangeException ex) {
                ex.printStackTrace();
            }
            System.out.println("Exit");
        }
    }

    public void pauseApp() {
    }

    protected void destroyApp(boolean unconditional) throws MIDletStateChangeException {
        /// @TODO: Store weights
        exitMIDlet();
    }

    public void exitMIDlet() {
        notifyDestroyed();
    }
}

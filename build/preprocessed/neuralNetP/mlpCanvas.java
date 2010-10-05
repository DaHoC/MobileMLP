package neuralNetP;

import javax.microedition.lcdui.Alert;
import javax.microedition.lcdui.AlertType;
import javax.microedition.lcdui.Display;
import javax.microedition.lcdui.Graphics;
import javax.microedition.lcdui.Image;
import javax.microedition.lcdui.TextBox;
import javax.microedition.lcdui.TextField;
import javax.microedition.lcdui.game.GameCanvas;
import javax.microedition.media.Manager;
import javax.microedition.media.Player;
import javax.microedition.media.control.VideoControl;

public class mlpCanvas extends GameCanvas implements Runnable {

    // Neural net specific
    // The patterns from the text file are stored in here, each line corresponds to 1 pattern
//    private static int[] teacherPatternsOrder; // This contains the order in which the patterns are presented to the mlp
//    private static double[][] teacherInputPatterns;
//    private static double[][] teacherOutputPatterns;
//    private static double[][] weightMatrix; // This matrix stores the weights, row i=from neuron i, column j=to neuron j
    private static int N; // Number of input neurons
    private static int H; // Number of hidden neurons
//    private static int H2; // Number of hidden neurons in 2nd hidden layer
    private static int M; // Number of output neurons
//    private static int P; // Number of patterns
    private static int WX; // width of weightmatrix
    private static int WY; // height of weightmatrix
    private static java.util.Random prng; // The pseudo-random number generator (PRNG) object
//    private static double[] netProgress; // This contains the accumulated mlp error per epoche
//    private static String separator = ","; // Separator for the read files, e.g. "," for CSV
    private static final long seed = 3015; // Seed for the pseudo-random number generator
//    private static final int epocheCount = 1000; // Number of epoches the net calculates over
    private static byte[] NNresult;
    private static Image resultImage;
    private static Player m_objPlayer;
    private static VideoControl m_objVideoControl;
    private boolean isRunning = true; // If the thread should continue or terminate
    private Graphics g;
    private volatile Thread thread;
    private mlpMidlet fParent;
    private int canvasWidth;
    private int canvasHeight;
    private int sleepTime = 50;
    private static Image screenCaptureImage;
    private static mlp myMobileNeuralNet;

    public mlpCanvas() {
        super(true);
    }

    public mlpCanvas(mlpMidlet m) {
        super(true);
        this.fParent = m;
        this.setFullScreenMode(true);
        this.initScreen();

//        Thread runner = new Thread(this);

        // Call and init PRNG with seed
        prng = new java.util.Random();
        prng.setSeed(seed);

        TextBox textBox = new TextBox("Initialization", "Init camera", 64, TextField.ANY);
        Display display = Display.getDisplay(this.fParent);
        display.setCurrent(textBox);
        display.flashBacklight(10000);

        // Init mobile phone camera
        this.initCameraCapture();
        byte[] capturedata = this.getScreenCapture();
        screenCaptureImage = Image.createImage(capturedata, 0, capturedata.length);

        //        Image mlpInputImage = Image.createImage(screenCaptureImage, 0, 0, screenCaptureImage.getWidth()/2, screenCaptureImage.getHeight()/2, Sprite.TRANS_NONE);

        N = capturedata.length/100; // 4587
        H = capturedata.length/1000;
        M = 2;
//        H = 20;
//        H2 = this.canvasHeight * this.canvasWidth;
//        M = capturedata.length/8; // this.canvasHeight * this.canvasWidth;

        textBox.setString("Init ANN");
        System.out.println("Init MLP");
        display.flashBacklight(10000);
        myMobileNeuralNet = new mlp(); // Init with weights
        textBox.setString("Adding input layer");
        System.out.println("Adding input layer");
        myMobileNeuralNet.addLayer(N);
        System.out.println("Adding hidden layer");
        textBox.setString("Adding hidden layer");
        display.flashBacklight(10000);
        myMobileNeuralNet.addLayer(H, (short)0, 0.25f);
//        this.myMobileNeuralNet.addLayer(H2, (short)0, 0.25);
        System.out.println("Adding output layer");
        textBox.setString("Adding output layer");
        display.flashBacklight(10000);
        myMobileNeuralNet.addLayer(M, (short)2, 0.5f);
        textBox.setString("Init random weights with seed");
        display.flashBacklight(10000);
        myMobileNeuralNet.initWeightsWithRandomValues(prng, (float) -0.5, (float) 0.5);
        textBox.setString("ANN init done!");
        System.out.println("Setting up ANN done!");
        display.flashBacklight(10000);

//        netProgress = new double[epocheCount];

    }

    private void initScreen() {
        System.out.println("Init screen");
        // Retrieve the screen width and height
        this.canvasWidth = getWidth();
        this.canvasHeight = getHeight();
        // Set up graphics environment
        this.g = getGraphics();
        // Set the background color
        this.g.setColor(0, 0, 0);
        this.g.fillRect(0, 0, this.canvasWidth, this.canvasHeight);
    }

    //Use this method to initialize
    // m_Form is the displayed Form
    private void initCameraCapture() {
        System.out.println("Init camera");
        try {
            m_objPlayer = Manager.createPlayer("capture://video");
            m_objPlayer.realize();
            m_objVideoControl = (VideoControl) m_objPlayer.getControl("VideoControl");
            if (m_objVideoControl != null) {
                m_objVideoControl.initDisplayMode(VideoControl.USE_GUI_PRIMITIVE, null); // VideoControl.USE_DIRECT_VIDEO, this
//                Form m_Form.append((Item) m_objVideoControl.initDisplayMode(VideoControl.USE_GUI_PRIMITIVE, null));
                m_objPlayer.start();
            }
        } catch (Exception exc) {
            /// @TODO: handle Exception
            exc.printStackTrace();
        }
        System.out.println("done");
    }

    private Image createThumbnail(Image image) {
        int sourceWidth = image.getWidth();
        int sourceHeight = image.getHeight();
        int thumbWidth = 64;
        int thumbHeight = -1;
        if (thumbHeight == -1) {
            thumbHeight = thumbWidth * sourceHeight / sourceWidth;
        }
        Image thumb = Image.createImage(thumbWidth, thumbHeight);
        Graphics g1 = thumb.getGraphics();
        for (int y = 0; y < thumbHeight; y++) {
            for (int x = 0; x < thumbWidth; x++) {
                g1.setClip(x, y, 1, 1);
                int dx = x * sourceWidth / thumbWidth;
                int dy = y * sourceHeight / thumbHeight;
                g1.drawImage(image, x - dx, y - dy, Graphics.LEFT | Graphics.TOP);
            }
        }
        Image immutableThumb = Image.createImage(thumb);
        return immutableThumb;
    }

    private byte[] getScreenCapture() {
//        System.out.println("Screencapture");
// m_objVideoControl.getSnapshot("encoding=png&width= 80&height=60"));
// m_objVideoControl.getSnapshot("encoding=bmp&width= 160&height=120"));
//        Image image = Image.createImage(raw, 0, raw.length);
        byte[] data = null;
        try {
            /// @TODO: Still need to know the resolution!!
            data = m_objVideoControl.getSnapshot(null); // PNG, native resolution ??!?!
//            data = m_objVideoControl.getSnapshot("width=320&height=240");
//            data = m_objVideoControl.getSnapshot("encoding=jpeg");
//            Image image = Image.createImage(data, 0, data.length);
//            g.drawImage(image, 0, 0, Graphics.TOP);

//            data = m_objVideoControl.getSnapshot("encoding=jpeg&qualtity=100&width=320&height=240");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return data;
    }

    public float[] convertByteArrayToFloatArray(byte[] byteArray) {
        System.out.println("Array Size: " + byteArray.length/100);
        float[] doubleArray = new float[byteArray.length/100];
        for (int i = 0; i < byteArray.length/100; i++) {
            doubleArray[i] = (float) byteArray[i];
        }
        return doubleArray;
    }

    public byte[] convertFloatArrayToByteArray(float[] doubleArray) {
        byte[] byteArray = new byte[doubleArray.length];
        /// @TODO: Normalize to -128..127 (byte scope)
        for (int i = 0; i < doubleArray.length; i++) {
            if (doubleArray[i] > 127) {
                byteArray[i] = 127;
            } else if (doubleArray[i] < -128) {
                byteArray[i] = -128;
            } else {
                byteArray[i] = (byte)(doubleArray[i]*60);
            }
        }
        return byteArray;
    }
    public int[] convertFloatArrayToIntArray(float[] doubleArray) {
        int[] intArray = new int[doubleArray.length];
        for (int i = 0; i < doubleArray.length; i++) {
            intArray[i] = (int)(doubleArray[i]*10);
        }
        return intArray;
    }

    // Allows the main loop to run in a separate thread
    public void run() {
        Display.getDisplay(this.fParent).setCurrent(new TextBox("Running", "Running", 64, TextField.ANY));
        while (isRunning) {
            byte[] capturedata = this.getScreenCapture();
            screenCaptureImage = Image.createImage(capturedata, 0, capturedata.length);

            myMobileNeuralNet.propagateInput(convertByteArrayToFloatArray(capturedata));
            System.out.println("------ input:");
            for (int i = 0; i < capturedata.length; i++) {
                System.out.print(capturedata[i]+" ");
            }
            System.out.println();

            // Convert float to string
            Display.getDisplay(this.fParent).setCurrent(new TextBox("ANN result", "Getting ANN result", 64, TextField.ANY));
            float NNresult1[] = myMobileNeuralNet.getOutput();
            String NNresStrings[] = new String[2];
            System.out.println("Output size "+ NNresult1.length + ", strarr size " + NNresStrings.length);
            for (short i = 0; i < M; i++) {
                System.out.println(" res "+i + ":" + NNresult1[i]);
                NNresStrings[i] = new Float(NNresult1[i]).toString();
            }
            Display.getDisplay(this.fParent).setCurrent(new TextBox("ANN result", "ANN result obtained", 64, TextField.ANY));
//            int[] NNresult2 = convertFloatArrayToIntArray(myMobileNeuralNet.getOutput());
//            NNresult = convertFloatArrayToByteArray(myMobileNeuralNet.getOutput());
/*            System.out.println("------ res:");
            for (int i = 0; i < NNresult.length; i++) {
                System.out.print(NNresult[i]+" ");
            }
 * 
 */
//            resultImage = Image.createImage(NNresult, 0, NNresult.length);
//            g.drawRGB(NNresult2, 0, NNresult.length, 0, 0, this.canvasWidth, this.canvasHeight, false);
            
            int state = getKeyStates();

            if ((state & DOWN_PRESSED) != 0) {
                isRunning = false;
            } else { // if ((state & UP_PRESSED) != 0)
                Display.getDisplay(this.fParent).setCurrent(new TextBox("ANN result", NNresStrings[0] +"\n" +NNresStrings[1], 64, TextField.ANY));
                Display.getDisplay(this.fParent).flashBacklight(10000);
                Display.getDisplay(this.fParent).vibrate(100);
//                g.drawString(NNresStrings[0] + "\n" + NNresStrings[1], this.canvasWidth / 2, this.canvasHeight / 2, Graphics.VCENTER | Graphics.HCENTER);
//                g.drawImage(this.screenCaptureImage, this.canvasWidth/2, this.canvasHeight/2, Graphics.VCENTER | Graphics.HCENTER);
//                g.drawImage(resultImage, this.canvasWidth / 2, this.canvasHeight / 2, Graphics.VCENTER | Graphics.HCENTER);
//                g.drawRGB(capturedata, state, state, j, j, state, j, isRunning);
            }


            flushGraphics();
            screenCaptureImage = null;

            // Now wait...
            try {
                Thread.sleep(sleepTime);
            } catch (Exception e) {
                this.crashHandler(e);
            }
        }
    }

    // Just output the sh**
    private void crashHandler(Exception e) {
        g.setColor(255, 255, 255);
        Alert errorAlert = new Alert("Fatal error", e.getMessage(), null, AlertType.ERROR);
        errorAlert.setTimeout(Alert.FOREVER);
//        graphics.drawString(null, UP, UP, UP)setCurrent(errorAlert);
        e.printStackTrace();
        this.thread = null; // Destroy thread, bail out of mail loop
    }

    // When the canvas is shown, start a thread to
    // run the game loop.
    protected void showNotify() {
        thread = new Thread(this);
        thread.start();
    }


    /*
    for (int epoche = 0; epoche < epocheCount; epoche++) {
    netProgress[epoche] = 0; // The initial error of this epoche is 0
    // Shuffle the patterns so that the net does not memorize the order of the patterns
    shuffle(teacherPatternsOrder);
    // Walk through teacher patterns
    for (int pattern = 0; pattern < P; pattern++) {
    // Generate a new instance of the neural net and feed the current pattern as input
    myNeuralNet.setTeacherOutputVector(teacherOutputPatterns[teacherPatternsOrder[pattern]]);
    //                System.out.format("Feeding net with input net pattern %3d%n", pattern);
    myNeuralNet.propagateInput(teacherInputPatterns[teacherPatternsOrder[pattern]]);
    //                myNeuralNet.printPatternInput();
    //                myNeuralNet.printOutput();
    //                System.out.format(" > Overall pattern net output error: %+3.5f%n", myNeuralNet.getErrorForOnePattern());
    netProgress[epoche] += myNeuralNet.getErrorForOnePattern(); // F = sum E(p), Gesamtfehler ist die Summe über alle Einzelfehler der Muster p
    myNeuralNet.backpropagationOfDelta();
    //                myNeuralNet.printWeightMatrix(WX,WY);
    myNeuralNet.updateWeights();
    //                myNeuralNet.printDifferencesWeightMatrix(WX, WY);
    //                myNeuralNet.printWeightMatrix(WX,WY);
    }
    //            System.out.format(" > Overall accumulated epoche error: %+3.5f%n", netProgress[epoche]);
    }
     *
     */
//        cardLogic cLogic = new cardLogic(this.fParent); // Init the actual game logic
//        cLogic.setFullScreenMode(true); // There can be a bug in P900, see http://devlinslab.blogspot.com/2007/10/making-fullscreen-canvas.html

    /*
    form = new Form("Initialisiere MLP...");
    // Call and init PRNG with seed
    prng = new java.util.Random();
    prng.setSeed(seed);
    form.append("...lade Gewichte");

    N =
    H =
    WX = N + H + 2; // rows=N+H+2 (+1 BIAS for each layer)
    WY = Math.max(H,M); // The biggest layer of the two determines the count of columns
    weightMatrix = generateRandomWeightMatrix(WX,WY,-0.5,0.5);

    /*
    dbparser weightdb = this.loadDatabase(WEIGHTS);
    if (weightdb == null) {

    }
    form.append("...lade Teacher");
    //        this.loadDatabase(TEACHERS);
    display.setCurrent(form);
    this.refreshForm();
    form.append("");

    int iKey = 0;
    System.err.println("- Initializing move options");

    //        Display.getDisplay(this).setCurrent(cardLogic);
    while (true) {

    /*
    Display.getDisplay(this).setCurrent(gCanvas);


    //restore the clipping rectangle to full screen
    g.setClip(0, 0, getWidth(), getHeight());

    g.setColor(0xFF0000); //set drawing color to black
    //fill the whole screen
    g.fillRect(0, 0, getWidth(), getHeight());
    flushGraphics();
     */
//            Display.getDisplay(this.fParent).setCurrent(cLogic);
//            cLogic.makeMove();

    /*
    try {
    Thread.currentThread().join();
    } catch(Exception ex) {
    }
     */
//            helpAlgos.waitMilliseconds(30);
//        g = null;
    // Simple Fisher-Yates shuffle for use with teacher input patterns
    public static void shuffle(int[] array) {
        int n = array.length;            // The number of items left to shuffle (loop invariant).
        while (n > 1) {
            n--;                         // n is now the last pertinent index
            int k = prng.nextInt(n + 1);  // 0 <= k <= n.
            // Simple swap of variables
            int tmp = array[k];
            array[k] = array[n];
            array[n] = tmp;
        }
    }

    // Need an approximation of tanh (none in J2ME)
    public static double tanh(double x) {
        return (1 - 2 / (pow(Math.E, 2 * x) + 1));
    }

    // Need an approximation of power (none in J2ME), see
    // http://today.java.net/pub/a/today/2007/11/06/creating-java-me-math-pow-method.html
    public static double pow(double a, double b) {
        // true if base is greater than 1
        boolean gt1 = (Math.sqrt((a - 1) * (a - 1)) <= 1) ? false : true;
        int oc = -1; // used to alternate math symbol (+,-)
        int iter = 20; // number of iterations
        double p, x, x2, sumX, sumY;              // is exponent a whole number?
        if ((b - Math.floor(b)) == 0) {         // return base^exponent
            p = a;
            for (int i = 1; i < b; i++) {
                p *= a;
            }
            return p;
        }
        x = (gt1)
                ? (a / (a - 1)) : // base is greater than 1
                (a - 1); // base is 1 or less
        sumX = (gt1)
                ? (1 / x) : // base is greater than 1
                x; // base is 1 or less
        for (int i = 2; i < iter; i++) {
            // find x^iteration
            p = x;
            for (int j = 1; j < i; j++) {
                p *= x;
            }
            double xTemp = (gt1)
                    ? (1 / (i * p)) : // base is greater than 1
                    (p / i); // base is 1 or less
            sumX = (gt1)
                    ? (sumX + xTemp) : // base is greater than 1
                    (sumX + (xTemp * oc)); // base is 1 or less
            oc *= -1; // change math symbol (+,-)
        }
        x2 = b * sumX;
        sumY = 1 + x2; // our estimate
        for (int i = 2; i <= iter; i++) {
            // find x2^iteration
            p = x2;
            for (int j = 1; j < i; j++) {
                p *= x2;
            }
            // multiply iterations (ex: 3 iterations = 3*2*1)
            int yTemp = 2;
            for (int j = i; j > 2; j--) {
                yTemp *= j;
            }
            // add to estimate (ex: 3rd iteration => (x2^3)/(3*2*1) )
            sumY += p / yTemp;
        }
        return sumY; // return our estimate
    }
}

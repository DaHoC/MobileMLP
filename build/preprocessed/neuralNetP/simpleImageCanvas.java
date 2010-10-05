package neuralNetP;

import javax.microedition.lcdui.Canvas;
import javax.microedition.lcdui.Graphics;
import javax.microedition.lcdui.Image;

public class simpleImageCanvas extends Canvas implements Runnable {

    Image image = null;

    public simpleImageCanvas(String imageRessource) {
        try {
            this.image = Image.createImage(imageRessource);
            new Thread(this).start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void run() {
        while (true) {
            repaint();
            try {
                synchronized (this) {
                    wait(50L);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    protected void paint(Graphics g) {
        g.setColor(0xffffff);
        g.fillRect(0, 0, getWidth(), getHeight());
        if (image != null) {
            g.drawImage(image, getWidth() / 2, getHeight() / 2, Graphics.HCENTER | Graphics.VCENTER);
        } else {
            g.setColor(0x000000);
            g.drawString("No image available", getWidth() / 2, getHeight() / 2, Graphics.HCENTER | Graphics.BASELINE);
        }
    }
}

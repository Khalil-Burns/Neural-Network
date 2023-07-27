import javax.imageio.ImageIO;
import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) {
        Network n = new Network(new int[]{784, 30, 10}, 0.1, 30, 10);
        //Network n = new Network(new int[]{2, 2, 2}, 0.5, 1, 1);
        //n.train("bleeblah", "bloobloo");
        n.allTests = n.readImages("t10k-images.idx3-ubyte");
        n.allLabels = n.readLabels("t10k-labels.idx1-ubyte");
        n.train("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        //n.testWithPrintError();
        n.test();
        /*try{
            FileInputStream in = new FileInputStream("train-images.idx3-ubyte");

            int magicNumber = (in.read() << 24) + (in.read() << 16) + (in.read() << 8) + in.read();
            int numOfImages = (in.read() << 24) + (in.read() << 16) + (in.read() << 8) + in.read();
            int rows = (in.read() << 24) + (in.read() << 16) + (in.read() << 8) + in.read();
            int cols = (in.read() << 24) + (in.read() << 16) + (in.read() << 8) + in.read();
            byte[] vals = new byte[rows * cols * numOfImages];
            in.read(vals);
            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < rows; j++) {
                    for (int k = 0; k < cols; k++) {
                        double x;
                        x = ((int)(vals[(i * 784) + (j * cols + k)] & 0xFF) / 255.0);
                        if (x == 1.0) {
                            System.out.print("██");
                        }
                        else if (x > 0.85 && x < 1.0) {
                            System.out.print("▓▓");
                        }
                        else if (x <= 0.85 && x >= 0.15) {
                            System.out.print("▒▒");
                        }
                        else if (x > 0 && x < 0.15) {
                            System.out.print("░░");
                        }
                        else {
                            System.out.print("  ");
                        }
                    }
                    System.out.println();
                }
                System.out.println();
                System.out.println();
                System.out.println();
                System.out.println();
                System.out.println();
            }
            in.close();
            System.out.println("done reading images");
        }
        catch (Exception e) {System.out.println(e);}*/
        //n.train("sample6.png",6);
        /*try {
            n.train(ImageIO.read(new File("sample6.png")), 6);
        }
        catch(IOException e) {}*/
    }
}
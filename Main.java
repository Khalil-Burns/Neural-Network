import javax.imageio.ImageIO;
import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) {
        Network n = new Network(new int[]{784, 30, 10}, 0.1, 30, 10);
        n.train("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        n.test("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
        //n.train("sample6.png",6);
        /*try {
            n.train(ImageIO.read(new File("sample6.png")), 6);
        }
        catch(IOException e) {}*/
    }
}
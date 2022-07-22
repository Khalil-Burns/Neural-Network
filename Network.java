import java.awt.image.BufferedImage;
import java.io.*;
import java.awt.Color;
import java.util.*;

import javax.imageio.ImageIO;

public class Network {

    int[] layerSizes;
    int size;
	Matrix input;
    Matrix[] allInputs;
	Matrix[] hidIn;
    Matrix[] hidOut;
	Matrix outIn;
    Matrix outOut;
    Matrix target;

    Matrix[] biases;
    Matrix[] adjustB; //adjustments that need to be subtracted from the biases. This will be calculated after the feed forward
    Matrix[] deltaB;
    Matrix[] weights;
    Matrix[] adjustW;
    Matrix[] deltaW;

    double[] correct = new double[1];

    double learnRate = 0;
    int epochs, miniBatchSize;

	public Network(int[] sizes, double learningRate, int epoch, int miniBatch) {
        this.learnRate = learningRate;
        this.epochs = epoch;
        this.miniBatchSize = miniBatch;
        this.size = sizes.length;
        this.layerSizes = sizes;
		this.input = new Matrix(sizes[0], 1, false);
        this.hidIn = new Matrix[this.size - 2]; //size - (input and output layers)
        this.hidOut = new Matrix[this.size - 2];
        for (int i = 0; i < this.hidIn.length; i++) {
            this.hidIn[i] = new Matrix(sizes[i + 1], 1, false); //i + 1 because layer 0 is the input
            this.hidOut[i] = new Matrix(sizes[i + 1], 1, false);
        }
        this.outIn = new Matrix(sizes[this.size - 1], 1, false);
        this.outOut = new Matrix(sizes[this.size - 1], 1, false);
        this.target = new Matrix(sizes[this.size - 1], 1, false);

        this.weights = new Matrix[this.size - 1]; //all but the input layer
        this.biases = new Matrix[this.size - 1];

        this.adjustW = new Matrix[this.size - 1];
        this.adjustB = new Matrix[this.size - 1];

        this.deltaW = new Matrix[this.size - 1];
        this.deltaB = new Matrix[this.size - 1];
        for (int i = 0; i < this.size - 1; i++) {
            this.weights[i] = new Matrix(sizes[i + 1], sizes[i], true); //width = number of nodes on previous layers, height = number of nodes on current layer (remember that I have switched width/x and height/y values)
            this.biases[i] = new Matrix(sizes[i + 1], 1, true);

            this.adjustW[i] = new Matrix(sizes[i + 1], sizes[i], true);
            this.adjustB[i] = new Matrix(sizes[i + 1], 1, true);

            this.deltaW[i] = new Matrix(sizes[i + 1], sizes[i], true);
            this.deltaB[i] = new Matrix(sizes[i + 1], 1, true);
        }
        writeWeights(0, -1);
        writeBiases(0, -1);
	}

    /*public int num(BufferedImage img) {
        readImg(img);
        this.hidIn[0] = this.weights[0].multiply(this.input);
        this.hidIn[0]  = this.hidIn[0].add(this.biases[0]);
        this.hidOut[0]  = this.hidIn[0].sigmoid();

        this.hidIn[1] = this.weights[1].multiply(this.hidOut[0]);
        this.hidIn[1]  = this.hidIn[1].add(this.biases[1]);
        this.hidOut[1]  = this.hidIn[1].sigmoid();

        this.outIn = this.weights[2].multiply(this.hidOut[1]);
        this.outIn  = this.outIn.add(this.biases[2]);
        this.outOut  = this.outIn.sigmoid();

        int maxIdx = 0;
        double maxN = 0;
        for (int i = 0; i < 10; i++) {
            if (this.outOut.values[i][0] > maxN) {
                maxN = this.outOut.values[i][0];
                maxIdx = i;
            }
            
        }
        System.out.println(maxIdx);
        return(maxIdx);
    }*/
    public void test(String fileImg, String fileCor) {
        readImages(fileImg);
        correct = readLabels(fileCor);

        int numCorrect = 0, numCorrectFirst = 0;
            for (int loop = 0; loop < this.allInputs.length; loop++) {
                for (int loop2 = 0; loop2 < 1; loop2++) {
                this.input.values = this.allInputs[loop].values;
                /*for (int i = 0; i < 784; i++) {
                    //System.out.println(this.input.values[i][0] + " " + this.allInputs[loop].values[i][0]);
                }*/
                for (int hidden = 0; hidden < this.hidIn.length; hidden++) {
                    if (hidden == 0) {
                        this.hidIn[0] = this.weights[0].multiply(this.input);
                    }
                    else {
                        this.hidIn[hidden] = this.weights[hidden].multiply(this.hidOut[hidden - 1]);
                    }
                    this.hidIn[hidden]  = this.hidIn[hidden].add(this.biases[hidden]);
                    this.hidOut[hidden]  = this.hidIn[hidden].sigmoid();
                }

                this.outIn = this.weights[this.hidIn.length].multiply(this.hidOut[this.hidIn.length - 1]);
                this.outIn  = this.outIn.add(this.biases[this.hidIn.length]);
                this.outOut  = this.outIn.sigmoid();

                int maxIdx = 0;
                double maxN = 0, cost = 0, actual = 0;;
                for (int i = 0; i < 10; i++) {
                    if (i == correct[loop]) {
                        actual = 1.0;
                    }
                    else {
                        actual = 0;
                    }
                    this.target.values[i][0] = actual;
                    cost += square(this.outOut.values[i][0] - actual);
                    if (this.outOut.values[i][0] > maxN) {
                        maxN = this.outOut.values[i][0];
                        maxIdx = i;
                    }
                    double z = this.outIn.values[i][0];
                    System.out.print(i + ": "+ (Math.round(this.outOut.values[i][0] * 10.0) / 10.0) + " ");
                    //backProp((this.size - 2), i, ((2.0 * (this.outOut.values[i][0] - actual)) * dSigmoid(z)));
                }
                System.out.println("  Guessed: " + maxIdx + ", correct: "+ correct[loop]);   
                /*backProp(this.size - 1);
                for (int i = 0; i < this.size - 1; i++) {
                    this.weights[i] = this.weights[i].subtract(this.adjustW[i]);
                    this.biases[i] = this.biases[i].subtract(this.adjustB[i]);
                }*/
                /*if (loop2 == 4) {
                    if (maxIdx == (int)correct[loop]) {
                        numCorrect++;
                    }
                }
                if (loop2 == 0) {
                    if (maxIdx == (int)correct[loop]) {
                        numCorrectFirst++;
                    }
                }
                System.out.println(maxIdx + " " + correct[loop] + ",   first try: " + numCorrectFirst + "/" + loop + ",   after: " + numCorrect + "/" + loop);*/
                if (maxIdx == (int)correct[loop]) {
                    numCorrect++;
                }
                //System.out.println(maxIdx + " " + correct[loop] + ",   Correct: " + numCorrect + "/" + loop + " = " + (numCorrect / (double)loop) + ",   epoch: " + epoch);
                }
            }
            System.out.println("Correct: " + (numCorrect / (double)(this.allInputs.length / 100.0)));
        //System.out.println(maxIdx);
    }

    public void train(String fileImg, String fileCor) {
        readImages(fileImg);
        correct = readLabels(fileCor);
        for (int epoch = 0; epoch < this.epochs; epoch++) {
            int numCorrect = 0;
            shuffleAllInputs();
            for (int loop = 0; loop < this.allInputs.length; loop += this.miniBatchSize) {
                double cost = 0;
                Matrix error = new Matrix(this.layerSizes[this.size - 1], 1, false);
                for (int curBatch = 0; curBatch < this.miniBatchSize; curBatch++) {
                    this.input.values = this.allInputs[loop + curBatch].values;
                    for (int hidden = 0; hidden < this.hidIn.length; hidden++) {
                        if (hidden == 0) {
                            this.hidIn[0] = this.weights[0].multiply(this.input);
                        }
                        else {
                            this.hidIn[hidden] = this.weights[hidden].multiply(this.hidOut[hidden - 1]);
                        }
                        this.hidIn[hidden]  = this.hidIn[hidden].add(this.biases[hidden]);
                        this.hidOut[hidden]  = this.hidIn[hidden].sigmoid();
                    }

                    this.outIn = this.weights[this.hidIn.length].multiply(this.hidOut[this.hidIn.length - 1]);
                    this.outIn  = this.outIn.add(this.biases[this.hidIn.length]);
                    this.outOut  = this.outIn.sigmoid();

                    int maxIdx = 0;
                    double maxN = 0, actual = 0;
                    cost = 0.0;
                    for (int i = 0; i < this.outOut.values.length; i++) {
                        if (curBatch == 0) {
                            error.values[i][0] = 0;
                        }
                        if (i == correct[loop + curBatch]) {
                            actual = 1.0;
                        }
                        else {
                            actual = 0;
                        }
                        this.target.values[i][0] = actual;
                        cost += square(this.outOut.values[i][0] - actual);
                        if (this.outOut.values[i][0] > maxN) {
                            maxN = this.outOut.values[i][0];
                            maxIdx = i;
                        }
                        //error.values[i][0] += this.outOut.values[i][0] - this.target.values[i][0];
                        error.values[i][0] = this.outOut.values[i][0] - this.target.values[i][0];
                        /*if (curBatch == this.miniBatchSize - 1) {
                            error.values[i][0] /= (double)this.miniBatchSize;
                        }*/
                        //backProp((this.size - 2), i, ((2.0 * (this.outOut.values[i][0] - actual)) * dSigmoid(z)));
                    }
                    backProp(this.size - 1, error);
                    /*if (curBatch == 0) {
                        this.adjustW = this.deltaW;
                        this.adjustB = this.deltaB;
                    }
                    else {
                        this.adjustW = this.adjustW
                    }*/
                    /*if (loop2 == 4) {
                        if (maxIdx == (int)correct[loop]) {
                            numCorrect++;
                        }
                    }
                    if (loop2 == 0) {
                        if (maxIdx == (int)correct[loop]) {
                            numCorrectFirst++;
                        }
                    }
                    System.out.println(maxIdx + " " + correct[loop] + ",   first try: " + numCorrectFirst + "/" + loop + ",   after: " + numCorrect + "/" + loop);*/
                    if (maxIdx == (int)correct[loop + curBatch]) {
                        numCorrect++;
                    }
                    //System.out.println(maxIdx + " " + correct[loop] + ",   Correct: " + numCorrect + "/" + loop + " = " + (numCorrect / (double)loop) + ",   epoch: " + epoch);
                }
                //backProp(this.size - 1, error);
                for (int i = 0; i < this.size - 1; i++) {
                    this.weights[i] = this.weights[i].subtract(this.adjustW[i]);
                    this.biases[i] = this.biases[i].subtract(this.adjustB[i]);
                }
                /*if (loop % 100 == 0) {
                    System.out.println(loop);
                }*/
            }
            System.out.println("Correct: " + (numCorrect / (double)(this.allInputs.length / 100.0)) + ",   epoch: " + epoch);
            writeWeights(Math.min(epoch + 1, 3), (numCorrect / (double)this.allInputs.length));
            writeBiases(Math.min(epoch + 1, 3), (numCorrect/ (double)this.allInputs.length));
        }
        //System.out.println(maxIdx);
    }
    // public void backProp(int layer, Matrix error) {
    //     Matrix gradient = this.outOut.dSigmoid();
    //     gradient = Matrix.multiply(gradient, error);
    //     gradient = gradient.scale(this.learnRate);
    //     //System.out.println("Gradient: " + gradient.values.length + " " + gradient.values[0].length);

    //     Matrix hiddenT = this.hidOut[0].transpose();
    //     //System.out.println("hiddenT: " + hiddenT.values.length + " " + hiddenT.values[0].length);
    //     Matrix who_delta = Matrix.multiply(gradient, hiddenT);
    //     //System.out.println("who_delta: " + who_delta.values.length + " " + who_delta.values[0].length);

    //     this.weights[1] = this.weights[1].add(who_delta);
    //     //System.out.println("weights[1]: " + weights[1].values.length + " " + weights[1].values[0].length);
    //     this.biases[1] = this.biases[1].add(gradient);
    //     //System.out.println("biases[1]: " + biases[1].values.length + " " + biases[1].values[0].length);

    //     Matrix who_T = this.weights[1].transpose();
    //     //System.out.println("whoT: " + who_T.values.length + " " + who_T.values[0].length);
	// 	Matrix hidden_errors = Matrix.multiply(who_T, error);
    //     //System.out.println("hidden_errors: " + hidden_errors.values.length + " " + hidden_errors.values[0].length);
		
	// 	Matrix h_gradient = hidOut[0].dSigmoid();
    //     //System.out.println("h_gradient: " + h_gradient.values.length + " " + h_gradient.values[0].length);
	// 	h_gradient = h_gradient.hadamard(hidden_errors);
	// 	h_gradient = h_gradient.scale(this.learnRate);
    //     //System.out.println("h_gradient: " + h_gradient.values.length + " " + h_gradient.values[0].length);
		
	// 	Matrix i_T = this.input.transpose();
    //     //System.out.println("i_T: " + i_T.values.length + " " + i_T.values[0].length);
	// 	Matrix wih_delta = Matrix.multiply(h_gradient, i_T);
    //     //System.out.println("wih_delta: " + wih_delta.values.length + " " + wih_delta.values[0].length);
		
	// 	weights[0] = weights[0].add(wih_delta);
    //     //System.out.println("weights[0]: " + weights[0].values.length + " " + weights[0].values[0].length);
	// 	biases[0] = biases[0].add(h_gradient);
    //     //System.out.println("biases[0]: " + biases[0].values.length + " " + biases[0].values[0].length);
    // }
    public void backProp(int layer, Matrix error) {
        if (layer <= 0) { //layer 0 is input layer
            return;
        }
        if (layer == this.size - 1) {
            for (int i = 0; i < biases[layer - 1].values.length; i++) {
                this.adjustB[layer - 1].values[i][0] = error.values[i][0] * dSigmoid(this.outIn.values[i][0]) * this.learnRate;
                for (int j = 0; j < this.weights[layer - 1].values[i].length; j++) {
                    this.adjustW[layer - 1].values[i][j] = this.adjustB[layer - 1].values[i][0] * this.hidOut[layer - 2].values[j][0];
                }
            }
        }
        else {
            for (int i = 0; i < this.biases[layer - 1].values.length; i++) {
                double sum = 0;
                for (int j = 0; j < this.adjustB[layer].values.length; j++) {
                    sum += this.adjustB[layer].values[j][0] * this.weights[layer].values[j][i];
                }
                this.adjustB[layer - 1].values[i][0] = sum * dSigmoid(hidIn[layer - 1].values[i][0]);
                for (int j = 0; j < this.weights[layer - 1].values[i].length; j++) {
                    if (layer == 1) {
                        this.adjustW[layer - 1].values[i][j] = this.adjustB[layer - 1].values[i][0] * this.input.values[j][0];
                    }
                    else {
                        this.adjustW[layer - 1].values[i][j] = this.adjustB[layer - 1].values[i][0] * this.hidOut[layer - 2].values[j][0];
                    }
                }
            }
        }
        backProp(layer - 1, error);
    }
    /*public void backProp(int layer, int idx, double curAdjust) {
        if (layer < 0) {
            return;
        }
        double adjustments = curAdjust;
        
        if (layer == 0) {
            this.adjustB[layer].values[idx][0] = applyLRate(adjustments);
            for (int i = 0; i < this.input.values.length; i++) {
                this.adjustW[layer].values[idx][i] = applyLRate(adjustments * this.input.values[i][0]);
            }
        }
        else {
            this.adjustB[layer].values[idx][0] = applyLRate(adjustments);
            for (int i = 0; i < this.hidOut[layer - 1].values.length; i++) {
                this.adjustW[layer].values[idx][i] = applyLRate(adjustments * this.hidOut[layer - 1].values[i][0]);
                double z = this.hidIn[layer - 1].values[i][0];
                z = dSigmoid(z);
                backProp(layer - 1, i, adjustments * z);
            }
        }
    }*/
    /*public void backProp(int layer, int idx, double curAdjust) {
        Matrix error = 
        double adjustments = curAdjust;
        
        
    }*/

    public static double randn() {
        Random random = new Random();
        return(random.nextGaussian());
    }
    public static double sigmoid(double x) {
        return(1.0 / (1.0 + Math.exp(-x)));
    }
    public static double dSigmoid(double x) {
        return(sigmoid(x) * (1.0 - sigmoid(x)));
    }
    public static double relu(double x) {
        // double a = Math.max(0, x);
        // a = Math.min(1, a);
        // return(a);
        return(Math.max(0, x));
    }
    public static double percep(double x) {
        return(x<0? 0:1);
    }
    public static double dRelu(double x) {
        return(x<0? 0:1);
    }
    public static double square(double x) {
        return(x * x);
    }
    public double applyLRate(double x) {
        return(x * this.learnRate);
    }
    public void shuffleAllInputs() {
        int index;
        Random rnd = new Random();
        for (int i = this.allInputs.length - 1; i > 0; i--) {
            index = rnd.nextInt(i + 1);
            if (index != i) {
                Matrix temp;
                temp = this.allInputs[index];
                this.allInputs[index] = this.allInputs[i];
                this.allInputs[i] = temp;
                double temp2;
                temp2 = this.correct[index];
                this.correct[index] = this.correct[i];
                this.correct[i] = temp2;
            }
        }
    }
    public void readImages(String file) {
        try{
            FileInputStream in = new FileInputStream(file);

            int magicNumber = (in.read() << 24) + (in.read() << 16) + (in.read() << 8) + in.read();
            int numOfImages = (in.read() << 24) + (in.read() << 16) + (in.read() << 8) + in.read();
            int rows = (in.read() << 24) + (in.read() << 16) + (in.read() << 8) + in.read();
            int cols = (in.read() << 24) + (in.read() << 16) + (in.read() << 8) + in.read();
            this.allInputs = new Matrix[numOfImages];
            byte[] vals = new byte[rows * cols * numOfImages];
            in.read(vals);
            for (int i = 0; i < this.allInputs.length; i++) {
                this.allInputs[i] = new Matrix(rows * cols, 1, true);
                for (int j = 0; j < rows; j++) {
                    for (int k = 0; k < cols; k++) {
                        /*double value = in.read();
                        value /= 255.0;
                        this.allInputs[i].values[j * cols + k][0] = value;*/
                        this.allInputs[i].values[j * cols + k][0] = (int)(vals[(i * 784) + (j * cols + k)] & 0xFF) / 255.0;
                    }
                }
                //System.out.println(i);
            }
            in.close();
            System.out.println("done reading images");
        }
        catch (Exception e) {System.out.println(e);}
    }
    public double[] readLabels(String file) {
        try{
            double[] labels;
            FileInputStream in = new FileInputStream(file);
            int magicNumber = (in.read() << 24) + (in.read() << 16) + (in.read() << 8) + in.read();
            int numOfLabels = (in.read() << 24) + (in.read() << 16) + (in.read() << 8) + in.read();
            labels = new double[numOfLabels];
            for (int i = 0; i < numOfLabels; i++) {
                double value = in.read();
                labels[i] = value;
            }
            in.close();
            System.out.println("done reading labels");
            return(labels);
        }
        catch (Exception e) {
            System.out.println(e);
            return(null);
        }
    }
    public void readImg(String file) {
        BufferedImage img;
        try {
            img = ImageIO.read(new File("sample6.png"));
            double avgC = 0;
            Color c;
            for (int i = 0; i < img.getHeight(); i++) {
                for (int j = 0; j < img.getWidth(); j++) {
                    c = new Color(img.getRGB(j, i));
                    avgC = c.getRed() + c.getGreen() + c.getBlue();
                    avgC /= 3.0;
                    avgC /= 255.0;
                    //this.allInputs[0].values[i * img.getWidth() + j][0] = avgC;
                    this.input.values[i * img.getWidth() + j][0] = avgC;
                }
            }
        }
        catch(IOException e) {}
    }
    public void writeWeights(int epoch, double accuracy) {
        try {
            File weights = new File(("weights" + epoch + ".txt"));
            PrintWriter out = new PrintWriter(weights);
            out.println(accuracy + "");
            for (int i = 0; i < this.weights.length; i++) {
                out.println(this.weights[i].toString());
            }
            out.close();
        }
        catch (IOException e){}
    }
    public void writeBiases(int epoch, double accuracy) {
        try {
            File biases = new File(("biases" + epoch + ".txt"));
            PrintWriter out = new PrintWriter(biases);
            out.println(accuracy + "");
            for (int i = 0; i < this.biases.length; i++) {
                out.println(this.biases[i].toString());
            }
            out.close();
        }
        catch (IOException e){}
    }

    public void train(String fileImg, double correct) {
        readImg(fileImg);
        int numCorrect = 0, numCorrectFirst = 0;
        for (int loop2 = 0; loop2 < 100; loop2++) {
            //this.input.values = this.allInputs[loop].values;
            /*for (int i = 0; i < 784; i++) {
                //System.out.println(this.input.values[i][0] + " " + this.allInputs[loop].values[i][0]);
            }*/
            for (int hidden = 0; hidden < this.hidIn.length; hidden++) {
                if (hidden == 0) {
                    this.hidIn[0] = this.weights[0].multiply(this.input);
                }
                else {
                    this.hidIn[hidden] = this.weights[hidden].multiply(this.hidOut[hidden - 1]);
                }
                this.hidIn[hidden]  = this.hidIn[hidden].add(this.biases[hidden]);
                this.hidOut[hidden]  = this.hidIn[hidden].sigmoid();
            }
            this.outIn = this.weights[this.hidIn.length].multiply(this.hidOut[this.hidIn.length - 1]);
            this.outIn  = this.outIn.add(this.biases[this.hidIn.length]);
            this.outOut  = this.outIn.sigmoid();

            int maxIdx = 0;
            double maxN = 0, cost = 0, actual = 0;;
            for (int i = 0; i < 10; i++) {
                if (i == correct) {
                    actual = 1.0;
                }
                else {
                    actual = 0;
                }
                this.target.values[i][0] = actual;
                cost += square(this.outOut.values[i][0] - actual);
                if (this.outOut.values[i][0] > maxN) {
                    maxN = this.outOut.values[i][0];
                    maxIdx = i;
                }
                System.out.print(i + ": "+ (Math.round(this.outOut.values[i][0] * 1000.0) / 1000.0) + " ");
                //backProp((this.size - 2), i, ((2.0 * (this.outOut.values[i][0] - actual)) * dSigmoid(z)));
            }
            System.out.println("  Guessed: " + maxIdx + ", correct: "+ correct); 
            System.out.print((Math.round(cost * 100000.0) / 100000.0) + "   ");
                //backProp((this.size - 2), i, ((2.0 * (this.outOut.values[i][0] - actual)) * dSigmoid(z)));
            backProp(this.size - 1, this.outOut.subtract(this.target));
            for (int i = 0; i < this.size - 1; i++) {
                this.weights[i] = this.weights[i].subtract(this.adjustW[i]);
                this.biases[i] = this.biases[i].subtract(this.adjustB[i]);
            }
            /*if (loop2 == 4) {
                if (maxIdx == (int)correct[loop]) {
                    numCorrect++;
                }
            }
            if (loop2 == 0) {
                if (maxIdx == (int)correct[loop]) {
                    numCorrectFirst++;
                }
            }
            System.out.println(maxIdx + " " + correct[loop] + ",   first try: " + numCorrectFirst + "/" + loop + ",   after: " + numCorrect + "/" + loop);*/
            //System.out.println(maxIdx + " " + correct[loop] + ",   Correct: " + numCorrect + "/" + loop + " = " + (numCorrect / (double)loop) + ",   epoch: " + epoch);
        }
        //System.out.println(maxIdx);
    }
}